"""Shared helpers for the Disaster Tweets study: data, text normalisation, the fixed folds, NB log-count ratios,
decision-threshold selection, a paired bootstrap for F1 differences, and storage of out-of-fold / test predictions."""
import json
import os

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import f1_score

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "results")
N_FOLDS = 5


def load():
    """Training frame, test frame, and the fixed fold assignment saved by the baseline notebook."""
    train = pd.read_csv(os.path.join(ROOT, "data/train.csv"))
    test = pd.read_csv(os.path.join(ROOT, "data/test.csv"))
    folds = pd.read_csv(os.path.join(RESULTS, "folds.csv"))
    assert (folds["id"].values == train["id"].values).all()
    return train, test, folds["fold"].values


def norm(t):
    """Lower-case, replace URLs and @mentions by the tokens `xurl` / `xuser`, drop everything but a-z 0-9 # '."""
    t = t.str.lower().str.replace(r"https?://\S+", " xurl ", regex=True).str.replace(r"@\w+", " xuser ", regex=True)
    return t.str.replace("&amp;", "&", regex=False).str.replace(r"[^a-z0-9#' ]", " ", regex=True)


def model_text(df):
    """Normalised text with the keyword prepended as the special token `kw_<keyword>`."""
    return "kw_" + df.keyword.fillna("none").str.replace("%20", "_") + " " + norm(df.text)


def nb_log_count_ratio(X, y, alpha=1.0):
    """Wang & Manning (2012): r = log((p/|p|_1) / (q/|q|_1)), p = alpha + sum of positive rows, q = alpha + sum of negative rows."""
    p = alpha + np.asarray(X[y == 1].sum(axis=0)).ravel()
    q = alpha + np.asarray(X[y == 0].sum(axis=0)).ravel()
    return np.log((p / np.abs(p).sum()) / (q / np.abs(q).sum()))


def best_threshold(y, p, grid=None):
    """Threshold on `p` that maximises F1 of the positive class; `p` may be any monotone score (probability or margin)."""
    grid = np.quantile(p, np.linspace(0.2, 0.8, 121)) if grid is None else grid
    f = [f1_score(y, p > t) for t in grid]
    i = int(np.argmax(f))
    return float(grid[i]), float(f[i])


def paired_bootstrap_f1(y, pred_a, pred_b, n=2000, seed=0):
    """Paired bootstrap over tweets of F1(a) - F1(b) for two sets of hard predictions (vectorised over resamples).
    Returns (mean difference, 95% interval, share of resamples with difference <= 0)."""
    rng = np.random.default_rng(seed); y = np.asarray(y).astype(bool); a = np.asarray(pred_a).astype(bool); b = np.asarray(pred_b).astype(bool)
    idx = rng.integers(0, len(y), (n, len(y))); yy = y[idx]
    f1 = lambda p: (lambda pp: 2 * (pp & yy).sum(1) / np.maximum(2 * (pp & yy).sum(1) + (pp & ~yy).sum(1) + (~pp & yy).sum(1), 1))(p[idx])
    d = f1(a) - f1(b); lo, hi = np.percentile(d, [2.5, 97.5])
    return float(d.mean()), (float(lo), float(hi)), float((d <= 0).mean())


def save_method(name, oof, test, meta):
    """Store out-of-fold scores, fold-averaged test scores and a small metadata record for one method."""
    os.makedirs(RESULTS, exist_ok=True)
    np.save(os.path.join(RESULTS, f"oof_{name}.npy"), np.asarray(oof, dtype=np.float64))
    np.save(os.path.join(RESULTS, f"test_{name}.npy"), np.asarray(test, dtype=np.float64))
    json.dump(meta, open(os.path.join(RESULTS, f"meta_{name}.json"), "w"), indent=2, default=float)


def load_method(name):
    return (np.load(os.path.join(RESULTS, f"oof_{name}.npy")), np.load(os.path.join(RESULTS, f"test_{name}.npy")),
            json.load(open(os.path.join(RESULTS, f"meta_{name}.json"))))


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -30, 30)))


def cv_run(fit_predict, n_train, folds, n_jobs=5):
    """Run `fit_predict(tr_idx, va_idx) -> (val_scores, test_scores)` over the fixed folds (one thread per fold; the linear
    solvers release the GIL). Returns the out-of-fold scores and the test scores averaged over the fold models."""
    from concurrent.futures import ThreadPoolExecutor
    nf = int(folds.max()) + 1
    splits = [(np.where(folds != k)[0], np.where(folds == k)[0]) for k in range(nf)]
    with ThreadPoolExecutor(max_workers=min(n_jobs, nf)) as ex:
        outs = list(ex.map(lambda s: fit_predict(*s), splits))
    oof = np.zeros(n_train); test_sum = 0
    for (tr, va), (sv, st) in zip(splits, outs):
        oof[va] = sv; test_sum = test_sum + st
    return oof, test_sum / nf


def evaluate(y, score, folds, n_boot=500, seed=42):
    """Threshold-free metrics, the F1-optimal threshold on out-of-fold scores, F1 with a bootstrap interval, per-fold F1."""
    from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score
    th, f1 = best_threshold(y, score)
    pred = (score > th).astype(int); rng = np.random.default_rng(seed); b = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y)); b.append(f1_score(y[i], pred[i]))
    return dict(auc=float(roc_auc_score(y, score)), ap=float(average_precision_score(y, score)), threshold=th, f1=f1,
                f1_ci95=[float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))], precision=float(precision_score(y, pred)),
                recall=float(recall_score(y, pred)), fold_f1=[float(f1_score(y[folds == k], pred[folds == k])) for k in range(int(folds.max()) + 1)])
