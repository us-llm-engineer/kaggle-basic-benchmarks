"""Shared helpers for the Spooky Author Identification study: data, fixed folds, multiclass metrics, a paired bootstrap on
log loss, a threaded fold runner, and storage of out-of-fold / test probabilities."""
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import StratifiedKFold

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "results")
AUTHORS = ["EAP", "HPL", "MWS"]  # Edgar Allan Poe, H. P. Lovecraft, Mary Shelley (the sample-submission column order)
N_FOLDS = 5


def load():
    """Training frame with integer labels `y`, the test frame, and the fixed five-fold assignment (created on first use)."""
    train = pd.read_csv(os.path.join(ROOT, "data/train.csv")); test = pd.read_csv(os.path.join(ROOT, "data/test.csv"))
    y = train.author.map({a: i for i, a in enumerate(AUTHORS)}).values
    path = os.path.join(RESULTS, "folds.csv"); os.makedirs(RESULTS, exist_ok=True)
    if not os.path.exists(path):
        f = np.zeros(len(train), int)
        for k, (_, va) in enumerate(StratifiedKFold(N_FOLDS, shuffle=True, random_state=42).split(train, y)): f[va] = k
        pd.DataFrame({"id": train.id, "fold": f}).to_csv(path, index=False)
    folds = pd.read_csv(path); assert (folds["id"].values == train["id"].values).all()
    return train, test, folds["fold"].values, y


def clip_norm(p, eps=1e-15):
    """Clip and renormalise probabilities, as the competition metric does."""
    p = np.clip(p, eps, 1 - eps); return p / p.sum(1, keepdims=True)


def sample_loss(y, p):
    return -np.log(clip_norm(p)[np.arange(len(y)), y])


def evaluate(y, p, folds, n_boot=1000, seed=42):
    """Multiclass log loss (with a bootstrap 95% interval over sentences), accuracy, macro F1, per-author F1, per-fold log loss."""
    p = clip_norm(p); nll = sample_loss(y, p); rng = np.random.default_rng(seed)
    boots = np.array([nll[rng.integers(0, len(y), len(y))].mean() for _ in range(n_boot)]); pred = p.argmax(1)
    return dict(log_loss=float(nll.mean()), log_loss_ci95=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                accuracy=float(accuracy_score(y, pred)), macro_f1=float(f1_score(y, pred, average="macro")),
                f1_per_author=dict(zip(AUTHORS, map(float, f1_score(y, pred, average=None)))),
                fold_log_loss=[float(nll[folds == k].mean()) for k in range(int(folds.max()) + 1)])


def paired_bootstrap_logloss(y, p_a, p_b, n=4000, seed=0):
    """Paired bootstrap of mean log loss difference (a - b) over sentences: (mean, 95% interval, share of resamples with a - b >= 0)."""
    d = sample_loss(y, p_a) - sample_loss(y, p_b); rng = np.random.default_rng(seed)
    m = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return float(d.mean()), (float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))), float((m >= 0).mean())


def cv_run(fit_predict, n_train, folds, n_classes=3, n_jobs=5):
    """Run `fit_predict(tr_idx, va_idx) -> (val_probs, test_probs)` over the folds (one thread per fold); returns the out-of-fold
    probabilities and the fold-averaged test probabilities."""
    from concurrent.futures import ThreadPoolExecutor
    nf = int(folds.max()) + 1; splits = [(np.where(folds != k)[0], np.where(folds == k)[0]) for k in range(nf)]
    with ThreadPoolExecutor(max_workers=min(n_jobs, nf)) as ex: outs = list(ex.map(lambda s: fit_predict(*s), splits))
    oof = np.zeros((n_train, n_classes)); tsum = 0
    for (tr, va), (pv, pt) in zip(splits, outs): oof[va] = pv; tsum = tsum + pt
    return oof, tsum / nf


def save_method(name, oof, test, meta):
    os.makedirs(RESULTS, exist_ok=True)
    np.save(os.path.join(RESULTS, f"oof_{name}.npy"), np.asarray(oof, np.float64)); np.save(os.path.join(RESULTS, f"test_{name}.npy"), np.asarray(test, np.float64))
    json.dump(meta, open(os.path.join(RESULTS, f"meta_{name}.json"), "w"), indent=2, default=float)


def load_method(name):
    return (np.load(os.path.join(RESULTS, f"oof_{name}.npy")), np.load(os.path.join(RESULTS, f"test_{name}.npy")),
            json.load(open(os.path.join(RESULTS, f"meta_{name}.json"))))


def apply_temperature(p, T):
    """Temperature scaling of probabilities: softmax(log p / T)."""
    z = np.log(clip_norm(p)) / T; z -= z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


def fit_temperature(p, y):
    """Temperature T > 0 minimising the log loss of softmax(log p / T) on (p, y)."""
    from scipy.optimize import minimize_scalar
    return float(minimize_scalar(lambda t: sample_loss(y, apply_temperature(p, t)).mean(), bounds=(0.3, 4.0), method="bounded").x)


def cv_temperature(p_oof, y, folds, p_test=None):
    """Honest temperature scaling: each fold's OOF probabilities are rescaled with a T fitted on the other folds' OOF probabilities.
    Returns (calibrated OOF, per-fold T, and the test probabilities rescaled with T fitted on all OOF rows if p_test is given)."""
    out, Ts = np.zeros_like(p_oof), []
    for k in range(int(folds.max()) + 1):
        tr, va = folds != k, folds == k; T = fit_temperature(p_oof[tr], y[tr]); Ts.append(T); out[va] = apply_temperature(p_oof[va], T)
    test_cal = None if p_test is None else apply_temperature(p_test, fit_temperature(p_oof, y))
    return out, Ts, test_cal
