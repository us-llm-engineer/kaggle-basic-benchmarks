"""Methodology 8: nested calibrated character TF-IDF linear SVM.

Every outer-fold prediction is made by a vocabulary, margin model and sigmoid
calibrator fitted without that fold.  This is CPU work; no GPU is requested.
"""
import json
import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC

import spooky_common as S

CFG = dict(ngram_range=(2, 5), min_df=2, sublinear_tf=True, max_features=180000,
           c_grid=(0.25, 0.75, 1.5), calibration_folds=3)
FOLD_WORKERS = int(os.environ.get("CPU_FOLD_WORKERS", min(S.N_FOLDS, os.cpu_count() or 1)))
RUNS = os.path.join(S.RESULTS, "char_svm_runs")


def _vectorizer():
    return TfidfVectorizer(analyzer="char", lowercase=False, ngram_range=CFG["ngram_range"],
                           min_df=CFG["min_df"], sublinear_tf=CFG["sublinear_tf"], max_features=CFG["max_features"])


def choose_c(text_train, y_train):
    """Choose C by inner calibrated log loss; each split owns its vocabulary."""
    scores, inner = [], StratifiedKFold(3, shuffle=True, random_state=17)
    for c in CFG["c_grid"]:
        losses = []
        for fit, check in inner.split(text_train, y_train):
            vec = _vectorizer(); x_fit = vec.fit_transform(text_train[fit]); x_check = vec.transform(text_train[check])
            cal = CalibratedClassifierCV(LinearSVC(C=c), method="sigmoid", cv=CFG["calibration_folds"], n_jobs=1)
            cal.fit(x_fit, y_train[fit])
            losses.append(log_loss(y_train[check], cal.predict_proba(x_check), labels=[0, 1, 2]))
        scores.append(float(np.mean(losses)))
    best = int(np.argmin(scores))
    return float(CFG["c_grid"][best]), scores


def fit_predict(text_train, y_train, text_val, text_test):
    """Fit all learned objects on text_train only, then return normalised probabilities."""
    best_c, inner_logloss = choose_c(text_train, y_train)
    vec = _vectorizer(); x_train = vec.fit_transform(text_train)
    # CalibratedClassifierCV creates its own stratified margin folds from only the outer training rows.
    cal = CalibratedClassifierCV(LinearSVC(C=best_c), method="sigmoid", cv=CFG["calibration_folds"], n_jobs=1)
    cal.fit(x_train, y_train)
    return S.clip_norm(cal.predict_proba(vec.transform(text_val))), S.clip_norm(cal.predict_proba(vec.transform(text_test))), dict(C=best_c, inner_logloss=inner_logloss, n_features=int(x_train.shape[1]))


def run_fold(k):
    train, test, folds, y = S.load(); txt = train.text.values; tr, va = folds != k, folds == k
    pv, ptest, info = fit_predict(txt[tr], y[tr], txt[va], test.text.values)
    os.makedirs(RUNS, exist_ok=True)
    np.savez(os.path.join(RUNS, f"f{k}.npz"), va=np.where(va)[0], pv=pv, pt=ptest)
    with open(os.path.join(RUNS, f"f{k}.json"), "w") as f: json.dump(dict(fold=k, **info), f, indent=2)
    return k, np.where(va)[0], pv, ptest, info


def completed_fold(k):
    path = os.path.join(RUNS, f"f{k}.npz")
    if not os.path.exists(path): return None
    z = np.load(path); info = json.load(open(os.path.join(RUNS, f"f{k}.json")))
    return k, z["va"], z["pv"], z["pt"], info


def main():
    train, test, folds, y = S.load(); oof = np.zeros((len(train), 3)); pt = 0; meta = []
    pending = [k for k in range(S.N_FOLDS) if completed_fold(k) is None]
    finished = [completed_fold(k) for k in range(S.N_FOLDS) if completed_fold(k) is not None]
    with ProcessPoolExecutor(max_workers=min(FOLD_WORKERS, len(pending) or 1), mp_context=get_context("spawn")) as ex:
        finished.extend(ex.map(run_fold, pending))
    for k, va, pv, ptest, info in sorted(finished):
            oof[va] = pv; pt = pt + ptest / S.N_FOLDS; meta.append(dict(fold=k, **info)); print(f"fold {k}: C={info['C']:.2f}, features={info['n_features']}", flush=True)
    ev = S.evaluate(y, oof, folds)
    S.save_method("char_svm", oof, pt, dict(method="nested log-loss-selected sigmoid-calibrated character TF-IDF linear SVM", config=CFG, fold_workers=FOLD_WORKERS, folds=meta, **ev))
    print(f"char_svm log loss {ev['log_loss']:.4f} {np.round(ev['log_loss_ci95'],4)} accuracy {ev['accuracy']:.4f}")


if __name__ == "__main__":
    main()
