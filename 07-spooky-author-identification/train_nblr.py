"""Resumable training of methodology 1 (Naive-Bayes-weighted n-gram logistic regression, one-vs-rest) on the five fixed folds.
Each (C, fold, class) fit is saved on completion to results/nblr_runs/; a re-invocation skips finished fits, and no new batch is started
once the time budget is used, so one invocation stays short:   python3 train_nblr.py [budget_seconds]
Requires the cache from spooky_features.py."""
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression

import spooky_common as S
from spooky_features import CACHE

C_GRID = [10, 30, 100]
RUNS = os.path.join(S.RESULTS, "nblr_runs")
_loaded, _lock = {}, threading.Lock()


def nb_ratio(X, yb, alpha=1.0):
    """Wang & Manning (2012) log-count ratio of the positive against the negative class for features X (rows = documents)."""
    p = alpha + np.asarray(X[yb == 1].sum(0)).ravel(); q = alpha + np.asarray(X[yb == 0].sum(0)).ravel()
    return np.log((p / p.sum()) / (q / q.sum()))


def fold_data(k):
    with _lock:
        if k not in _loaded: _loaded[k] = [sparse.load_npz(os.path.join(CACHE, f"f{k}_{n}.npz")).tocsr() for n in ("train", "val", "test")]
    return _loaded[k]


def run_job(job, y, folds):
    C, k, c = job; Xa, Xv, Xt = fold_data(k); tr = np.where(folds != k)[0]; yb = (y[tr] == c).astype(int)
    r = nb_ratio(Xa, yb); m = LogisticRegression(C=C, solver="liblinear", max_iter=200).fit(Xa.multiply(r).tocsr(), yb)
    np.savez(os.path.join(RUNS, f"C{C}_f{k}_c{c}.npz"), val=m.predict_proba(Xv.multiply(r).tocsr())[:, 1], test=m.predict_proba(Xt.multiply(r).tocsr())[:, 1], n_iter=m.n_iter_)


def assemble(C, folds, n_test):
    """Normalised one-vs-rest probabilities for a given C: (out-of-fold, fold-averaged test); None if fits are missing."""
    oof, tst = np.zeros((len(folds), 3)), np.zeros((n_test, 3))
    for k in range(5):
        va = np.where(folds == k)[0]
        for c in range(3):
            p = os.path.join(RUNS, f"C{C}_f{k}_c{c}.npz")
            if not os.path.exists(p): return None
            z = np.load(p); oof[va, c] = z["val"]; tst[:, c] += z["test"] / 5
    return oof / oof.sum(1, keepdims=True), tst / tst.sum(1, keepdims=True)


def main(budget):
    t0 = time.time(); train, test, folds, y = S.load(); os.makedirs(RUNS, exist_ok=True)
    todo = [(C, k, c) for k in range(5) for C in C_GRID for c in range(3) if not os.path.exists(os.path.join(RUNS, f"C{C}_f{k}_c{c}.npz"))]  # fold-major: few folds loaded per invocation
    print(f"{len(todo)} fits left")
    for i in range(0, len(todo), 3):
        if time.time() - t0 > budget: print(f"budget reached with {len(todo) - i} fits left: run again to continue"); return False
        with ThreadPoolExecutor(3) as ex: list(ex.map(lambda j: run_job(j, y, folds), todo[i:i + 3]))
        print(f"  {min(i + 3, len(todo))}/{len(todo)} fits done, {time.time() - t0:.0f}s elapsed")
    print("all fits complete"); return True


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 50.0)
