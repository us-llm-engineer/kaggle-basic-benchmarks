"""Fit one character language model per author on each fold's training sentences and score that fold's held-out sentences and the test sentences
at every order 1..N. One process per fold:   python3 train_charlm.py [N]     (default N = 7)
Writes results/charlm_runs/f{k}.npz with the per-sentence log-likelihoods, shape (sentences, authors, orders)."""
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np

import spooky_charlm as CL
import spooky_common as S

RUNS = os.path.join(S.RESULTS, "charlm_runs"); CHUNK = 3000


def score(lms, texts):
    out = np.zeros((len(texts), len(lms), lms[0].N))
    for a, lm in enumerate(lms):
        for i in range(0, len(texts), CHUNK): out[i:i + CHUNK, a] = lm.loglik(texts[i:i + CHUNK])
    return out


def run_fold(k, N):
    t0 = time.time(); train, test, folds, y = S.load(); txt = train.text.values; cmap = CL.char_map(list(txt) + list(test.text.values))
    tr, va = np.where(folds != k)[0], np.where(folds == k)[0]; lms = [CL.CharLM(list(txt[tr[y[tr] == a]]), cmap, N) for a in range(3)]
    os.makedirs(RUNS, exist_ok=True); np.savez(os.path.join(RUNS, f"f{k}.npz"), va=va, ll_val=score(lms, list(txt[va])), ll_test=score(lms, list(test.text.values)), n_chars=len(cmap))
    return k, round(time.time() - t0, 1)


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 7; t0 = time.time()
    with ProcessPoolExecutor(5, mp_context=get_context("spawn")) as ex:
        for k, sec in ex.map(run_fold, range(5), [N] * 5): print(f"fold {k}: fitted and scored in {sec}s", flush=True)
    print(f"done in {time.time() - t0:.0f}s")
