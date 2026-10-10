"""Fit one word-level modified Kneser-Ney LM per author on each fold's training sentences; score the fold's held-out sentences and the test sentences at orders 1..N.
    python3 train_wordlm.py <fold> <N> <lower 0|1>        ->  results/wordlm_runs/{lower|cased}_f{fold}.npz   (ll_val, ll_test: shape (sentences, authors, orders))"""
import os, sys, time
import numpy as np
import spooky_common as S
import spooky_wordlm as W

RUNS = os.path.join(S.RESULTS, "wordlm_runs")

def run(k, N, lower):
    t0 = time.time(); train, test, folds, y = S.load(); tok = lambda x: W.tokens(x, lower)
    allv = {w for t in list(train.text) + list(test.text) for w in tok(t)}; tr, va = np.where(folds != k)[0], np.where(folds == k)[0]
    lms = [W.WordLM([tok(t) for t in train.text.values[tr[y[tr] == a]]], N, len(allv)) for a in range(3)]
    score = lambda texts: np.array([[lm.logprobs(tok(t)) for lm in lms] for t in texts])
    os.makedirs(RUNS, exist_ok=True); tag = "lower" if lower else "cased"
    np.savez(os.path.join(RUNS, f"{tag}_f{k}.npz"), va=va, ll_val=score(train.text.values[va]), ll_test=score(test.text.values), vocab=len(allv))
    print(f"fold {k} {tag} N={N}: {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    run(int(sys.argv[1]), int(sys.argv[2]), bool(int(sys.argv[3])))
