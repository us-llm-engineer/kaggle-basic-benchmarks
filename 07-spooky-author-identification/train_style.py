"""Resumable training of methodology 2 (stylometric features + latent topics, LightGBM) on the five fixed folds, with and without the topic features.
Each (variant, fold) fit is saved on completion to results/style_runs/; a re-invocation skips finished fits and starts no new batch once the time
budget is used:   python3 train_style.py [budget_seconds]"""
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

import spooky_common as S
import spooky_style as ST

RUNS = os.path.join(S.RESULTS, "style_runs"); CKPT = os.path.join(S.ROOT, "checkpoints"); CACHE = os.path.join(S.ROOT, "cache")
PARAMS = dict(objective="multiclass", num_class=3, learning_rate=0.05, num_leaves=15, min_child_samples=25, feature_fraction=0.5, bagging_fraction=0.8,
              bagging_freq=1, lambda_l2=5.0, max_bin=127, verbose=-1, num_threads=2, seed=7)
VARIANTS = ["full", "style"]  # full = stylometry + 100 latent topics; style = stylometry only (ablation)


def features():
    train, test, folds, y = S.load(); vocab = ST.function_word_vocab(list(train.text) + list(test.text))
    return train, test, folds, y, ST.style_features(train.text, vocab), ST.style_features(test.text, vocab)


def run_fold(variant, k, train, test, folds, y, Fa, Ft):
    tr, va = np.where(folds != k)[0], np.where(folds == k)[0]; A, V, Te = Fa.values[tr], Fa.values[va], Ft.values; names = list(Fa.columns)
    if variant == "full":
        tv = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2); W = tv.fit_transform(train.text.values[tr]); svd = TruncatedSVD(100, random_state=7, n_iter=4).fit(W)
        A, V, Te = np.hstack([A, svd.transform(W)]), np.hstack([V, svd.transform(tv.transform(train.text.values[va]))]), np.hstack([Te, svd.transform(tv.transform(test.text.values))])
        names += [f"topic_{i}" for i in range(100)]
    i_tr, i_in = train_test_split(np.arange(len(tr)), test_size=0.1, stratify=y[tr], random_state=k); rec = {}
    m = lgb.train(PARAMS, lgb.Dataset(A[i_tr], y[tr][i_tr]), 3000, valid_sets=[lgb.Dataset(A[i_in], y[tr][i_in])], valid_names=["inner"],
                  callbacks=[lgb.early_stopping(80, verbose=False), lgb.record_evaluation(rec)])
    its = [i for i in range(10, m.current_iteration() + 80, 10) if i <= m.num_trees() // 3]; mon = np.array([(i, S.sample_loss(y[va], m.predict(V, num_iteration=i)).mean()) for i in its])
    os.makedirs(CKPT, exist_ok=True); os.makedirs(CACHE, exist_ok=True)
    m.save_model(os.path.join(CKPT, f"style_{variant}_f{k}.txt")); np.save(os.path.join(CACHE, f"style_{variant}_V{k}.npy"), V)
    np.savez(os.path.join(RUNS, f"{variant}_f{k}.npz"), va=va, pv=m.predict(V, num_iteration=m.best_iteration), pt=m.predict(Te, num_iteration=m.best_iteration), best=m.best_iteration,
             inner=np.array(rec["inner"]["multi_logloss"]), mon=mon, gain=m.feature_importance("gain"), names=np.array(names))


def load_runs(variant):
    """List of per-fold records for a variant (None if a fit is missing)."""
    out = []
    for k in range(5):
        p = os.path.join(RUNS, f"{variant}_f{k}.npz")
        if not os.path.exists(p): return None
        z = np.load(p, allow_pickle=True); out.append({key: z[key] for key in z.files} | {"k": k})
    return out


_DATA = None


def job(args):
    """Worker entry point: one (variant, fold) fit in its own process (features are computed once per process)."""
    global _DATA
    if _DATA is None: _DATA = features()
    run_fold(args[0], args[1], *_DATA); return args


def main(budget):
    t0 = time.time(); os.makedirs(RUNS, exist_ok=True)
    for variant in VARIANTS:
        todo = [k for k in range(5) if not os.path.exists(os.path.join(RUNS, f"{variant}_f{k}.npz"))]
        if not todo: continue
        if time.time() - t0 > budget: print(f"budget reached before variant '{variant}': run again to continue"); return False
        with ProcessPoolExecutor(len(todo), mp_context=get_context("spawn")) as ex: list(ex.map(job, [(variant, k) for k in todo]))
        print(f"variant '{variant}' done ({len(todo)} folds), {time.time() - t0:.0f}s elapsed", flush=True)
    print("all fits complete"); return True


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 30.0)
