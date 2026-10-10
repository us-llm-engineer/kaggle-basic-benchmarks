"""Fold-wise word and character n-gram TF-IDF features, cached to cache/fold{k}.npz (fitted on each fold's training rows only).
Run once before notebook 02:  python3 spooky_features.py"""
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

import spooky_common as S

CACHE = os.path.join(S.ROOT, "cache")


def build_fold(k):
    train, test, folds, y = S.load(); tr, va = np.where(folds != k)[0], np.where(folds == k)[0]; txt, ttxt = train.text.values, test.text.values
    wv = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2)
    cv = TfidfVectorizer(analyzer="char", ngram_range=(1, 5), sublinear_tf=True, min_df=3, lowercase=False, max_features=150000)
    Xa = sparse.hstack([wv.fit_transform(txt[tr]), cv.fit_transform(txt[tr])]).tocsr()
    f = lambda s: sparse.hstack([wv.transform(s), cv.transform(s)]).tocsr()
    os.makedirs(CACHE, exist_ok=True)
    sparse.save_npz(os.path.join(CACHE, f"f{k}_train.npz"), Xa, compressed=False); sparse.save_npz(os.path.join(CACHE, f"f{k}_val.npz"), f(txt[va]), compressed=False); sparse.save_npz(os.path.join(CACHE, f"f{k}_test.npz"), f(ttxt), compressed=False)
    np.save(os.path.join(CACHE, f"f{k}_names.npy"), np.array(["w:" + n for n in wv.get_feature_names_out()] + ["c:" + n for n in cv.get_feature_names_out()]))
    return k, Xa.shape


if __name__ == "__main__":
    t0 = time.time()
    with ProcessPoolExecutor(5) as ex:
        for k, shape in ex.map(build_fold, range(5)): print(f"fold {k}: train features {shape}")
    print(f"done in {time.time() - t0:.0f}s")
