"""Independent reference + leakage control on the project's fixed folds. (a) plain sklearn word-TFIDF multinomial LR and MultinomialNB;
(b) the same LR with SHUFFLED training labels (must score near the class-frequency log loss 1.0875 if no leakage). python3 audit/audit_reference.py"""
import sys, time; sys.path.insert(0, ".")
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import spooky_common as S
train, test, folds, y = S.load(); txt = train.text.values; t0 = time.time()
def fold(k, shuffle=False, kind="lr"):
    tr, va = np.where(folds != k)[0], np.where(folds == k)[0]; ytr = y[tr].copy()
    if shuffle: ytr = np.random.default_rng(k).permutation(ytr)
    if kind == "lr":
        v = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2); m = LogisticRegression(C=30, max_iter=300)
    else:
        v = CountVectorizer(ngram_range=(1, 2), min_df=2); m = MultinomialNB(alpha=0.1)
    m.fit(v.fit_transform(txt[tr]), ytr); return va, m.predict_proba(v.transform(txt[va]))
for name, kw in [("LR with SHUFFLED labels (leak control)", dict(shuffle=True))]:
    with ThreadPoolExecutor(5) as ex: outs = list(ex.map(lambda k: fold(k, **kw), range(5)))
    oof = np.zeros((len(y), 3))
    for va, p in outs: oof[va] = p
    e = S.evaluate(y, oof, folds, 100); print(f"{name:42s} OOF log loss {e['log_loss']:.4f} acc {e['accuracy']:.4f}  ({time.time()-t0:.0f}s)", flush=True)
