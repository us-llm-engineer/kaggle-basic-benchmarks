"""Diagnostic of the levers that could lower the out-of-fold log loss of the three Spooky methodologies. Reads the saved out-of-fold probabilities
(results/oof_*.npy); every fitted quantity (blend weights, calibration map) is fitted on four folds and scored on the fifth, so no sentence scores itself."""
import os
import sys

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import spooky_common as S

train, test, folds, y = S.load(); K = ["nblr", "style_gbdt", "cnn"]; P = {k: S.load_method(k)[0] for k in K}
print("== Where does the log loss of each methodology come from? (out-of-fold, temperature-scaled)")
for k in K:
    nll = S.sample_loss(y, P[k]); order = np.sort(nll)[::-1]; tot = nll.sum(); wrong = P[k].argmax(1) != y
    print(f"{k:10s} mean {nll.mean():.4f} | errors {wrong.mean():.1%} of sentences carry {nll[wrong].sum() / tot:.1%} of total loss | worst 5% of sentences carry {order[:int(.05 * len(y))].sum() / tot:.1%} | correct sentences mean loss {nll[~wrong].mean():.4f}, wrong {nll[wrong].mean():.4f} | mean confidence on errors {P[k].max(1)[wrong].mean():.3f}")
print("\n== Constant-confidence reference: a model that is right with probability equal to its accuracy and spreads the rest evenly, on every sentence")
for k in K:
    acc = (P[k].argmax(1) == y).mean(); print(f"{k:10s} accuracy {acc:.3f}: constant-confidence log loss {-(acc * np.log(acc) + (1 - acc) * np.log((1 - acc) / 2)):.4f} against actual {S.sample_loss(y, P[k]).mean():.4f}")
print("\n== Diagnostic: cross-validated convex blend of the three methodologies' probabilities (weights fitted on 4 folds, scored on the 5th; NOT a fourth methodology)")
def blend(pl, w): return S.clip_norm(sum(wi * p for wi, p in zip(w, pl)))
oof = np.zeros((len(y), 3)); Ws = []
for k in range(5):
    tr = np.where(folds != k)[0]; f = lambda z: S.sample_loss(y[tr], blend([P[m][tr] for m in K], np.exp(z) / np.exp(z).sum())).mean(); z = minimize(f, np.zeros(3), method="Nelder-Mead").x
    w = np.exp(z) / np.exp(z).sum(); Ws.append(w); va = folds == k; oof[va] = blend([P[m][va] for m in K], w)
print("fold weights (nblr, style, cnn):", [np.round(w, 3).tolist() for w in Ws]); print(f"blend log loss {S.sample_loss(y, oof).mean():.4f} | accuracy {(oof.argmax(1) == y).mean():.4f} (best single: nblr {S.sample_loss(y, P['nblr']).mean():.4f})")
d, ci, _ = S.paired_bootstrap_logloss(y, oof, P["nblr"], n=2000); print(f"paired difference blend minus nblr: {d:+.4f} 95% interval [{ci[0]:+.4f}, {ci[1]:+.4f}]")
print("\n== Diagnostic: log-linear (geometric) pooling with the same cross-validated fitting")
def geo(pl, w): z = sum(wi * np.log(S.clip_norm(p)) for wi, p in zip(w, pl)); z -= z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)
og = np.zeros((len(y), 3))
for k in range(5):
    tr = np.where(folds != k)[0]; f = lambda z: S.sample_loss(y[tr], geo([P[m][tr] for m in K], z)).mean(); z = minimize(f, np.ones(3) / 3, method="Nelder-Mead").x; va = folds == k; og[va] = geo([P[m][va] for m in K], z)
    print(f"  fold {k} exponents {np.round(z, 3).tolist()}")
print(f"geometric pool log loss {S.sample_loss(y, og).mean():.4f} | accuracy {(og.argmax(1) == y).mean():.4f}"); d, ci, _ = S.paired_bootstrap_logloss(y, og, P["nblr"], n=2000); print(f"paired difference geometric pool minus nblr: {d:+.4f} 95% interval [{ci[0]:+.4f}, {ci[1]:+.4f}]")
print("\n== Diagnostic: Dirichlet-style linear calibration on log-probabilities (multinomial logistic regression with L2, cross-validated across folds) for the best methodology")
Z = np.log(S.clip_norm(P["nblr"])); oc = np.zeros_like(Z)
for k in range(5):
    tr, va = folds != k, folds == k; oc[va] = LogisticRegression(C=10.0, max_iter=500).fit(Z[tr], y[tr]).predict_proba(Z[va])
print(f"nblr temperature-scaled {S.sample_loss(y, P['nblr']).mean():.4f} -> Dirichlet-style linear calibration {S.sample_loss(y, oc).mean():.4f}"); d, ci, _ = S.paired_bootstrap_logloss(y, oc, P["nblr"], n=2000); print(f"paired difference Dirichlet-style calibration minus nblr: {d:+.4f} 95% interval [{ci[0]:+.4f}, {ci[1]:+.4f}]")
