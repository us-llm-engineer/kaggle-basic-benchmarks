"""Assemble the members of the best modelling pipeline on a (cleaned) working folder: nblr (C = 30), style_gbdt (with topics), glm (assemble_glm.py), charlm (assemble_charlm.py), wordlm (lowercase, order 4), fasttext (word, 30 hidden), char_svm (saved by its trainer). Same calibration as the originals: cross-validated temperature, or a beta fitted on the other folds for the language models."""
import json, os, numpy as np
from scipy.optimize import minimize_scalar
import spooky_common as S, train_nblr as TN, train_style as TS
train, test, folds, y = S.load()
oof, tst = TN.assemble(30, folds, len(test)); cal, Ts, tcal = S.cv_temperature(oof, y, folds, tst); e = S.evaluate(y, cal, folds, 100); S.save_method("nblr", cal, tcal, dict(method="NB-weighted n-gram LR (C=30)", **e)); print(f"nblr       {e['log_loss']:.4f} acc {e['accuracy']:.4f}")
full = TS.load_runs("full"); assert full; oof, tst = np.zeros((len(y), 3)), 0
for r in full: oof[r["va"]] = r["pv"]; tst = tst + r["pt"] / 5
cal, Ts, tcal = S.cv_temperature(oof, y, folds, tst); e = S.evaluate(y, cal, folds, 100); S.save_method("style_gbdt", cal, tcal, dict(method="stylometry + topics LightGBM", **e)); print(f"style_gbdt {e['log_loss']:.4f} acc {e['accuracy']:.4f}")
def softmax(z): z = z - z.max(1, keepdims=True); ez = np.exp(z); return ez / ez.sum(1, keepdims=True)
fb = lambda Z, yy: float(minimize_scalar(lambda b: S.sample_loss(yy, softmax(b * Z)).mean(), bounds=(1e-3, 3.0), method="bounded").x)
lp = {k: np.log(np.bincount(y[folds != k], minlength=3) / (folds != k).sum()) for k in range(5)}; R = [np.load(os.path.join(S.RESULTS, "wordlm_runs", f"lower_f{k}.npz")) for k in range(5)]; o = 3   # order 4
Z = np.zeros((len(y), 3))
for k in range(5): Z[R[k]["va"]] = R[k]["ll_val"][:, :, o] + lp[k]
cal = np.zeros_like(Z)
for k in range(5): b = fb(Z[folds != k], y[folds != k]); cal[folds == k] = softmax(b * Z[folds == k])
b = fb(Z, y); tp = np.mean([softmax(b * (R[k]["ll_test"][:, :, o] + lp[k])) for k in range(5)], 0); e = S.evaluate(y, cal, folds, 100); S.save_method("wordlm", cal, tp, dict(method="word MKN LM lower order 4", **e)); print(f"wordlm     {e['log_loss']:.4f} acc {e['accuracy']:.4f}")
oof, tst = np.zeros((len(y), 3)), 0
for k in range(5): z = np.load(os.path.join(S.RESULTS, "fasttext_runs", f"word_h30_f{k}.npz")); oof[z["va"]] = z["pv"]; tst = tst + z["pt"] / 5
cal, Ts, tcal = S.cv_temperature(oof, y, folds, tst); e = S.evaluate(y, cal, folds, 100); S.save_method("fasttext", cal, tcal, dict(method="fastText-style word h30", **e)); print(f"fasttext   {e['log_loss']:.4f} acc {e['accuracy']:.4f}")
