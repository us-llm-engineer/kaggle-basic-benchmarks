"""Calibrate the GLM word members (p = softmax(beta (ll + log prior)), beta fitted on the other folds), report every grid row, save the best as method 'glm'."""
import glob, os, numpy as np
from scipy.optimize import minimize_scalar
import spooky_common as S
train, test, folds, y = S.load(); lp = {k: np.log(np.bincount(y[folds != k], minlength=3) / (folds != k).sum()) for k in range(5)}
def sm(z): z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)
fb = lambda Z, yy: float(minimize_scalar(lambda b: S.sample_loss(yy, sm(b * Z)).mean(), bounds=(1e-3, 3), method="bounded").x)
best = None; tags = sorted({os.path.basename(f).rsplit("_f", 1)[0] for f in glob.glob(os.path.join(S.RESULTS, "glm_runs", "*.npz"))})
for tag in tags:
    R = [np.load(os.path.join(S.RESULTS, "glm_runs", f"{tag}_f{k}.npz")) for k in range(5)]; Z = np.zeros((len(y), 3))
    for k in range(5): Z[R[k]["va"]] = R[k]["ll_val"] + lp[k]
    cal = np.zeros_like(Z)
    for k in range(5): b = fb(Z[folds != k], y[folds != k]); cal[folds == k] = sm(b * Z[folds == k])
    e = S.evaluate(y, cal, folds, 100); print(f"{tag:18s} calibrated log loss {e['log_loss']:.4f} acc {e['accuracy']:.4f}")
    if best is None or e["log_loss"] < best[0]: best = (e["log_loss"], tag, cal, R, Z)
_, tag, cal, R, Z = best; b = fb(Z, y); tp = np.mean([sm(b * (R[k]["ll_test"] + lp[k])) for k in range(5)], 0); ev = S.evaluate(y, cal, folds)
S.save_method("glm", cal, tp, dict(method="per-author word-level Generalized Language Model (skipped n-grams + modified Kneser-Ney, arXiv 1404.3377), from scratch", config=tag, beta_all_rows=b, **ev)); print(f"CHOSEN {tag}: {ev['log_loss']:.4f} {np.round(ev['log_loss_ci95'],4)} acc {ev['accuracy']:.4f}")
