"""Turn the per-author character log-likelihoods into a calibrated probability member for each order 1..N, choose the order, and save it as method 'charlm'.
p(author | sentence) = softmax(beta * (log-likelihood + log prior)); beta (a temperature on a generative score) is fitted on the other folds' rows only."""
import json
import os

import numpy as np
from scipy.optimize import minimize_scalar

import spooky_common as S

train, test, folds, y = S.load(); R = [np.load(os.path.join(S.RESULTS, "charlm_runs", f"f{k}.npz")) for k in range(5)]; N = R[0]["ll_val"].shape[2]
logprior = {k: np.log(np.bincount(y[folds != k], minlength=3) / (folds != k).sum()) for k in range(5)}
oof_ll = np.zeros((len(y), 3, N))
for k in range(5): oof_ll[R[k]["va"]] = R[k]["ll_val"] + logprior[k][None, :, None]
def softmax(z): z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)
def fit_beta(Z, yy): return float(minimize_scalar(lambda b: S.sample_loss(yy, softmax(b * Z)).mean(), bounds=(1e-3, 3.0), method="bounded").x)
table = []; best = None
for o in range(N):
    Z = oof_ll[:, :, o]; cal = np.zeros_like(Z); betas = []
    for k in range(5):
        b = fit_beta(Z[folds != k], y[folds != k]); betas.append(b); cal[folds == k] = softmax(b * Z[folds == k])
    e = S.evaluate(y, cal, folds, 200); table.append(dict(order=o + 1, log_loss=e["log_loss"], accuracy=e["accuracy"], macro_f1=e["macro_f1"], mean_beta=float(np.mean(betas)), raw_log_loss=S.sample_loss(y, softmax(Z)).mean(), raw_accuracy=float((Z.argmax(1) == y).mean())))
    if best is None or e["log_loss"] < best[1]: best = (o, e["log_loss"], cal)
o, _, cal = best; Z = oof_ll[:, :, o]; beta_all = fit_beta(Z, y)
test_p = np.mean([softmax(beta_all * (R[k]["ll_test"][:, :, o] + logprior[k][None, :])) for k in range(5)], axis=0); ev = S.evaluate(y, cal, folds)
S.save_method("charlm", cal, test_p, dict(method="per-author character n-gram language models (interpolated Witten-Bell), from scratch", order=o + 1, beta_all_rows=beta_all, order_table=table, **ev))
for t in table: print(f"order {t['order']}: calibrated log loss {t['log_loss']:.4f} accuracy {t['accuracy']:.4f} macro-F1 {t['macro_f1']:.4f} | raw (beta = 1) log loss {t['raw_log_loss']:.3f} accuracy {t['raw_accuracy']:.4f} | mean beta {t['mean_beta']:.3f}")
print(f"CHOSEN order {o + 1}: log loss {ev['log_loss']:.4f} {np.round(ev['log_loss_ci95'], 4)} accuracy {ev['accuracy']:.4f} macro-F1 {ev['macro_f1']:.4f} per-author F1 { {a: round(v, 3) for a, v in ev['f1_per_author'].items()} }")
