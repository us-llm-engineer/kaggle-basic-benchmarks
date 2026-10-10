"""Turn the remotely computed fold outputs of the two replacement methodologies into calibrated members.
 wordlm  : per-author word modified-Kneser-Ney log-likelihoods -> p = softmax(beta (ll + log prior)), beta fitted on the other folds (as for charlm); order and casing chosen on OOF.
 fasttext: fastText-style probabilities (best inner epoch per fold) -> temperature fitted on the other folds (S.cv_temperature); configuration chosen on OOF.
Saves methods 'wordlm' and 'fasttext' (oof, test, meta) and prints every grid row, chosen or not."""
import json, os
import numpy as np
from scipy.optimize import minimize_scalar
import spooky_common as S
train, test, folds, y = S.load()
def softmax(z): z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)
logprior = {k: np.log(np.bincount(y[folds != k], minlength=3) / (folds != k).sum()) for k in range(5)}
def fit_beta(Z, yy): return float(minimize_scalar(lambda b: S.sample_loss(yy, softmax(b * Z)).mean(), bounds=(1e-3, 3.0), method="bounded").x)
rows, best = [], None
for tag in ("cased", "lower"):
    R = [np.load(os.path.join(S.RESULTS, "wordlm_runs", f"{tag}_f{k}.npz")) for k in range(5)]; N = R[0]["ll_val"].shape[2]; oof_ll = np.zeros((len(y), 3, N))
    for k in range(5): oof_ll[R[k]["va"]] = R[k]["ll_val"] + logprior[k][None, :, None]
    for o in range(N):
        Z = oof_ll[:, :, o]; cal = np.zeros_like(Z); betas = []
        for k in range(5): b = fit_beta(Z[folds != k], y[folds != k]); betas.append(b); cal[folds == k] = softmax(b * Z[folds == k])
        e = S.evaluate(y, cal, folds, 100); rows.append(dict(tag=tag, order=o + 1, log_loss=e["log_loss"], accuracy=e["accuracy"], mean_beta=float(np.mean(betas))))
        print(f"wordlm {tag:5s} order {o+1}: calibrated log loss {e['log_loss']:.4f} acc {e['accuracy']:.4f} mean beta {np.mean(betas):.3f}")
        if best is None or e["log_loss"] < best[0]: best = (e["log_loss"], tag, o, cal, R, betas)
_, tag, o, cal, R, betas = best; Zall = np.zeros((len(y), 3))
for k in range(5): Zall[R[k]["va"]] = R[k]["ll_val"][:, :, o] + logprior[k][None, :]
b_all = fit_beta(Zall, y); tp = np.mean([softmax(b_all * (R[k]["ll_test"][:, :, o] + logprior[k][None, :])) for k in range(5)], 0); ev = S.evaluate(y, cal, folds)
S.save_method("wordlm", cal, tp, dict(method="per-author word-level interpolated modified Kneser-Ney LM, from scratch (reproduces MKN of arXiv 1404.3377 App.A; skip patterns not implemented)", casing=tag, order=o + 1, beta_all_rows=b_all, grid=rows, **ev))
print(f"CHOSEN wordlm {tag} order {o+1}: log loss {ev['log_loss']:.4f} {np.round(ev['log_loss_ci95'],4)} acc {ev['accuracy']:.4f}")
rows, best = [], None
for v in ("word", "char"):
    for h in (10, 30):
        oof, tst, bes = np.zeros((len(y), 3)), 0, []
        for k in range(5):
            z = np.load(os.path.join(S.RESULTS, "fasttext_runs", f"{v}_h{h}_f{k}.npz")); oof[z["va"]] = z["pv"]; tst = tst + z["pt"] / 5; bes.append(json.load(open(os.path.join(S.RESULTS, "fasttext_runs", f"{v}_h{h}_f{k}.json")))["best_epoch"])
        cal, Ts, tcal = S.cv_temperature(oof, y, folds, tst); e = S.evaluate(y, cal, folds, 100); raw = S.evaluate(y, oof, folds, 10)
        rows.append(dict(variant=v, h=h, log_loss=e["log_loss"], raw_log_loss=raw["log_loss"], accuracy=e["accuracy"], best_epochs=bes, temperatures=Ts))
        print(f"fasttext {v:4s} h={h:2d}: raw log loss {raw['log_loss']:.4f} -> temperature-scaled {e['log_loss']:.4f} acc {e['accuracy']:.4f} best epochs {bes} T {np.round(Ts,2).tolist()}")
        if best is None or e["log_loss"] < best[0]: best = (e["log_loss"], v, h, cal, tcal)
_, v, h, cal, tcal = best; ev = S.evaluate(y, cal, folds)
S.save_method("fasttext", cal, tcal, dict(method=f"fastText-style averaged hashed word-bigram{' + char 3-5gram' if v == 'char' else ''} embeddings, from scratch (arXiv 1607.01759 Sec.2; char n-grams inspired_by)", variant=v, hidden=h, grid=rows, **ev))
print(f"CHOSEN fasttext {v} h={h}: log loss {ev['log_loss']:.4f} {np.round(ev['log_loss_ci95'],4)} acc {ev['accuracy']:.4f}")
