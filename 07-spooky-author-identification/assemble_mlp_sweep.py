"""Assemble the MLP sweep (N, lr, input dropout; alpha 1e-4; 4 inner evaluations per epoch): per-configuration temperature-scaled OOF log loss, then the geometric-mean ensemble of the configurations of the best N (a seed/config ensemble in log space, fitted nothing), saved as methods mlp_sweep_best and mlp_sweep_ens."""
import glob, json, os, re, sys
import numpy as np
import spooky_common as S
RUNS = os.path.join(S.RESULTS, "char_mlp_runs"); train, test, folds, y = S.load(); cfgs = sorted({re.sub(r"_f\d\.npz$", "", os.path.basename(f)) for f in glob.glob(os.path.join(RUNS, "N*_lr*_dp*_a*_s7_f0.npz"))}); res = {}
for c in cfgs:
    if not all(os.path.exists(os.path.join(RUNS, f"{c}_f{k}.npz")) for k in range(5)): print("incomplete", c); continue
    oof, tst, be, ne = np.zeros((len(y), 3)), 0, [], []
    for k in range(5):
        z = np.load(os.path.join(RUNS, f"{c}_f{k}.npz")); oof[z["va"]] = z["pv"]; tst = tst + z["pt"] / 5; h = json.load(open(os.path.join(RUNS, f"{c}_f{k}_history.json")))["history"]; be.append(h[int(np.argmin([r["inner_loss"] for r in h]))]["real_epoch"]); ne.append(len(h))
    cal, Ts, tcal = S.cv_temperature(oof, y, folds, tst); e = S.evaluate(y, cal, folds, 100); raw = S.evaluate(y, oof, folds, 10); res[c] = (cal, tcal, e["log_loss"])
    print(f"{c:34s} raw {raw['log_loss']:.4f} -> T-scaled {e['log_loss']:.4f} acc {e['accuracy']:.4f} | best real epoch {np.round(be,1).tolist()} | evals run {ne}")
best = min(res, key=lambda c: res[c][2]); S.save_method("mlp_sweep_best", res[best][0], res[best][1], dict(config=best, log_loss=res[best][2])); print("BEST", best, round(res[best][2], 4))
def geo(ps): z = np.mean([np.log(S.clip_norm(p)) for p in ps], 0); z -= z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)
for N in ("N5", "N4"):
    cs = [c for c in res if c.startswith(N + "_")]
    if cs: o = geo([res[c][0] for c in cs]); t = geo([res[c][1] for c in cs]); e = S.evaluate(y, o, folds, 100); print(f"ensemble of {len(cs)} {N} configurations (geometric mean): log loss {e['log_loss']:.4f} {np.round(e['log_loss_ci95'],4)} acc {e['accuracy']:.4f}"); S.save_method(f"mlp_sweep_ens_{N}", o, t, dict(configs=cs, **e))
