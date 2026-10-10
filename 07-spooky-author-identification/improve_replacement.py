"""Pools with the two replacement members, fully nested (pool exponents and, where used, the Dirichlet-style map are fitted on the other folds only; same code path as improve_bc.py).
The reference pool4 / pool4_dir out-of-fold predictions are READ from results/improvement_bc.npz, never recomputed. Paired bootstrap against pool4_dir on the same sentences."""
import json, os, time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import numpy as np
import spooky_common as S, spooky_pool as PL
CONFIGS = {"pool5": ["nblr", "style_gbdt", "cnn", "charlm", "wordlm"], "pool6": ["nblr", "style_gbdt", "cnn", "charlm", "wordlm", "fasttext"], "pool_nodeep": ["nblr", "style_gbdt", "charlm", "wordlm", "fasttext"]}
def job(a):
    name, d = a; t0 = time.time(); train, test, folds, y = S.load(); keys = CONFIGS[name]; P = [S.load_method(k)[0] for k in keys]; T = [S.load_method(k)[1] for k in keys]
    w = PL.fit_exponents(P, y); pt = PL.pool(T, w)
    if not d: oof, _ = PL.nested_pool(P, y, folds); return name, d, oof, pt, w, round(time.time() - t0)
    oof = PL.nested_pool_dirichlet(P, y, folds); inner, _ = PL.nested_pool(P, y, folds); m = PL.dirichlet_map(inner, y); return name, d, oof, PL.apply_map(m, pt), w, round(time.time() - t0)
if __name__ == "__main__":
    train, test, folds, y = S.load(); ref = np.load(os.path.join(S.RESULTS, "improvement_bc.npz")); out, meta = {}, {}
    for r in ("pool4", "pool4_dir"): e = S.evaluate(y, ref[f"oof_{r}"], folds, 300); print(f"REFERENCE {r:11s} log loss {e['log_loss']:.4f} {np.round(e['log_loss_ci95'],4)} acc {e['accuracy']:.4f}", flush=True)
    jobs = [(n, d) for n in CONFIGS for d in (False, True)]
    with ProcessPoolExecutor(6, mp_context=get_context("spawn")) as ex:
        for name, d, oof, tst, w, sec in ex.map(job, jobs):
            tag = f"{name}{'_dir' if d else ''}"; out[f"oof_{tag}"], out[f"test_{tag}"] = oof, tst; e = S.evaluate(y, oof, folds, 300)
            dm, ci, p = S.paired_bootstrap_logloss(y, oof, ref["oof_pool4_dir"]); meta[tag] = dict(members=CONFIGS[name], log_loss=e["log_loss"], ci95=e["log_loss_ci95"], accuracy=e["accuracy"], macro_f1=e["macro_f1"], exponents_all_rows=[float(v) for v in w],
                                                                                          fold_log_loss=e["fold_log_loss"], paired_vs_pool4_dir=dict(mean_diff=dm, ci95=list(ci), share_nonneg=p), seconds=sec)
            print(f"{tag:15s} log loss {e['log_loss']:.4f} {np.round(e['log_loss_ci95'],4)} acc {e['accuracy']:.4f} | vs pool4_dir diff {dm:+.4f} {np.round(ci,4)} | exps {np.round(w,3).tolist()} ({sec}s)", flush=True)
    np.savez(os.path.join(S.RESULTS, "improvement_replacement.npz"), **out); json.dump(meta, open(os.path.join(S.RESULTS, "improvement_replacement.json"), "w"), indent=2, default=float); print("done")
