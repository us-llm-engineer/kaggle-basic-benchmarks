"""Improvements B and C: pool the methodologies with and without the character language model (C), each with and without the Dirichlet-style calibration map (B).
All fitting is nested: every pooled or calibrated prediction for a sentence uses parameters fitted on the other folds only. Saves out-of-fold and test probabilities."""
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np

import spooky_common as S
import spooky_pool as PL

CONFIGS = {"pool2": ["nblr", "charlm"], "pool3": ["nblr", "style_gbdt", "cnn"], "pool4": ["nblr", "style_gbdt", "cnn", "charlm"]}


def job(args):
    name, dirichlet = args; t0 = time.time(); train, test, folds, y = S.load(); keys = CONFIGS[name]; P = [S.load_method(k)[0] for k in keys]; T = [S.load_method(k)[1] for k in keys]
    w_all = PL.fit_exponents(P, y); pooled_test = PL.pool(T, w_all)
    if not dirichlet:
        oof, ws = PL.nested_pool(P, y, folds); return name, dirichlet, oof, pooled_test, w_all, round(time.time() - t0, 1)
    oof = PL.nested_pool_dirichlet(P, y, folds); inner, _ = PL.nested_pool(P, y, folds)   # inner: out-of-fold pooled probabilities of all rows -> training data of the final map
    m = PL.dirichlet_map(inner, y); return name, dirichlet, oof, PL.apply_map(m, pooled_test), w_all, round(time.time() - t0, 1)


if __name__ == "__main__":
    t0 = time.time(); train, test, folds, y = S.load(); jobs = [(n, d) for n in CONFIGS for d in (False, True)]; out = {}; meta = {}
    with ProcessPoolExecutor(6, mp_context=get_context("spawn")) as ex:
        for name, d, oof, tst, w, sec in ex.map(job, jobs):
            tag = f"{name}{'_dir' if d else ''}"; out[f"oof_{tag}"], out[f"test_{tag}"] = oof, tst; e = S.evaluate(y, oof, folds, 300)
            meta[tag] = dict(log_loss=e["log_loss"], ci95=e["log_loss_ci95"], accuracy=e["accuracy"], macro_f1=e["macro_f1"], exponents_all_rows=[float(v) for v in w], fold_log_loss=e["fold_log_loss"], seconds=sec)
            print(f"{tag:11s} log loss {e['log_loss']:.4f} {np.round(e['log_loss_ci95'], 4)} accuracy {e['accuracy']:.4f} macro-F1 {e['macro_f1']:.4f} exponents {np.round(w, 3).tolist()} ({sec}s)", flush=True)
    np.savez(os.path.join(S.RESULTS, "improvement_bc.npz"), **out); json.dump(meta, open(os.path.join(S.RESULTS, "improvement_bc.json"), "w"), indent=2); print(f"done in {time.time() - t0:.0f}s")
