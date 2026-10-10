"""Per-author word-level Generalized Language Model on one fold: python3 train_glm.py <fold> <N> <lower 0|1> <disc cont|raw> [n_val_limit]
Writes results/glm_runs/{lower|cased}_N{N}_{disc}_f{fold}.npz with ll_val (sentences, authors) and ll_test."""
import os, sys, time, re
import numpy as np
import spooky_common as S, spooky_glm as G
RUNS = os.path.join(S.RESULTS, "glm_runs"); TOKEN = re.compile(r"\w+|[^\w\s]")
def run(k, N, lower, disc, limit=None):
    t0 = time.time(); train, test, folds, y = S.load(); tok = lambda x: TOKEN.findall(x.lower() if lower else x)
    V = len({w for t in list(train.text) + list(test.text) for w in tok(t)}) + 2; tr, va = np.where(folds != k)[0], np.where(folds == k)[0]
    if limit: va = va[:limit]
    lms = [G.GLM([tok(t) for t in train.text.values[tr[y[tr] == a]]], N, V, disc) for a in range(3)]; t1 = time.time()
    score = lambda texts, name: np.array([[lm.loglik(tok(t)) for lm in lms] for i, t in enumerate(texts)])
    llv = score(train.text.values[va], "val"); t2 = time.time()
    if limit: print(f"fold {k} N={N} {disc}: tables {t1-t0:.0f}s, {len(va)} val sentences {t2-t1:.1f}s -> est. full val+test {(t2-t1)/len(va)*(3916+8392):.0f}s"); return
    llt = score(test.text.values, "test"); os.makedirs(RUNS, exist_ok=True); tag = ("lower" if lower else "cased") + f"_N{N}_{disc}"
    np.savez(os.path.join(RUNS, f"{tag}_f{k}.npz"), va=va, ll_val=llv, ll_test=llt); print(f"fold {k} {tag}: tables {t1-t0:.0f}s total {time.time()-t0:.0f}s", flush=True)
if __name__ == "__main__":
    a = sys.argv; run(int(a[1]), int(a[2]), bool(int(a[3])), a[4], int(a[5]) if len(a) > 5 else None)
