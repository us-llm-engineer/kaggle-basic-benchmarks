"""Resumable training of the small neural text classifiers on the five fixed folds (two seeds per fold) on the local GPU.
Every (fold, seed) run is saved on completion (per-epoch statistics, predictions, weights); a re-invocation skips finished runs and
no new run is started once the time budget is used, so a single invocation stays short:
    python3 train_dl.py textcnn|fasttext [budget_seconds]"""
import json
import os
import sys
import time

import numpy as np
import torch
from sklearn.model_selection import train_test_split

import nlp_common as N
import nlp_dl as D

BUCKETS = 50000
SPECS = {
    "textcnn": dict(cfg=dict(lr=2e-3, wd=1e-2, batch=256, epochs=30, patience=6, lr_patience=2), seeds=[0, 1], dim=64, channels=64, encode="words",
                    build=lambda V: D.TextCNN(len(V), dim=64, channels=64)),
    "fasttext": dict(cfg=dict(lr=2e-3, wd=1e-2, batch=128, epochs=40, patience=6, lr_patience=2), seeds=[0, 1], dim=16, dropout=0.5, buckets=BUCKETS, encode="ngrams",
                     build=lambda V: D.FastText(len(V) + BUCKETS, dim=16, dropout=0.5)),
}
RUNS_DIR = os.path.join(N.RESULTS, "dl_runs")
dev = "cuda" if torch.cuda.is_available() else "cpu"


def prepare_fold(name, k, train, test, folds, device=dev):
    """Vocabulary, index split and GPU tensors of fold k (deterministic: the inner split uses random_state=k)."""
    y = train.target.values; X, TX = N.model_text(train), N.model_text(test)
    tr, va = np.where(folds != k)[0], np.where(folds == k)[0]
    i_tr, i_in = train_test_split(np.arange(len(tr)), test_size=0.1, stratify=y[tr], random_state=k)
    V = D.Vocab(X.iloc[tr[i_tr]], 2)
    if SPECS[name]["encode"] == "words": enc = lambda s: torch.tensor(np.array([V.encode(t) for t in s]), device=device)
    else: enc = lambda s: torch.tensor(D.encode_ngrams(list(s), V, BUCKETS), device=device)
    T = lambda idx: torch.tensor(y[idx], device=device)
    return dict(V=V, tr=tr, va=va, i_tr=i_tr, i_in=i_in, Xa=enc(X.iloc[tr[i_tr]]), Xi=enc(X.iloc[tr[i_in]]), Xv=enc(X.iloc[va]), Xte=enc(TX),
                ya=T(tr[i_tr]), yi=T(tr[i_in]), yv=T(va), texts=list(X.iloc[tr[i_tr]]))


def load_model(name, d, k=0, seed=0, device=dev):
    m = SPECS[name]["build"](d["V"]).to(device); m.load_state_dict(torch.load(os.path.join(N.ROOT, "checkpoints", f"{name}_f{k}_s{seed}.pt"), map_location=device)); return m.eval()


def assemble(name):
    """Combine the saved runs into out-of-fold logits per seed, fold-averaged test logits per seed and the run records; None if runs are missing."""
    train, test, folds = N.load(); seeds = SPECS[name]["seeds"]
    oof_s, test_s, runs = np.zeros((len(seeds), len(train))), np.zeros((len(seeds), len(test))), []
    for k in range(5):
        for si, seed in enumerate(seeds):
            p = os.path.join(RUNS_DIR, f"{name}_f{k}_s{seed}")
            if not os.path.exists(p + ".npz"): return None
            z = np.load(p + ".npz"); oof_s[si, z["va"]] = z["logit_va"]; test_s[si] += z["logit_test"] / 5
            runs.append(json.load(open(p + ".json")))
    D.write_history(name, runs)
    return oof_s, test_s, runs


def main(name, budget):
    t0 = time.time(); spec = SPECS[name]; train, test, folds = N.load()
    os.makedirs(RUNS_DIR, exist_ok=True); os.makedirs(os.path.join(N.ROOT, "checkpoints"), exist_ok=True); log = D.RunLog(name)
    for k in range(5):
        d = None
        for seed in spec["seeds"]:
            p = os.path.join(RUNS_DIR, f"{name}_f{k}_s{seed}")
            if os.path.exists(p + ".npz"): continue
            if time.time() - t0 > budget: log(f"budget {budget}s used; stopping before fold {k} seed {seed}"); print(f"budget reached at fold {k} seed {seed}: run again to continue"); return False
            d = d or prepare_fold(name, k, train, test, folds)
            torch.manual_seed(seed); m = spec["build"](d["V"]).to(dev)
            m, hist = D.fit(m, d["Xa"], d["ya"], d["Xi"], d["yi"], d["Xv"], d["yv"], dict(spec["cfg"], seed=seed), log=lambda s: log(f"fold {k} seed {seed} " + s))
            best = min(hist, key=lambda r: r["inner_loss"]); torch.save(m.state_dict(), os.path.join(N.ROOT, "checkpoints", f"{name}_f{k}_s{seed}.pt"))
            json.dump(dict(fold=k, seed=seed, best_epoch=best["epoch"], n_epochs=len(hist), history=hist), open(p + ".json", "w"))
            np.savez(p + ".npz", va=d["va"], logit_va=D.predict_logits(m, d["Xv"]), logit_test=D.predict_logits(m, d["Xte"]))
            log(f"fold {k} seed {seed} done: {len(hist)} epochs, best epoch {best['epoch']}, inner loss {best['inner_loss']:.4f}")
            print(f"fold {k} seed {seed}: {len(hist)} epochs, best {best['epoch']}, {time.time()-t0:.0f}s elapsed")
    print("all runs complete"); return True


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 55.0)
