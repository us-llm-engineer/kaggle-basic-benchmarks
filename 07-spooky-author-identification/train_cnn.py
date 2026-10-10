"""Methodology 3: a character + word convolutional network trained from scratch (no pre-trained weights), one model per fold of the five fixed folds.

Two branches read the sentence: a character branch (embeddings of characters, case and punctuation preserved, convolutions of width 3/5/7)
and a word branch (embeddings of lower-cased word and punctuation tokens, widths 1/2/3). Each branch is max-pooled over time, the pooled vectors
are concatenated, passed through dropout and a linear layer to three logits. An inner 10% of each fold's training rows selects the best epoch
(lowest log loss) and stops training; the outer validation fold is recorded every epoch and never selects anything.

    python3 train_cnn.py [epochs] [threads_per_fold]        # runs the five folds in parallel processes
    SMOKE=1 python3 train_cnn.py                            # tiny smoke test
"""
import json
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

import spooky_common as S

RUNS = os.path.join(S.RESULTS, "cnn_runs")
MAX_CHARS, MAX_WORDS = 256, 64
CFG = dict(lr=2e-3, wd=1e-2, batch=128, patience=3, char_dim=32, word_dim=100, char_channels=96, word_channels=64, dropout=0.5)
TOKEN = re.compile(r"\w+|[^\w\s]")


class Vocab:
    """Index of the items occurring at least `min_count` times; 0 = padding, 1 = unknown."""

    def __init__(self, items, min_count):
        c = Counter(items); self.itos = ["<pad>", "<unk>"] + [w for w, n in c.most_common() if n >= min_count]; self.stoi = {w: i for i, w in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, seq, maxlen):
        ids = [self.stoi.get(w, 1) for w in seq][:maxlen]; return ids + [0] * (maxlen - len(ids))


def words_of(text):
    return TOKEN.findall(text.lower())


class CharWordCNN(nn.Module):
    def __init__(self, n_chars, n_words, cd, wd, cch, wch, dropout):
        super().__init__()
        self.ce, self.we = nn.Embedding(n_chars, cd, padding_idx=0), nn.Embedding(n_words, wd, padding_idx=0)
        self.cc = nn.ModuleList([nn.Conv1d(cd, cch, w) for w in (3, 5, 7)]); self.wc = nn.ModuleList([nn.Conv1d(wd, wch, w) for w in (1, 2, 3)])
        self.drop = nn.Dropout(dropout); self.out = nn.Linear(3 * cch + 3 * wch, 3)

    def forward(self, xc, xw):
        ec, ew = self.ce(xc).transpose(1, 2), self.we(xw).transpose(1, 2)
        h = torch.cat([F.relu(c(ec)).max(2).values for c in self.cc] + [F.relu(c(ew)).max(2).values for c in self.wc], 1)
        return self.out(self.drop(h))


@torch.no_grad()
def logits_of(model, xc, xw, bs=512):
    model.eval(); return torch.cat([model(xc[i:i + bs], xw[i:i + bs]) for i in range(0, len(xc), bs)])


def run_fold(k, epochs, threads, smoke):
    torch.set_num_threads(threads); torch.manual_seed(7); t_start = time.time()
    train, test, folds, y = S.load(); tr, va = np.where(folds != k)[0], np.where(folds == k)[0]
    if smoke: tr, va = tr[:600], va[:200]
    i_tr, i_in = train_test_split(np.arange(len(tr)), test_size=0.1, stratify=y[tr], random_state=k)
    txt = train.text.values; cv = Vocab([ch for t in txt[tr[i_tr]] for ch in t], 3); wv = Vocab([w for t in txt[tr[i_tr]] for w in words_of(t)], 2)
    enc = lambda texts: (torch.tensor([cv.encode(t, MAX_CHARS) for t in texts]), torch.tensor([wv.encode(words_of(t), MAX_WORDS) for t in texts]))
    (Xc_a, Xw_a), (Xc_i, Xw_i), (Xc_v, Xw_v) = enc(txt[tr[i_tr]]), enc(txt[tr[i_in]]), enc(txt[va]); y_a, y_i, y_v = (torch.tensor(y[idx]) for idx in (tr[i_tr], tr[i_in], va))
    model = CharWordCNN(len(cv), len(wv), CFG["char_dim"], CFG["word_dim"], CFG["char_channels"], CFG["word_channels"], CFG["dropout"])
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["wd"]); sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=1)
    hist, best, best_state, bad = [], np.inf, None, 0; g = torch.Generator().manual_seed(7)
    for ep in range(epochs):
        t0 = time.time(); model.train(); perm = torch.randperm(len(y_a), generator=g); tot, gn, nb = 0.0, 0.0, 0
        for i in range(0, len(perm), CFG["batch"]):
            idx = perm[i:i + CFG["batch"]]; loss = F.cross_entropy(model(Xc_a[idx], Xw_a[idx]), y_a[idx])
            opt.zero_grad(set_to_none=True); loss.backward(); gn += float(nn.utils.clip_grad_norm_(model.parameters(), 5.0)); opt.step(); tot += float(loss.detach()) * len(idx); nb += 1
        zi, zv = logits_of(model, Xc_i, Xw_i), logits_of(model, Xc_v, Xw_v); li, lv = float(F.cross_entropy(zi, y_i)), float(F.cross_entropy(zv, y_v)); sched.step(li)
        hist.append(dict(epoch=ep, seconds=time.time() - t0, lr=opt.param_groups[0]["lr"], train_loss=tot / len(y_a), grad_norm=gn / nb, inner_loss=li, inner_acc=float((zi.argmax(1) == y_i).float().mean()),
                         mon_loss=lv, mon_acc=float((zv.argmax(1) == y_v).float().mean())))
        if li < best - 1e-4: best, bad, best_state = li, 0, {n: v.clone() for n, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= CFG["patience"]: break
    model.load_state_dict(best_state); os.makedirs(RUNS, exist_ok=True); os.makedirs(os.path.join(S.ROOT, "checkpoints"), exist_ok=True)
    pv = F.softmax(logits_of(model, Xc_v, Xw_v), 1).numpy(); Xc_t, Xw_t = enc(test.text.values[:200] if smoke else test.text.values); pt = F.softmax(logits_of(model, Xc_t, Xw_t), 1).numpy()
    torch.save(model.state_dict(), os.path.join(S.ROOT, "checkpoints", f"cnn_f{k}.pt")); best_ep = int(np.argmin([h["inner_loss"] for h in hist]))
    json.dump(dict(fold=k, best_epoch=best_ep, n_epochs=len(hist), seconds=time.time() - t_start, n_char_vocab=len(cv), n_word_vocab=len(wv), history=hist), open(os.path.join(RUNS, f"f{k}.json"), "w"))
    np.savez(os.path.join(RUNS, f"f{k}.npz"), va=va, pv=pv, pt=pt)
    return k, best_ep, len(hist), round(time.time() - t_start), float(hist[best_ep]["mon_loss"])


def assemble():
    """Out-of-fold probabilities, fold-averaged test probabilities and the per-fold run records; None if a fold is missing."""
    train, test, folds, y = S.load(); oof, tst, runs = np.zeros((len(y), 3)), 0, []
    for k in range(5):
        p = os.path.join(RUNS, f"f{k}")
        if not os.path.exists(p + ".npz"): return None
        z = np.load(p + ".npz"); oof[z["va"]] = z["pv"]; tst = tst + z["pt"] / 5; runs.append(json.load(open(p + ".json")))
    return oof, tst, runs


if __name__ == "__main__":
    smoke = bool(os.environ.get("SMOKE")); epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 10; threads = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    t0 = time.time(); folds_to_run = [0] if smoke else range(5)
    with ProcessPoolExecutor(len(list(folds_to_run)), mp_context=get_context("spawn")) as ex:
        for k, be, ne, sec, ml in ex.map(run_fold, folds_to_run, [epochs] * 5, [threads] * 5, [smoke] * 5): print(f"fold {k}: {ne} epochs, best epoch {be}, held-out log loss {ml:.4f}, {sec}s", flush=True)
    print(f"done in {time.time() - t0:.0f}s")
