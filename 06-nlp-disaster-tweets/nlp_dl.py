"""Small neural text classifiers trained from scratch (no pre-trained weights): fastText-style bag of hashed n-grams and a
Kim-style text CNN with random embeddings. Training collects per-epoch statistics only (no figures) and writes a log line
and a history JSON after every epoch, so a running job can be inspected."""
import json
import os
import time
import zlib
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score

from nlp_common import ROOT

PAD, UNK = 0, 1
MAXLEN = 40


def tokens(text):
    return text.split()


class Vocab:
    """Word vocabulary from the training rows of one fold only (min_count filters rare words)."""

    def __init__(self, texts, min_count=2):
        c = Counter(w for t in texts for w in tokens(t))
        self.itos = ["<pad>", "<unk>"] + [w for w, n in c.most_common() if n >= min_count]
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, text, maxlen=MAXLEN):
        ids = [self.stoi.get(w, UNK) for w in tokens(text)][:maxlen]
        return ids + [PAD] * (maxlen - len(ids))


def bigram_bucket(a, b, n_buckets):
    """Deterministic hash of a word pair into one of n_buckets (zlib.crc32; Python's hash() is salted per process)."""
    return zlib.crc32(f"{a} {b}".encode()) % n_buckets


def encode_ngrams(texts, vocab, n_buckets, maxlen=MAXLEN):
    """fastText input: unigram ids, then hashed-bigram ids offset by len(vocab); padded with 0 to 2*maxlen."""
    out = np.zeros((len(texts), 2 * maxlen), dtype=np.int64)
    for i, t in enumerate(texts):
        ws = tokens(t)[:maxlen]
        uni = [vocab.stoi.get(w, UNK) for w in ws]
        bi = [len(vocab) + bigram_bucket(a, b, n_buckets) for a, b in zip(ws, ws[1:])]
        ids = uni + bi
        out[i, :len(ids)] = ids
    return out


class FastText(nn.Module):
    """Mean of embeddings of unigrams and hashed bigrams -> linear. Padding (id 0) is excluded from the mean."""

    def __init__(self, n_ids, dim=32, dropout=0.3):
        super().__init__()
        self.emb = nn.Embedding(n_ids, dim, padding_idx=PAD)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(dim, 1)
        nn.init.uniform_(self.emb.weight, -0.05, 0.05)
        with torch.no_grad():
            self.emb.weight[PAD].zero_()

    def forward(self, x):
        m = (x != PAD).float().unsqueeze(-1)
        h = (self.emb(x) * m).sum(1) / m.sum(1).clamp(min=1.0)
        return self.out(self.drop(h)).squeeze(-1)


class TextCNN(nn.Module):
    """Kim (2014) CNN-rand: random embeddings, parallel convolutions over widths, max-over-time pooling, dropout, linear."""

    def __init__(self, n_words, dim=100, widths=(2, 3, 4), channels=100, dropout=0.5, emb_dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(n_words, dim, padding_idx=PAD)
        self.convs = nn.ModuleList([nn.Conv1d(dim, channels, w) for w in widths])
        self.emb_drop, self.drop = nn.Dropout(emb_dropout), nn.Dropout(dropout)
        self.out = nn.Linear(channels * len(widths), 1)

    def forward_emb(self, e):
        e = self.emb_drop(e).transpose(1, 2)
        h = torch.cat([F.relu(c(e)).max(dim=2).values for c in self.convs], dim=1)
        return self.out(self.drop(h)).squeeze(-1)

    def forward(self, x):
        return self.forward_emb(self.emb(x))


@torch.no_grad()
def predict_logits(model, X, bs=1024):
    model.eval()
    return torch.cat([model(X[i:i + bs]) for i in range(0, len(X), bs)]).float().cpu().numpy()


def fit(model, Xtr, ytr, Xin, yin, Xmon, ymon, cfg, log=None):
    """Train with AdamW + BCE. The inner hold-out (Xin, yin) selects the epoch (lowest log loss) and stops training;
    (Xmon, ymon) is the outer validation fold, recorded as a trace only and never used to select anything.
    Returns the model restored to its best epoch and a list of per-epoch statistics."""
    dev = Xtr.device
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=cfg.get("lr_patience", 2))
    ytr_t = ytr.float()
    hist, best, best_state, bad = [], np.inf, None, 0
    g = torch.Generator(device="cpu").manual_seed(cfg["seed"])
    for ep in range(cfg["epochs"]):
        t0 = time.time(); model.train(); perm = torch.randperm(len(Xtr), generator=g).to(dev); tot, gn, nb = 0.0, 0.0, 0
        for i in range(0, len(perm), cfg["batch"]):
            idx = perm[i:i + cfg["batch"]]
            loss = F.binary_cross_entropy_with_logits(model(Xtr[idx]), ytr_t[idx])
            opt.zero_grad(set_to_none=True); loss.backward()
            gn += float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)); opt.step()
            tot += float(loss.detach()) * len(idx); nb += 1
        zin, zmon = predict_logits(model, Xin), predict_logits(model, Xmon)
        pin, pmon = 1 / (1 + np.exp(-zin)), 1 / (1 + np.exp(-zmon)); yin_n, ymon_n = yin.cpu().numpy(), ymon.cpu().numpy()
        vloss = float(-np.mean(yin_n * np.log(pin + 1e-9) + (1 - yin_n) * np.log(1 - pin + 1e-9)))
        sched.step(vloss)
        rec = dict(epoch=ep, seconds=time.time() - t0, lr=opt.param_groups[0]["lr"], train_loss=tot / len(Xtr), grad_norm=gn / nb,
                   inner_loss=vloss, inner_auc=float(roc_auc_score(yin_n, pin)), inner_f1=float(f1_score(yin_n, pin > .5)),
                   mon_auc=float(roc_auc_score(ymon_n, pmon)), mon_f1=float(f1_score(ymon_n, pmon > .5)),
                   emb_norm=float(model.emb.weight.detach().norm()))
        hist.append(rec)
        if log: log(f"ep {ep:02d} train {rec['train_loss']:.4f} inner {vloss:.4f} auc {rec['inner_auc']:.4f} mon_f1 {rec['mon_f1']:.4f} gn {rec['grad_norm']:.2f} lr {rec['lr']:.1e}")
        if vloss < best - 1e-4:
            best, bad = vloss, 0; best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg["patience"]: break
    model.load_state_dict(best_state)
    return model, hist


class RunLog:
    """Append-only text log, flushed after every line, so progress is visible while a job runs."""

    def __init__(self, name):
        os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
        self.path = os.path.join(ROOT, "logs", f"{name}.log"); self.f = open(self.path, "a")

    def __call__(self, msg):
        self.f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}\n"); self.f.flush()


def write_history(name, runs):
    """Rewrite results/dl_history_<name>.json atomically with the statistics of every (fold, seed) run so far."""
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    p = os.path.join(ROOT, "results", f"dl_history_{name}.json"); tmp = p + ".tmp"
    json.dump(runs, open(tmp, "w")); os.replace(tmp, p)
