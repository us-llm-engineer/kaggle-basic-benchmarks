"""Methodology 6 (replacement for the scratch deep screens): a fastText-style classifier, from scratch (Joulin et al. 2016, arXiv 1607.01759, Sec.2).
Words and hashed word bigrams are embedded (look-up matrix A, dimension h), AVERAGED into a text vector, passed through one linear layer B and a softmax; the loss is the
negative log-likelihood -1/N sum y log f(BAx); SGD with a linearly decaying learning rate (Sec.2, Sec.3). Per the paper's sentiment setting h is small and the epoch count is 5; here h, the
learning rate and the epoch count are chosen on an INNER 10% split of each fold's training rows (the outer fold selects nothing), and the best inner-loss epoch is kept.
Fidelity: reproduces the architecture and optimiser, no hierarchical softmax (3 classes). Character n-grams are NOT in this paper (they come from later subword work), so the optional
'chargrams' variant is labelled inspired_by.
    python3 train_fasttext.py <fold> <variant: word|char> <hidden> <lr> <epochs>  ->  results/fasttext_runs/{variant}_h{hidden}_f{fold}.json/npz"""
import json, os, re, sys, time, zlib
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import train_test_split
import spooky_common as S

RUNS = os.path.join(S.RESULTS, "fasttext_runs"); TOKEN = re.compile(r"\w+|[^\w\s]"); BUCKETS = 2_000_000

def feats(text, variant):
    w = TOKEN.findall(text.lower()); f = ["u:" + t for t in w] + ["b:" + a + " " + b for a, b in zip(w, w[1:])]
    if variant == "char":
        t = "<" + text + ">"; f += ["c:" + t[i:i + n] for n in (3, 4, 5) for i in range(len(t) - n + 1)]
    return [zlib.crc32(x.encode()) % BUCKETS for x in f] or [0]

def bag(texts, variant):
    ids, off = [], [0]
    for t in texts: f = feats(t, variant); ids += f; off.append(off[-1] + len(f))
    return torch.tensor(ids), torch.tensor(off[:-1])

def batches(texts, variant, idx):
    f = [feats(texts[i], variant) for i in idx]; off = np.cumsum([0] + [len(x) for x in f])[:-1]
    return torch.tensor([i for x in f for i in x]), torch.tensor(off)

class FT(nn.Module):
    def __init__(self, h):
        super().__init__(); self.A = nn.EmbeddingBag(BUCKETS, h, mode="mean", sparse=True); self.B = nn.Linear(h, 3); nn.init.uniform_(self.A.weight, -1 / h, 1 / h)
    def forward(self, ids, off): return self.B(self.A(ids, off))

@torch.no_grad()
def predict(m, texts, variant, bs=2048):
    m.eval(); return torch.cat([F.softmax(m(*batches(texts, variant, range(i, min(i + bs, len(texts))))), 1) for i in range(0, len(texts), bs)]).numpy()

def run(k, variant, h, lr0, epochs):
    torch.set_num_threads(1); torch.manual_seed(7 + k); t0 = time.time(); train, test, folds, y = S.load(); txt = train.text.values
    tr, va = np.where(folds != k)[0], np.where(folds == k)[0]; i_tr, i_in = train_test_split(np.arange(len(tr)), test_size=0.1, stratify=y[tr], random_state=k)
    a, b = tr[i_tr], tr[i_in]; m = FT(h); opt_s = torch.optim.SGD(m.A.parameters(), lr=lr0); opt_d = torch.optim.SGD(m.B.parameters(), lr=lr0); g = np.random.default_rng(k)
    steps = epochs * int(np.ceil(len(a) / 32)); step = 0; best, best_state, hist = 9, None, []; os.makedirs(RUNS, exist_ok=True); tag = f"{variant}_h{h}"
    for ep in range(epochs):
        m.train(); perm = g.permutation(len(a)); tot = 0.0
        for i in range(0, len(perm), 32):
            lr = lr0 * (1 - step / steps); step += 1
            for o in (opt_s, opt_d):
                for pg in o.param_groups: pg["lr"] = lr
            idx = a[perm[i:i + 32]]; loss = F.cross_entropy(m(*batches(txt, variant, idx)), torch.tensor(y[idx]), reduction="sum")   # sum = per-example SGD steps of the paper, not a batch mean
            opt_s.zero_grad(); opt_d.zero_grad(); loss.backward(); opt_s.step(); opt_d.step(); tot += float(loss.detach())
        pin = predict(m, txt[b], variant); li = float(S.sample_loss(y[b], pin).mean()); hist.append(dict(epoch=ep, train_loss=tot / len(a), inner_loss=li, inner_acc=float((pin.argmax(1) == y[b]).mean()), seconds=time.time() - t0))
        print(f"{tag} fold {k} ep {ep} train {tot/len(a):.4f} inner {li:.4f} acc {hist[-1]['inner_acc']:.4f} {time.time()-t0:.0f}s", flush=True)
        if li < best: best, best_state = li, {n: v.clone() for n, v in m.state_dict().items()}
    m.load_state_dict(best_state); pv, pt = predict(m, txt[va], variant), predict(m, test.text.values, variant)
    np.savez(os.path.join(RUNS, f"{tag}_f{k}.npz"), va=va, pv=pv, pt=pt); json.dump(dict(fold=k, best_epoch=int(np.argmin([x["inner_loss"] for x in hist])), history=hist, h=h, lr=lr0, epochs=epochs, variant=variant), open(os.path.join(RUNS, f"{tag}_f{k}.json"), "w"))

if __name__ == "__main__":
    run(int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5]))
