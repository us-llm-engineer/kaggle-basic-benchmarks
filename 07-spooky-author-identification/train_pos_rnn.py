"""Methodology 7: syntactic (POS-only) neural network for authorship attribution, from scratch, after Jafariakinabad et al. 2019 (arXiv 1902.09723).
Input: Penn Treebank POS tags of ONE sentence (NLTK, label-free) -> trainable tag embedding theta_P (no pretrained weights) -> a sentence encoder -> softmax over the 3 authors.
Variants (the paper's hierarchy needs 100-sentence segments; our rows are single independent sentences, so the document-level BiLSTM/attention has nothing to aggregate and is dropped - stated deviation):
  cnn   : parallel 1-D convolutions with receptive fields 3 and 5, ReLU, temporal max-pool, concatenated            (paper's 'syntactic CNN', sentence level)
  bilstm: BiLSTM, sentence vector = UNWEIGHTED SUM of the hidden states                                             (paper's 'syntactic BiLSTM', sentence level)
  fusion: both paper encoders on the same tags, concatenated (CNN vector + summed-BiLSTM vector); inspired_by (the paper compares them, never fuses them)
  attn  : BiLSTM + learned-context-vector attention OVER TAGS, u_i = tanh(W h_i + b), a_i = softmax(u_i . u_s)       (the paper's attention, moved from sentences to tags: inspired_by)
Optimiser NAdam with L2 weight decay (paper: Nadam + L2 regularisation); EPOCH CAP 120 (paper: 30) with early stop after 20 epochs without inner-loss improvement. d_p, d_l, lambda, batch are not reported by the paper: chosen here, selection only on an inner 10% split.
The outer fold is MONITORED every epoch and never selects anything. Statistics for interpretation are written every epoch (jsonl log, atomically replaced history json, checkpoints).
    python3 train_pos_rnn.py <fold> <variant> [epochs] [out_dir]"""
import json, os, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import train_test_split
import spooky_common as S

MAXLEN = 64; CFG = dict(lr=2e-3, wd=1e-4, batch=128, d_p=32, d_l=64, ch=96, dropout=0.3)

class Net(nn.Module):
    def __init__(self, n_tags, variant, c=CFG):
        super().__init__(); self.v = variant; self.emb = nn.Embedding(n_tags, c["d_p"], padding_idx=0); self.drop = nn.Dropout(c["dropout"])
        if variant in ("cnn", "fusion"): self.conv = nn.ModuleList([nn.Linear(c["d_p"] * r, c["ch"]) for r in (3, 5)]); self.rs = (3, 5); dim = 2 * c["ch"]   # 1-D convolution of receptive field r written as unfold + linear: identical function and parameter count (checked to 2e-7), avoids a slow cuDNN path on tiny channels
        if variant == "fusion": self.rnn = nn.LSTM(c["d_p"], c["d_l"], batch_first=True, bidirectional=True); dim += 2 * c["d_l"]
        elif variant != "cnn":
            self.rnn = nn.LSTM(c["d_p"], c["d_l"], batch_first=True, bidirectional=True); dim = 2 * c["d_l"]
            if variant == "attn": self.W = nn.Linear(dim, dim); self.us = nn.Parameter(torch.randn(dim) * 0.1)
        self.out = nn.Linear(dim, 3)
    def forward(self, x, lens=None):
        m = max(int(x.ne(0).sum(1).max()), 1) if lens is None else int(lens.max()); x = x[:, :min(((m + 7) // 8) * 8, x.shape[1])]   # trim to the batch maximum rounded up to a multiple of 8: few distinct shapes keep cuDNN plans cached
        mask = x.ne(0); e = self.drop(self.emb(x))
        if self.v in ("cnn", "fusion"): h = torch.cat([F.relu(cv(F.pad(e, (0, 0, r // 2, r // 2)).unfold(1, r, 1).transpose(2, 3).reshape(e.shape[0], e.shape[1], -1))).masked_fill(~mask.unsqueeze(-1), -1e4).max(1).values for r, cv in zip(self.rs, self.conv)], 1)
        if self.v == "fusion":
            o, _ = self.rnn(e); h = torch.cat([h, (o * mask.unsqueeze(-1)).sum(1)], 1)
        elif self.v != "cnn":
            o, _ = self.rnn(e)
            if self.v == "bilstm": h = (o * mask.unsqueeze(-1)).sum(1)
            else: a = torch.softmax((torch.tanh(self.W(o)) @ self.us).masked_fill(~mask, -1e9), 1); self.last_attn_entropy = float(-(a * a.clamp_min(1e-12).log()).sum(1).mean()); h = (a.unsqueeze(-1) * o).sum(1)
        return self.out(self.drop(h))

def extra_stats(z, t, model):
    """Interpretation aids: inner confusion matrix, logit margin (top1 - top2), share of the log loss carried by misclassified rows, per-layer weight norms."""
    p = F.softmax(z, 1); pred = p.argmax(1); nll = -torch.log(p.gather(1, t[:, None]).squeeze(1).clamp_min(1e-12)); top2 = z.topk(2, 1).values
    cm = torch.zeros(3, 3, dtype=torch.long); 
    for a, b in zip(t.cpu().tolist(), pred.cpu().tolist()): cm[a, b] += 1
    wrong = pred != t
    return dict(inner_confusion=cm.tolist(), inner_margin_mean=float((top2[:, 0] - top2[:, 1]).mean()), inner_margin_wrong=float((top2[:, 0] - top2[:, 1])[wrong].mean()) if wrong.any() else 0.0,
                inner_loss_share_from_errors=float(nll[wrong].sum() / nll.sum()), weight_norms={n: float(q.detach().norm()) for n, q in model.named_parameters()})

def ece(p, t, bins=10):
    conf, pred = p.max(1).values, p.argmax(1); ok = (pred == t).float(); r = 0.0
    for lo in np.linspace(0, 0.9, bins):
        m = (conf >= lo) & (conf < lo + 0.1)
        if m.any(): r += float(m.float().mean() * (ok[m].mean() - conf[m].mean()).abs())
    return r

@torch.no_grad()
def logits_of(model, x, lens, bs=1024):
    model.eval(); return torch.cat([model(x[i:i + bs], lens[i:i + bs]) for i in range(0, len(x), bs)])

def run(k, variant, epochs=120, out="results/pos_rnn_runs", device=None, limit=None, patience=20):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu"); torch.manual_seed(1000 + k); np.random.seed(k); t_start = time.time()
    train, test, folds, y = S.load(); tags = json.load(open(os.path.join(S.RESULTS, "pos_tags.json"))); vocab = {t: i + 1 for i, t in enumerate(sorted({a for s in tags for a in s}))}
    enc = lambda seqs: torch.tensor([[vocab[a] for a in s[:MAXLEN]] + [0] * max(0, MAXLEN - len(s)) for s in seqs])
    n = len(train); Xtr_all, Xte = enc(tags[:n]), enc(tags[n:]); tr, va = np.where(folds != k)[0], np.where(folds == k)[0]
    if limit: tr, va = tr[:limit], va[:limit // 4]
    i_a, i_i = train_test_split(np.arange(len(tr)), test_size=0.1, stratify=y[tr], random_state=k); a_idx, i_idx = tr[i_a], tr[i_i]
    L = lambda X: X.ne(0).sum(1).clamp_min(1).cpu(); xa, xi, xv, xt = (z.to(device) for z in (Xtr_all[a_idx], Xtr_all[i_idx], Xtr_all[va], Xte)); la, li_, lv_, lt_ = (L(z) for z in (Xtr_all[a_idx], Xtr_all[i_idx], Xtr_all[va], Xte)); ya, yi, yv = (torch.tensor(y[z], device=device) for z in (a_idx, i_idx, va))
    model = Net(len(vocab) + 1, variant).to(device); opt = torch.optim.NAdam(model.parameters(), lr=CFG["lr"], weight_decay=CFG["wd"]); sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=4)
    os.makedirs(out, exist_ok=True); tag = f"{variant}_f{k}"; hist, best, best_ep = [], 9.0, 0; logf = open(os.path.join(out, f"{tag}.log"), "w")
    p0 = torch.cat([p.detach().flatten() for p in model.parameters()]); n_params = int(p0.numel()); chk = dict()
    for ep in range(epochs):
        t0 = time.time(); model.train(); perm = torch.randperm(len(ya)); tot, gns, upd = 0.0, [], []; nstep = 0
        if device == "cuda": torch.cuda.reset_peak_memory_stats()
        for i in range(0, len(perm), CFG["batch"]):
            j = perm[i:i + CFG["batch"]]; jg = j.to(device); loss = F.cross_entropy(model(xa[jg], la[j]), ya[jg]); opt.zero_grad(set_to_none=True); loss.backward()
            gn = nn.utils.clip_grad_norm_(model.parameters(), 5.0); sample = nstep % 10 == 0; w0 = torch.cat([p.detach().flatten() for p in model.parameters()]) if sample else None; opt.step(); nstep += 1
            gns.append(gn.detach()); tot += loss.detach() * len(j)
            if sample: upd.append(((torch.cat([p.detach().flatten() for p in model.parameters()]) - w0).norm() / (w0.norm() + 1e-12)).detach())
        tot = float(tot); gns = [float(g) for g in gns]; upd = [float(u) for u in upd]; zi, zv = logits_of(model, xi, li_), logits_of(model, xv, lv_); pi, pv = F.softmax(zi, 1), F.softmax(zv, 1); li, lv = float(F.cross_entropy(zi, yi)), float(F.cross_entropy(zv, yv)); sched.step(li)
        pcls = [float((pi.argmax(1)[yi == c] == c).float().mean()) for c in range(3)]; w = torch.cat([p.detach().flatten() for p in model.parameters()])
        row = dict(epoch=ep, train_loss=tot / len(ya), inner_loss=li, inner_acc=float((pi.argmax(1) == yi).float().mean()), inner_mean_conf=float(pi.max(1).values.mean()), inner_entropy=float((-(pi * pi.clamp_min(1e-12).log()).sum(1)).mean()),
                   inner_ece=ece(pi, yi), inner_acc_per_class=pcls, inner_wrong_conf_gt_09=int(((pi.max(1).values > 0.9) & (pi.argmax(1) != yi)).sum()), mon_outer_loss=lv, mon_outer_acc=float((pv.argmax(1) == yv).float().mean()),
                   grad_norm_mean=float(np.mean(gns)), grad_norm_max=float(np.max(gns)), update_ratio_mean=float(np.mean(upd)), param_norm=float(w.norm()), param_dist_from_init=float((w - p0).norm()), lr=opt.param_groups[0]["lr"],
                   **extra_stats(zi, yi, model), attn_entropy=getattr(model, "last_attn_entropy", None), seconds=time.time() - t0, gpu_peak_mib=(torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else 0.0), rows_per_second=len(ya) / (time.time() - t0)); hist.append(row)
        logf.write(json.dumps(row) + "\n"); logf.flush(); tmp = os.path.join(out, f".{tag}.tmp")
        json.dump(dict(fold=k, variant=variant, cfg=CFG, n_params=n_params, epochs_done=ep + 1, history=hist), open(tmp, "w")); os.replace(tmp, os.path.join(out, f"{tag}_history.json"))
        if li < best - 1e-5: best, best_ep = li, ep; torch.save(dict(epoch=ep, state=model.state_dict(), vocab=vocab), os.path.join(out, f"{tag}_best.pt"))
        if ep % 10 == 0 or ep == epochs - 1: torch.save(dict(epoch=ep, state=model.state_dict()), os.path.join(out, f"{tag}_epoch{ep:03d}.pt"))
        if ep - best_ep >= patience: print(f"{tag} stop: no inner improvement for {patience} epochs", flush=True); json.dump(dict(fold=k, variant=variant, cfg=CFG, n_params=n_params, epochs_done=ep + 1, stopped_by_patience=True, history=hist), open(tmp, "w")); os.replace(tmp, os.path.join(out, f"{tag}_history.json")); break
        print(f"{tag} ep {ep} train {row['train_loss']:.4f} inner {li:.4f} acc {row['inner_acc']:.4f} conf {row['inner_mean_conf']:.3f} ece {row['inner_ece']:.3f} mon {lv:.4f} {row['seconds']:.1f}s", flush=True)
    st = torch.load(os.path.join(out, f"{tag}_best.pt"), map_location=device, weights_only=False); model.load_state_dict(st["state"]); pvf, ptf = F.softmax(logits_of(model, xv, lv_), 1).cpu().numpy(), F.softmax(logits_of(model, xt, lt_), 1).cpu().numpy()
    np.savez(os.path.join(out, f"{tag}.npz"), va=va, pv=pvf, pt=ptf, best_epoch=st["epoch"]); print(f"{tag} done best epoch {st['epoch']} outer loss {S.sample_loss(y[va], pvf).mean():.4f} in {time.time()-t_start:.0f}s", flush=True)

if __name__ == "__main__":
    a = sys.argv; run(int(a[1]), a[2], int(a[3]) if len(a) > 3 else 80, a[4] if len(a) > 4 else "results/pos_rnn_runs", limit=(int(os.environ["LIMIT"]) if os.environ.get("LIMIT") else None))
