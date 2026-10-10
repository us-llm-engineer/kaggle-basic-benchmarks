"""Methodology 8: shallow ANN on TF-IDF character n-grams, from scratch, after Oldies but Goldies (arXiv 2506.15650): TfidfVectorizer(analyzer='char') for ONE n-gram size N, or N = 25 meaning the concatenated N = 2..5 matrix of the paper's Sec.3.3.4 (the paper evaluates N = 2..5 separately in its tables; 4 and 5 are best),
ALL n-grams retained (min_df=1, no feature cap), original case, hidden layers (100, 50) with ReLU, softmax output. Paper does not report optimiser / epochs / regularisation / batch size: here Adam (lr 1e-3), L2 alpha = 1e-4 and batch 200
(scikit-learn MLPClassifier defaults, the library the paper uses), EPOCH CAP 120 with early stop after 20 epochs without inner-loss improvement, best epoch chosen on an inner 10% split of the fold's training rows (the outer fold is only monitored). The vectoriser is fitted on the training rows of each fold only.
Statistics collected every epoch (jsonl + atomically replaced history + checkpoints): losses, accuracy, confidence, entropy, ECE, per-class accuracy, confident errors, gradient and update norms per layer, dead-ReLU fractions, hidden activation norms, parameter drift, throughput, GPU memory.
    python3 train_char_mlp.py <fold> <N> [epochs] [out_dir] [seed]"""
import json, os, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import spooky_common as S

CFG = dict(lr=1e-3, alpha=1e-4, batch=200, h1=100, h2=50)

class MLP(nn.Module):
    def __init__(self, n_in, c=CFG):
        super().__init__(); self.W1 = nn.Parameter(torch.empty(n_in, c["h1"])); nn.init.xavier_uniform_(self.W1); self.b1 = nn.Parameter(torch.zeros(c["h1"])); self.l2 = nn.Linear(c["h1"], c["h2"]); self.out = nn.Linear(c["h2"], 3)
    def forward(self, X, stats=False):
        h1 = F.relu(torch.sparse.mm(X, self.W1) + self.b1); h2 = F.relu(self.l2(h1)); z = self.out(h2)
        return (z, h1, h2) if stats else z

def to_t(X, device): X = X.tocsr().astype(np.float32); return torch.sparse_csr_tensor(torch.from_numpy(X.indptr).long(), torch.from_numpy(X.indices).long(), torch.from_numpy(X.data), size=X.shape).to(device)
class GPUCSR:
    """Whole CSR matrix resident on the GPU; a batch of rows is gathered with GPU ops only (no scipy slicing, no host copies per step)."""
    def __init__(self, X, device):
        X = X.tocsr().astype(np.float32); self.ip = torch.from_numpy(X.indptr).long().to(device); self.ix = torch.from_numpy(X.indices).long().to(device); self.dt = torch.from_numpy(X.data).to(device); self.shape = X.shape; self.device = device
    def rows(self, j):
        st = self.ip[j]; ln = self.ip[j + 1] - st; crow = torch.cat([torch.zeros(1, dtype=torch.long, device=self.device), ln.cumsum(0)]); tot = int(crow[-1])
        pos = torch.arange(tot, device=self.device) - torch.repeat_interleave(crow[:-1], ln) + torch.repeat_interleave(st, ln)
        return torch.sparse_csr_tensor(crow, self.ix[pos], self.dt[pos], size=(len(j), self.shape[1]))

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
def forward_all(model, X, bs=1024):
    model.eval(); zs, d1, d2 = [], [], []
    for i in range(0, X.shape[0], bs): z, h1, h2 = model(to_t(X[i:i + bs], next(model.parameters()).device), True); zs.append(z); d1.append(float((h1 == 0).float().mean())); d2.append(float((h2 == 0).float().mean()))
    return torch.cat(zs), float(np.mean(d1)), float(np.mean(d2))

def run(k, N, epochs=120, out="results/char_mlp_runs", seed=7, device=None, limit=None, patience=20):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu"); torch.manual_seed(seed + 100 * k); t_start = time.time()
    train, test, folds, y = S.load(); txt = train.text.values; tr, va = np.where(folds != k)[0], np.where(folds == k)[0]
    if limit: tr, va = tr[:limit], va[:limit // 4]
    i_a, i_i = train_test_split(np.arange(len(tr)), test_size=0.1, stratify=y[tr], random_state=k); a_idx, i_idx = tr[i_a], tr[i_i]
    vec = TfidfVectorizer(analyzer="char", ngram_range=((2, 5) if N == 25 else (N, N)), lowercase=False); Xa = vec.fit_transform(txt[a_idx]).tocsr(); Xi, Xv, Xt = vec.transform(txt[i_idx]), vec.transform(txt[va]), vec.transform(test.text.values)
    ya, yi, yv = (torch.tensor(y[z], device=device) for z in (a_idx, i_idx, va)); Ga = GPUCSR(Xa, device); model = MLP(Xa.shape[1]).to(device); opt = torch.optim.Adam(model.parameters(), lr=CFG["lr"], weight_decay=CFG["alpha"])
    os.makedirs(out, exist_ok=True); tag = f"N{N}_s{seed}_f{k}"; hist, best, best_ep = [], 9.0, 0; logf = open(os.path.join(out, f"{tag}.log"), "w"); p0 = torch.cat([p.detach().flatten() for p in model.parameters()]); n_params = int(p0.numel()); g = torch.Generator().manual_seed(seed + k)
    for ep in range(epochs):
        t0 = time.time(); model.train(); perm = torch.randperm(Xa.shape[0], generator=g).numpy(); tot, gn, nb = 0.0, {"W1": [], "l2": [], "out": []}, 0
        if device == "cuda": torch.cuda.reset_peak_memory_stats()
        for i in range(0, len(perm), CFG["batch"]):
            j = perm[i:i + CFG["batch"]]; jg = torch.from_numpy(j).to(device); loss = F.cross_entropy(model(Ga.rows(jg)), ya[jg]); opt.zero_grad(set_to_none=True); loss.backward()
            gn["W1"].append(model.W1.grad.norm()); gn["l2"].append(model.l2.weight.grad.norm()); gn["out"].append(model.out.weight.grad.norm()); opt.step(); tot += loss.detach() * len(j)
        tot = float(tot); gn = {a: [float(v) for v in b] for a, b in gn.items()}; zi, d1, d2 = forward_all(model, Xi); zv, _, _ = forward_all(model, Xv); pi = F.softmax(zi, 1); li, lv = float(F.cross_entropy(zi, yi)), float(F.cross_entropy(zv, yv)); w = torch.cat([p.detach().flatten() for p in model.parameters()])
        row = dict(epoch=ep, train_loss=tot / len(ya), inner_loss=li, inner_acc=float((pi.argmax(1) == yi).float().mean()), inner_mean_conf=float(pi.max(1).values.mean()), inner_entropy=float((-(pi * pi.clamp_min(1e-12).log()).sum(1)).mean()), inner_ece=ece(pi, yi),
                   inner_acc_per_class=[float((pi.argmax(1)[yi == c] == c).float().mean()) for c in range(3)], inner_wrong_conf_gt_09=int(((pi.max(1).values > 0.9) & (pi.argmax(1) != yi)).sum()), mon_outer_loss=lv, mon_outer_acc=float((zv.argmax(1) == yv).float().mean()),
                   grad_norm_W1=float(np.mean(gn["W1"])), grad_norm_l2=float(np.mean(gn["l2"])), grad_norm_out=float(np.mean(gn["out"])), dead_relu_h1=d1, dead_relu_h2=d2, param_norm=float(w.norm()), param_dist_from_init=float((w - p0).norm()),
                   **extra_stats(zi, yi, model), seconds=time.time() - t0, rows_per_second=len(ya) / (time.time() - t0), gpu_peak_mib=(torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else 0.0)); hist.append(row)
        logf.write(json.dumps(row) + "\n"); logf.flush(); tmp = os.path.join(out, f".{tag}.tmp"); json.dump(dict(fold=k, N=N, seed=seed, cfg=CFG, n_features=int(Xa.shape[1]), n_params=n_params, epochs_done=ep + 1, history=hist), open(tmp, "w")); os.replace(tmp, os.path.join(out, f"{tag}_history.json"))
        if li < best - 1e-5: best, best_ep = li, ep; torch.save(dict(epoch=ep, state=model.state_dict()), os.path.join(out, f"{tag}_best.pt"))
        if ep - best_ep >= patience: print(f"{tag} stop: no inner improvement for {patience} epochs", flush=True); break
        print(f"{tag} ep {ep} train {row['train_loss']:.4f} inner {li:.4f} acc {row['inner_acc']:.4f} conf {row['inner_mean_conf']:.3f} ece {row['inner_ece']:.3f} dead {d1:.2f}/{d2:.2f} mon {lv:.4f} {row['seconds']:.1f}s", flush=True)
    st = torch.load(os.path.join(out, f"{tag}_best.pt"), map_location=device, weights_only=False); model.load_state_dict(st["state"]); pvf = F.softmax(forward_all(model, Xv)[0], 1).cpu().numpy(); ptf = F.softmax(forward_all(model, Xt)[0], 1).cpu().numpy()
    np.savez(os.path.join(out, f"{tag}.npz"), va=va, pv=pvf, pt=ptf, best_epoch=st["epoch"]); print(f"{tag} done best epoch {st['epoch']} outer loss {S.sample_loss(y[va], pvf).mean():.4f} in {time.time()-t_start:.0f}s", flush=True)

if __name__ == "__main__":
    a = sys.argv; run(int(a[1]), int(a[2]), int(a[3]) if len(a) > 3 else 80, a[4] if len(a) > 4 else "results/char_mlp_runs", int(a[5]) if len(a) > 5 else 7, limit=(int(os.environ["LIMIT"]) if os.environ.get("LIMIT") else None))
