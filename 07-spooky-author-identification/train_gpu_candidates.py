"""GPU-only candidate methods: scratch character BiGRU attention and Transformer encoder.

Run this source on a CUDA machine only. Both loops use cuda placement, checkpoint each
epoch, append a progress log, and atomically replace a JSON history file.
"""
import json, os, re, time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import spooky_common as S

TOKEN = re.compile(r"\w+|[^\w\s]")


class Vocab:
    def __init__(self, seqs, limit=128):
        c = Counter(x for s in seqs for x in s); self.stoi = {"<pad>": 0, "<unk>": 1}
        self.stoi.update({x: i + 2 for i, (x, _) in enumerate(c.most_common(limit - 2))})
    def encode(self, seq, n): return [self.stoi.get(x, 1) for x in seq[:n]] + [0] * max(0, n-len(seq))


class CharAttention(nn.Module):
    """Bidirectional character GRU with learned attention over a short passage."""
    def __init__(self, n):
        super().__init__(); self.emb=nn.Embedding(n, 64, padding_idx=0); self.rnn=nn.GRU(64, 96, batch_first=True, bidirectional=True); self.att=nn.Linear(192, 1); self.drop=nn.Dropout(0.25); self.out=nn.Linear(192,3)
    def forward(self, x):
        h,_=self.rnn(self.emb(x)); a=torch.softmax(self.att(h).masked_fill(x.unsqueeze(-1).eq(0), -1e9),1); return self.out(self.drop((a*h).sum(1)))


class CharTransformer(nn.Module):
    def __init__(self, n, length=256):
        super().__init__(); self.e=nn.Embedding(n,128,padding_idx=0); self.p=nn.Embedding(length,128); layer=nn.TransformerEncoderLayer(128,4,256,0.15,batch_first=True,norm_first=True); self.enc=nn.TransformerEncoder(layer,3); self.out=nn.Linear(128,3)
    def forward(self,x):
        pos=torch.arange(x.shape[1],device=x.device); h=self.e(x)+self.p(pos)[None]; h=self.enc(h,src_key_padding_mask=x.eq(0)); return self.out(h.masked_fill(x.unsqueeze(-1).eq(0),0).sum(1)/x.ne(0).sum(1,keepdim=True).clamp_min(1))


@torch.no_grad()
def probabilities(model, x, batch):
    """Bound inference memory: test/validation arrays must never become one GPU batch."""
    model.eval()
    return torch.cat([torch.softmax(model(x[i:i + batch]), 1) for i in range(0, len(x), batch)]).cpu().numpy()


@torch.no_grad()
def logits_batched(model, x, batch):
    model.eval()
    return torch.cat([model(x[i:i + batch]) for i in range(0, len(x), batch)])


def calibration_error(prob, target, bins=10):
    """Expected calibration error on the inner holdout, for monitoring only."""
    conf, pred = prob.max(1).values, prob.argmax(1); correct = (pred == target).float(); out = torch.zeros((), device=prob.device)
    for lo in torch.linspace(0, 0.9, bins, device=prob.device):
        mask = (conf >= lo) & (conf < lo + 0.1)
        if mask.any(): out += mask.float().mean() * (correct[mask].mean() - conf[mask].mean()).abs()
    return float(out)


def save_monitor_plot(history, path, title):
    """Overwrite a compact, inspectable training dashboard from complete epoch records."""
    if not history: return
    e = [r["epoch"] + 1 for r in history]; fig, ax = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    for a, keys, label in ((ax[0,0], ("train_loss", "inner_loss"), "loss"), (ax[0,1], ("inner_accuracy", "mean_confidence"), "fraction"),
                           (ax[0,2], ("entropy", "ece"), "uncertainty"), (ax[1,0], ("grad_norm",), "norm"),
                           (ax[1,1], ("rows_per_second",), "rows/sec"), (ax[1,2], ("gpu_peak_mib",), "MiB")):
        for key in keys: a.plot(e, [r[key] for r in history], label=key)
        a.set(xlabel="epoch", ylabel=label); a.grid(alpha=.25); a.legend(fontsize=8)
    fig.suptitle(title); tmp = path + ".tmp.png"; fig.savefig(tmp, dpi=140); plt.close(fig); os.replace(tmp, path)


def run_fold(kind, fold, epochs=80, batch=None):
    assert torch.cuda.is_available(), "CUDA GPU required"; device=torch.device("cuda"); torch.manual_seed(100+fold)
    train,test,folds,y=S.load(); tr=np.where(folds!=fold)[0]; va=np.where(folds==fold)[0]; inner,hold=train_test_split(tr,test_size=.1,stratify=y[tr],random_state=fold)
    if kind not in {"char_gru", "char_transformer"}:
        raise ValueError("kind must be char_gru or char_transformer")
    seq=lambda texts: [list(t) for t in texts]
    vocab=Vocab(seq(train.text.values[inner]))
    length=128 if kind == "char_gru" else 160
    enc=lambda rows: torch.tensor([vocab.encode(s,length) for s in seq(rows)], dtype=torch.long)
    xa,xi,xv,xt=map(lambda z:z.to(device),(enc(train.text.values[inner]),enc(train.text.values[hold]),enc(train.text.values[va]),enc(test.text.values))); ya,yi,yv=(torch.tensor(y[z],device=device) for z in (inner,hold,va))
    model=(CharAttention(len(vocab.stoi)) if kind=='char_gru' else CharTransformer(len(vocab.stoi),length)).to(device)
    batch = batch or (512 if kind == "char_gru" else 96)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=.01)
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)
    root=os.path.join(S.RESULTS,'gpu_candidates'); os.makedirs(root,exist_ok=True); log=os.path.join(root,f'{kind}_f{fold}.log'); hist=[]; best=1e9
    for ep in range(epochs):
        epoch_start=time.time(); torch.cuda.reset_peak_memory_stats(device); model.train(); perm=torch.randperm(len(ya),device=device); total=0.; norms=[]
        for i in range(0,len(perm),batch):
            j=perm[i:i+batch]; loss=F.cross_entropy(model(xa[j]),ya[j]); opt.zero_grad(set_to_none=True); loss.backward(); norms.append(float(nn.utils.clip_grad_norm_(model.parameters(),5))); opt.step(); total+=float(loss.detach())*len(j)
        model.eval()
        z_inner=logits_batched(model,xi,batch); prob=torch.softmax(z_inner,1); li=float(F.cross_entropy(z_inner,yi)); conf=prob.max(1).values
        sched.step(li); row=dict(epoch=ep,train_loss=total/len(ya),inner_loss=li,inner_accuracy=float((prob.argmax(1)==yi).float().mean()),mean_confidence=float(conf.mean()),entropy=float((-(prob*prob.clamp_min(1e-12).log()).sum(1)).mean()),ece=calibration_error(prob,yi),grad_norm=float(np.mean(norms)),rows_per_second=float(len(ya)/(time.time()-epoch_start)),gpu_peak_mib=round(torch.cuda.max_memory_allocated(device)/2**20,2),lr=opt.param_groups[0]["lr"]); hist.append(row)
        with open(log,'a') as f: f.write(json.dumps(row)+'\n')
        tmp=os.path.join(root,f'.{kind}_f{fold}.tmp')
        with open(tmp,'w') as f: json.dump(hist,f)
        os.replace(tmp,os.path.join(root,f'{kind}_f{fold}_history.json'))
        if ep % 5 == 0 or ep == epochs - 1: save_monitor_plot(hist, os.path.join(root,f'{kind}_f{fold}_monitor.png'), f'{kind}, fold {fold}')
        if ep % 10 == 0 or ep == epochs - 1: torch.save(dict(epoch=ep,state=model.state_dict(),vocab=vocab.stoi,length=length),os.path.join(root,f'{kind}_f{fold}_epoch{ep:03d}.pt'))
        if li<best: best=li; torch.save(dict(epoch=ep,state=model.state_dict(),vocab=vocab.stoi,length=length),os.path.join(root,f'{kind}_f{fold}.pt'))
        print(kind,fold,row,flush=True)
    state=torch.load(os.path.join(root,f'{kind}_f{fold}.pt'),map_location=device,weights_only=False); model.load_state_dict(state['state']); model.eval()
    pv=probabilities(model,xv,batch); pt=probabilities(model,xt,batch)
    final=dict(fold=fold,best_inner_loss=best,outer_log_loss=float(-np.log(S.clip_norm(pv)[np.arange(len(va)),y[va]]).mean()),outer_accuracy=float((pv.argmax(1)==y[va]).mean()))
    with open(os.path.join(root,f'{kind}_f{fold}_summary.json'),'w') as f: json.dump(final,f,indent=2)
    return va, pv, pt, final


def main(kind, epochs=80, batch=None):
    train, test, folds, y=S.load(); oof=np.zeros((len(train),3)); ptest=np.zeros((len(test),3)); meta=[]
    for fold in range(S.N_FOLDS):
        va,pv,pt,info=run_fold(kind,fold,epochs,batch); oof[va]=pv; ptest+=pt/S.N_FOLDS; meta.append(info)
    ev=S.evaluate(y,oof,folds)
    S.save_method(kind,oof,ptest,dict(method=kind,config=dict(epochs=epochs,batch=batch),folds=meta,**ev))
    print(json.dumps(dict(method=kind,**ev),indent=2),flush=True)


if __name__ == "__main__":
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument("kind",choices=("char_gru","char_transformer")); parser.add_argument("--epochs",type=int,default=80); parser.add_argument("--batch",type=int,default=None)
    args=parser.parse_args(); main(args.kind,args.epochs,args.batch)

