"""Shared code for the v4 methodologies (v4a recipe, v4b Adaptive Wing + DARK, v4c capacity) and the seed control.

Coordinates follow the area convention: pixel i spans [i, i+1), image centre 48.0, flip x -> 96 - x (as v1/v2).
Data lives on the GPU and is augmented there per batch, so several training processes can share one L4 without
competing for CPU data-loader workers. Every run logs one line per epoch to logs/<run>.log and checkpoints on every
held-out improvement to checkpoints/<run>.pt, so a time cap or crash never loses the best weights (there is no automatic
resume: a restarted run starts from scratch unless its checkpoint is loaded by hand).
"""
import json
import math
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

IMG = 96
N_KP = 15
KEYPOINT_COLS = [
    "left_eye_center_x", "left_eye_center_y", "right_eye_center_x", "right_eye_center_y",
    "left_eye_inner_corner_x", "left_eye_inner_corner_y", "left_eye_outer_corner_x", "left_eye_outer_corner_y",
    "right_eye_inner_corner_x", "right_eye_inner_corner_y", "right_eye_outer_corner_x", "right_eye_outer_corner_y",
    "left_eyebrow_inner_end_x", "left_eyebrow_inner_end_y", "left_eyebrow_outer_end_x", "left_eyebrow_outer_end_y",
    "right_eyebrow_inner_end_x", "right_eyebrow_inner_end_y", "right_eyebrow_outer_end_x", "right_eyebrow_outer_end_y",
    "nose_tip_x", "nose_tip_y",
    "mouth_left_corner_x", "mouth_left_corner_y", "mouth_right_corner_x", "mouth_right_corner_y",
    "mouth_center_top_lip_x", "mouth_center_top_lip_y", "mouth_center_bottom_lip_x", "mouth_center_bottom_lip_y",
]
FLIP_SWAP_PAIRS = [
    ("left_eye_center", "right_eye_center"), ("left_eye_inner_corner", "right_eye_inner_corner"),
    ("left_eye_outer_corner", "right_eye_outer_corner"), ("left_eyebrow_inner_end", "right_eyebrow_inner_end"),
    ("left_eyebrow_outer_end", "right_eyebrow_outer_end"), ("mouth_left_corner", "mouth_right_corner"),
]
_ci = {c: i for i, c in enumerate(KEYPOINT_COLS)}
SWAP = list(range(30))
for _a, _b in FLIP_SWAP_PAIRS:
    for _s in ("_x", "_y"):
        SWAP[_ci[_a + _s]], SWAP[_ci[_b + _s]] = _ci[_b + _s], _ci[_a + _s]

LEGACY_RMSE = 2.7384
V2_RMSE = 2.4385


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------ data
def decode(df):
    return np.stack(df["Image"].apply(lambda s: np.array(s.split(), dtype=np.float32)).values).reshape(-1, IMG, IMG) / 255.0


def load_data(data_dir="data"):
    """Same split as v1/v2: train_test_split(test_size=0.12, random_state=42) -> idx_tr (88%), idx_va (held out)."""
    df = pd.read_csv(os.path.join(data_dir, "training.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    imgs = decode(df).astype(np.float32)
    test_imgs = decode(test_df).astype(np.float32)
    y = df[KEYPOINT_COLS].values.astype(np.float32)
    idx_tr, idx_va = train_test_split(np.arange(len(df)), test_size=0.12, random_state=42)
    return dict(imgs=imgs, y=y, idx_tr=idx_tr, idx_va=idx_va, test_imgs=test_imgs, test_ids=test_df["ImageId"].values)


def to_gpu(imgs, y, dev):
    x = torch.from_numpy(imgs).unsqueeze(1).to(dev)
    m = torch.from_numpy(~np.isnan(y)).float().to(dev)
    t = torch.from_numpy(np.nan_to_num(y)).to(dev)
    return x, t, m


# ------------------------------------------------------------------ augmentation (GPU, per batch)
def flip_batch(x, y, m, sel):
    """Horizontal flip for samples where sel is True; x' = IMG - x (v1/v2 convention), left/right identities swapped."""
    if not sel.any():
        return x, y, m
    xf = torch.where(sel[:, None, None, None], x.flip(-1), x)
    yf, mf = y[:, SWAP], m[:, SWAP]
    yf = yf.clone()
    yf[:, 0::2] = IMG - yf[:, 0::2]
    y = torch.where(sel[:, None], yf, y)
    m = torch.where(sel[:, None], mf, m)
    return xf, y, m


def augment(x, y, m, cfg):
    """Random affine (rotation, scale, translation) applied identically to image and keypoints, plus flip, photometric
    jitter, noise, blur and cutout occlusion. Keypoints pushed out of frame lose their mask."""
    B, dev = x.shape[0], x.device
    x, y, m = flip_batch(x, y, m, torch.rand(B, device=dev) < 0.5)
    if cfg.get("affine", True):
        rot = (torch.rand(B, device=dev) * 2 - 1) * math.radians(cfg.get("rot_deg", 30))
        sc = 1 + (torch.rand(B, device=dev) * 2 - 1) * cfg.get("scale", 0.20)
        tr = (torch.rand(B, 2, device=dev) * 2 - 1) * cfg.get("translate", 0.06) * 2   # in normalized [-1,1] units
        cos, sin = torch.cos(rot), torch.sin(rot)
        A = torch.stack([torch.stack([cos, -sin], 1), torch.stack([sin, cos], 1)], 1) * sc[:, None, None]  # forward
        Ainv = torch.linalg.inv(A)
        theta = torch.cat([Ainv, -(Ainv @ tr[:, :, None])], 2)                     # output -> input sampling
        grid = F.affine_grid(theta, x.shape, align_corners=False)
        x = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)
        p = y.view(B, N_KP, 2)
        pn = p / (IMG / 2) - 1                                                           # pixel -> normalized (area convention)
        pn = torch.einsum("bij,bkj->bki", A, pn) + tr[:, None, :]
        p = (pn + 1) * (IMG / 2)
        y = p.reshape(B, 30)
        inside = ((p >= 0) & (p <= IMG)).all(-1).float()                              # (B,15)
        m = m * inside.repeat_interleave(2, dim=1)
        y = y * m
    if cfg.get("photometric", True):
        a = 1 + (torch.rand(B, 1, 1, 1, device=dev) * 2 - 1) * 0.3
        b = (torch.rand(B, 1, 1, 1, device=dev) * 2 - 1) * 0.1
        x = (x - 0.5) * a + 0.5 + b
        noisy = (torch.rand(B, 1, 1, 1, device=dev) < 0.3).float()
        x = x + noisy * torch.randn_like(x) * 0.03
        blur_sel = torch.rand(B, device=dev) < 0.2
        if blur_sel.any():
            k = torch.tensor([1., 2., 1.], device=dev)
            k = (k[:, None] * k[None, :]) / 16
            xb = F.conv2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), k.view(1, 1, 3, 3))
            x = torch.where(blur_sel[:, None, None, None], xb, x)
    if cfg.get("cutout", True):
        sel = torch.rand(B, device=dev) < 0.4
        size = torch.randint(16, 25, (B,), device=dev)
        cx = torch.randint(0, IMG, (B,), device=dev)
        cy = torch.randint(0, IMG, (B,), device=dev)
        ar = torch.arange(IMG, device=dev)
        mx = ((ar[None, :] >= (cx - size // 2)[:, None]) & (ar[None, :] < (cx + size // 2)[:, None]))
        my = ((ar[None, :] >= (cy - size // 2)[:, None]) & (ar[None, :] < (cy + size // 2)[:, None]))
        box = (my[:, :, None] & mx[:, None, :]) & sel[:, None, None]
        x = torch.where(box[:, None], x.mean(dim=(2, 3), keepdim=True), x)
    return x.clamp(0, 1), y, m


# ------------------------------------------------------------------ heatmaps (pixel <-> cell: pixel = (cell + 0.5) * stride)
def make_heatmaps(y, m, H, sigma, mapping="centered"):
    B = y.shape[0]
    stride = IMG / H
    p = y.view(B, N_KP, 2)
    c = p / stride - 0.5 if mapping == "centered" else p / stride   # 'v2' mapping reproduces v2 exactly
    ar = torch.arange(H, device=y.device, dtype=y.dtype)
    gx = torch.exp(-((ar[None, None, :] - c[..., 0:1]) ** 2) / (2 * sigma ** 2))
    gy = torch.exp(-((ar[None, None, :] - c[..., 1:2]) ** 2) / (2 * sigma ** 2))
    hm = gy[..., :, None] * gx[..., None, :]
    kp = m.view(B, N_KP, 2)[..., 0]
    return hm * kp[..., None, None], kp


# ------------------------------------------------------------------ models
class CoordConv(nn.Module):
    def forward(self, x):
        B, _, H, W = x.shape
        ys = torch.linspace(-1, 1, H, device=x.device, dtype=x.dtype).view(1, 1, H, 1).expand(B, 1, H, W)
        xs = torch.linspace(-1, 1, W, device=x.device, dtype=x.dtype).view(1, 1, 1, W).expand(B, 1, H, W)
        return torch.cat([x, xs, ys], 1)


class SmallBackbone(nn.Module):
    """v2's backbone: three conv/BN/ReLU/max-pool blocks, 96 -> 12, 128 channels."""
    def __init__(self, in_ch=1):
        super().__init__()
        self.out_ch = 128
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.net(x)


class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False); self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False); self.b2 = nn.BatchNorm2d(cout)
        self.sc = (nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))
                   if stride != 1 or cin != cout else nn.Identity())

    def forward(self, x):
        return F.relu(self.b2(self.c2(F.relu(self.b1(self.c1(x))))) + self.sc(x))


class ResBackbone(nn.Module):
    """ResNet-18-style for 1-channel 96x96: no early max-pool. Returns the 24x24 (128ch) and 12x12 (256ch) maps."""
    def __init__(self, in_ch=1):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_ch, 32, 3, 1, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU())
        self.l1 = nn.Sequential(BasicBlock(32, 64, 2), BasicBlock(64, 64, 1))      # 48
        self.l2 = nn.Sequential(BasicBlock(64, 128, 2), BasicBlock(128, 128, 1))   # 24
        self.l3 = nn.Sequential(BasicBlock(128, 256, 2), BasicBlock(256, 256, 1))  # 12
        self.out_ch = 256

    def forward(self, x):
        f24 = self.l2(self.l1(self.stem(x)))
        return f24, self.l3(f24)


class IntegralHead(nn.Module):
    """Deconv(s) to an HxH heatmap, then soft-argmax with temperature beta (learnable unless fixed)."""
    def __init__(self, cin, H, n_deconv, beta=10.0, learn_beta=True, mapping="centered"):
        super().__init__()
        layers, c = [], cin
        for _ in range(n_deconv):
            layers += [nn.ConvTranspose2d(c, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU()]
            c = 256
        self.up = nn.Sequential(*layers)
        self.hm = nn.Conv2d(c, N_KP, 1)
        self.H, self.mapping = H, mapping
        lb = torch.tensor(math.log(beta))
        self.log_beta = nn.Parameter(lb) if learn_beta else nn.Parameter(lb, requires_grad=False)
        ar = torch.arange(H, dtype=torch.float32)
        self.register_buffer("cells", ar)

    def coords(self, logits):
        B, K, H, W = logits.shape
        prob = torch.softmax(self.log_beta.exp() * logits.float().view(B, K, -1), -1).view(B, K, H, W)
        ex = (prob.sum(2) * self.cells).sum(-1)
        ey = (prob.sum(3) * self.cells).sum(-1)
        stride = IMG / H
        if self.mapping == "centered":
            ex, ey = (ex + 0.5) * stride, (ey + 0.5) * stride
        else:
            ex, ey = ex * stride, ey * stride
        return torch.stack([ex, ey], -1).view(B, K * 2)

    def forward(self, feat):
        logits = self.hm(self.up(feat))
        return self.coords(logits), logits


class SingleStageModel(nn.Module):
    def __init__(self, H=24, n_deconv=1, coordconv=False, beta=10.0, learn_beta=True, mapping="centered"):
        super().__init__()
        self.cc = CoordConv() if coordconv else nn.Identity()
        self.backbone = SmallBackbone(3 if coordconv else 1)
        self.head = IntegralHead(128, H, n_deconv, beta, learn_beta, mapping)

    def forward(self, x):
        coords, logits = self.head(self.backbone(self.cc(x)))
        return {"coords": coords, "heatmaps": [logits]}


class TwoStageModel(nn.Module):
    """Residual backbone, stage-1 integral head on 12x12 features, stage-2 refinement on the 24x24 features concatenated
    with stage-1 heatmaps; both stages supervised (MS-I1), stage 2 is the output."""
    def __init__(self, beta=10.0):
        super().__init__()
        self.backbone = ResBackbone(1)
        self.head1 = IntegralHead(256, 24, 1, beta)
        self.refine = nn.Sequential(nn.Conv2d(128 + N_KP, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(),
                                    BasicBlock(128, 128, 1), BasicBlock(128, 128, 1))
        self.head2 = IntegralHead(128, 24, 0, beta)

    def forward(self, x):
        f24, f12 = self.backbone(x)
        c1, h1 = self.head1(f12)
        c2, h2 = self.head2(self.refine(torch.cat([f24, h1], 1)))
        return {"coords": c2, "heatmaps": [h1, h2], "coords_stage1": c1}


# ------------------------------------------------------------------ losses (fp32)
def masked_l1(pred, y, m):
    return ((pred.float() - y).abs() * m).sum() / m.sum().clamp(min=1)


def heatmap_mse(logits, gt, kp):
    se = (logits.float() - gt) ** 2 * kp[..., None, None]
    return se.sum() / (kp.sum() * gt.shape[-1] * gt.shape[-2]).clamp(min=1)


def adaptive_wing(logits, gt, kp, omega=14.0, theta=0.5, eps=1.0, alpha=2.1, W=10.0):
    """Adaptive Wing loss (Wang, Bo, Li Fuxin; Oregon State University; arXiv 1904.07399, Eq. 3) with the
    Weighted Loss Map (Eq. 4-5: 3x3 grey dilation of the target, threshold 0.2, weight W)."""
    yhat, y = logits.float(), gt
    d = (y - yhat).abs()
    p = alpha - y
    A = omega * (1 / (1 + (theta / eps) ** p)) * p * ((theta / eps) ** (p - 1)) / eps
    C = theta * A - omega * torch.log1p((theta / eps) ** p)
    loss = torch.where(d < theta, omega * torch.log1p((d / eps) ** p), A * d - C)
    dil = F.max_pool2d(y, 3, 1, 1)
    weight = W * (dil >= 0.2).float() + 1
    loss = loss * weight * kp[..., None, None]
    return loss.sum() / (kp.sum() * y.shape[-1] * y.shape[-2]).clamp(min=1)


# ------------------------------------------------------------------ decoders (eval-time)
def decode_argmax_shift(logits, mapping="centered"):
    """Argmax plus a quarter-cell shift toward the higher neighbour (Newell et al.; used by Adaptive Wing at test time)."""
    B, K, H, W = logits.shape
    hm = logits.float()
    flat = hm.view(B, K, -1).argmax(-1)
    yi, xi = (flat // W), (flat % W)
    xs, ys = xi.float(), yi.float()
    bi = torch.arange(B, device=hm.device)[:, None].expand(B, K)
    ki = torch.arange(K, device=hm.device)[None, :].expand(B, K)
    xl, xr = (xi - 1).clamp(0, W - 1), (xi + 1).clamp(0, W - 1)
    yu, yd = (yi - 1).clamp(0, H - 1), (yi + 1).clamp(0, H - 1)
    xs = xs + 0.25 * torch.sign(hm[bi, ki, yi, xr] - hm[bi, ki, yi, xl])
    ys = ys + 0.25 * torch.sign(hm[bi, ki, yd, xi] - hm[bi, ki, yu, xi])
    return _cells_to_pixels(xs, ys, H, mapping)


def decode_dark(logits, sigma, mapping="centered"):
    """DARK (Zhang et al., arXiv 1910.06278): modulate with a 3x3 Gaussian (sigma of the targets) and rescale to the
    original max/min (Eq. 10-11), take log, then one Taylor step mu = m - H^-1 g at the argmax (Eq. 6-9)."""
    B, K, H, W = logits.shape
    h = logits.float().clamp(min=0)
    k1 = torch.exp(-torch.tensor([-1., 0., 1.], device=h.device) ** 2 / (2 * sigma ** 2))
    k = (k1[:, None] * k1[None, :]); k = k / k.sum()
    hm = F.conv2d(h.view(B * K, 1, H, W), k.view(1, 1, 3, 3), padding=1).view(B, K, H, W)
    mn = hm.amin((2, 3), keepdim=True); mx = hm.amax((2, 3), keepdim=True)
    hm = (hm - mn) / (mx - mn).clamp(min=1e-10) * h.amax((2, 3), keepdim=True)
    L = torch.log(hm.clamp(min=1e-10))
    flat = L.view(B, K, -1).argmax(-1)
    yi, xi = flat // W, flat % W
    inner = (xi > 0) & (xi < W - 1) & (yi > 0) & (yi < H - 1)
    xi_c, yi_c = xi.clamp(1, W - 2), yi.clamp(1, H - 2)
    bi = torch.arange(B, device=h.device)[:, None].expand(B, K)
    ki = torch.arange(K, device=h.device)[None, :].expand(B, K)
    g = lambda dy, dx: L[bi, ki, yi_c + dy, xi_c + dx]
    dx = 0.5 * (g(0, 1) - g(0, -1)); dy = 0.5 * (g(1, 0) - g(-1, 0))
    dxx = g(0, 1) - 2 * g(0, 0) + g(0, -1); dyy = g(1, 0) - 2 * g(0, 0) + g(-1, 0)
    dxy = 0.25 * (g(1, 1) - g(1, -1) - g(-1, 1) + g(-1, -1))
    det = dxx * dyy - dxy ** 2
    ok = inner & (det.abs() > 1e-8) & (dxx < 0)
    ox = torch.where(ok, -(dyy * dx - dxy * dy) / det.where(ok, torch.ones_like(det)), torch.zeros_like(det))
    oy = torch.where(ok, -(dxx * dy - dxy * dx) / det.where(ok, torch.ones_like(det)), torch.zeros_like(det))
    ox, oy = ox.clamp(-1, 1), oy.clamp(-1, 1)
    return _cells_to_pixels(xi.float() + ox, yi.float() + oy, H, mapping)


def _cells_to_pixels(xs, ys, H, mapping):
    stride = IMG / H
    if mapping == "centered":
        xs, ys = (xs + 0.5) * stride, (ys + 0.5) * stride
    else:
        xs, ys = xs * stride, ys * stride
    return torch.stack([xs, ys], -1).view(xs.shape[0], -1)


# ------------------------------------------------------------------ evaluation
@torch.no_grad()
def predict(model, x, bs=512, tta=False, decoder="softargmax", sigma=1.0, mapping="centered"):
    model.eval()
    outs = []
    for i in range(0, len(x), bs):
        xb = x[i:i + bs]
        variants = [(xb, False)] + ([(xb.flip(-1), True)] if tta else [])
        acc = 0
        for xv, flipped in variants:
            o = model(xv)
            h = o["heatmaps"][-1]
            if decoder == "softargmax":
                c = o["coords"].float()
            elif decoder == "argmax_shift":
                c = decode_argmax_shift(h, mapping)
            else:
                c = decode_dark(h, sigma, mapping)
            if flipped:
                c = c[:, SWAP].clone(); c[:, 0::2] = IMG - c[:, 0::2]
            acc = acc + c
        outs.append(acc / len(variants))
    return torch.cat(outs)


def rmse(pred, y, m):
    return float(torch.sqrt(((pred - y) ** 2 * m).sum() / m.sum()))


def per_keypoint_rmse(pred, y, m):
    return torch.sqrt(((pred - y) ** 2 * m).sum(0) / m.sum(0).clamp(min=1)).cpu().numpy()


def shrinkage_slopes(pred, y, m):
    """OLS slope of predicted vs true per coordinate; < 1 means predictions are pulled toward the mean (centre-pull)."""
    p, t, mm = pred.cpu().numpy(), y.cpu().numpy(), m.cpu().numpy().astype(bool)
    return np.array([np.polyfit(t[mm[:, j], j], p[mm[:, j], j], 1)[0] for j in range(30)])


# ------------------------------------------------------------------ logging / checkpoints
class RunLog:
    def __init__(self, name):
        os.makedirs("logs", exist_ok=True)
        self.path = f"logs/{name}.log"
        self.f = open(self.path, "a", buffering=1)

    def __call__(self, msg):
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}"
        print(line, flush=True)
        self.f.write(line + "\n")


# ------------------------------------------------------------------ in-training statistics (no figures; drawn post-training)
@torch.no_grad()
def val_stats(model, xva, yva, mva, mapping, tta=False):
    """One held-out pass: RMSE (overall and per coordinate), centre-pull slope, softmax entropy and peak probability."""
    model.eval()
    heads = [m for m in model.modules() if isinstance(m, IntegralHead)]
    beta = heads[-1].log_beta.exp() if heads else torch.tensor(1.0)
    coords, ent, mxp = [], [], []
    for i in range(0, len(xva), 512):
        o = model(xva[i:i + 512])
        coords.append(o["coords"].float())
        lg = o["heatmaps"][-1].float(); B, K = lg.shape[:2]
        p = torch.softmax(beta * lg.view(B, K, -1), -1)
        ent.append(-(p * torch.log(p.clamp(min=1e-12))).sum(-1)); mxp.append(p.amax(-1))
    pred = torch.cat(coords)
    err2 = (pred - yva) ** 2 * mva
    per = torch.sqrt(err2.sum(0) / mva.sum(0).clamp(min=1))
    t = yva * mva; n = mva.sum(0).clamp(min=1)
    mt, mp = t.sum(0) / n, (pred * mva).sum(0) / n
    cov = (((yva - mt) * (pred - mp)) * mva).sum(0); var = (((yva - mt) ** 2) * mva).sum(0).clamp(min=1e-9)
    out = {"rmse": float(torch.sqrt(err2.sum() / mva.sum())), "per_kp": per.cpu().tolist(),
           "slope_mean": float((cov / var).mean()), "entropy": float(torch.cat(ent).mean()), "maxprob": float(torch.cat(mxp).mean())}
    if tta:
        out["rmse_tta"] = rmse(predict(model, xva, tta=True, mapping=mapping), yva, mva)
    return out


class History:
    """Per-epoch statistics, rewritten atomically to results/<run>_history.json after every epoch."""
    KEYS = ("epoch", "stage", "seconds", "lr", "beta", "grad_norm", "train_loss", "train_l1", "train_hm", "val_hm",
            "val_rmse", "val_rmse_tta", "val_slope_mean", "val_entropy", "val_maxprob", "val_per_kp")

    def __init__(self, name):
        self.path = f"results/{name}_history.json"
        self.d = {k: [] for k in self.KEYS}
        self.d["keypoints"] = KEYPOINT_COLS
        os.makedirs("results", exist_ok=True)

    def add(self, **row):
        for k in self.KEYS:
            self.d[k].append(row.get(k))
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.d, f)
        os.replace(tmp, self.path)


def save_ckpt(name, model, opt, epoch, stage, best_val, history):
    os.makedirs("checkpoints", exist_ok=True)
    tmp = f"checkpoints/{name}.pt.tmp"
    torch.save({"model_state": model.state_dict(), "opt_state": opt.state_dict(), "epoch": epoch, "stage": stage,
                "val_rmse": best_val, "history": history}, tmp)
    os.replace(tmp, f"checkpoints/{name}.pt")


# ------------------------------------------------------------------ training
def train(name, model, data, cfg):
    """Stage A: heatmap loss only, until the held-out heatmap loss plateaus (<= a_max epochs).
    Stage B: L1 on soft-argmax coordinates (+ a heatmap term fixed at hm_frac of the L1 term when hm_frac > 0), in one
    phase (step LR decay, capped so the run uses <= cfg['epochs']) or in cfg['b_phases']. Early stop on held-out RMSE.
    No figures are drawn: every epoch appends statistics to results/<name>_history.json and a line to logs/<name>.log,
    and the best weights are checkpointed on every improvement."""
    dev = device()
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    torch.backends.cudnn.benchmark = True
    log, hist = RunLog(name), History(name)
    xtr, ytr, mtr = to_gpu(data["imgs"][data["idx_tr"]], data["y"][data["idx_tr"]], dev)
    xva, yva, mva = to_gpu(data["imgs"][data["idx_va"]], data["y"][data["idx_va"]], dev)
    model = model.to(dev)
    use_amp = dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    bs, sigma, mapping = cfg["batch"], cfg["sigma"], cfg.get("mapping", "centered")
    hm_loss = adaptive_wing if cfg.get("heatmap_loss") == "awing" else heatmap_mse
    best, best_state, bad = float("inf"), None, 0
    state = {"hm_weight": None}
    log(f"[{name}] start | params={sum(p.numel() for p in model.parameters()):,} | cfg={json.dumps(cfg)}")

    def heat_terms(out, yb, mb):
        tot = 0.0
        for h in out["heatmaps"]:
            gt, kp = make_heatmaps(yb, mb, h.shape[-1], sigma, mapping)
            tot = tot + hm_loss(h, gt, kp)
        return tot

    def epoch_pass(opt, stage):
        model.train()
        perm = torch.randperm(len(xtr), device=dev)
        s = {"loss": 0.0, "l1": 0.0, "hm": 0.0, "n": 0, "grad_norm": float("nan")}
        n_batches = (len(perm) + bs - 1) // bs
        for bi, i in enumerate(range(0, len(perm), bs)):
            idx = perm[i:i + bs]
            xb, yb, mb = augment(xtr[idx], ytr[idx], mtr[idx], cfg["aug"])
            with torch.autocast("cuda", enabled=use_amp):
                out = model(xb)
            l1 = torch.zeros((), device=dev); ht = torch.zeros((), device=dev)
            if stage == "A":
                ht = heat_terms(out, yb, mb); loss = ht
            else:
                l1 = masked_l1(out["coords"], yb, mb)
                if "coords_stage1" in out:
                    l1 = l1 + 0.5 * masked_l1(out["coords_stage1"], yb, mb)
                loss = l1
                if cfg.get("hm_frac", 0) > 0:
                    ht = heat_terms(out, yb, mb)
                    if state["hm_weight"] is None:   # fixed once: heatmap term starts at hm_frac of the L1 term
                        state["hm_weight"] = cfg["hm_frac"] * float(l1) / max(float(ht), 1e-12)
                        log(f"[{name}:B] heatmap weight fixed at {state['hm_weight']:.4g} (hm_frac {cfg['hm_frac']})")
                    loss = l1 + state["hm_weight"] * ht
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if bi == n_batches - 1:              # one gradient-norm reading per epoch keeps CPU dispatch overhead low
                scaler.unscale_(opt)
                s["grad_norm"] = float(torch.norm(torch.stack([p.grad.norm() for p in model.parameters() if p.grad is not None])))
            scaler.step(opt); scaler.update()
            k = len(idx); s["loss"] += float(loss) * k; s["l1"] += float(l1) * k; s["hm"] += float(ht) * k; s["n"] += k
        return {k: (v / s["n"] if k in ("loss", "l1", "hm") else v) for k, v in s.items() if k != "n"}

    def beta_now():
        heads = [m for m in model.modules() if isinstance(m, IntegralHead)]
        return float(heads[-1].log_beta.exp()) if heads else float("nan")

    def record(epoch, stage, t0, opt, tr, vs, val_hm=None):
        hist.add(epoch=epoch, stage=stage, seconds=time.time() - t0, lr=opt.param_groups[0]["lr"], beta=beta_now(),
                 grad_norm=tr["grad_norm"], train_loss=tr["loss"], train_l1=tr["l1"], train_hm=tr["hm"], val_hm=val_hm,
                 val_rmse=vs["rmse"], val_rmse_tta=vs.get("rmse_tta"), val_slope_mean=vs["slope_mean"],
                 val_entropy=vs["entropy"], val_maxprob=vs["maxprob"], val_per_kp=vs["per_kp"])

    epoch = 0
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("wd", 1e-4)) if cfg.get("opt", "adamw") == "adamw" \
        else torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    # ---- Stage A ----
    best_a, bad_a = float("inf"), 0
    while epoch < cfg["a_max"]:
        t0 = time.time()
        tr = epoch_pass(opt, "A")
        model.eval()
        with torch.no_grad():
            va = 0.0
            for i in range(0, len(xva), 512):
                out = model(xva[i:i + 512]); va += float(heat_terms(out, yva[i:i + 512], mva[i:i + 512])) * len(out["coords"])
            va /= len(xva)
        vs = val_stats(model, xva, yva, mva, mapping)
        record(epoch, "A", t0, opt, tr, vs, val_hm=va)
        log(f"[{name}:A] epoch {epoch:3d} train_hm {tr['hm']:.5f} val_hm {va:.5f} val_rmse {vs['rmse']:.4f} beta {beta_now():.2f} {time.time()-t0:.1f}s")
        epoch += 1
        if va < best_a - 1e-6:
            best_a, bad_a = va, 0
        else:
            bad_a += 1
        if bad_a >= cfg.get("a_patience", 6):
            log(f"[{name}:A] heatmap loss plateaued at epoch {epoch-1}")
            break
    save_ckpt(name, model, opt, epoch - 1, "A_done", best, hist.d)
    # ---- Stage B ----
    phases = cfg.get("b_phases") or [{"epochs": cfg["epochs"] - epoch, "lr": cfg["lr_b"], "sched": cfg.get("sched", "step"),
                                       "plateau_patience": cfg.get("plateau_patience", 6), "patience": cfg.get("patience", 30)}]
    for ph_i, ph in enumerate(phases):
        if ph_i > 0 and best_state is not None:
            model.load_state_dict(best_state)
            log(f"[{name}:B] phase {ph_i + 1}: resumed from best weights ({best:.4f})")
        opt = torch.optim.AdamW(model.parameters(), lr=ph["lr"], weight_decay=cfg.get("wd", 1e-4)) if cfg.get("opt", "adamw") == "adamw" \
            else torch.optim.Adam(model.parameters(), lr=ph["lr"], weight_decay=1e-4)
        if ph["sched"] == "step":
            sched = torch.optim.lr_scheduler.MultiStepLR(opt, [int(ph["epochs"] * 0.6), int(ph["epochs"] * 0.85)], 0.1)
        else:
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", 0.5, ph["plateau_patience"])
        bad = 0
        for eb in range(ph["epochs"]):
            t0 = time.time()
            tr = epoch_pass(opt, "B")
            vs = val_stats(model, xva, yva, mva, mapping, tta=(eb % 10 == 0))
            vr = vs["rmse"]
            lr_used = opt.param_groups[0]["lr"]
            sched.step() if ph["sched"] == "step" else sched.step(vr)
            record(epoch, f"B{ph_i + 1}", t0, opt, tr, vs)
            if vr < best:
                best, bad = vr, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                save_ckpt(name, model, opt, epoch, f"B{ph_i + 1}", best, hist.d)
            else:
                bad += 1
            log(f"[{name}:B{ph_i + 1}] epoch {epoch:3d} loss {tr['loss']:.4f} (l1 {tr['l1']:.4f} hm {tr['hm']:.4g}) val_rmse {vr:.4f} "
                f"best {best:.4f} slope {vs['slope_mean']:.3f} lr {lr_used:.1e} beta {beta_now():.2f} gnorm {tr['grad_norm']:.2f} {time.time()-t0:.1f}s")
            epoch += 1
            if bad >= ph["patience"]:
                log(f"[{name}:B{ph_i + 1}] early stop at epoch {epoch-1} (no improvement for {ph['patience']} epochs)")
                break
    model.load_state_dict(best_state)
    log(f"[{name}] done | best held-out RMSE {best:.4f} px | epochs used {epoch}")
    return model, hist.d, best


# ------------------------------------------------------------------ evaluation report + submission
def evaluate_and_save(name, model, data, cfg, decoders=("softargmax",)):
    dev = device()
    xva, yva, mva = to_gpu(data["imgs"][data["idx_va"]], data["y"][data["idx_va"]], dev)
    model = model.to(dev)
    res = {"run": name, "n_val": int(len(xva))}
    for d in decoders:
        for tta in (False, True):
            p = predict(model, xva, tta=tta, decoder=d, sigma=cfg["sigma"], mapping=cfg.get("mapping", "centered"))
            res[f"rmse_{d}{'_tta' if tta else ''}"] = rmse(p, yva, mva)
    p = predict(model, xva, tta=False, mapping=cfg.get("mapping", "centered"))
    res["per_keypoint_rmse"] = dict(zip(KEYPOINT_COLS, map(float, per_keypoint_rmse(p, yva, mva))))
    sl = shrinkage_slopes(p, yva, mva)
    res["shrinkage_slope_mean"] = float(sl.mean()); res["shrinkage_slopes"] = dict(zip(KEYPOINT_COLS, map(float, sl)))
    heads = [m for m in model.modules() if isinstance(m, IntegralHead)]
    res["beta_final"] = float(heads[-1].log_beta.exp()) if heads else None
    os.makedirs("results", exist_ok=True)
    json.dump(res, open(f"results/{name}_eval.json", "w"), indent=2)
    return res


def write_submission(name, model, data, cfg, tta=True, decoder="softargmax"):
    dev = device()
    x = torch.from_numpy(data["test_imgs"]).unsqueeze(1).to(dev)
    p = predict(model.to(dev), x, tta=tta, decoder=decoder, sigma=cfg["sigma"], mapping=cfg.get("mapping", "centered"))
    p = p.clamp(0, IMG).cpu().numpy()
    pred = pd.DataFrame(p, columns=KEYPOINT_COLS); pred.insert(0, "ImageId", data["test_ids"])
    look = pd.read_csv("data/IdLookupTable.csv").merge(pred, on="ImageId", how="left")
    look["Location"] = [row[f] for row, f in zip(look.to_dict("records"), look["FeatureName"])]
    look[["RowId", "Location"]].to_csv(f"submission_{name}.csv", index=False)
    return len(look)
