"""Mutation runs for the v4 suites: inject each plausible defect and show the corresponding check fails (killed).
A check that still passes under its mutant proves nothing. CPU, seconds."""
import os, sys
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fkd_v4_common as C  # noqa: E402

torch.manual_seed(0)
out = []


def report(target, killed, detail):
    out.append((target, killed, detail))


# ---- shared checks (same criteria as tests/test_fkd_v4_common.py) ----
def dot_image(points):
    ar = torch.arange(C.IMG, dtype=torch.float32) + 0.5
    img = torch.zeros(len(points), 1, C.IMG, C.IMG)
    for b, (x, y) in enumerate(points):
        img[b, 0] = torch.exp(-((ar[None, :] - x) ** 2 + (ar[:, None] - y) ** 2) / (2 * 1.5 ** 2))
    return img


def centroid(img):
    ar = torch.arange(C.IMG, dtype=torch.float32) + 0.5
    w = img[:, 0].clamp(min=0) ** 4
    return (w.sum(1) * ar).sum(-1) / w.sum((1, 2)), (w.sum(2) * ar).sum(-1) / w.sum((1, 2))


def affine_check(aug_fn):
    pts = [(30.3, 40.7), (60.2, 35.5), (48.0, 70.1), (52.6, 52.6)] * 8
    x = dot_image(pts); y = torch.zeros(len(pts), 30); m = torch.zeros(len(pts), 30)
    for b, (px, py) in enumerate(pts):
        y[b, 20], y[b, 21] = px, py; m[b, 20] = m[b, 21] = 1
    cfg = {"affine": True, "rot_deg": 30, "scale": 0.2, "translate": 0.06, "photometric": False, "cutout": False}
    torch.manual_seed(1)
    xa, ya, ma = aug_fn(x.clone(), y.clone(), m.clone(), cfg)
    cx, cy = centroid(xa); keep = ma[:, 20] > 0
    err = torch.sqrt((cx - ya[:, 20]) ** 2 + (cy - ya[:, 21]) ** 2)[keep]
    return bool((err < 1.0).all()), float(err.max())


def awing_check(fn):
    for yv in (0.0, 0.3, 0.7, 1.0):
        gt = torch.full((1, 1, 1, 1), yv); kp = torch.ones(1, 1)
        if abs(float(fn(gt - (0.5 - 1e-5), gt, kp, W=0.0)) - float(fn(gt - (0.5 + 1e-5), gt, kp, W=0.0))) >= 1e-3:
            return False
    return True


def decoder_check(decode):
    true = torch.tensor([[33.7, 58.2] * 15]); hm, _ = C.make_heatmaps(true, torch.ones(1, 30), 24, 2.0)
    return float((decode(hm) - true).abs().max()) < 0.05 * 4


def softargmax_check(head_cls):
    true = torch.tensor([[33.7, 58.2] * 15]); hm, _ = C.make_heatmaps(true, torch.ones(1, 30), 24, 2.0)
    with torch.no_grad():
        return float((head_cls(1, 24, 0, beta=12.0).coords(hm) - true).abs().max()) < 0.1 * 4


def flip_check(flip_fn):
    y = torch.zeros(1, 30); m = torch.ones(1, 30); y[0, 0], y[0, 2] = 30.0, 66.0
    _, yf, _ = flip_fn(torch.zeros(1, 1, 96, 96), y, m, torch.tensor([True]))
    return abs(float(yf[0, 0]) - 30.0) < 1e-6 and abs(float(yf[0, 2]) - 66.0) < 1e-6


def stage2_uses_stage1(model_fn):
    model = model_fn(); x = torch.rand(2, 1, 96, 96)
    model(x)["coords"].sum().backward()
    g = model.head1.hm.weight.grad
    return g is not None and float(g.abs().sum()) > 0


# sanity: every check passes on the real code
base = {"affine": affine_check(C.augment)[0], "awing": awing_check(C.adaptive_wing),
        "dark": decoder_check(lambda h: C.decode_dark(h, 2.0)), "softargmax": softargmax_check(C.IntegralHead),
        "flip": flip_check(C.flip_batch), "twostage": stage2_uses_stage1(lambda: C.TwoStageModel())}
print("baseline (all must be True):", base)
assert all(base.values())

# S5 mutant: keypoints not transformed with the image (image rotates, labels stay)
def aug_mutant(x, y, m, cfg):
    _, y0, m0 = x, y.clone(), m.clone()
    xa, _, _ = C.augment(x, y, m, cfg)
    return xa, y0, m0
ok, e = affine_check(aug_mutant)
report("S5 augmentation moves the image but not the keypoints", not ok, f"max err {e:.2f}px")

# S5 mutant 2: flip mirrors x but forgets to swap left/right identities
def flip_mutant(x, y, m, sel):
    xf = torch.where(sel[:, None, None, None], x.flip(-1), x); yf = y.clone(); yf[:, 0::2] = C.IMG - yf[:, 0::2]
    return xf, torch.where(sel[:, None], yf, y), m
report("S5 flip without left/right swap", not flip_check(flip_mutant), "")

# S6 mutant: Adaptive Wing continuity constant C dropped
def awing_mutant(logits, gt, kp, omega=14.0, theta=0.5, eps=1.0, alpha=2.1, W=10.0):
    yhat, y = logits.float(), gt; d = (y - yhat).abs(); p = alpha - y
    A = omega * (1 / (1 + (theta / eps) ** p)) * p * ((theta / eps) ** (p - 1)) / eps
    loss = torch.where(d < theta, omega * torch.log1p((d / eps) ** p), A * d)
    return (loss * kp[..., None, None]).sum() / (kp.sum() * y.shape[-1] * y.shape[-2])
report("S6 Adaptive Wing without the continuity constant C", not awing_check(awing_mutant), "")

# S7 mutant: stage 2 ignores the stage-1 heatmaps
class TwoStageMutant(C.TwoStageModel):
    def forward(self, x):
        f24, f12 = self.backbone(x); c1, h1 = self.head1(f12)
        c2, h2 = self.head2(self.refine(torch.cat([f24, torch.zeros_like(h1)], 1)))
        return {"coords": c2, "heatmaps": [h1, h2], "coords_stage1": c1}
report("S7 stage-2 refinement ignores stage-1 heatmaps", not stage2_uses_stage1(lambda: TwoStageMutant()), "")

# S8 mutant: temperature ignored (beta silently 1)
class HeadNoBeta(C.IntegralHead):
    def coords(self, logits):
        saved = self.log_beta.data.clone(); self.log_beta.data.zero_()
        try:
            return super().coords(logits)
        finally:
            self.log_beta.data.copy_(saved)
report("S8 soft-argmax ignores the temperature", not softargmax_check(HeadNoBeta), "")

# S10 mutant: DARK decoding with the x and y axes confused (heatmap read transposed)
report("S10 DARK decoding with x/y axes swapped", not decoder_check(lambda h: C.decode_dark(h.transpose(-1, -2), 2.0)), "")

for t, k, d in out:
    print("KILLED" if k else "SURVIVED", t, d)
print(f"\n{sum(k for _, k, _ in out)}/{len(out)} mutants killed")
sys.exit(0 if all(k for _, k, _ in out) else 1)
