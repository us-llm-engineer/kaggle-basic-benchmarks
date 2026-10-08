"""Unit checks for fkd_v4_common (CPU, seconds). Run: python3 tests/test_fkd_v4_common.py"""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fkd_v4_common as C  # noqa: E402

torch.manual_seed(0)
results = []


def check(name, ok, detail=""):
    results.append((ok, name, detail))


def dot_image(points):
    """Images with one bright Gaussian blob per keypoint, centred at a continuous (area-convention) coordinate."""
    B = len(points)
    ar = torch.arange(C.IMG, dtype=torch.float32) + 0.5           # pixel i centre at i + 0.5
    img = torch.zeros(B, 1, C.IMG, C.IMG)
    for b, (x, y) in enumerate(points):
        img[b, 0] = torch.exp(-((ar[None, :] - x) ** 2 + (ar[:, None] - y) ** 2) / (2 * 1.5 ** 2))
    return img


def centroid(img):
    ar = torch.arange(C.IMG, dtype=torch.float32) + 0.5
    w = img[:, 0].clamp(min=0) ** 4
    return (w.sum(1) * ar).sum(-1) / w.sum((1, 2)), (w.sum(2) * ar).sum(-1) / w.sum((1, 2))


# 1. affine + flip move the keypoint exactly where they move the image (one keypoint per sample, others masked)
pts = [(30.3, 40.7), (60.2, 35.5), (48.0, 70.1), (52.6, 52.6)] * 8
x = dot_image(pts)
y = torch.zeros(len(pts), 30); m = torch.zeros(len(pts), 30)
for b, (px, py) in enumerate(pts):
    y[b, 20], y[b, 21] = px, py          # nose tip (not a swapped pair)
    m[b, 20] = m[b, 21] = 1
cfg = {"affine": True, "rot_deg": 30, "scale": 0.2, "translate": 0.06, "photometric": False, "cutout": False}
xa, ya, ma = C.augment(x.clone(), y.clone(), m.clone(), cfg)
cx, cy = centroid(xa)
keep = ma[:, 20] > 0
err = torch.sqrt((cx - ya[:, 20]) ** 2 + (cy - ya[:, 21]) ** 2)[keep]
check("affine+flip: transformed keypoint lands on the transformed dot (<1px)", bool((err < 1.0).all()), f"max err {err.max():.3f}px over {int(keep.sum())}")

# 2. flip alone: x -> 96 - x, and left/right identities swap
y2 = torch.zeros(1, 30); m2 = torch.ones(1, 30)
y2[0, 0], y2[0, 2] = 30.0, 66.0      # left eye x, right eye x
xf, yf, mf = C.flip_batch(torch.zeros(1, 1, 96, 96), y2, m2, torch.tensor([True]))
check("flip: left/right eye swap and mirror (x -> 96 - x)", abs(float(yf[0, 0]) - 30.0) < 1e-6 and abs(float(yf[0, 2]) - 66.0) < 1e-6,
      f"left={float(yf[0,0])} right={float(yf[0,2])}")
xd = dot_image([(30.3, 40.0)])
cxf, _ = centroid(xd.flip(-1))
check("flip: image mirror matches 96 - x", abs(float(cxf) - (96 - 30.3)) < 0.05, f"centroid {float(cxf):.3f} vs {96-30.3:.3f}")

# 3. Adaptive Wing is continuous at |error| = theta for every target value
for yv in (0.0, 0.3, 0.7, 1.0):
    gt = torch.full((1, 1, 1, 1), yv); kp = torch.ones(1, 1)
    lo = C.adaptive_wing(gt - (0.5 - 1e-5), gt, kp, W=0.0)
    hi = C.adaptive_wing(gt - (0.5 + 1e-5), gt, kp, W=0.0)
    check(f"awing continuous at theta (y={yv})", abs(float(lo) - float(hi)) < 1e-3, f"{float(lo):.6f} vs {float(hi):.6f}")
check("awing zero at zero error", float(C.adaptive_wing(torch.ones(1, 1, 3, 3), torch.ones(1, 1, 3, 3), torch.ones(1, 1))) < 1e-9)

# 4. decoders on an ideal Gaussian heatmap at a sub-pixel location
true = torch.tensor([[33.7, 58.2] * 15])
mfull = torch.ones(1, 30)
for H, sigma, beta in ((24, 2.0, 12.0), (48, 2.0, 15.0)):   # the configs the v4 runs use (measured floor)
    hm, _ = C.make_heatmaps(true, mfull, H, sigma)
    dark = C.decode_dark(hm, sigma)
    arg = C.decode_argmax_shift(hm)
    head = C.IntegralHead(1, H, 0, beta=beta)
    soft10 = head.coords(hm)
    head1 = C.IntegralHead(1, H, 0, beta=1.0)
    soft1 = head1.coords(hm)
    e = lambda p: float((p.detach() - true).abs().max())
    stride = 96 / H
    check(f"DARK recovers sub-pixel centre (H={H})", e(dark) < 0.05 * stride, f"max err {e(dark):.4f}px (cell={stride}px)")
    check(f"argmax+1/4 shift error is larger than DARK (H={H})", e(arg) > e(dark), f"argmax {e(arg):.3f}px")
    check(f"soft-argmax beta={beta} near-exact (H={H}, sigma={sigma})", e(soft10) < 0.1 * stride, f"{e(soft10):.4f}px")
    check(f"soft-argmax beta=1 shrinks toward centre (H={H})", e(soft1) > 5 * max(e(soft10), 1e-3), f"beta=1 err {e(soft1):.2f}px (centre-pull)")

# 5. one forward/backward per model on CPU
for name, model, H in (("small24", C.SingleStageModel(24, 1), 24),
                       ("awing48_coordconv", C.SingleStageModel(48, 2, coordconv=True), 48),
                       ("twostage", C.TwoStageModel(), 24),
                       ("v2compat", C.SingleStageModel(24, 1, beta=1.0, learn_beta=False, mapping="v2"), 24)):
    xb = torch.rand(4, 1, 96, 96); yb = torch.rand(4, 30) * 90; mb = torch.ones(4, 30)
    out = model(xb)
    gt, kp = C.make_heatmaps(yb, mb, H, 1.0)
    loss = C.masked_l1(out["coords"], yb, mb) + C.adaptive_wing(out["heatmaps"][-1], gt, kp) + C.heatmap_mse(out["heatmaps"][-1], gt, kp)
    loss.backward()
    ok = out["coords"].shape == (4, 30) and out["heatmaps"][-1].shape[-1] == H and torch.isfinite(loss)
    check(f"forward/backward {name}", bool(ok), f"heatmap {tuple(out['heatmaps'][-1].shape)}")

# 6. learnable temperature actually receives a gradient; fixed one does not
m1 = C.SingleStageModel(24, 1)
C.masked_l1(m1(torch.rand(2, 1, 96, 96))["coords"], torch.rand(2, 30) * 90, torch.ones(2, 30)).backward()
check("learnable beta gets a gradient", m1.head.log_beta.grad is not None and float(m1.head.log_beta.grad.abs()) > 0)
m2 = C.SingleStageModel(24, 1, beta=1.0, learn_beta=False)
check("fixed beta is frozen", not m2.head.log_beta.requires_grad)

bad = [r for r in results if not r[0]]
for ok, n, d in results:
    print("PASS" if ok else "FAIL", n, d)
print(f"\n{len(results) - len(bad)}/{len(results)} passed")
sys.exit(1 if bad else 0)
