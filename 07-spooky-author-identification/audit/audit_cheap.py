"""Audit of the existing methodologies: cheap integrity checks that need no training. Run: python3 -I audit/audit_cheap.py (from project root)."""
import sys, json, glob; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from sklearn.metrics import log_loss
import spooky_common as S
train, test, folds, y = S.load(); n = len(y)
print("rows", n, "test", len(test), "fold sizes", np.bincount(folds).tolist(), "class counts", np.bincount(y).tolist())
print("duplicate texts in train:", int(train.text.duplicated().sum()), "| train/test text overlap:", int(train.text.isin(test.text).sum()))
L = train.text.str.len(); print("chars: mean %.0f median %.0f p90 %.0f max %d" % (L.mean(), L.median(), L.quantile(.9), L.max()))
for k in (128, 160, 256): print(f"  share of sentences longer than {k} chars: {(L > k).mean():.3f}")
print("\n--- stored OOF probabilities: independent metric re-derivation (sklearn) ---")
for name in ["nblr", "style_gbdt", "cnn", "charlm", "char_svm", "pool_geo"]:
    try: oof, tst, meta = S.load_method(name)
    except Exception as e: print(name, "missing", e); continue
    ok = np.allclose(oof.sum(1), 1, atol=1e-6) and (oof >= 0).all() and np.isfinite(oof).all() and len(oof) == n
    ours = S.evaluate(y, oof, folds, 10)["log_loss"]; ref = log_loss(y, oof / oof.sum(1, keepdims=True), labels=[0, 1, 2])
    print(f"{name:11s} valid_probs={ok} ours={ours:.4f} sklearn={ref:.4f} acc={(oof.argmax(1)==y).mean():.4f} test_rows={len(tst)} test_mean_class_share={np.round(tst.mean(0),3).tolist()}")
print("\n--- CNN fold histories ---")
for k in range(5):
    r = json.load(open(f"results/cnn_runs/f{k}.json")); h = r["history"]; b = r["best_epoch"]
    print(f"fold {k}: epochs {r['n_epochs']} best {b} train_loss@best {h[b]['train_loss']:.3f} inner {h[b]['inner_loss']:.3f} outer_mon {h[b]['mon_loss']:.3f} final train {h[-1]['train_loss']:.3f} outer_mon_min {min(x['mon_loss'] for x in h):.3f}")
print("\n--- GRU/Transformer screen records ---")
for f in sorted(glob.glob("results/gpu_candidates/*_history.json")):
    h = json.load(open(f)); print(f.split('/')[-1], "epochs", len(h), "best inner %.3f" % min(x["inner_loss"] for x in h), "best epoch", int(np.argmin([x["inner_loss"] for x in h])))
