"""Collect the POS-RNN fold outputs: temperature-scale each variant (T fitted on the other folds' OOF rows), save methods pos_cnn / pos_bilstm / pos_attn, print metrics and training-dynamics summaries from the per-epoch histories."""
import json, os, sys
import numpy as np
import spooky_common as S
RUNS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(S.RESULTS, "pos_rnn_runs"); train, test, folds, y = S.load()
for v in ("cnn", "bilstm", "attn", "fusion"):
    oof, tst, hist = np.zeros((len(y), 3)), 0, []
    for k in range(5):
        z = np.load(os.path.join(RUNS, f"{v}_f{k}.npz")); oof[z["va"]] = z["pv"]; tst = tst + z["pt"] / 5; hist.append(json.load(open(os.path.join(RUNS, f"{v}_f{k}_history.json"))))
    cal, Ts, tcal = S.cv_temperature(oof, y, folds, tst); e, raw = S.evaluate(y, cal, folds), S.evaluate(y, oof, folds, 10); H = [h["history"] for h in hist]
    be = [int(np.argmin([r["inner_loss"] for r in h])) for h in H]; gap = [h[b]["inner_loss"] - h[b]["train_loss"] for h, b in zip(H, be)]
    S.save_method(f"pos_{v}", cal, tcal, dict(method=f"syntactic POS-only {v} (arXiv 1902.09723 sentence-level adaptation), from scratch", best_epochs=be, temperatures=Ts, n_params=hist[0]["n_params"], **e))
    print(f"pos_{v:6s}: raw {raw['log_loss']:.4f} -> temperature-scaled {e['log_loss']:.4f} {np.round(e['log_loss_ci95'],4)} acc {e['accuracy']:.4f} | best epochs {be} | inner-minus-train loss at best {np.round(gap,3).tolist()} | final ECE {np.mean([h[-1]['inner_ece'] for h in H]):.3f} | params {hist[0]['n_params']} | epochs run {[len(h) for h in H]}")
    ep = min(len(h) for h in H); pick = lambda key, idx: [round(float(np.mean([h[j][key] for h in H])), 4) for j in idx if j < ep]
    print(f"   mean over folds at epochs 0/10/20/40/{ep - 1} (shortest fold): train", pick("train_loss", (0, 10, 20, 40, ep - 1)), "inner", pick("inner_loss", (0, 10, 20, 40, ep - 1)), "update ratio", pick("update_ratio_mean", (0, 10, 40, ep - 1)))
