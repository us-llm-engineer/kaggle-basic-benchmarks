"""Draw training-dynamics figures AFTER training from the per-epoch history files (statistics are collected during training, never plotted in the loop): python3 plot_pos_rnn.py [runs_dir]"""
import json, os, sys
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, numpy as np
import spooky_common as S
RUNS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(S.RESULTS, "pos_rnn_runs"); os.makedirs(os.path.join(S.ROOT, "assets"), exist_ok=True)
panels = [("train_loss", "inner_loss", "loss"), ("inner_acc", "inner_mean_conf", "accuracy vs confidence"), ("inner_ece", "inner_wrong_conf_gt_09", "ECE / confident errors"), ("grad_norm_mean", "update_ratio_mean", "gradient norm / update ratio"), ("param_norm", "param_dist_from_init", "parameter norm"), ("mon_outer_loss", "lr", "monitor outer loss / lr")]
fig, ax = plt.subplots(len(panels), 4, figsize=(20, 3 * len(panels)), constrained_layout=True)
for c, v in enumerate(("cnn", "bilstm", "attn", "fusion")):
    H = [json.load(open(os.path.join(RUNS, f"{v}_f{k}_history.json")))["history"] for k in range(5)]
    for r, (a, b, t) in enumerate(panels):
        for key, ls in ((a, "-"), (b, "--")):
            m = np.mean([[row[key] for row in h] for h in H], 0); ax[r, c].plot(m, ls, label=key)
        ax[r, c].set_title(f"{v}: {t}", fontsize=9); ax[r, c].legend(fontsize=7); ax[r, c].grid(alpha=.25)
        if r in (4, 5): ax[r, c].set_yscale("log") if panels[r][1] == "lr" else None
fig.savefig(os.path.join(S.ROOT, "assets", "pos_rnn_training_dynamics.png"), dpi=110); print("saved assets/pos_rnn_training_dynamics.png")
