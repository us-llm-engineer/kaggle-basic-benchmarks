"""In-training figures, drawn AFTER training from the per-epoch statistics files (nothing is plotted inside a training loop).
 train_01_mlp.py-style panels for the char-TF-IDF MLP; train_02 for the POS networks; train_03 for the CNN / fastText-style / BiGRU / Transformer histories.
    python3 viz_in_training.py  ->  assets/train_*.png"""
import glob, json, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, numpy as np
import spooky_common as S
R, A = S.RESULTS, os.path.join(S.ROOT, "assets"); os.makedirs(A, exist_ok=True); plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": .25, "axes.spines.top": False, "axes.spines.right": False})
C5 = ["#4C72B0", "#C44E52", "#55A868", "#8172B2", "#CCB974"]
def H(path): return json.load(open(path))["history"]
def mean_curve(hs, key, x="epoch"):
    n = min(len(h) for h in hs); return np.array([h[i][x] for i, h in zip([0] * len(hs), hs)][0:1] and [hs[0][i][x] for i in range(n)]), np.array([[h[i][key] for i in range(n)] for h in hs])
def panel(ax, hs, key, label, color, x="epoch", logy=False):
    xs, ys = mean_curve(hs, key, x); m, s = ys.mean(0), ys.std(0); ax.plot(xs, m, color=color, label=label); ax.fill_between(xs, m - s, m + s, color=color, alpha=.15)
    if logy: ax.set_yscale("log")

# ---- 1. MLP on TF-IDF character n-grams (best configuration N=5, lr 1e-4, input dropout .5), mean +- sd over five folds, x = real epochs (4 inner evaluations per epoch)
hs = [H(f"{R}/char_mlp_runs/N5_lr0.0001_dp0.5_a0.0001_s7_f{k}_history.json") for k in range(5)]
fig, ax = plt.subplots(2, 4, figsize=(20, 8.5), constrained_layout=True)
panel(ax[0, 0], hs, "train_loss", "training loss", C5[0], "real_epoch"); panel(ax[0, 0], hs, "inner_loss", "inner hold-out loss", C5[1], "real_epoch"); panel(ax[0, 0], hs, "mon_outer_loss", "outer fold (monitor only)", C5[2], "real_epoch"); ax[0, 0].set(title="Loss", xlabel="epoch", ylim=(0, 1.0)); ax[0, 0].legend(fontsize=8)
panel(ax[0, 1], hs, "inner_acc", "inner accuracy", C5[0], "real_epoch"); panel(ax[0, 1], hs, "inner_mean_conf", "mean confidence", C5[1], "real_epoch"); ax[0, 1].set(title="Accuracy vs confidence (the gap is over-confidence)", xlabel="epoch"); ax[0, 1].legend(fontsize=8)
panel(ax[0, 2], hs, "inner_ece", "ECE", C5[0], "real_epoch"); panel(ax[0, 2], hs, "inner_entropy", "mean entropy", C5[1], "real_epoch"); ax[0, 2].set(title="Calibration error and entropy", xlabel="epoch"); ax[0, 2].legend(fontsize=8)
panel(ax[0, 3], hs, "inner_wrong_conf_gt_09", "errors made with confidence > 0.9", C5[3], "real_epoch"); ax[0, 3].set(title="Confident errors on the inner hold-out", xlabel="epoch"); ax[0, 3].legend(fontsize=8)
for k, c in zip(("grad_norm_W1", "grad_norm_l2", "grad_norm_out"), C5): panel(ax[1, 0], hs, k, k.replace("grad_norm_", "gradient norm "), c, "real_epoch", True)
ax[1, 0].set(title="Gradient norm per layer", xlabel="epoch"); ax[1, 0].legend(fontsize=8)
panel(ax[1, 1], hs, "dead_relu_h1", "dead ReLU, layer 1", C5[0], "real_epoch"); panel(ax[1, 1], hs, "dead_relu_h2", "dead ReLU, layer 2", C5[1], "real_epoch"); ax[1, 1].set(title="Share of hidden units that never fire", xlabel="epoch"); ax[1, 1].legend(fontsize=8)
panel(ax[1, 2], hs, "param_norm", "parameter norm", C5[0], "real_epoch"); panel(ax[1, 2], hs, "param_dist_from_init", "distance from initialisation", C5[1], "real_epoch"); ax[1, 2].set(title="Parameter drift", xlabel="epoch"); ax[1, 2].legend(fontsize=8)
panel(ax[1, 3], hs, "inner_loss_share_from_errors", "share of loss from misclassified rows", C5[3], "real_epoch"); panel(ax[1, 3], hs, "inner_margin_wrong", "logit margin on errors", C5[4], "real_epoch"); ax[1, 3].set(title="Where the loss comes from", xlabel="epoch"); ax[1, 3].legend(fontsize=8)
for a in ax.ravel(): [a.axvline(np.mean([h[int(np.argmin([r["inner_loss"] for r in h]))]["real_epoch"] for h in hs]), color="grey", ls=":", lw=1)]
fig.suptitle("In-training statistics: shallow ANN (100, 50) on TF-IDF character 5-grams, input dropout 0.5, lr 1e-4 (mean +- sd of 5 folds; dotted line = mean selected epoch)", fontsize=12); fig.savefig(f"{A}/train_01_char_mlp.png", bbox_inches="tight"); plt.close(fig)

# ---- 2. MLP sweep: how the selected epoch and the overfit gap move with learning rate and dropout
cfgs = sorted({os.path.basename(p).rsplit("_f", 1)[0] for p in glob.glob(f"{R}/char_mlp_runs/N5_lr*_dp*_a*_s7_f0_history.json")}); fig, ax = plt.subplots(1, 3, figsize=(18, 4.8), constrained_layout=True)
for c, col in zip(cfgs, plt.cm.viridis(np.linspace(0, .9, len(cfgs)))):
    hh = [H(f"{R}/char_mlp_runs/{c}_f{k}_history.json") for k in range(5)]; lab = c.replace("N5_", "").replace("_a0.0001_s7", "").replace("lr", "lr ").replace("_dp", "  dropout ")
    panel(ax[0], hh, "inner_loss", lab, col, "real_epoch"); panel(ax[1], hh, "train_loss", lab, col, "real_epoch", True); panel(ax[2], hh, "inner_mean_conf", lab, col, "real_epoch")
ax[0].set(title="Inner hold-out loss", xlabel="epoch", xscale="log", ylim=(.3, .9)); ax[1].set(title="Training loss (log scale)", xlabel="epoch", xscale="log"); ax[2].set(title="Mean confidence on the inner hold-out", xlabel="epoch", xscale="log"); ax[0].legend(fontsize=7)
fig.suptitle("In-training: learning-rate and input-dropout sweep for the 5-gram MLP (smaller learning rates reach the same minimum later; dropout lowers the minimum slightly)"); fig.savefig(f"{A}/train_02_char_mlp_sweep.png", bbox_inches="tight"); plt.close(fig)

# ---- 3. POS-tag networks (four encoders), mean +- sd over 5 folds
fig, ax = plt.subplots(2, 4, figsize=(20, 8.5), constrained_layout=True); V = ["cnn", "bilstm", "attn", "fusion"]
for v, c in zip(V, C5):
    hh = [H(f"{R}/pos_rnn_runs/{v}_f{k}_history.json") for k in range(5)]
    panel(ax[0, 0], hh, "train_loss", v, c); panel(ax[0, 1], hh, "inner_loss", v, c); panel(ax[0, 2], hh, "inner_acc", v, c); panel(ax[0, 3], hh, "inner_ece", v, c)
    panel(ax[1, 0], hh, "grad_norm_mean", v, c, logy=True); panel(ax[1, 1], hh, "update_ratio_mean", v, c, logy=True); panel(ax[1, 2], hh, "param_dist_from_init", v, c); panel(ax[1, 3], hh, "inner_mean_conf", v, c)
for a, t in zip(ax.ravel(), ("Training loss", "Inner hold-out loss", "Inner accuracy", "Expected calibration error", "Gradient norm", "Relative update size", "Distance from initialisation", "Mean confidence")): a.set(title=t, xlabel="epoch"); a.legend(fontsize=8)
fig.suptitle("In-training statistics: POS-tag networks (CNN / BiLSTM / BiLSTM with tag attention / CNN+BiLSTM fusion), mean +- sd of 5 folds, runs stop after 20 epochs without inner improvement"); fig.savefig(f"{A}/train_03_pos_networks.png", bbox_inches="tight"); plt.close(fig)

# ---- 4. other neural screens: char+word CNN, fastText-style, char BiGRU (80 epochs, fold 0), char Transformer (19 epochs, fold 0)
fig, ax = plt.subplots(1, 4, figsize=(21, 4.6), constrained_layout=True)
hc = [H_ for H_ in [json.load(open(f"{R}/cnn_runs/f{k}.json"))["history"] for k in range(5)]]; panel(ax[0], hc, "train_loss", "train", C5[0]); panel(ax[0], hc, "inner_loss", "inner hold-out", C5[1]); panel(ax[0], hc, "mon_loss", "outer fold (monitor)", C5[2]); ax[0].set(title="Char+word CNN (8 epochs, 5 folds)", xlabel="epoch"); ax[0].legend(fontsize=8)
hf = [json.load(open(f"{R}/fasttext_runs/word_h30_f{k}.json"))["history"] for k in range(5)]; panel(ax[1], hf, "train_loss", "train", C5[0]); panel(ax[1], hf, "inner_loss", "inner hold-out", C5[1]); ax[1].set(title="fastText-style, 30 hidden units (20 epochs)", xlabel="epoch"); ax[1].legend(fontsize=8)
g = json.load(open(f"{R}/gpu_candidates/char_gru_f0_history.json")); xe = [r["epoch"] for r in g]
for k, c in (("train_loss", C5[0]), ("inner_loss", C5[1])): ax[2].plot(xe, [r[k] for r in g], color=c, label=k)
a2 = ax[2].twinx(); a2.plot(xe, [r["ece"] for r in g], color=C5[3], ls="--", label="ECE"); a2.set_ylabel("ECE"); ax[2].set(title="Char BiGRU + attention, fold 0 (80 epochs): memorises, then over-confident", xlabel="epoch"); ax[2].legend(fontsize=8, loc="upper right")
t = json.load(open(f"{R}/gpu_candidates/char_transformer_f0_history.json")); xe = [r["epoch"] for r in t]
for k, c in (("train_loss", C5[0]), ("inner_loss", C5[1])): ax[3].plot(xe, [r[k] for r in t], color=c, label=k)
ax[3].set(title="Char Transformer encoder, fold 0 (19 epochs, run stopped early)", xlabel="epoch"); ax[3].legend(fontsize=8)
fig.savefig(f"{A}/train_04_other_neural_screens.png", bbox_inches="tight"); plt.close(fig); print("saved", sorted(os.path.basename(p) for p in glob.glob(A + "/train_*.png")))
