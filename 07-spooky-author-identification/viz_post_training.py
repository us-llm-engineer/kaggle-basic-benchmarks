"""Post-training figures for the Spooky Author Identification study (12 PNGs in assets/, plus two CSVs in results/).
Run from this folder: python3 viz_post_training.py"""
import glob, os, re, time
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import rankdata  # noqa: F401
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_fscore_support
import spooky_common as S

plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12})
ASSETS = "assets"; os.makedirs(ASSETS, exist_ok=True)
ALL = ["nblr", "style_gbdt", "cnn", "charlm", "wordlm", "glm", "fasttext", "char_svm", "mlp_N5", "mlp_N4", "mlp_sweep_best",
       "pos_fusion", "pos_cnn", "pos_bilstm", "pos_attn"]
M8 = ["nblr", "style_gbdt", "cnn", "charlm", "wordlm", "glm", "fasttext", "char_svm"]
M7 = [m for m in M8 if m != "cnn"]
CLASSICAL = {"nblr", "style_gbdt", "charlm", "wordlm", "glm", "char_svm"}
NEURAL = {"cnn", "fasttext", "mlp_N5", "mlp_N4", "mlp_sweep_best", "pos_fusion", "pos_cnn", "pos_bilstm", "pos_attn"}
NAMES = dict(glm="Word GLM (skip n-grams + MKN)", wordlm="Word MKN language model", nblr="NB-weighted n-gram LR",
             charlm="Char 9-gram language model", char_svm="Char TF-IDF linear SVM", style_gbdt="Stylometry + topics GBDT",
             mlp_sweep_best="Char 5-gram TF-IDF MLP (lr 1e-4, dropout 0.5)",
             mlp_N5="Char 5-gram MLP, first round (lr 1e-3)", mlp_N4="Char 4-gram MLP, first round (lr 1e-3)",
             pos_cnn="POS-tag CNN", pos_bilstm="POS-tag BiLSTM", pos_attn="POS-tag BiLSTM + attention", fasttext="fastText-style", cnn="Char+word CNN",
             pos_fusion="POS-tag CNN+BiLSTM", stack8="Stack of 8")
COL = {"classical": "tab:blue", "neural": "tab:orange", "stack": "tab:green"}
FAM_LABEL = {"classical": "classical ML", "neural": "neural", "stack": "stack"}


def family(m):
    return "stack" if m == "stack8" else "classical" if m in CLASSICAL else "neural"


def nm(m):
    return NAMES.get(m, m)


def save(fig, name):
    fig.savefig(os.path.join(ASSETS, name), dpi=110, bbox_inches="tight"); plt.close(fig); print("wrote", name)


def stack_oof(P, members, y, folds, C=0.01):
    """Out-of-fold stack: multinomial LR on log-probabilities of members, fit on folds != k, predict fold k."""
    X = np.hstack([np.log(S.clip_norm(P[m])) for m in members]); out = np.zeros((len(y), 3))
    for k in range(5):
        tr, va = folds != k, folds == k
        out[va] = LogisticRegression(C=C, max_iter=1000).fit(X[tr], y[tr]).predict_proba(X[va])
    return out


def legend_families(ax, fams, loc="lower right"):
    ax.legend(handles=[Patch(color=COL[f], label=FAM_LABEL[f]) for f in fams], loc=loc)


def fig01(P, y, ev):
    ms = sorted(ev, key=lambda m: ev[m]["log_loss"], reverse=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    for i, m in enumerate(ms):
        e = ev[m]; ll = e["log_loss"]; lo, hi = e["log_loss_ci95"]
        ax.barh(i, ll, xerr=[[ll - lo], [hi - ll]], color=COL[family(m)], capsize=3, alpha=.9)
        ax.text(hi + 0.004, i, f"{ll:.4f}", va="center", fontsize=10)
    ax.set_yticks(range(len(ms))); ax.set_yticklabels([nm(m) for m in ms])
    ax.axvline(ev["stack8"]["log_loss"], ls="--", color="k", lw=1, label="stack value")
    ax.set_xlabel("Out-of-fold log loss (lower is better), 95% bootstrap CI"); ax.set_title("Out-of-fold log loss of all members and the stack")
    h, l = ax.get_legend_handles_labels()
    ax.legend(handles=h + [Patch(color=COL[f], label=FAM_LABEL[f]) for f in ("classical", "neural", "stack")], loc="upper right")
    ax.set_xlim(0, 0.9)
    save(fig, "post_01_leaderboard.png")


def fig02(P, y, ev):
    best = sorted([m for m in ALL], key=lambda m: ev[m]["log_loss"])[:8] + ["stack8"]; n = len(best)
    D = np.zeros((n, n)); sig = np.zeros((n, n), bool)
    for i in range(n):
        for j in range(i + 1, n):
            d, (lo, hi), _ = S.paired_bootstrap_logloss(y, P[best[i]], P[best[j]], n=300)
            D[i, j], D[j, i] = d, -d; sig[i, j] = sig[j, i] = (lo > 0 or hi < 0)
    fig, ax = plt.subplots(figsize=(11, 9)); v = np.abs(D).max()
    im = ax.imshow(D, cmap="RdBu", vmin=-v, vmax=v)
    for i in range(n):
        for j in range(n):
            if i != j: ax.text(j, i, f"{D[i,j]:+.3f}" + ("*" if sig[i, j] else ""), ha="center", va="center", fontsize=9)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels([nm(m) for m in best], rotation=45, ha="right"); ax.set_yticklabels([nm(m) for m in best])
    ax.set_xlabel("Column model (b)"); ax.set_ylabel("Row model (a)")
    ax.set_title("Paired log-loss difference (row minus column); negative = row better\n* = 95% bootstrap interval excludes 0")
    fig.colorbar(im, ax=ax, label="mean log-loss difference")
    save(fig, "post_02_paired_differences.png")


def fig03(P, y):
    ms = ["stack8", "glm", "nblr", "mlp_sweep_best", "fasttext", "cnn"]
    fig, axs = plt.subplots(2, 3, figsize=(15, 9.5))
    for ax, m in zip(axs.ravel(), ms):
        cm = confusion_matrix(y, P[m].argmax(1)); cn = cm / cm.sum(1, keepdims=True) * 100
        ax.imshow(cn, cmap="Blues", vmin=0, vmax=100)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{cn[i,j]:.1f}%", ha="center", va="center", color="w" if cn[i, j] > 50 else "k", fontsize=12)
        ax.set_xticks(range(3)); ax.set_yticks(range(3)); ax.set_xticklabels(S.AUTHORS); ax.set_yticklabels(S.AUTHORS)
        ax.set_xlabel("Predicted author"); ax.set_ylabel("True author"); ax.set_title(f"{nm(m)}\naccuracy {np.mean(y == P[m].argmax(1)):.3f}")
    fig.suptitle("Row-normalised confusion matrices (out-of-fold)", fontsize=15, y=1.0)
    fig.tight_layout(); save(fig, "post_03_confusion_matrices.png")


def fig04(P, y):
    ms = ["stack8", "glm", "mlp_sweep_best", "cnn"]; edges = np.linspace(0, 1, 11)
    fig, axs = plt.subplots(1, 4, figsize=(18, 4.8), sharey=True)
    for ax, m in zip(axs, ms):
        p = S.clip_norm(P[m]); conf = p.max(1); corr = (p.argmax(1) == y).astype(float)
        b = np.clip(np.digitize(conf, edges) - 1, 0, 9); ece = 0; xs, acc, cnt = [], [], []
        for k in range(10):
            s = b == k
            if s.sum() == 0: continue
            xs.append(conf[s].mean()); acc.append(corr[s].mean()); cnt.append(s.sum())
            ece += s.mean() * abs(corr[s].mean() - conf[s].mean())
        ax.bar((edges[:-1] + .05)[[k for k in range(10) if (b == k).any()]], acc, width=.09, color="tab:blue", alpha=.7, label="accuracy in bin")
        ax.plot([0, 1], [0, 1], "k--", label="perfect calibration")
        for x, a, c in zip(xs, acc, cnt): ax.text(x, min(a + .02, .97), f"n={c}", rotation=90, ha="center", fontsize=7)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.05); ax.set_xlabel("Predicted max probability"); ax.set_title(f"{nm(m)}\nECE = {ece:.4f}")
    axs[0].set_ylabel("Observed accuracy"); axs[0].legend(loc="upper left", fontsize=9)
    fig.suptitle("Reliability diagrams (10 equal-width bins; n = bin count)", y=1.03); save(fig, "post_04_reliability.png")


def fig05(P, y):
    wrong = sum((P[m].argmax(1) != y).astype(int) for m in M8); loss = S.sample_loss(y, P["stack8"])
    ks = np.arange(9); cnt = np.array([(wrong == k).sum() for k in ks]); share = np.array([loss[wrong == k].sum() for k in ks]) / loss.sum()
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    axs[0].bar(ks, cnt, color="tab:blue")
    for k in ks: axs[0].text(k, cnt[k], f"{cnt[k]}\n({cnt[k]/len(y):.1%})", ha="center", va="bottom", fontsize=9)
    axs[0].set_yscale("log"); axs[0].set_xlabel("Number of the 8 members that misclassify the sentence"); axs[0].set_ylabel("Sentences (log scale)")
    axs[0].set_title("How many members get each sentence wrong"); axs[0].set_ylim(top=cnt.max() * 5)
    axs[1].bar(ks, share * 100, color="tab:green")
    for k in ks: axs[1].text(k, share[k] * 100, f"{share[k]*100:.1f}%", ha="center", va="bottom", fontsize=9)
    axs[1].set_xlabel("Number of the 8 members that misclassify the sentence"); axs[1].set_ylabel("Share of stack total log loss (%)")
    axs[1].set_title("Where the stack's log loss comes from"); save(fig, "post_05_error_overlap.png")


def fig06(P, y, train):
    L = train.text.str.len().values; q = pd.qcut(L, 8, labels=False, duplicates="drop"); nb = q.max() + 1
    med = [np.median(L[q == k]) for k in range(nb)]; ms = ["stack8", "glm", "mlp_sweep_best", "nblr"]
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    for m in ms:
        nll = S.sample_loss(y, P[m]); ok = P[m].argmax(1) == y
        axs[0].plot(med, [nll[q == k].mean() for k in range(nb)], "o-", label=nm(m), lw=2.5 if m == "stack8" else 1.5)
        axs[1].plot(med, [ok[q == k].mean() for k in range(nb)], "o-", label=nm(m), lw=2.5 if m == "stack8" else 1.5)
    for a, t in zip(axs, ["Mean per-sentence log loss", "Accuracy"]):
        a.set_xscale("log"); a.set_xlabel("Median sentence length in bin (characters, log scale)"); a.set_ylabel(t)
        a.set_title(f"{t} by sentence length (8 quantile bins)"); a.grid(alpha=.3)
    axs[0].legend(); save(fig, "post_06_loss_by_length.png")


def fig07(P, y):
    X = np.hstack([np.log(S.clip_norm(P[m])) for m in M8]); lr = LogisticRegression(C=0.01, max_iter=1000).fit(X, y); W = lr.coef_
    cols = [f"{m}|{a}" for m in M8 for a in S.AUTHORS]
    fig, axs = plt.subplots(1, 2, figsize=(17, 5.5), gridspec_kw=dict(width_ratios=[2.3, 1]))
    v = np.abs(W).max(); im = axs[0].imshow(W, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
    axs[0].set_xticks(range(len(cols))); axs[0].set_xticklabels(cols, rotation=90, fontsize=8)
    axs[0].set_yticks(range(3)); axs[0].set_yticklabels(S.AUTHORS)
    for i in range(3):
        for j in range(len(cols)): axs[0].text(j, i, f"{W[i,j]:.2f}", ha="center", va="center", fontsize=6)
    axs[0].set_xlabel("Member | class of its log-probability input"); axs[0].set_ylabel("Predicted author (output class)")
    axs[0].set_title("Stack logistic-regression coefficients (C=0.01, fit on all rows)"); fig.colorbar(im, ax=axs[0], label="coefficient")
    s = np.abs(W).reshape(3, len(M8), 3).sum((0, 2)); o = np.argsort(s)
    axs[1].barh(range(len(M8)), s[o], color=[COL[family(M8[i])] for i in o]); axs[1].set_yticks(range(len(M8)))
    axs[1].set_yticklabels([nm(M8[i]) for i in o]); axs[1].set_xlabel("Sum of absolute coefficients"); axs[1].set_title("Total weight per member")
    legend_families(axs[1], ["classical", "neural"]); fig.tight_layout(); save(fig, "post_07_stack_weights.png")


def fig08(P, y):
    p = S.clip_norm(P["stack8"]); fig, axs = plt.subplots(1, 2, figsize=(14, 5.5))
    for k, a in enumerate(S.AUTHORS):
        fpr, tpr, _ = roc_curve(y == k, p[:, k]); axs[0].plot(fpr, tpr, lw=2, label=f"{a} vs rest (AUC = {auc(fpr, tpr):.4f})")
    axs[0].plot([0, 1], [0, 1], "k--", lw=1); axs[0].set_xlabel("False positive rate"); axs[0].set_ylabel("True positive rate")
    axs[0].set_title("Stack of 8: one-vs-rest ROC per author"); axs[0].legend(loc="lower right")
    pr, rc, f1, _ = precision_recall_fscore_support(y, p.argmax(1)); x = np.arange(3); w = .27
    for i, (v, n) in enumerate([(pr, "Precision"), (rc, "Recall"), (f1, "F1")]):
        b = axs[1].bar(x + (i - 1) * w, v, w, label=n)
        for r in b: axs[1].text(r.get_x() + r.get_width() / 2, r.get_height(), f"{r.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    axs[1].set_xticks(x); axs[1].set_xticklabels(S.AUTHORS); axs[1].set_ylim(.8, 1.0); axs[1].set_xlabel("Author"); axs[1].set_ylabel("Score")
    axs[1].set_title("Stack of 8: precision / recall / F1 per author"); axs[1].legend(loc="lower right"); save(fig, "post_08_per_author.png")


def fig09(P, y):
    fig, axs = plt.subplots(1, 2, figsize=(14, 5)); p = S.clip_norm(P["stack8"]); conf = p.max(1); ok = p.argmax(1) == y; bins = np.linspace(1 / 3, 1, 31)
    axs[0].hist(conf[ok], bins, alpha=.7, color="tab:green", label=f"correct (n={ok.sum()})"); axs[0].hist(conf[~ok], bins, alpha=.7, color="tab:red", label=f"wrong (n={(~ok).sum()})")
    axs[0].set_yscale("log"); axs[0].set_xlabel("Maximum predicted probability"); axs[0].set_ylabel("Sentences (log scale)")
    axs[0].set_title("Stack of 8: confidence, correct vs wrong"); axs[0].legend(loc="upper left")
    for m in ["stack8", "glm"]:
        l = np.sort(S.sample_loss(y, P[m]))[::-1]; c = np.cumsum(l) / l.sum(); x = np.arange(1, len(l) + 1) / len(l) * 100
        axs[1].plot(x, c * 100, lw=2, label=nm(m))
    axs[1].plot([0, 100], [0, 100], "k:", label="uniform"); axs[1].set_xlabel("Worst x% of sentences (by log loss)"); axs[1].set_ylabel("Cumulative share of total log loss (%)")
    axs[1].set_title("Concentration of log loss in the hardest sentences"); axs[1].legend(loc="lower right"); axs[1].grid(alpha=.3); save(fig, "post_09_confidence_vs_correctness.png")


def fig10(y, folds):
    ds = [("original", S.RESULTS), ("c1", "results/clean_runs/c1/results"), ("c2", "results/clean_runs/c2/results"), ("c3", "results/clean_runs/c3/results")]
    lab = {"original": "original", "c1": "c1: fmt + singleton words -> UNK", "c2": "c2: c1 + capitalised names -> NAME", "c3": "c3: fmt + names + rare-word masking"}
    D = {d: {m: np.load(f"{p}/oof_{m}.npy") for m in M7} for d, p in ds}
    fig, axs = plt.subplots(1, 2, figsize=(17, 5.8), gridspec_kw=dict(width_ratios=[2.4, 1]))
    w = .2; x = np.arange(len(M7)); cols = ["gray", "tab:blue", "tab:orange", "tab:purple"]
    for i, (d, _) in enumerate(ds):
        axs[0].bar(x + (i - 1.5) * w, [S.sample_loss(y, D[d][m]).mean() for m in M7], w, label=lab[d], color=cols[i])
    axs[0].set_xticks(x); axs[0].set_xticklabels([nm(m) for m in M7], rotation=25, ha="right"); axs[0].set_ylabel("Out-of-fold log loss")
    axs[0].set_xlabel("Member"); axs[0].set_title("Effect of text cleaning on each member"); axs[0].legend(fontsize=9); axs[0].set_ylim(0.2, None)
    st = {d: stack_oof(D[d], M7, y, folds) for d, _ in ds}
    vals = [S.sample_loss(y, st[d]).mean() for d, _ in ds]
    axs[1].bar(range(4), vals, color=cols)
    for i, (d, _) in enumerate(ds):
        t = f"{vals[i]:.4f}"
        if d != "original":
            dm, (lo, hi), _ = S.paired_bootstrap_logloss(y, st[d], st["original"], n=1000); t += f"\nvs orig {dm:+.4f}\n[{lo:+.4f}, {hi:+.4f}]"
        axs[1].text(i, vals[i], t, ha="center", va="bottom", fontsize=9)
    axs[1].set_xticks(range(4)); axs[1].set_xticklabels(["original", "c1", "c2", "c3"]); axs[1].set_ylim(min(vals) - .03, max(vals) + .02)
    axs[1].set_xlabel("Dataset"); axs[1].set_ylabel("Out-of-fold log loss"); axs[1].set_title("Stack of 7 members per dataset\n(paired bootstrap diff vs original, 95% CI)")
    fig.tight_layout(); save(fig, "post_10_cleaning_study.png")


def fig11(y, folds):
    rx = re.compile(r"^N(\d+)_lr([\d.e+-]+)_dp([\d.]+)_a[\d.e+-]+_s7_f(\d)\.npz$"); cfg = {}
    for f in glob.glob(os.path.join(S.RESULTS, "char_mlp_runs", "N*.npz")):
        m = rx.match(os.path.basename(f))
        if not m: continue
        N, lr, dp, k = int(m[1]), m[2], m[3], int(m[4]); cfg.setdefault((N, lr, dp), {})[k] = f
    res = {}
    for key, fs in cfg.items():
        if len(fs) < 5: continue
        oof = np.zeros((len(y), 3))
        for k, f in fs.items():
            z = np.load(f); oof[z["va"]] = z["pv"]
        res[key] = S.sample_loss(y, S.cv_temperature(oof, y, folds)[0]).mean()
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, N in zip(axs, (4, 5)):
        ks = [k for k in res if k[0] == N]
        if not ks: ax.set_title(f"N={N}: no data"); continue
        lrs = sorted({k[1] for k in ks}, key=float); dps = sorted({k[2] for k in ks}, key=float)
        M = np.full((len(lrs), len(dps)), np.nan)
        for k in ks: M[lrs.index(k[1]), dps.index(k[2])] = res[k]
        im = ax.imshow(M, cmap="viridis_r")
        for i in range(len(lrs)):
            for j in range(len(dps)):
                if not np.isnan(M[i, j]): ax.text(j, i, f"{M[i,j]:.4f}", ha="center", va="center", color="w", fontsize=10)
        ax.set_xticks(range(len(dps))); ax.set_xticklabels(dps); ax.set_yticks(range(len(lrs))); ax.set_yticklabels(lrs)
        ax.set_xlabel("Dropout"); ax.set_ylabel("Learning rate"); ax.set_title(f"Char {N}-gram TF-IDF MLP: temperature-scaled OOF log loss")
        fig.colorbar(im, ax=ax, label="log loss")
    fig.tight_layout(); save(fig, "post_11_mlp_sweep.png")


def fig12():
    df = pd.DataFrame([("stack8", .2492, .25663, .23384), ("pool9_dir", .2496, .25784, .23470), ("pool7_dir", .2499, .25700, .23495)],
                      columns=["submission", "oof_log_loss", "public_score", "private_score"])
    df.to_csv(os.path.join(S.RESULTS, "kaggle_scores.csv"), index=False)
    fig, ax = plt.subplots(figsize=(10, 5.5)); x = np.arange(3); w = .26
    for i, (c, n) in enumerate([("oof_log_loss", "Our OOF estimate"), ("public_score", "Kaggle public"), ("private_score", "Kaggle private")]):
        b = ax.bar(x + (i - 1) * w, df[c], w, label=n)
        for r in b: ax.text(r.get_x() + r.get_width() / 2, r.get_height(), f"{r.get_height():.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(df.submission); ax.set_ylim(.22, .265); ax.set_xlabel("Submission"); ax.set_ylabel("Multiclass log loss")
    ax.set_title("Out-of-fold estimate vs Kaggle public and private scores"); ax.legend(loc="upper left"); save(fig, "post_12_kaggle_scores.png")


def main():
    t0 = time.time(); train, test, folds, y = S.load()
    P = {m: S.load_method(m)[0] for m in ALL}; P["stack8"] = stack_oof(P, M8, y, folds)
    ev = {m: S.evaluate(y, P[m], folds, 1000) for m in P}
    rows = [dict(model=m, display=nm(m), family=family(m), oof_log_loss=e["log_loss"], ci_low=e["log_loss_ci95"][0], ci_high=e["log_loss_ci95"][1],
                 accuracy=e["accuracy"], macro_f1=e["macro_f1"]) for m, e in ev.items()]
    df = pd.DataFrame(rows); df.to_csv(os.path.join(S.RESULTS, "final_metrics_v2.csv"), index=False); print(df.round(4).to_string(index=False))
    for f, a in [(fig01, (P, y, ev)), (fig02, (P, y, ev)), (fig03, (P, y)), (fig04, (P, y)), (fig05, (P, y)), (fig06, (P, y, train)),
                 (fig07, (P, y)), (fig08, (P, y)), (fig09, (P, y)), (fig10, (y, folds)), (fig11, (y, folds)), (fig12, ())]:
        try: f(*a)
        except Exception as e: print("FAILED", f.__name__, repr(e))
    print(f"runtime {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
