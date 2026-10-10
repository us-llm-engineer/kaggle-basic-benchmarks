"""Original data vs the three cleaned datasets: per-member out-of-fold log loss and the 7-member stack (logistic regression C = 0.01 on member log-probabilities, fitted on the other folds), with a paired bootstrap on identical sentences/folds."""
import os, numpy as np
from sklearn.linear_model import LogisticRegression
import spooky_common as S
train, test, folds, y = S.load(); M = ["nblr", "style_gbdt", "charlm", "wordlm", "glm", "fasttext", "char_svm"]
def load(src, m): return np.load(os.path.join(src, f"oof_{m}.npy"))
def stack(src):
    X = np.hstack([np.log(S.clip_norm(load(src, m))) for m in M]); o = np.zeros((len(y), 3))
    for k in range(5): tr, va = folds != k, folds == k; o[va] = LogisticRegression(C=0.01, max_iter=2000).fit(X[tr], y[tr]).predict_proba(X[va])
    return o
srcs = {"original": S.RESULTS, "c1 fmt+unk": "results/clean_runs/c1/results", "c2 fmt+names+unk": "results/clean_runs/c2/results", "c3 fmt+names+distort": "results/clean_runs/c3/results"}
print(f"{'member':12s}" + "".join(f"{k:>22s}" for k in srcs))
for m in M: print(f"{m:12s}" + "".join(f"{S.evaluate(y, load(s, m), folds, 10)['log_loss']:22.4f}" for s in srcs.values()))
st = {k: stack(s) for k, s in srcs.items()}; ref = st["original"]
for k, o in st.items():
    e = S.evaluate(y, o, folds, 300); dm, ci, p = S.paired_bootstrap_logloss(y, o, ref) if k != "original" else (0, (0, 0), 0)
    print(f"STACK {k:22s} log loss {e['log_loss']:.4f} {np.round(e['log_loss_ci95'],4)} acc {e['accuracy']:.4f}" + (f" | vs original {dm:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]" if k != "original" else ""))
