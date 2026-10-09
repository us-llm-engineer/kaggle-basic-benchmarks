"""Post-hoc diagnostics saved to results/: (1) accuracy versus F1 per model at both thresholds, (2) a convergence audit of every
model, (3) near-duplicate structure and an implied label-noise estimate. Run: python3 diagnostics.py"""
import glob, json, re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
import nlp_common as N

train, test, folds = N.load(); y = train.target.values; R = "results/"
MODELS = ["svm_cost", "nbsvm", "cnb", "gbdt", "fasttext", "textcnn"]

# 1. accuracy (= micro-F1, the leaderboard metric for two classes) versus positive-class F1
acc = {}
for k in MODELS:
    o, _, m = N.load_method(k); th = m["threshold"]; grid = np.quantile(o, np.linspace(0.2, 0.8, 241)); a = [accuracy_score(y, o > t) for t in grid]; ta = grid[int(np.argmax(a))]
    acc[k] = dict(f1_threshold=dict(f1=f1_score(y, o > th), accuracy=accuracy_score(y, o > th), positive_rate=float((o > th).mean())),
                  accuracy_threshold=dict(f1=f1_score(y, o > ta), accuracy=float(max(a)), positive_rate=float((o > ta).mean())))
json.dump(acc, open(R + "accuracy_vs_f1.json", "w"), indent=2, default=float)

# 2. convergence audit
audit = {"solver_warnings_in_notebooks": {}}
for f in sorted(glob.glob("0*.ipynb")):
    n = 0
    for c in json.load(open(f))["cells"]:
        for o in c.get("outputs", []):
            n += len(re.findall(r"ConvergenceWarning|failed to converge|did not converge", "".join(o.get("text", "")) if o.get("output_type") == "stream" else ""))
    audit["solver_warnings_in_notebooks"][f] = n
for name, cap, pat in [("fasttext", 40, 6), ("textcnn", 30, 6)]:
    rows = []
    for r in json.load(open(R + f"dl_history_{name}.json")):
        h, b = r["history"], r["best_epoch"]
        rows.append(dict(best_epoch=b, epochs=len(h), stopped_by_patience=len(h) - 1 - b >= pat, hit_epoch_cap=len(h) >= cap, train_inner_loss_gap=h[b]["inner_loss"] - h[b]["train_loss"]))
    d = pd.DataFrame(rows); audit[name] = dict(best_epoch_range=[int(d.best_epoch.min()), int(d.best_epoch.max())], runs=len(d), stopped_by_patience=int(d.stopped_by_patience.sum()),
                                              hit_epoch_cap=int(d.hit_epoch_cap.sum()), mean_train_inner_loss_gap=float(d.train_inner_loss_gap.mean()))
audit["gbdt"] = dict(best_iterations=json.load(open(R + "meta_gbdt.json"))["best_iterations"], iteration_cap=1500)
json.dump(audit, open(R + "convergence_audit.json", "w"), indent=2, default=float)

# 3. near-duplicates and the implied label-noise level
txt = lambda df: N.norm(df.text).str.replace(r"\bxurl\b|\bxuser\b", " ", regex=True)
A = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit_transform(pd.concat([txt(train), txt(test)])).astype(np.float32).tocsr(); Atr, Ate = A[:len(train)], A[len(train):]
def nearest(Q, Rm, self_):
    s, ix = np.zeros(Q.shape[0], np.float32), np.zeros(Q.shape[0], int)
    for i in range(0, Q.shape[0], 1000):
        S = (Q[i:i + 1000] @ Rm.T).toarray()
        if self_: S[np.arange(S.shape[0]), np.arange(i, i + S.shape[0])] = -1
        ix[i:i + 1000], s[i:i + 1000] = S.argmax(1), S.max(1)
    return s, ix
s_tr, i_tr = nearest(Atr, Atr, True); s_te, _ = nearest(Ate, Atr, False); dup = {}
for th in [0.6, 0.7, 0.8, 0.9]:
    m = s_tr >= th; dup[str(th)] = dict(train_share=float(m.mean()), test_share=float((s_te >= th).mean()), pairs=int(m.sum()), label_differs=float(np.mean(y[m] != y[i_tr[m]])))
d9 = dup["0.9"]["label_differs"]; eps = (1 - np.sqrt(max(1 - 2 * d9, 0))) / 2
sv = N.load_method("svm_cost"); pred = (sv[0] > sv[2]["threshold"]).astype(int); cross = folds[i_tr] != folds; bands = {}
for lo, hi in [(0.6, 0.8), (0.8, 0.9), (0.9, 1.01)]:
    m = (s_tr >= lo) & (s_tr < hi) & cross; bands[f"{lo}-{min(hi, 1.0)}"] = dict(n=int(m.sum()), svm_accuracy=float(np.mean(pred[m] == y[m])), neighbour_label_accuracy=float(np.mean(y[i_tr[m]] == y[m])))
m = s_tr < 0.6; bands["below_0.6"] = dict(n=int(m.sum()), svm_accuracy=float(np.mean(pred[m] == y[m])))
json.dump(dict(similarity_thresholds=dup, implied_per_label_noise_if_independent=float(eps), cross_fold_bands=bands), open(R + "near_duplicates.json", "w"), indent=2, default=float)
print("diagnostics written: accuracy_vs_f1.json, convergence_audit.json, near_duplicates.json")
