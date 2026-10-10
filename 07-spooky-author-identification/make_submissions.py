"""Write the three best modellings as Kaggle submissions (columns id,EAP,HPL,MWS; each row sums to one).
 A stack8 : logistic regression (C=0.01) on the log-probabilities of 8 members, fitted on ALL out-of-fold rows, applied to the members' fold-averaged test probabilities. OOF log loss 0.2492
 B pool9_dir : nested log-linear pool of 9 members + Dirichlet-style map (improvement_mlp.npz). OOF 0.2496
 C pool7_dir : nested log-linear pool of 7 members without the CNN + Dirichlet-style map (improvement_glm.npz). OOF 0.2499"""
import os, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
import spooky_common as S
train, test, folds, y = S.load(); sub = pd.read_csv("data/sample_submission.csv"); assert (sub["id"].values == test["id"].values).all(); cols = list(sub.columns[1:]); assert cols == S.AUTHORS
names = ["nblr", "style_gbdt", "cnn", "charlm", "wordlm", "glm", "fasttext", "char_svm"]
X = np.hstack([np.log(S.clip_norm(S.load_method(n)[0])) for n in names]); Xt = np.hstack([np.log(S.clip_norm(S.load_method(n)[1])) for n in names])
pA = LogisticRegression(C=0.01, max_iter=2000).fit(X, y).predict_proba(Xt)
pB = np.load("results/improvement_mlp.npz")["test_pool_nodeep_glm_mlp_dir"]; pC = np.load("results/improvement_glm.npz")["test_pool_nodeep_glm_dir"]
for name, p in (("stack8", pA), ("pool9_dir", pB), ("pool7_dir", pC)):
    p = S.clip_norm(p); assert p.shape == (len(test), 3) and np.isfinite(p).all(); out = sub.copy(); out[cols] = p; path = f"submissions/submission_{name}.csv"; out.to_csv(path, index=False)
    print(f"{path}: rows {len(out)}, mean probs {np.round(p.mean(0),3).tolist()}, mean max prob {p.max(1).mean():.3f}, argmax shares {np.round(np.bincount(p.argmax(1),minlength=3)/len(p),3).tolist()}")
print("pairwise mean |diff|: A-B %.4f A-C %.4f B-C %.4f" % (np.abs(pA-S.clip_norm(pB)).mean(), np.abs(pA-S.clip_norm(pC)).mean(), np.abs(S.clip_norm(pB)-S.clip_norm(pC)).mean()))
