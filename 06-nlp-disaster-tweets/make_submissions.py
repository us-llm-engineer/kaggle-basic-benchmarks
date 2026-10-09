"""Write one submission CSV per model: fold-averaged test scores, standardised with the out-of-fold mean/std,
thresholded at the model's out-of-fold F1-optimal threshold (the same pipeline as the comparison notebook)."""
import numpy as np, pandas as pd
import nlp_common as N

train, test, folds = N.load(); y = train.target.values
KIND = {"tfidf_lr": "prob", "nbsvm": "margin", "cnb": "margin", "svm_cost": "margin", "gbdt": "prob", "fasttext": "margin", "textcnn": "margin"}
logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
for k, kind in KIND.items():
    oof, tst = (np.load("results/v1_tfidf_oof.npy"), np.load("results/v1_tfidf_test_proba.npy")) if k == "tfidf_lr" else N.load_method(k)[:2]
    o, t = (logit(oof), logit(tst)) if kind == "prob" else (oof, tst)
    z, zt = (o - o.mean()) / o.std(), (t - o.mean()) / o.std(); th = N.evaluate(y, z, folds, n_boot=10)["threshold"]
    sub = pd.DataFrame({"id": test.id, "target": (zt > th).astype(int)}); sub.to_csv(f"submission_{k}.csv", index=False)
    print(f"{k:9s} threshold(z) {th:+.3f}  test positive rate {sub.target.mean():.4f}  file submission_{k}.csv")
