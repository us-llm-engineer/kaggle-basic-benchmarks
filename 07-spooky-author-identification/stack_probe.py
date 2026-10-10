"""Probe (new script): nested stacking of the member log-probabilities with a multinomial logistic regression (3*M inputs) and a LightGBM meta-learner. For outer fold k the meta-learner is trained on the other folds' out-of-fold member predictions only."""
import sys, numpy as np, lightgbm as lgb
from sklearn.linear_model import LogisticRegression
import spooky_common as S
train, test, folds, y = S.load(); names = ["nblr", "style_gbdt", "cnn", "charlm", "wordlm", "glm", "fasttext", "char_svm"]
X = np.hstack([np.log(S.clip_norm(S.load_method(n)[0])) for n in names]); L = train.text.str.len().values[:, None] / 150.0
def run(make, Xm):
    oof = np.zeros((len(y), 3))
    for k in range(5):
        tr, va = folds != k, folds == k; m = make().fit(Xm[tr], y[tr]); oof[va] = m.predict_proba(Xm[va])
    return oof
for C in (0.01, 0.03, 0.1, 1.0):
    o = run(lambda: LogisticRegression(C=C, max_iter=2000), X); e = S.evaluate(y, o, folds, 50); print(f"stack LR on {len(names)} members C={C}: log loss {e['log_loss']:.4f} acc {e['accuracy']:.4f}", flush=True)
o = run(lambda: LogisticRegression(C=0.1, max_iter=2000), np.hstack([X, L, X * L])); e = S.evaluate(y, o, folds, 50); print(f"stack LR + length interactions: log loss {e['log_loss']:.4f} acc {e['accuracy']:.4f}", flush=True)
P = dict(objective="multiclass", num_class=3, learning_rate=0.03, num_leaves=7, min_child_samples=50, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=4, seed=7)
class G:
    def fit(self, a, b): self.m = lgb.train(P, lgb.Dataset(a, b), 250); return self
    def predict_proba(self, a): return self.m.predict(a)
o = run(G, np.hstack([X, L])); e = S.evaluate(y, o, folds, 50); print(f"stack LightGBM (250 rounds): log loss {e['log_loss']:.4f} acc {e['accuracy']:.4f}", flush=True)
