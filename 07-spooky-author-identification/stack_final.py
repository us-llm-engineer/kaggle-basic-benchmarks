"""Final stacker of the best modelling so far: multinomial logistic regression (C = 0.01) on the concatenated log-probabilities of the member methods, fitted on the other folds' out-of-fold rows only. Usage: python3 stack_final.py [member,member,...]  (reads results/oof_*.npy of the working folder)"""
import sys, numpy as np
from sklearn.linear_model import LogisticRegression
import spooky_common as S
train, test, folds, y = S.load(); names = (sys.argv[1] if len(sys.argv) > 1 else "nblr,style_gbdt,cnn,charlm,wordlm,glm,fasttext,char_svm").split(",")
X = np.hstack([np.log(S.clip_norm(S.load_method(n)[0])) for n in names]); oof = np.zeros((len(y), 3))
for k in range(5): tr, va = folds != k, folds == k; oof[va] = LogisticRegression(C=0.01, max_iter=2000).fit(X[tr], y[tr]).predict_proba(X[va])
e = S.evaluate(y, oof, folds, 300); print(f"stack of {len(names)} [{','.join(names)}]: log loss {e['log_loss']:.4f} {np.round(e['log_loss_ci95'],4)} acc {e['accuracy']:.4f}")
