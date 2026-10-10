"""Negative control for the character language model pipeline: with the author labels permuted (all else identical), the held-out log loss must stay at the
class-prior level. A value clearly below it would mean held-out text or labels leak into the models. Fold 0, order 6."""
import os
import sys

import numpy as np
from scipy.optimize import minimize_scalar

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import spooky_charlm as CL
import spooky_common as S

train, test, folds, y = S.load(); rng = np.random.default_rng(0); txt = train.text.values; cmap = CL.char_map(list(txt) + list(test.text.values)); N = 6
tr, va = np.where(folds != 0)[0], np.where(folds == 0)[0]
def run(labels):
    lms = [CL.CharLM(list(txt[tr[labels[tr] == a]]), cmap, N) for a in range(3)]; ll = np.stack([lm.loglik(list(txt[va])) for lm in lms], 1)[:, :, N - 1]
    sm = lambda z: (lambda e: e / e.sum(1, keepdims=True))(np.exp(z - z.max(1, keepdims=True)))
    prior = np.log(np.bincount(labels[tr], minlength=3) / len(tr)); Z = ll + prior; b = minimize_scalar(lambda b: S.sample_loss(labels[va], sm(b * Z)).mean(), bounds=(1e-3, 3), method="bounded").x
    return S.sample_loss(labels[va], sm(b * Z)).mean(), (Z.argmax(1) == labels[va]).mean(), b
prior_loss = S.sample_loss(y[va], np.tile(np.bincount(y[tr]) / len(tr), (len(va), 1))).mean()
real = run(y); shuf = run(rng.permutation(y))
print(f"class-prior log loss on fold 0: {prior_loss:.4f}")
print(f"real labels:     log loss {real[0]:.4f}, accuracy {real[1]:.4f}, beta {real[2]:.3f}")
print(f"shuffled labels: log loss {shuf[0]:.4f}, accuracy {shuf[1]:.4f}, beta {shuf[2]:.3f}")
ok = shuf[0] > prior_loss - 0.02 and shuf[1] < 0.45 and real[0] < prior_loss - 0.5
print("PASS negative control: shuffled labels give no signal (loss within 0.02 of the class prior, accuracy near chance), real labels do" if ok else "FAIL negative control: held-out signal survives label shuffling")
sys.exit(0 if ok else 1)
