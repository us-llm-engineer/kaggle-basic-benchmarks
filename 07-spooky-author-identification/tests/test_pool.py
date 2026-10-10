"""Unit tests of the pooling module: algebra, recovery of a known miscalibration, and a leakage test with a mutation check."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import spooky_common as S
import spooky_pool as PL

ok = []
def check(name, cond): ok.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name)

rng = np.random.default_rng(0); n = 3000; y = rng.integers(0, 3, n); folds = np.tile(np.arange(5), n // 5 + 1)[:n]
def member(strength, noise):  # softmax of (strength * one-hot + noise): a miscalibrated but informative member
    z = strength * np.eye(3)[y] + rng.normal(0, noise, (n, 3)); e = np.exp(z - z.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)
A, B = member(2.0, 1.0), member(1.0, 1.5)

# algebra
check("pool of one member with exponent 1 returns that member", np.allclose(PL.pool([A], [1.0]), S.clip_norm(A)))
check("pool of two identical members with exponents summing to 1 returns the member", np.allclose(PL.pool([A, A], [0.4, 0.6]), S.clip_norm(A)))
check("pool rows sum to one for arbitrary exponents", np.allclose(PL.pool([A, B], [0.7, 1.9]).sum(1), 1))
check("a larger common exponent sharpens: confidence rises", PL.pool([A, B], [1.0, 1.0]).max(1).mean() > PL.pool([A, B], [0.5, 0.5]).max(1).mean())

# recovery of a known miscalibration: sharpening a member by a factor 3 must divide its optimal exponent by 3
wA = PL.fit_exponents([A], y); wO = PL.fit_exponents([PL.pool([A], [3.0])], y)
check("fitted exponent of a member sharpened 3x is one third of the member's own optimum", abs(3 * wO[0] - wA[0]) < 0.03 * max(wA[0], 1))
check("fitted pool never scores worse than equal weights on the fitting data", S.sample_loss(y, PL.pool([A, B], PL.fit_exponents([A, B], y))).mean() <= S.sample_loss(y, PL.pool([A, B], [0.5, 0.5])).mean() + 1e-9)

# leakage: the pooled prediction of fold 0 must not depend on the labels of fold 0
def leak_free(nested_fn):
    y2 = y.copy(); y2[folds == 0] = rng.permutation(y2[folds == 0]); a = nested_fn([A, B], y, folds)[0][folds == 0]; b = nested_fn([A, B], y2, folds)[0][folds == 0]; return np.allclose(a, b, atol=1e-9)
def mutant_nested_pool(prob_list, y, folds):  # MUTANT: exponents fitted on all rows, scored rows included
    w = PL.fit_exponents(prob_list, y); return PL.pool(prob_list, w), np.array([w])
check("nested pooling: fold-0 predictions do not depend on fold-0 labels", leak_free(PL.nested_pool))
check("MUTANT (exponents fitted on all rows) is detected by the leakage test", not leak_free(mutant_nested_pool))
oof, ws = PL.nested_pool([A, B], y, folds)
check("nested pooling returns 5 exponent vectors and normalised rows", ws.shape == (5, 2) and np.allclose(oof.sum(1), 1))

# improvement B: the Dirichlet-style map must also be fitted without the scored fold, and must help when the pool has a class bias
def leak_free_dir(fn):
    y2 = y.copy(); y2[folds == 0] = rng.permutation(y2[folds == 0]); a = fn([A, B], y, folds)[folds == 0]; b = fn([A, B], y2, folds)[folds == 0]; return np.allclose(a, b, atol=1e-7)
def mutant_dir(prob_list, y_, folds_, C=10.0):  # MUTANT: the map is fitted on all rows, scored fold included
    pl = PL.pool(prob_list, PL.fit_exponents(prob_list, y_)); m = PL.dirichlet_map(pl, y_, C); return PL.apply_map(m, pl)
check("nested pool + Dirichlet: fold-0 predictions do not depend on fold-0 labels", leak_free_dir(PL.nested_pool_dirichlet))
check("MUTANT (Dirichlet map fitted on all rows) is detected by the leakage test", not leak_free_dir(mutant_dir))
biased = [np.clip(A * np.array([1.8, 1.0, 0.55]), 1e-9, None)]; biased = [biased[0] / biased[0].sum(1, keepdims=True)]
check("the Dirichlet map lowers the log loss of a class-biased member that exponents alone cannot fix", S.sample_loss(y, PL.nested_pool_dirichlet(biased, y, folds)).mean() < S.sample_loss(y, PL.nested_pool(biased, y, folds)[0]).mean() - 0.01)
check("the calibrated rows are normalised", np.allclose(PL.nested_pool_dirichlet([A, B], y, folds).sum(1), 1))
print(f"{sum(ok)}/{len(ok)} passed"); sys.exit(0 if all(ok) else 1)
