"""Improvement methodology A: log-linear (geometric) pooling of several methodologies' probabilities with jointly fitted exponents.
    p_pool(c | x) proportional to  prod_m  p_m(c | x) ** w_m
The exponents w_m double as a joint calibration (a common scale on all members is a temperature on the pool). They are fitted by minimising
log loss, and every score reported for a sentence uses exponents fitted on the other folds only."""
import numpy as np
from scipy.optimize import minimize

import spooky_common as S


def pool(prob_list, w):
    """Normalised weighted geometric mean of the members' probability matrices."""
    z = sum(wi * np.log(S.clip_norm(p)) for wi, p in zip(w, prob_list)); z = z - z.max(1, keepdims=True); e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fit_exponents(prob_list, y):
    """Exponents minimising the log loss of the pool on (prob_list, y); starts from equal weights summing to one."""
    f = lambda w: S.sample_loss(y, pool(prob_list, w)).mean()
    return minimize(f, np.ones(len(prob_list)) / len(prob_list), method="Nelder-Mead", options=dict(xatol=1e-4, fatol=1e-8, maxiter=800)).x


def nested_pool(prob_list, y, folds):
    """Out-of-fold pooled probabilities and the exponents used for each fold (fitted on the other folds' rows only)."""
    out, ws = np.zeros_like(prob_list[0]), []
    for k in range(int(folds.max()) + 1):
        tr, va = folds != k, folds == k; w = fit_exponents([p[tr] for p in prob_list], y[tr]); ws.append(w); out[va] = pool([p[va] for p in prob_list], w)
    return out, np.array(ws)


def dirichlet_map(pool_probs, y, C=10.0):
    """Improvement B: Dirichlet-style calibration map = multinomial logistic regression with L2 on the log-probabilities (k^2 + k parameters for k classes)."""
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=C, max_iter=1000).fit(np.log(S.clip_norm(pool_probs)), y)


def apply_map(m, pool_probs):
    return S.clip_norm(m.predict_proba(np.log(S.clip_norm(pool_probs))))


def nested_pool_dirichlet(prob_list, y, folds, C=10.0):
    """Fully nested pool-then-Dirichlet. For each outer fold k, everything is fitted on the other folds' rows only: the pool exponents (on all of them) and the calibration
    map (on pooled predictions that are themselves out-of-fold *within* those rows, exponents refitted per inner fold). Returns the calibrated out-of-fold probabilities."""
    out, ids = np.zeros_like(prob_list[0]), np.unique(folds)
    for k in ids:
        tr, va = folds != k, folds == k; inner = np.zeros((int(tr.sum()), prob_list[0].shape[1])); tr_idx = np.where(tr)[0]; fo = folds[tr_idx]
        for j in ids[ids != k]:
            fit, ap = fo != j, fo == j; w = fit_exponents([p[tr_idx][fit] for p in prob_list], y[tr_idx][fit]); inner[ap] = pool([p[tr_idx][ap] for p in prob_list], w)
        m = dirichlet_map(inner, y[tr_idx], C); w_all = fit_exponents([p[tr] for p in prob_list], y[tr]); out[va] = apply_map(m, pool([p[va] for p in prob_list], w_all))
    return out
