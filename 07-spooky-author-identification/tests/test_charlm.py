"""Tests of the vectorised Witten-Bell character LM against a plain-Python reference, plus normalisation and sanity checks."""
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import spooky_charlm as CL

ok = []
def check(name, cond): ok.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name)

rng = np.random.default_rng(0); ALPHA = list("abcde ")
def gen(n, weights):  # sentences of a toy author: characters drawn with author-specific probabilities
    return ["".join(rng.choice(ALPHA, size=int(rng.integers(8, 30)), p=weights)) for _ in range(n)]
A_tr, B_tr = gen(200, [.35, .3, .1, .05, .05, .15]), gen(200, [.05, .1, .3, .35, .1, .1]); held_A, held_B = gen(60, [.35, .3, .1, .05, .05, .15]), gen(60, [.05, .1, .3, .35, .1, .1])
cmap = CL.char_map(A_tr + B_tr + ["z"]); N = 4

def reference(train, test, N, V):
    """Plain-Python interpolated Witten-Bell model with the same padding, end symbol and add-one unigram."""
    cnt = [defaultdict(float) for _ in range(N + 1)]; tot = [defaultdict(float) for _ in range(N + 1)]; typ = [defaultdict(float) for _ in range(N + 1)]; uni = defaultdict(float)
    enc = lambda t: [CL.PAD] * (N - 1) + [cmap.get(c, CL.UNK) for c in t] + [CL.END]
    for t in train:
        x = enc(t)
        for i in range(N - 1, len(x)):
            uni[x[i]] += 1
            for k in range(2, N + 1):
                h, c = tuple(x[i - k + 1:i]), x[i]
                if cnt[k][h + (c,)] == 0: typ[k][h] += 1
                cnt[k][h + (c,)] += 1; tot[k][h] += 1
    n_uni = sum(uni.values())
    def prob(h, c, k):  # P_k(c | last k-1 symbols of h)
        if k == 1: return (uni[c] + 1) / (n_uni + V)
        hh = tuple(h[len(h) - k + 1:]); lower = prob(h, c, k - 1)
        return lower if tot[k][hh] == 0 else (cnt[k][hh + (c,)] + typ[k][hh] * lower) / (tot[k][hh] + typ[k][hh])
    out = np.zeros((len(test), N))
    for s, t in enumerate(test):
        x = enc(t)
        for i in range(N - 1, len(x)):
            for k in range(1, N + 1): out[s, k - 1] += np.log(prob(x[:i], x[i], k))
    return out, prob, enc

lm = CL.CharLM(A_tr, cmap, N); ref, prob, enc = reference(A_tr, held_A + ["abz ab", "e e"], N, lm.V)
check("vectorised log-likelihoods equal the plain-Python reference at every order (held-out text, unseen contexts, unknown character)", np.allclose(lm.loglik(held_A + ["abz ab", "e e"]), ref, atol=1e-9))
check("the same on the training sentences themselves", np.allclose(lm.loglik(A_tr[:30]), reference(A_tr, A_tr[:30], N, lm.V)[0], atol=1e-9))
# normalisation: for several contexts, the probabilities of all predictable symbols sum to one at every order
syms = [CL.END, CL.UNK] + sorted(cmap.values()); worst = 0.0
for h in [[CL.PAD] * 3, [CL.PAD, CL.PAD, cmap["a"]], [cmap["a"], cmap["b"], cmap["c"]], [cmap["e"], cmap["e"], cmap["e"]], [cmap["d"], cmap[" "], cmap["d"]]]:
    for k in range(1, N + 1): worst = max(worst, abs(sum(prob(h, c, k) for c in syms) - 1))
check("P_k(. | h) sums to one over all predictable symbols for seen and unseen contexts and every order", worst < 1e-9)
# sanity: a model of author A prefers A's text, and more data / higher order helps in-domain
lmB = CL.CharLM(B_tr, cmap, N)
check("each author's model gives its own held-out text the higher log-likelihood on most sentences", (lm.loglik(held_A)[:, -1] > lmB.loglik(held_A)[:, -1]).mean() > 0.9 and (lmB.loglik(held_B)[:, -1] > lm.loglik(held_B)[:, -1]).mean() > 0.9)
# higher orders must help on text with sequential structure (a noisy repeating pattern), which an order-1 model cannot capture
def pat(n):  # a repeating 'abcde' pattern, random phase per sentence, 8% of characters replaced by 'a'
    out = []
    for _ in range(n):
        ph, L = int(rng.integers(0, 5)), int(rng.integers(20, 40)); out.append("".join("a" if rng.random() < 0.08 else "abcde"[(i + ph) % 5] for i in range(L)))
    return out
S_tr, S_held = pat(200), pat(60); cm2 = CL.char_map(S_tr); ll = CL.CharLM(S_tr, cm2, 4).loglik(S_held)
check("on text with sequential structure the order-4 model scores held-out text higher than the order-1 model", ll[:, 3].sum() > ll[:, 0].sum() + 100)
check("unknown characters are handled without error and receive finite probability", np.isfinite(lm.loglik(["zzz ab"])).all())
grams = {tuple(int(v) for v in rng.integers(4, 60, 4)) for _ in range(6000)}; keys = {int(CL.kgram_keys(np.array(g, dtype=np.int64), 4)[-1]) for g in grams}
check("k-gram keys are exact: 6000 distinct 4-grams give 6000 distinct keys", len(keys) == len(grams))
g = np.array([5, 17, 30, 44], dtype=np.int64); check("the key of a k-gram divided by the base is the key of its (k-1)-gram context", CL.kgram_keys(g, 4)[-1] // CL.BASE == CL.kgram_keys(g, 3)[-2])
print(f"{sum(ok)}/{len(ok)} passed"); sys.exit(0 if all(ok) else 1)
