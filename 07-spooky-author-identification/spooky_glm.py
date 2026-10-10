"""Methodology 5b: the Generalized Language Model of Pickhardt et al. 2014 (arXiv 1404.3377), per author, word level, from scratch. Differs from spooky_wordlm.py (plain MKN) by the paper's actual contribution:
the highest order interpolates with ALL single-skip lower patterns, uniformly (Sec.3.2, Eq.4); every lower pattern interpolates with all single skips of its remaining positions (Eq.7).
Counts per pattern mask m over the N-1 context positions (as in the paper's worked n=3 example):
  cnt_m(ctx_m, w)  = raw count aggregated over the skipped words,
  cont_m(ctx_m, w) = N1+ = number of DISTINCT tuples of skipped words with c(full n-gram) > 0,        den_m(ctx_m) = sum_w cont_m(ctx_m, w).
P_top = max(c - D(c), 0)/c(ctx) + gamma * mean_p P_{top minus p};  P_m = max(cont - D, 0)/den + gamma_m * mean_{p in m} P_{m minus p};  P_empty = unigram continuation count interpolated with uniform 1/V.
Deviation from the paper's closed-form gamma: gamma(ctx) = sum_w min(count, D)/total, which is the paper's gamma when D <= count and keeps every distribution normalised otherwise (checked in tests/test_glm.py).
disc='cont' picks D1/D2/D3+ by the continuation count (standard MKN); disc='raw' picks D by the aggregated raw count c (the paper's D(c(skip n-gram)))."""
from collections import Counter, defaultdict

import numpy as np

BOS, EOS = "<s>", "</s>"


def discounts(counts):
    n = Counter(v for v in counts if v <= 4); n1, n2, n3, n4 = (max(n[i], 1) for i in (1, 2, 3, 4)); Y = n1 / (n1 + 2 * n2)
    return (min(max(1 - 2 * Y * n2 / n1, 0.05), 0.95), min(max(2 - 3 * Y * n3 / n2, 0.05), 1.95), min(max(3 - 4 * Y * n4 / n3, 0.05), 2.95))


class GLM:
    def __init__(self, sents, N, V, disc="cont"):
        self.N, self.V, self.disc = N, V, disc; full = Counter()
        for s in sents:
            t = [BOS] * (N - 1) + s + [EOS]
            for i in range(N - 1, len(t)): full[tuple(t[i - N + 1:i + 1])] += 1
        self.top = (1 << (N - 1)) - 1; self.masks = sorted(range(1 << (N - 1)), key=lambda m: bin(m).count("1"))
        self.kept = {m: [i for i in range(N - 1) if m >> i & 1] for m in self.masks}; self.cnt, self.cont, self.den, self.D, self.gam = {}, {}, {}, {}, {}
        for m in self.masks:
            kept, skipped = self.kept[m], [i for i in range(N - 1) if not m >> i & 1]; cnt, dist = Counter(), defaultdict(set)
            for g, c in full.items():
                key = (tuple(g[i] for i in kept), g[-1]); cnt[key] += c
                if m != self.top: dist[key].add(tuple(g[i] for i in skipped))
            cont = cnt if m == self.top else {k: len(v) for k, v in dist.items()}; den = Counter()
            for (ctx, w), v in cont.items(): den[ctx] += v
            self.cnt[m], self.cont[m], self.den[m] = cnt, cont, den
            self.D[m] = discounts(cnt.values() if disc == "raw" else cont.values()); gsum = Counter()
            for key, v in cont.items(): gsum[key[0]] += min(v, self._d(m, cnt[key], v))
            self.gam[m] = {ctx: gsum[ctx] / den[ctx] for ctx in den}

    def _d(self, m, c, v):
        D = self.D[m]; return D[min(c if self.disc == "raw" else v, 3) - 1]

    def prob(self, w, ctx):
        """P_GLM(w | ctx) for the N-1 context words ctx (oldest first)."""
        P = {}
        for m in self.masks:
            ck = tuple(ctx[i] for i in self.kept[m]); key = (ck, w); den = self.den[m].get(ck)
            lower = [P[m & ~(1 << p)] for p in self.kept[m]]; low = sum(lower) / len(lower) if lower else 1.0 / self.V
            if den is None: P[m] = low; continue
            v = self.cont[m].get(key, 0); c = self.cnt[m].get(key, 0); d = self._d(m, c, v) if v else 0.0
            P[m] = max(v - d, 0.0) / den + self.gam[m][ck] * low
        return P[self.top]

    def loglik(self, s):
        t = [BOS] * (self.N - 1) + s + [EOS]; return float(sum(np.log(self.prob(t[i], t[i - self.N + 1:i])) for i in range(self.N - 1, len(t))))
