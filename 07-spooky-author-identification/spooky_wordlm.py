"""Methodology 5 (replacement for the scratch deep screens): a per-author WORD-level interpolated modified Kneser-Ney n-gram language model, from scratch.
Formulas follow Pickhardt et al. 2014 (arXiv 1404.3377) Sec.3.2 and Appendix A (discounts D1, D2, D3+ from count-of-counts, Y = n1 / (n1 + 2 n2)):
    P_k(w | h) = max(c(hw) - D(c(hw)), 0) / c(h)  +  gamma(h) * P_{k-1}(w | h'),    gamma(h) = (D1 N1(h.) + D2 N2(h.) + D3+ N3+(h.)) / c(h)
with raw counts at the highest order and continuation counts N1+(. h w) at lower orders (n-grams that start at the sentence start keep raw counts). The unigram level is
interpolated with a uniform distribution over the label-free vocabulary (the paper's treatment of unseen words). This is the STANDARD modified Kneser-Ney recursion; the paper's
skip-pattern interpolation (its actual contribution) is NOT implemented here, so the fidelity is 'reproduces MKN, inspired_by GLM'. One pass gives the sentence log-likelihood at every order 1..N."""
import re
from collections import Counter, defaultdict

import numpy as np

TOKEN = re.compile(r"\w+|[^\w\s]")
BOS, EOS = "<s>", "</s>"


def tokens(text, lower):
    return TOKEN.findall(text.lower() if lower else text)


def discounts(counts):
    """D1, D2, D3+ from the numbers n1..n4 of n-grams occurring exactly 1..4 times (Appendix A, Eq. 10a-10c)."""
    n = Counter(v for v in counts.values() if v <= 4); n1, n2, n3, n4 = (max(n[i], 1) for i in (1, 2, 3, 4)); Y = n1 / (n1 + 2 * n2)
    return (min(max(1 - 2 * Y * n2 / n1, 0.05), 0.95), min(max(2 - 3 * Y * n3 / n2, 0.05), 1.95), min(max(3 - 4 * Y * n4 / n3, 0.05), 2.95))


class WordLM:
    def __init__(self, sents, N, vocab_size):
        self.N, self.V = N, vocab_size + 1; raw = [Counter() for _ in range(N + 1)]
        for s in sents:
            t = [BOS] * (N - 1) + s + [EOS]
            for k in range(1, N + 1):
                for i in range(N - 1 if k > 1 else N - 1, len(t)):   # targets only (never a padding symbol as the predicted word)
                    if i - k + 1 >= 0: raw[k][tuple(t[i - k + 1:i + 1])] += 1
        self.cnt = [None] * (N + 1)
        self.cnt[N] = raw[N]
        for k in range(N - 1, 0, -1):   # continuation counts N1+(. h w): distinct left extensions from the order above
            left = defaultdict(set)
            for g in raw[k + 1]: left[g[1:]].add(g[0])
            c = Counter({g: len(v) for g, v in left.items()})
            for g, v in raw[k].items():
                if g[0] == BOS: c[g] = v   # sentence-initial n-grams cannot be extended on the left: keep raw counts
            self.cnt[k] = c
        self.D, self.ctx = [None] * (N + 1), [None] * (N + 1)
        for k in range(1, N + 1):
            self.D[k] = discounts(self.cnt[k]); D = self.D[k]; tot, n123 = defaultdict(float), defaultdict(lambda: [0, 0, 0])
            for g, v in self.cnt[k].items():
                h = g[:-1]; tot[h] += v; n123[h][min(v, 3) - 1] += 1
            self.ctx[k] = {h: (tot[h], (D[0] * n[0] + D[1] * n[1] + D[2] * n[2]) / tot[h]) for h, n in n123.items()}

    def logprobs(self, s):
        """Log-likelihood of the sentence token list at every order 1..N: array (N,)."""
        t = [BOS] * (self.N - 1) + s + [EOS]; out = np.zeros(self.N)
        for i in range(self.N - 1, len(t)):
            w = t[i]; p = 1.0 / self.V
            for k in range(1, self.N + 1):
                h = tuple(t[i - k + 1:i]); info = self.ctx[k].get(h)
                if info is not None:
                    tot, gamma = info; c = self.cnt[k].get(h + (w,), 0)
                    D = self.D[k]; d = 0.0 if c == 0 else D[min(c, 3) - 1]; p = max(c - d, 0.0) / tot + gamma * p
                out[k - 1] += np.log(p)
        return out
