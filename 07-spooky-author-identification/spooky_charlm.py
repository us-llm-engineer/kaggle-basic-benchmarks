"""Improvement C: a per-author character n-gram language model, trained from scratch, in the style of the PPM and n-gram baselines.
Interpolated Witten-Bell smoothing:   P_k(c | h) = (count(h c) + T(h) * P_{k-1}(c | h')) / (count(h) + T(h)),
where h is the previous k-1 characters, h' drops the oldest one, count(h) counts the continuations of h and T(h) is the number of distinct continuations.
Order 1 is add-one smoothed. Characters are mapped to integers and every k-gram becomes one exact int64 key (base 128, so k <= 9), which lets the scoring of
every position of every sentence run as a few vectorised sorted-array look-ups. One pass gives the sentence log-likelihood at every order from 1 to N."""
import numpy as np

BASE, PAD, END, UNK = 128, 1, 2, 3


def char_map(texts):
    """Character -> integer id (4 and up), built from label-free text; characters beyond the capacity of one base-128 digit map to UNK."""
    chars = sorted({c for t in texts for c in t}); assert len(chars) < BASE - 4, "too many distinct characters for the base-128 keys"
    return {c: i + 4 for i, c in enumerate(chars)}


def stream(texts, cmap, N):
    """All sentences as one int64 array, each padded with N-1 start symbols and one end symbol. Returns (ids, sentence index per position, is-target mask)."""
    parts, sid, tgt = [], [], []
    for s, t in enumerate(texts):
        ids = np.array([cmap.get(c, UNK) for c in t] + [END], dtype=np.int64); parts.append(np.concatenate([np.full(N - 1, PAD, dtype=np.int64), ids]))
        sid.append(np.full(len(ids) + N - 1, s, dtype=np.int64)); m = np.zeros(len(ids) + N - 1, bool); m[N - 1:] = True; tgt.append(m)
    return np.concatenate(parts), np.concatenate(sid), np.concatenate(tgt)


def kgram_keys(x, k):
    """key_k[i] encodes x[i-k+1..i] with the last symbol as the lowest base-128 digit; valid wherever the k-gram stays inside the padded sentence."""
    key = x.copy()
    for _ in range(k - 1):
        prev = np.concatenate([[0], key[:-1]]); key = x + BASE * prev
    return key


class CharLM:
    """Order-N interpolated Witten-Bell character model of one author's text."""

    def __init__(self, texts, cmap, N):
        self.N, self.cmap = N, cmap; x, _, tgt = stream(texts, cmap, N); self.grams, self.ctx = {}, {}
        for k in range(1, N + 1):
            u, c = np.unique(kgram_keys(x, k)[tgt], return_counts=True); self.grams[k] = (u, c)
            cu, inv = np.unique(u // BASE, return_inverse=True); self.ctx[k] = (cu, np.bincount(inv, weights=c), np.bincount(inv).astype(np.float64))
        uni = np.zeros(BASE); u1, c1 = self.grams[1]; uni[u1] = c1; self.V = len(cmap) + 2   # symbols that can be predicted: the characters, the end symbol and UNK
        self.p1 = (uni + 1.0) / (uni.sum() + self.V)   # add-one smoothing over those symbols

    def loglik(self, texts):
        """Per-sentence log-likelihood at every order 1..N, shape (n_sentences, N)."""
        x, sid, tgt = stream(texts, self.cmap, self.N); n = len(texts); out = np.zeros((n, self.N)); keys = {k: kgram_keys(x, k) for k in range(1, self.N + 1)}
        pos = np.where(tgt)[0]; P = self.p1[x[pos]]; out[:, 0] = np.bincount(sid[pos], weights=np.log(P), minlength=n)
        for k in range(2, self.N + 1):
            cu, tot, typ = self.ctx[k]; ctx = keys[k - 1][pos - 1]; i = np.minimum(np.searchsorted(cu, ctx), len(cu) - 1); hit = cu[i] == ctx
            t, ty = np.where(hit, tot[i], 0.0), np.where(hit, typ[i], 0.0)
            gu, gc = self.grams[k]; g = keys[k][pos]; j = np.minimum(np.searchsorted(gu, g), len(gu) - 1); cnt = np.where(gu[j] == g, gc[j], 0.0)
            P = np.where(t > 0, (cnt + ty * P) / (t + ty + (t == 0)), P); out[:, k - 1] = np.bincount(sid[pos], weights=np.log(P), minlength=n)
        return out
