"""Cleaned-dataset experiment on the best single methodology (word-level Generalized Language Model, spooky_glm.py, lowercase order 5, continuation-count discounts: the configuration chosen on the uncleaned data).
Variants (each a deterministic text transformation; vocabulary-dependent ones use the FOLD'S TRAINING rows only):
  fmt     : Unicode NFKD with combining marks removed, ae/oe ligatures expanded, 'word ,' -> 'word,' (removes edition artefacts: diacritics occur only in EAP/HPL, space-before-punctuation is 61% HPL)
  unk     : fmt + words seen at most once in the fold's training rows -> <unk>   (rare-word noise)
  distort : fmt + every alphabetic word outside the 2000 most frequent training words replaced by '*' x its length  (text distortion, keeps function words, punctuation and word length)
  ner     : fmt + capitalised alphabetic words that are not sentence-initial and not 'I' -> <cap>  (character-name / place-name content masking)
    python3 train_glm_clean.py <fold> <variant>  ->  results/glm_clean_runs/{variant}_f{fold}.npz  (ll_val, ll_test: sentences x authors)"""
import os, re, sys, time, unicodedata
from collections import Counter
import numpy as np
import spooky_common as S, spooky_glm as G
RUNS = os.path.join(S.RESULTS, "glm_clean_runs"); TOKEN = re.compile(r"\w+|[^\w\s]"); N, DISC, TOPK = 5, "cont", 2000

def fmt(t):
    t = t.replace("æ", "ae").replace("Æ", "AE").replace("œ", "oe").replace("Œ", "OE")
    t = "".join(c for c in unicodedata.normalize("NFKD", t) if not unicodedata.combining(c)); return re.sub(r" ([,.;:!?])", r"\1", t)

def base_tokens(text, variant):
    toks = TOKEN.findall(fmt(text))
    if variant == "ner": return ["<cap>" if (i > 0 and toks[i - 1] not in (".", "!", "?", '"') and w[0].isupper() and w != "I" and w.isalpha()) else w.lower() for i, w in enumerate(toks)]
    return [w.lower() for w in toks]

def run(k, variant):
    t0 = time.time(); train, test, folds, y = S.load(); txt = train.text.values; tr, va = np.where(folds != k)[0], np.where(folds == k)[0]
    tok = [base_tokens(t, variant) for t in txt]; ttok = [base_tokens(t, variant) for t in test.text.values]; cnt = Counter(w for i in tr for w in tok[i]); top = {w for w, _ in cnt.most_common(TOPK)}
    if variant == "unk": m = lambda ws: [w if cnt[w] > 1 else "<unk>" for w in ws]
    elif variant == "distort": m = lambda ws: [w if (not w.isalpha() or w in top) else "*" * len(w) for w in ws]
    else: m = lambda ws: ws
    tok = [m(ws) for ws in tok]; ttok = [m(ws) for ws in ttok]; V = len({w for ws in tok + ttok for w in ws}) + 2
    lms = [G.GLM([tok[i] for i in tr[y[tr] == a]], N, V, DISC) for a in range(3)]; t1 = time.time()
    score = lambda seqs: np.array([[lm.loglik(s) for lm in lms] for s in seqs]); llv = score([tok[i] for i in va]); llt = score(ttok)
    os.makedirs(RUNS, exist_ok=True); np.savez(os.path.join(RUNS, f"{variant}_f{k}.npz"), va=va, ll_val=llv, ll_test=llt, vocab=V)
    print(f"fold {k} {variant}: vocab {V}, tables {t1-t0:.0f}s, total {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    run(int(sys.argv[1]), sys.argv[2])
