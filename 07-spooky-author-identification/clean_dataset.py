"""Build cleaned copies of the competition text (text-level transformations, so every method sees the same cleaned sentences). Word-frequency tables are label-free and use train + test text together.
  fmt     : Unicode NFKD with combining marks removed, ae/oe ligatures expanded, 'word ,' -> 'word,'  (edition artefacts: diacritics occur only in EAP/HPL; a row with a space before punctuation is 61% HPL)
  ner     : capitalised alphabetic words that do not follow . ! ? or a double quote, and are not 'I', -> NAME   (character and place names = content/topic words)
  unk     : words occurring once in train + test (case-insensitive) -> UNK                                        (hapax noise)
  distort : after ner, every alphabetic word outside the 5000 most frequent words -> '*' x its length            (text distortion, mild: keeps function words, punctuation, word length)
Datasets: c1 = fmt + unk ; c2 = fmt + ner + unk ; c3 = fmt + ner + distort.   python3 clean_dataset.py  ->  data_clean/{c1,c2,c3}/{train,test}.csv (same ids and row order as data/)"""
import os, re, unicodedata
from collections import Counter
import pandas as pd
ROOT = os.path.dirname(os.path.abspath(__file__)); WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
def fmt(t):
    t = t.replace("\u00e6", "ae").replace("\u00c6", "AE").replace("\u0153", "oe").replace("\u0152", "OE")
    t = "".join(c for c in unicodedata.normalize("NFKD", t) if not unicodedata.combining(c))
    return re.sub(r" ([,.;:!?])", lambda m: m.group(1), t)
TITLES = {"Mr", "Mrs", "Ms", "Dr", "St", "Messrs", "Mme", "Mlle", "M", "Sir", "Lady", "Madame"}
def ner(t):
    out, last = [], 0
    for m in WORD.finditer(t):
        head = t[:m.start()].rstrip(); prev = head[-1:]; w = m.group(0); after_title = prev == "." and WORD.findall(head[:-1])[-1:] and WORD.findall(head[:-1])[-1] in TITLES
        sentence_start = (not prev) or (prev in '.!?"' and not after_title)
        out.append(t[last:m.start()]); out.append("NAME" if (not sentence_start and w[0].isupper() and w != "I" and w.isalpha() and w not in TITLES) or (after_title and w[0].isupper() and w.isalpha()) else w); last = m.end()
    out.append(t[last:]); return "".join(out)
def mask(t, keep): return WORD.sub(lambda m: m.group(0) if (m.group(0).lower() in keep or m.group(0) in ("NAME", "UNK")) else "*" * len(m.group(0)), t)
if __name__ == "__main__":
    tr, te = pd.read_csv(os.path.join(ROOT, "data/train.csv")), pd.read_csv(os.path.join(ROOT, "data/test.csv")); F = lambda s: s.map(fmt)
    for name, steps in {"c1": ("unk",), "c2": ("ner", "unk"), "c3": ("ner", "distort")}.items():
        a, b = F(tr.text), F(te.text)
        if "ner" in steps: a, b = a.map(ner), b.map(ner)
        cnt = Counter(w.lower() for t in pd.concat([a, b]) for w in WORD.findall(t) if w != "NAME")
        if "unk" in steps: sub = lambda t: WORD.sub(lambda m: "UNK" if cnt[m.group(0).lower()] == 1 and m.group(0) != "NAME" else m.group(0), t); a, b = a.map(sub), b.map(sub)
        if "distort" in steps: keep = {w for w, _ in cnt.most_common(5000)}; a, b = a.map(lambda t: mask(t, keep)), b.map(lambda t: mask(t, keep))
        d = os.path.join(ROOT, "data_clean", name); os.makedirs(d, exist_ok=True); tr.assign(text=a).to_csv(os.path.join(d, "train.csv"), index=False); te.assign(text=b).to_csv(os.path.join(d, "test.csv"), index=False)
        ch = (a != tr.text).mean(); print(f"{name} {steps}: rows changed {ch*100:.1f}%  mean length {a.str.len().mean():.1f} (orig {tr.text.str.len().mean():.1f})"); print("   e.g.", a.iloc[1][:110])
