"""Dataset noise audit (label-free except the per-author breakdowns): non-ASCII characters, whitespace anomalies, digits, case, very long rows, quote/dash variants, per-author artefacts."""
import re, collections, sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
import spooky_common as S
train, test, folds, y = S.load(); allt = pd.concat([train.text, test.text])
print("rows", len(train), len(test))
na = collections.Counter(c for t in allt for c in t if ord(c) > 127); print("non-ASCII chars:", dict(na.most_common(25)), "rows containing any:", int(allt.map(lambda t: any(ord(c) > 127 for c in t)).sum()))
for name, pat in [("double space", r"  "), ("leading/trailing space", r"^\s|\s$"), ("space before punct", r" [,.;:!?]"), ("digits", r"\d"), ("straight dq", r'"'), ("curly dq/sq", r"[‘’“”]"), ("apostrophe '", r"'"), ("hyphen -", r"-"), ("em dash", r"—"), ("ellipsis ...", r"\.\.\."), ("ALLCAPS word>1", r"\b[A-Z]{2,}\b"), ("underscore", r"_"), ("parenthesis", r"[()]")]:
    m = allt.str.contains(pat, regex=True); print(f"{name:24s} rows {int(m.sum()):6d} ({m.mean()*100:4.1f}%)")
L = allt.str.len(); print("length: mean %.0f p50 %d p99 %d max %d | rows >600 chars: %d | rows <30 chars: %d" % (L.mean(), L.median(), L.quantile(.99), L.max(), (L > 600).sum(), (L < 30).sum()))
print("\nper-author (train): mean chars, share with digits, share with non-ASCII, share ending with .!?, share starting lowercase")
for a, name in enumerate(S.AUTHORS):
    t = train.text[y == a]; print(f"{name}: {t.str.len().mean():6.1f} {t.str.contains(r'\d').mean():.4f} {t.map(lambda s: any(ord(c)>127 for c in s)).mean():.4f} {t.str[-1].isin(list('.!?')).mean():.3f} {t.str[0].str.islower().mean():.4f}")
tok = collections.Counter(w for t in allt.str.lower() for w in re.findall(r"\w+|[^\w\s]", t)); print("\ntokens", sum(tok.values()), "types", len(tok), "hapax", sum(1 for v in tok.values() if v == 1), "types with count<=2", sum(1 for v in tok.values() if v <= 2))
print("examples with non-ASCII:"); [print("  ", repr(t[:110])) for t in allt[allt.map(lambda t: any(ord(c) > 127 for c in t))].head(4)]
print("long row example:", repr(allt[L.idxmax() if False else L.values.argmax()][:200]))
