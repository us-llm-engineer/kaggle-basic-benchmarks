"""Label-free POS tagging (NLTK averaged perceptron, Penn Treebank tags) of every train and test sentence -> results/pos_tags.json (list of tag lists, train rows then test rows)."""
import json, os, re, sys, time
import nltk
import spooky_common as S
TOKEN = re.compile(r"\w+|[^\w\s]")
train, test, folds, y = S.load(); texts = list(train.text) + list(test.text); t0 = time.time(); out = []
for i, t in enumerate(texts):
    out.append([tag for _, tag in nltk.pos_tag(TOKEN.findall(t))])
    if i % 4000 == 0: print(i, len(texts), f"{time.time()-t0:.0f}s", flush=True)
json.dump(out, open(os.path.join(S.RESULTS, "pos_tags.json"), "w")); print("done", len(out), f"{time.time()-t0:.0f}s")
