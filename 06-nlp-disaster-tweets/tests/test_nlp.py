"""Unit checks for the helpers the Disaster Tweets models depend on."""
import os, sys
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nlp_common as N, nlp_dl as D

ok = []
def check(name, cond): ok.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name)

# normaliser keeps URL / mention markers and strips punctuation
t = N.norm(pd.Series(["Forest FIRE near La Ronge!! http://t.co/abc @user1 &amp; #help"]))[0].split()
check("norm keeps xurl and xuser tokens", "xurl" in t and "xuser" in t)
check("norm lower-cases, keeps hashtags, strips '!'", "fire" in t and "#help" in t and not any("!" in w for w in t))
check("model_text prepends the keyword token", N.model_text(pd.DataFrame({"keyword": ["wild%20fires"], "text": ["x"]}))[0].startswith("kw_wild_fires "))

# NB log-count ratio on a toy corpus, against the definition
X = np.array([[2, 0, 1], [1, 0, 1], [0, 3, 1]], float); y = np.array([1, 1, 0])
p, q = 1 + X[:2].sum(0), 1 + X[2:].sum(0); r_ref = np.log((p / p.sum()) / (q / q.sum()))
check("NB log-count ratio equals the definition", np.allclose(N.nb_log_count_ratio(X, y, 1.0), r_ref))
check("NB ratio sign: word 0 -> positive class, word 1 -> negative class", r_ref[0] > 0 > r_ref[1])

# threshold search and paired bootstrap
rng = np.random.default_rng(0); yy = rng.integers(0, 2, 400); s = yy + rng.normal(0, .7, 400)
th, f = N.best_threshold(yy, s); check("best_threshold beats the 0.5-style midpoint F1", f >= N.f1_score(yy, s > 0.5) - 1e-12 and 0 < th < 1.2)
d, ci, pv = N.paired_bootstrap_f1(yy, (s > th).astype(int), (s > th).astype(int), n=200)
check("bootstrap of a model against itself is exactly zero", d == 0 and ci == (0.0, 0.0))
d2, ci2, _ = N.paired_bootstrap_f1(yy, (s > th).astype(int), 1 - (s > th).astype(int), n=300)
check("bootstrap detects a much better model (CI above 0)", d2 > 0.3 and ci2[0] > 0)

# hashing is deterministic across processes and in range
check("bigram hash deterministic and bounded", D.bigram_bucket("a", "b", 1000) == D.bigram_bucket("a", "b", 1000) and 0 <= D.bigram_bucket("a", "b", 1000) < 1000)
import subprocess
h = lambda: subprocess.run([sys.executable, "-c", "import nlp_dl as D; print(D.bigram_bucket('forest','fire',100000))"], capture_output=True, text=True, cwd=N.ROOT).stdout.strip()
check("hash identical in two separate interpreter processes", h() == h() != "")

# vocabulary / encoding
V = D.Vocab(["fire in town", "fire fire in town", "rare once"], min_count=2)
check("vocab keeps words with count >= 2 only", "fire" in V.stoi and "town" in V.stoi and "rare" not in V.stoi)
e = V.encode("fire rare"); check("encode pads to MAXLEN and maps unseen to <unk>", len(e) == D.MAXLEN and e[0] == V.stoi["fire"] and e[1] == D.UNK and e[2] == D.PAD)
ng = D.encode_ngrams(["fire in town"], V, 1000); check("ngram ids: 3 unigrams + 2 bigrams, bigrams offset by vocab size", (ng > 0).sum() == 5 and (ng[0, 3:5] >= len(V)).all())

# fastText: mean ignores padding; CNN output shape
m = D.FastText(50, dim=4, dropout=0.0).eval(); x = torch.tensor([[3, 5, 0, 0]]); xx = torch.tensor([[3, 5, 0, 0, 0, 0]])
manual = (m.emb.weight[3] + m.emb.weight[5]) / 2
check("fastText mean excludes padding and is length-invariant", torch.allclose(m(x), m.out(manual.unsqueeze(0)).squeeze(-1)) and torch.allclose(m(x), m(xx)))
check("fastText padding row stays zero", float(m.emb.weight[0].detach().abs().sum()) == 0.0)
cnn = D.TextCNN(30).eval(); check("TextCNN returns one logit per tweet", cnn(torch.randint(1, 30, (7, D.MAXLEN))).shape == (7,))

# training loop restores the best epoch and records statistics, on a learnable toy problem
torch.manual_seed(0); Xt = torch.randint(2, 120, (600, D.MAXLEN)); yt = (Xt == 7).any(1).long()
net = D.TextCNN(120, channels=16); cfg = dict(lr=5e-3, wd=0.0, batch=64, epochs=25, patience=6, seed=0)
net, hist = D.fit(net, Xt[:400], yt[:400], Xt[400:500], yt[400:500], Xt[500:], yt[500:], cfg)
check("training records per-epoch statistics", len(hist) >= 1 and {"train_loss", "grad_norm", "inner_loss", "inner_auc", "mon_f1", "lr"} <= set(hist[0]))
check("CNN learns a token-presence task (best inner AUC > 0.9)", max(h["inner_auc"] for h in hist) > 0.9)
bi = min(range(len(hist)), key=lambda i: hist[i]["inner_loss"]); zin = D.predict_logits(net, Xt[400:500]); pin = 1 / (1 + np.exp(-zin)); yi = yt[400:500].numpy()
ll = float(-np.mean(yi * np.log(pin + 1e-9) + (1 - yi) * np.log(1 - pin + 1e-9)))
check("model is restored to the best inner-loss epoch", abs(ll - hist[bi]["inner_loss"]) < 1e-4)
print(f"{sum(ok)}/{len(ok)} passed"); sys.exit(0 if all(ok) else 1)
