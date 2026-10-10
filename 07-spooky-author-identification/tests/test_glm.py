"""spooky_glm.py: every conditional distribution must sum to one over the vocabulary (seen, partly seen and unseen contexts), for orders 2-4 and both discount rules."""
import sys, os, itertools; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pytest
import spooky_glm as G
CORPUS = [s.split() for s in ["the old house stood on the hill", "the old man saw the dark house", "a dark and stormy night on the hill", "the man on the hill saw the old night", "dark house old man the night"] * 3 + ["a man a house a hill"]]
VOCAB = sorted({w for s in CORPUS for w in s} | {G.EOS})
@pytest.mark.parametrize("N", [2, 3, 4])
@pytest.mark.parametrize("disc", ["cont", "raw"])
def test_normalised(N, disc):
    m = G.GLM(CORPUS, N, len(VOCAB), disc)
    for ctx in itertools.islice(itertools.product(VOCAB + ["zzz", G.BOS], repeat=N - 1), 0, 400, 7):
        assert abs(sum(m.prob(w, ctx) for w in VOCAB) - 1) < 1e-9, (N, disc, ctx)
def test_unseen_word_has_floor():
    m = G.GLM(CORPUS, 3, len(VOCAB) + 1, "cont"); assert m.prob("zzz", ("the", "old")) > 0
