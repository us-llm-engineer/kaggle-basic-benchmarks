# Provenance — 04_digit_recognizer.ipynb

- **Executed on:** Modal GPU sandbox, 1x Tesla T4, 15360 MiB VRAM. Terminated immediately after
  the run; confirmed exited.
- **Training wall time:** 14.2s for 12 epochs on the full 37,800-image training split.
- **Final validation accuracy:** 0.98881 (10% held-out split, stratified).
- **Baseline comparison:** PCA(50 components) + multinomial logistic regression on the same
  split scored 0.9033 — the CNN's error rate is ~12% of the baseline's.
- **Leaderboard score:** 0.98860 (public; this is an evergreen "Getting Started" competition
  with no private/public leaderboard split).
- **Verification before accepting this run:** downloaded notebook inspected — 20 code cells, 0
  error outputs, all 11 expected plot cells produced an embedded image, `submission.csv` has
  28,000 rows matching `test.csv`.
