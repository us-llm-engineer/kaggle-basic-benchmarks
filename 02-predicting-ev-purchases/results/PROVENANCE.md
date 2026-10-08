# Provenance — 02_ev_purchases.ipynb

- **Executed on:** Modal CPU sandbox, 16 cores, 8192 MiB memory. Terminated immediately after
  the run; confirmed exited.
- **Parallelization:** the 5 CV folds trained as 5 separate OS processes (`joblib`, `loky`
  backend, LightGBM `num_threads=1` per process) — same approach validated in
  [`01-predicting-airline-satisfaction`](../01-predicting-airline-satisfaction/).
- **Measured training wall time:** 72.8s for all 5 folds (parallel).
- **Final CV ROC AUC:** 0.94164 ± 0.00080 (OOF-recomputed: 0.94163). Per-fold:
  [0.94037, 0.94130, 0.94273, 0.94218, 0.94163].
- **Leaderboard score (late submission, competition closed before this run):** public 0.94144,
  private 0.94093 — both within the CV estimate's band.
- **Verification before accepting this run:** downloaded notebook inspected — 19 code cells, 0
  error outputs, all 9 expected plot cells produced an embedded image, `submission.csv` has
  286,571 rows matching `test.csv`.
