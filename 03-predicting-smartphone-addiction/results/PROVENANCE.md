# Provenance — 03_smartphone_addiction.ipynb

- **Executed on:** Modal CPU sandbox, 16 cores, 8192 MiB memory. Terminated immediately after
  the run; confirmed exited.
- **Parallelization:** the 5 CV folds trained as 5 separate OS processes (`joblib`, `loky`
  backend, LightGBM `num_threads=1` per process).
- **Measured training wall time:** 354.3s for all 5 folds (parallel) — longer than the first two
  competitions in this series; several folds needed close to the full 2000-round budget before
  early stopping triggered, consistent with a noisier, more heavily-missing dataset.
- **Final CV ROC AUC:** 0.96364 ± 0.00056. Per-fold: [0.96276, 0.96361, 0.96388, 0.96447, 0.96349].
- **Leaderboard score (late submission, competition closed before this run):** 0.96506, public
  0.96524 — both above the CV estimate, consistent with CV being a slightly conservative
  (not overfit) estimate here.
- **Verification before accepting this run:** downloaded notebook inspected — 20 code cells, 0
  error outputs, all 10 expected plot cells produced an embedded image, `submission.csv` has
  296,302 rows matching `test.csv`.
