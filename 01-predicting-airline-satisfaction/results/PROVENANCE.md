# Provenance — 01_airline_satisfaction.ipynb

- **Executed on:** Modal CPU sandbox (handle `5f89b4ea`), app `kaggle-s6e10`, 8 cores (scaling_factor=64), 4096 MiB memory, no GPU.
- **Not offloaded:** nothing — the whole notebook ran in one kernel on the sandbox, since the heavy modeling cell depends on in-kernel state (features, splits) built by the earlier EDA/feature cells; those cells cost single-digit seconds regardless of host.
- **Parallelization:** the 5 CV folds are independent units, trained as 5 separate OS processes (`joblib`, `loky` backend, LightGBM `num_threads=1` per process) rather than relying on LightGBM's own in-process threading.
- **Why process-parallel folds, not a bigger box:** a local benchmark (this repo, `bench_threads.py`) on the same dataset showed LightGBM's own thread scaling saturates at 6 threads (12.0s/300 rounds) and *regresses* at 8/12 threads (18.6s/36.1s) from thread-sync overhead + contention with other load on that shared machine. Modal's CPU sandbox max is 16 cores — not enough headroom over local to help via that axis — so the fix was splitting work across folds (true multi-process parallelism) instead.
- **Measured training wall time:** 124.3s for all 5 folds (parallel).
- **Final CV ROC AUC:** 0.95880 ± 0.00058 (OOF-recomputed: 0.95879). Per-fold: [0.95899, 0.95771, 0.95942, 0.95881, 0.95907].
- **Sandbox packages (pip-installed fresh on the sandbox, latest at install time):** pandas, numpy, scikit-learn, lightgbm, matplotlib, seaborn, shap, joblib, jupyter/nbconvert/ipykernel/nbformat (for execution only).
- **Verification before accepting this run:** downloaded notebook inspected — 20 code cells, 0 error outputs, 10/10 expected plot cells produced an embedded image, `submission.csv` has 299,844 rows matching `test.csv`.
- **Sandbox terminated** immediately after the run and verified exited (exit code 137) — not left to idle-timeout.
- **Not yet done (pending user review of the metric above):** Kaggle submission, git commit/push.
