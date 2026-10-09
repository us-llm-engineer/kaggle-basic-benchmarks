# Provenance — Disaster Tweets

- **Data:** the three competition files (`train.csv`, `test.csv`, `sample_submission.csv`), not redistributed here. 7,613 training and 3,263 test tweets.
- **Folds:** `results/folds.csv`, five stratified folds (`random_state=42`), created in notebook 02 and used by every later model.
- **Hardware:** all steps ran locally in under 90 seconds each. fastText and the CNN trained on an NVIDIA GTX 1650 Ti (4 GB); everything else on CPU (12 threads).
  Neural training: 5 folds × 2 seeds per model (about 45 s for fastText and 30 s for the CNN of GPU time); `results/dl_runs/` holds each run's per-epoch statistics and predictions.
- **Selection protocol:** hyper-parameters chosen by out-of-fold AUC; thresholds chosen on out-of-fold scores; neural epochs chosen on an inner 10% hold-out of each fold's training rows (the outer fold is recorded but never used for selection). Test scores are averages over the fold models.
- **Verification before publishing:** all notebooks executed with 0 error outputs; 19 unit tests pass; the comparison notebook was re-executed from a clean copy containing only the published files plus the competition data and reproduced the published stack submission byte for byte.
- **Kaggle public scores** (`kaggle_public_scores.json`): stack 0.80937, SVM 0.80692, NB-SVM 0.79926, fastText 0.78700, text CNN 0.78118 (micro-F1 = accuracy; no private split is reported for this competition). Complement NB, GBDT and the baseline were not submitted.
