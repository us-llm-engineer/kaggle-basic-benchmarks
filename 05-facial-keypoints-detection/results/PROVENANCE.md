# Provenance — Facial Keypoints Detection

- **v1 – v2b:** single NVIDIA T4. **v3 and all v4 runs:** single NVIDIA L4 (24 GB), mixed precision, data and
  augmentation resident on the GPU. v4a, v4b and the seed control trained concurrently on the same L4 (≈46 min wall).
- **Split:** every model after v1 holds out the same 12% of training rows (`train_test_split(test_size=0.12,
  random_state=42)`); v3 is the only exception (trained on 100%, so it has no honest validation number).
- **Stopping:** early stop after 30 epochs without held-out improvement, 240-epoch cap. Epochs used: v4a 230,
  v4b 224, seed control 162. Best weights checkpointed on every improvement.
- **Per-epoch statistics** (no figures during training): `results/<run>_history.json` — learning rate, β, gradient
  norm, train L1 / heatmap loss, held-out RMSE (+ flip test every 10 epochs), centre-pull slope, softmax entropy and
  peak probability, per-coordinate RMSE. Evaluation summaries: `results/<run>_eval.json`;
  all post-training numbers: `results/posttraining_summary.json`.
- **Kaggle scores (private / public):** v1 2.75978 / 2.95000 · v2b 2.46539 / 2.65896 · v3 2.62105 / 2.86097 ·
  v4a 2.24540 / 2.37053 · **v4b 1.61419 / 1.94381**. `submission.csv` is v4b's (27,124 rows).
- **Verification before accepting each run:** executed notebooks have 0 error outputs; 22 unit tests and 6 mutation
  tests pass; the post-training notebook reloads v2 and reproduces its 2.4385px before comparing anything against it.
