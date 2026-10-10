# #7 Spooky Author Identification — count-based models against small neural nets, no pre-training

Given one sentence from a horror story, which author wrote it: Edgar Allan Poe (EAP), H. P. Lovecraft (HPL) or Mary Shelley (MWS)? 19,579 training and 8,392 test sentences,
scored by **multiclass log loss** (lower is better; the class-frequency baseline is 1.0875). Every model here is trained **from scratch** (no pre-trained language models), scored on the **same five stratified folds**,
and judged by out-of-fold log loss with bootstrap intervals; every probability is calibrated on the other folds only.

- **Best Kaggle result: private 0.23384, public 0.25663** (stack of eight members; out-of-fold estimate 0.2492).
- **Honest outcome:** a target near 0.15 was **not** reached. Gains came from adding *different kinds of evidence* (word-level and character-level likelihood models), not from deeper networks.

## The three best machine-learning models and the three best deep-learning models

Out-of-fold log loss on 19,579 sentences (`results/final_metrics_v2.csv`; 95% bootstrap interval in brackets).

| | Model | Log loss | Accuracy | Notes |
| --- | --- | --- | --- | --- |
| **ML 1** | Word Generalized Language Model (skip n-grams + modified Kneser-Ney), one per author | **0.3155** [0.3064, 0.3252] | 0.8746 | best single model |
| **ML 2** | Word modified Kneser-Ney language model, one per author | 0.3363 [0.3272, 0.3459] | 0.8656 | |
| **ML 3** | NB-weighted n-gram logistic regression | 0.3441 [0.3343, 0.3547] | 0.8719 | |
| **DL 1** | Shallow ANN (100, 50) on TF-IDF character 5-grams | **0.3754** [0.3662, 0.3851] | 0.8550 | lr 1e-4, input dropout 0.5 |
| **DL 2** | fastText-style averaged hashed word/bigram embeddings | 0.4723 [0.4597, 0.4841] | 0.8281 | one hidden layer |
| **DL 3** | Character + word CNN | 0.4822 [0.4720, 0.4924] | 0.8066 | 8 epochs |
| ensemble | Stack of eight members (logistic regression, C = 0.01) | **0.2492** [0.2409, 0.2580] | 0.9021 | Kaggle private 0.23384 |

The best sequence network, a POS-tag CNN+BiLSTM, reaches 0.7302 (`results/final_metrics_v2.csv`). "Deep learning" here means trained neural networks; DL 1 and DL 2 are shallow (one to two hidden layers) and the rest are
convolutional or recurrent. Classical members of the stack that are not in the top three: character 9-gram language model 0.3784, character TF-IDF linear SVM 0.4062, stylometry + latent topics GBDT 0.6276.

| Kaggle submission | Out-of-fold estimate | Public | Private |
| --- | --- | --- | --- |
| Stack of 8 (`submissions/submission_stack8.csv`) | 0.2492 | 0.25663 | **0.23384** |
| 9-member log-linear pool + Dirichlet-style map (no CNN; adds two MLPs) | 0.2496 | 0.25784 | 0.23470 |
| 7-member pool without the CNN + Dirichlet-style map | 0.2499 | 0.25700 | 0.23495 |

(`results/kaggle_scores.csv`; a screenshot of the submissions page is in `assets/kaggle_submissions.png`.) The competition had closed, so these are **late submissions**: Kaggle scores them (status "Complete (after deadline)") but they do not enter the leaderboard. The three differ by about 0.001, which is inside the noise; the public and private subsets sit about 0.007 above and 0.015 below the out-of-fold estimate.

## What was built

| Member | Idea | File |
| --- | --- | --- |
| Word GLM | Per-author word model; the highest order interpolates with every single-skip lower pattern (Pickhardt et al.), modified Kneser-Ney discounts; sentence log-likelihood → softmax with a fitted scale | `spooky_glm.py`, `train_glm.py`, `assemble_glm.py` |
| Word MKN LM | Same without skip patterns; orders 1–4, lowercase | `spooky_wordlm.py`, `train_wordlm.py` |
| NB-weighted LR | One-vs-rest logistic regression on word 1–2-grams + character 1–5-grams scaled by the NB log-count ratio (Wang & Manning) | `train_nblr.py`, `spooky_features.py` |
| Char LM | Per-author character n-gram model, interpolated Witten-Bell, order 9 | `spooky_charlm.py`, `train_charlm.py` |
| Char SVM | Sigmoid-calibrated linear SVM on character TF-IDF n-grams, nested selection of `C` | `train_char_svm.py` |
| Stylometry GBDT | Length, punctuation, case, function-word rates + 100 latent topics, LightGBM | `spooky_style.py`, `train_style.py` |
| Char MLP | TF-IDF character n-grams (one `N`, all kept), ReLU (100, 50), Adam, L2, input dropout; inner loss evaluated 4× per epoch | `train_char_mlp.py` |
| fastText-style | Hashed word unigrams and bigrams, averaged embeddings, linear softmax, SGD with linear decay (Joulin et al.) | `train_fasttext.py` |
| Char+word CNN | Parallel character and word convolutions, max-pooled | `train_cnn.py` |
| POS-tag networks | NLTK POS tags → embedding → CNN (filters 3, 5) / BiLSTM (summed states) / BiLSTM + tag attention / fusion (Jafariakinabad et al., adapted to single sentences) | `train_pos_rnn.py`, `pos_tag_all.py` |
| Char BiGRU, char Transformer | Screens, fold 0 only | `train_gpu_candidates.py` |

**Language-model classifier.** For author $a$ with prior $\pi_a$ and sentence $x$, $p(a\mid x)=\operatorname{softmax}_a\big(\beta\,[\log p_a(x)+\log\pi_a]\big)$, where the scale $\beta$ (a temperature on a generative score) is fitted on the other folds.
The interpolated modified Kneser-Ney probability is $P(w\mid h)=\frac{\max(c(hw)-D(c),0)}{c(h)}+\gamma(h)\,P(w\mid h')$ with $D_1=1-2Y\frac{n_2}{n_1}$, $D_2=2-3Y\frac{n_3}{n_2}$, $D_{3+}=3-4Y\frac{n_4}{n_3}$, $Y=\frac{n_1}{n_1+2n_2}$.
The GLM replaces the single lower-order term by the uniform average over all single-skip patterns. *Deviation from the paper:* $\gamma(h)=\sum_w\min(\text{count},D)/\text{total}$ so every conditional distribution sums to one (checked in `tests/test_glm.py`).

**Calibration, stacking and pooling.** Temperature scaling $\operatorname{softmax}(\log p/T)$ with $T$ fitted on the other folds; log-linear pool $p\propto\prod_m p_m^{w_m}$ with exponents fitted on the other folds; a Dirichlet-style map is a multinomial
logistic regression on log-probabilities ($k^2+k=12$ parameters); the final stacker is a multinomial logistic regression (C = 0.01) on the concatenated log-probabilities, fitted on the other folds' out-of-fold rows.

## Findings

1. **Different evidence beats a deeper network.** Adding the character language model took the three-method pool from 0.3373 to 0.2661, and the word GLM took the seven-member pool (with the CNN) to 0.2512 (`results/improvement_bc.json`, `results/improvement_glm.json`).
   Dropping the CNN from the eight-member stack changes the log loss by 0.0001 (0.2492 with it, 0.2493 without; `results/stack_final.txt`), and adding the two first-round MLPs to the seven-member pool without the CNN changes it by 0.0003 (0.2499 to 0.2496; `results/improvement_glm.json`, `results/improvement_mlp.json`).
2. **The combiner is not the limit.** A logistic-regression stacker (0.2492) matches the pooled map (0.2496-0.2499) within 0.0007, so how the members are combined is not what limits the log loss (`results/stack_final.txt`); the error-overlap figure shows how many members fail on each sentence.
3. **The neural models overfit rather than fail.** The 5-gram MLP reaches its best inner loss near epoch 22 and then its training loss falls toward 0.01 while its confidence keeps rising (`assets/train_01_char_mlp.png`);
   the BiGRU memorises (training loss 0.263 against inner loss 0.971 at epoch 80, best at epoch 27); the CNN stops after 8 epochs with a training loss of 0.20–0.35 against 0.47–0.51 held out.
   Epoch caps were 8 (CNN), 20 (fastText-style), 80 (BiGRU) and 120 with early stopping after 20 stale epochs (MLP, POS networks, 120 × 4 inner evaluations); early stopping always ended the runs first.
4. **Learning rate and dropout matter little.** Over three learning rates × two dropouts, the 5-gram MLP spans 0.3754–0.3822 (`results/assemble_mlp_sweep.txt`); input dropout 0.5 gives the minimum in every pair.
5. **Cleaning the text hurts.** Four cleaning variants rebuilt the text for the whole seven-member pipeline: replacing words seen once by a placeholder (+0.0036, interval +0.0021 to +0.0049), additionally masking capitalised names (+0.0425), masking rare words (+0.0865). All are worse than the original (`results/compare_clean.txt`);
   diacritics occur only in EAP and HPL sentences (shares 0.0165 and 0.0119; MWS 0.0000, `audit/audit_text_noise.txt`), so such "noise" is partly signal. Reading: formatting, names and rare words are author evidence here.
6. **No leakage.** A scikit-learn logistic regression fitted on shuffled labels scores 1.5782, worse than the class-frequency baseline (1.0875), and a plain word-bigram TF-IDF logistic regression scores 0.4159 on the same folds (`audit/audit_leak_control.txt`, `audit/audit_reference.txt`); no training sentence appears in the test file.

## Experiments in numbers

**Word language models.** Lowercase beat cased text; order 4 (MKN) and order 5 (GLM) were best (`results/assemble_new_members.txt`, `results/assemble_glm.txt`).

| Variant | Log loss | Variant | Log loss |
| --- | --- | --- | --- |
| MKN lower, order 4 | 0.3363 | GLM lower N=5, continuation discounts | **0.3155** |
| MKN cased, order 4 | 0.3413 | GLM lower N=4, continuation discounts | 0.3161 |
| MKN lower, order 3 | 0.3375 | GLM lower N=4, raw-count discounts | 0.3184 |
| MKN lower, order 2 | 0.3518 | GLM cased N=4 | 0.3194 |

**Neural models, all folds, out-of-fold, temperature-scaled** (`results/final_metrics_v2.csv`, `results/assemble_char_mlp.txt`, `results/assemble_pos_rnn.txt`).

| Model | Log loss | Accuracy | Epochs run per fold |
| --- | --- | --- | --- |
| 5-gram MLP, lr 1e-4, dropout 0.5 | 0.3754 | 0.8550 | about 40 (best near 22) |
| 5-gram MLP, first round (lr 1e-3, no dropout) | 0.3782 | 0.8527 | 22 (best at 1) |
| 4-gram MLP, first round | 0.3967 | 0.8426 | 22 |
| fastText-style, 30 hidden units | 0.4723 | 0.8281 | 20 |
| Char+word CNN | 0.4822 | 0.8066 | 8 |
| POS-tag CNN+BiLSTM fusion | 0.7302 | 0.6787 | 48–77 |
| POS-tag BiLSTM | 0.7455 | 0.6752 | 55–73 |
| POS-tag CNN | 0.7480 | 0.6696 | 64–113 |
| POS-tag BiLSTM + attention | 0.7585 | 0.6630 | 36–46 |

**Cleaning study** (`results/compare_clean.txt`, same folds, same stacker; the three cleaned datasets are rebuilt from the text by `clean_dataset.py`).

| Dataset | Stack log loss | Paired difference vs original (95% interval) |
| --- | --- | --- |
| original | 0.2493 | — |
| c1: format fixes + words seen once → `UNK` | 0.2529 | +0.0036 (+0.0021, +0.0049) |
| c2: c1 + capitalised names → `NAME` | 0.2919 | +0.0425 (+0.0391, +0.0459) |
| c3: format fixes + names + masking words outside the 5,000 most frequent | 0.3358 | +0.0865 (+0.0813, +0.0916) |

## Visualizations

**Data and cleaning**

<table>
<tr>
<td width="50%"><img src="assets/bc_01_charlm_order_curve.png"><br><sub>Character language model: log loss against n-gram order (`results/meta_charlm.json`).</sub></td>
<td width="50%"><img src="assets/post_10_cleaning_study.png"><br><sub>Cleaning study: every member and the 7-member stack on the original and three cleaned datasets (`results/compare_clean.txt`).</sub></td>
</tr>
</table>

**In-training statistics (collected per epoch, drawn afterwards)**

<table>
<tr>
<td width="50%"><img src="assets/train_01_char_mlp.png"><br><sub>Char 5-gram MLP: losses, confidence vs accuracy, ECE, confident errors, gradient norms, dead ReLUs, drift (`results/char_mlp_runs/N5_lr0.0001_dp0.5_*_history.json`).</sub></td>
<td width="50%"><img src="assets/train_02_char_mlp_sweep.png"><br><sub>MLP sweep over learning rate and input dropout (`results/char_mlp_runs/*_history.json`).</sub></td>
</tr>
<tr>
<td width="50%"><img src="assets/train_03_pos_networks.png"><br><sub>Four POS-tag networks: losses, accuracy, ECE, gradient and update norms (`results/pos_rnn_runs/*_history.json`).</sub></td>
<td width="50%"><img src="assets/train_04_other_neural_screens.png"><br><sub>Char+word CNN, fastText-style, char BiGRU and char Transformer histories (`results/cnn_runs`, `fasttext_runs`, `gpu_candidates`).</sub></td>
</tr>
<tr>
<td width="50%"><img src="assets/cnn_01_training_statistics.png"><br><sub>Char+word CNN training statistics from the first round (`results/cnn_runs/f*.json`).</sub></td>
<td width="50%"><img src="assets/style_02_training_curves.png"><br><sub>Stylometry + topics GBDT: per-iteration inner log loss.</sub></td>
</tr>
<tr>
<td width="50%"><img src="assets/nblr_01_regularisation_sweep.png"><br><sub>NB-weighted LR: log loss against the regularisation constant `C`.</sub></td>
</tr>
</table>

**Per-member post-training diagnostics**

<table>
<tr>
<td width="50%"><img src="assets/nblr_02_post_training_diagnostics.png"><br><sub>NB-weighted LR: confusion, calibration and ROC on out-of-fold rows.</sub></td>
<td width="50%"><img src="assets/nblr_03_log_count_ratios.png"><br><sub>NB-weighted LR: the most author-specific n-grams by log-count ratio.</sub></td>
</tr>
<tr>
<td width="50%"><img src="assets/cnn_02_post_training_diagnostics.png"><br><sub>Char+word CNN: confusion, calibration and ROC on out-of-fold rows.</sub></td>
<td width="50%"><img src="assets/cnn_03_token_saliency.png"><br><sub>Char+word CNN: token saliency on held-out sentences.</sub></td>
</tr>
<tr>
<td width="50%"><img src="assets/style_01_author_style_distributions.png"><br><sub>Stylometric statistics per author (semicolons, commas, word length).</sub></td>
<td width="50%"><img src="assets/style_03_post_training_diagnostics.png"><br><sub>Stylometry + topics GBDT: confusion, calibration and ROC.</sub></td>
</tr>
<tr>
<td width="50%"><img src="assets/style_04_feature_use.png"><br><sub>Stylometry + topics GBDT: split gain by feature, topics versus style.</sub></td>
</tr>
</table>

**Comparison, pooling and stacking**

<table>
<tr>
<td width="50%"><img src="assets/cmp_01_final_metrics.png"><br><sub>First-round out-of-fold metrics of the three original methodologies (`results/final_metrics.csv`).</sub></td>
<td width="50%"><img src="assets/cmp_02_confusion_matrices.png"><br><sub>First-round confusion matrices.</sub></td>
</tr>
<tr>
<td width="50%"><img src="assets/cmp_03_diagnostics.png"><br><sub>First-round diagnostics of the three original methodologies: log loss by author and by sentence length, calibration, correlation of the true-author probability, error overlap.</sub></td>
<td width="50%"><img src="assets/pool_01_final_metrics.png"><br><sub>Log-linear pool of the first three methodologies (`results/pool_summary.json`).</sub></td>
</tr>
<tr>
<td width="50%"><img src="assets/pool_02_error_behaviour.png"><br><sub>Pooling barely changes how confident the wrong sentences are.</sub></td>
<td width="50%"><img src="assets/pool_03_where_gain_lands.png"><br><sub>Where the pooling gain falls (sentences the best single model misclassifies).</sub></td>
</tr>
<tr>
<td width="50%"><img src="assets/bc_02_complementarity.png"><br><sub>A fourth member (the character language model) lowers the pool from 0.3373 to 0.2661; how many sentences all four members get wrong, and where the pool helps or hurts (`results/improvement_bc.json`).</sub></td>
<td width="50%"><img src="assets/bc_03_loss_decomposition.png"><br><sub>Calibration loss versus refinement loss: a post-hoc map removes little (`results/improvement_bc.json`).</sub></td>
</tr>
<tr>
<td width="50%"><img src="assets/post_01_leaderboard.png"><br><sub>Out-of-fold log loss with bootstrap intervals for every member and the stack (`results/final_metrics_v2.csv`).</sub></td>
<td width="50%"><img src="assets/post_02_paired_differences.png"><br><sub>Paired log-loss differences between the best members; asterisk = 95% interval excludes 0.</sub></td>
</tr>
<tr>
<td width="50%"><img src="assets/post_07_stack_weights.png"><br><sub>Stacker coefficients and total weight per member.</sub></td>
</tr>
</table>

**Post-training diagnostics of the final stack and its members**

<table>
<tr>
<td width="50%"><img src="assets/post_03_confusion_matrices.png"><br><sub>Row-normalised confusion matrices of the stack and five members.</sub></td>
<td width="50%"><img src="assets/post_04_reliability.png"><br><sub>Reliability diagrams and ECE.</sub></td>
</tr>
<tr>
<td width="50%"><img src="assets/post_05_error_overlap.png"><br><sub>How many of the eight members get each sentence wrong, and the loss those sentences carry.</sub></td>
<td width="50%"><img src="assets/post_06_loss_by_length.png"><br><sub>Log loss and accuracy by sentence length.</sub></td>
</tr>
<tr>
<td width="50%"><img src="assets/post_08_per_author.png"><br><sub>Per-author ROC curves and precision / recall / F1 of the stack.</sub></td>
<td width="50%"><img src="assets/post_09_confidence_vs_correctness.png"><br><sub>Confidence of correct versus wrong predictions and the concentration of the total loss.</sub></td>
</tr>
<tr>
<td width="50%"><img src="assets/post_11_mlp_sweep.png"><br><sub>Out-of-fold log loss of the MLP over learning rate and dropout (N = 4 and 5).</sub></td>
<td width="50%"><img src="assets/kaggle_submissions.png"><br><sub>Kaggle submissions page: private and public scores of the three submissions (late submissions; values in `results/kaggle_scores.csv`).</sub></td>
</tr>
<tr>
<td width="50%"><img src="assets/post_12_kaggle_scores.png"><br><sub>Out-of-fold estimate versus Kaggle public and private scores (`results/kaggle_scores.csv`).</sub></td>
</tr>
</table>
In-training figures are drawn from per-epoch statistics files written during training (losses, accuracy, confidence, ECE, gradient and update norms, dead-ReLU share, parameter drift, confusion and margins); nothing is plotted inside a training loop.
The shaded bands are ±1 standard deviation over the five folds; curves stop at the shortest fold's last epoch.

## How to reproduce

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('averaged_perceptron_tagger_eng')"
kaggle competitions download -c spooky-author-identification -p data && unzip data/*.zip -d data   # data/train.csv, test.csv, sample_submission.csv
python -m pytest tests/test_glm.py -q                      # GLM: every conditional distribution sums to one (7 tests)
python tests/test_pool.py && python tests/test_charlm.py   # pooling algebra and leakage (13 checks), character LM keys and scoring (8 checks)
python train_wordlm.py 0 4 1 && python train_glm.py 0 5 1 cont       # one fold of the word models (about 1-2 minutes each); loop over folds 0-4
python assemble_glm.py && python assemble_new_members.py             # calibrated members -> results/oof_*.npy
python stack_final.py nblr,style_gbdt,cnn,charlm,wordlm,glm,fasttext,char_svm   # the stack; reads results/oof_*.npy
python make_submissions.py                                           # writes submissions/*.csv
python viz_in_training.py && python viz_post_training.py             # every figure from the shipped result files
```

Every command above stays under 90 seconds per fold on a laptop CPU, except the character n-gram MLP and the POS networks, which were trained on a GPU (`results/PROVENANCE.md`).
`results/oof_*.npy`, `results/test_*.npy`, the pooled test probabilities and the history files are shipped, so the figure scripts, the stack and `make_submissions.py` run without re-training
(checked from a clean copy: the three submissions regenerate with a maximum absolute difference of 0.0; 11 of the 12 post-training figures are pixel-identical, the cleaning-study figure differs by 0.003 grey levels because the shipped cleaned-run scores are stored as float32). `pipeline/run_pipeline.sh` re-runs the seven-member pipeline on a dataset placed in `./data`
(used for the cleaning study); `clean_dataset.py` rebuilds the three cleaned copies from the competition text.

## Limitations and caveats

- The target near 0.15 was not reached; the best private score is 0.23384. All models here are from scratch, which is the main restriction on the neural side.
- The submissions were made after the deadline, so they are scored but not ranked on the leaderboard.
- The three submissions differ by about 0.001, inside the noise; the ranking among them is weak evidence.
- Calibration scales (temperatures, language-model scales, pool exponents, stacker coefficients) are fitted on the other folds only, but the choice of order, learning rate and dropout used all out-of-fold scores of the other folds' search, so the reported log losses of the selected configurations carry a small selection effect.
- The GLM departs from the paper's closed-form interpolation weight (see above) and uses one set of discounts per order; the paper's own gains are for perplexity, not classification.
- Only fold 0 was run for the character BiGRU and Transformer screens; they are not comparable with the five-fold numbers.
- POS tags come from the NLTK averaged-perceptron tagger; its errors are not modelled.

## References

- Kull, M., Perello-Nieto, M., Kängsepp, M., Silva Filho, T., Song, H., Flach, P. (2019). *Beyond temperature scaling: Obtaining well-calibrated multiclass probabilities with Dirichlet calibration.* NeurIPS. arXiv:1910.12656.
- Pickhardt, R., Gottron, T., Körner, M., Staab, S., Wagner, P. G., Speicher, T. (2014). *A Generalized Language Model as the Combination of Skipped n-grams and Modified Kneser-Ney Smoothing.* ACL. arXiv:1404.3377.
- Joulin, A., Grave, E., Bojanowski, P., Mikolov, T. (2016). *Bag of Tricks for Efficient Text Classification.* arXiv:1607.01759.
- Jafariakinabad, F., Tarnpradab, S., Hua, K. A. (2019). *Syntactic Recurrent Neural Network for Authorship Attribution.* arXiv:1902.09723.
- Wang, S., Manning, C. D. (2012). *Baselines and Bigrams: Simple, Good Sentiment and Topic Classification.* ACL.
- Lupșa, D., Avram, S.-M., Lupșa, R. (2025). *Oldies but Goldies: The Potential of Character N-grams for Romanian Texts.* arXiv:2506.15650.
- Fredsgaard, L., Schmidt, M. N. (2025). *On Joint Regularization and Calibration in Deep Ensembles.* arXiv:2511.04160.
- Huang, W., Murakami, A., Grieve, J. (2024). *ALMs: Authorial Language Models for Authorship Attribution.* arXiv:2401.12005.
- Camacho-Collados, J., Pilehvar, M. T. (2018). *On the Role of Text Preprocessing in Neural Network Architectures.* arXiv:1707.01780.
- Altakrori, M. H., Cheung, J. C. K., Fung, B. C. M. (2021). *The Topic Confusion Task: A Novel Evaluation Scenario for Authorship Attribution.* arXiv:2104.08530.
- Du, M., He, F., Zou, N., Tao, D., Hu, X. (2022). *Shortcut Learning of Large Language Models in Natural Language Understanding.* arXiv:2208.11857.
