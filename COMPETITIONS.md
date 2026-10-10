# Competition roadmap

A 21-competition bundle spanning multiple ML/DL skill areas — tabular classification/regression,
computer vision, NLP, LLM fine-tuning — rather than repeating one pipeline on near-identical
tabular datasets. RL/agent/simulation competitions (card-game battle agents, maze navigation) are
intentionally excluded; they don't fit the EDA → model → diagnostics → submission shape the rest
of this repo follows.

| # | Competition | Link | Shape | Status |
|---|---|---|---|---|
| 1 | Predicting Airline Satisfaction | https://www.kaggle.com/competitions/playground-series-s6e10 | Tabular classification (ROC AUC) | Done — [`01-predicting-airline-satisfaction/`](01-predicting-airline-satisfaction/), LB 0.95827 |
| 2 | Predicting Electric Vehicle Purchases | https://www.kaggle.com/competitions/playground-series-s6e9 | Tabular classification (ROC AUC) | Done — [`02-predicting-ev-purchases/`](02-predicting-ev-purchases/), LB 0.94093 |
| 3 | Predicting Smartphone Addiction | https://www.kaggle.com/competitions/playground-series-s6e8 | Tabular classification (ROC AUC) | Done — [`03-predicting-smartphone-addiction/`](03-predicting-smartphone-addiction/), LB 0.96506 |
| 4 | Digit Recognizer | https://www.kaggle.com/competitions/digit-recognizer | Computer vision (CNN, MNIST) | Done — [`04-digit-recognizer/`](04-digit-recognizer/), LB 0.98860 |
| 5 | Facial Keypoints Detection | https://www.kaggle.com/competitions/facial-keypoints-detection | Computer vision (image regression — keypoints) | Done — [`05-facial-keypoints-detection/`](05-facial-keypoints-detection/), RMSE 1.61419 private / 1.94381 public |
| 6 | Natural Language Processing with Disaster Tweets | https://www.kaggle.com/competitions/nlp-getting-started | NLP text classification | Done — [`06-nlp-disaster-tweets/`](06-nlp-disaster-tweets/), public score 0.80937 |
| 7 | Spooky Author Identification | https://www.kaggle.com/competitions/spooky-author-identification | NLP authorship classification (multiclass log loss) | Done — [`07-spooky-author-identification/`](07-spooky-author-identification/), private 0.23384 / public 0.25663 |
| 8 | Predicting Student Health Risk | https://www.kaggle.com/competitions/playground-series-s6e7 | Tabular classification (Balanced Accuracy) | Not started |
| 9 | Predicting Stellar Class | https://www.kaggle.com/competitions/playground-series-s6e6 | Tabular multi-class (Balanced Accuracy) | Not started |
| 10 | Predicting F1 Pit Stops | https://www.kaggle.com/competitions/playground-series-s6e5 | Tabular classification (ROC AUC) | Not started |
| 11 | Predicting Irrigation Need | https://www.kaggle.com/competitions/playground-series-s6e4 | Tabular classification (Balanced Accuracy) | Not started |
| 12 | Predict Customer Churn | https://www.kaggle.com/competitions/playground-series-s6e3 | Tabular classification (ROC AUC) | Not started |
| 13 | Predicting Heart Disease | https://www.kaggle.com/competitions/playground-series-s6e2 | Tabular classification (ROC AUC) | Not started |
| 14 | Predicting Student Test Scores | https://www.kaggle.com/competitions/playground-series-s6e1 | Tabular regression (RMSE) | Not started |
| 15 | Diabetes Prediction Challenge | https://www.kaggle.com/competitions/playground-series-s5e12 | Tabular classification (ROC AUC) | Not started |
| 16 | Predicting Loan Payback | https://www.kaggle.com/competitions/playground-series-s5e11 | Tabular classification (ROC AUC) | Not started |
| 17 | Predicting Road Accident Risk | https://www.kaggle.com/competitions/playground-series-s5e10 | Tabular regression (MSE) | Not started |
| 18 | Predicting the Beats-per-Minute of Songs | https://www.kaggle.com/competitions/playground-series-s5e9 | Tabular regression (MSE) | Not started |
| 19 | Binary Classification with a Bank Dataset | https://www.kaggle.com/competitions/playground-series-s5e8 | Tabular classification (ROC AUC) | Not started |
| 20 | House Prices - Advanced Regression Techniques | https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques | Tabular regression (RMSE-log), classic | Not started |
| 21 | Titanic - Machine Learning from Disaster | https://www.kaggle.com/competitions/titanic | Tabular classification (accuracy), classic | Not started |

LLM Classification Finetuning is deferred past this list and will slot in once GPU-based language
model work is back in scope.

Folder-per-competition convention: `NN-<short-slug>/`, numbered by the `#` column above. Each
entry gets its own notebook, diagnostics gallery, and scored submission.
