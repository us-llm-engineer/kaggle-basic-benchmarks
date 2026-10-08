import time
import pandas as pd
import lightgbm as lgb

train = pd.read_csv("data/train.csv")
cat_cols = ["Gender", "Customer Type", "Type of Travel", "Class"]
y = train["satisfaction"].astype(int)
X = train.drop(columns=["id", "satisfaction"]).copy()
for c in cat_cols:
    X[c] = X[c].astype("category")

params_base = dict(
    objective="binary",
    metric="auc",
    learning_rate=0.05,
    num_leaves=63,
    min_data_in_leaf=50,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=1.0,
    verbosity=-1,
    seed=42,
)

dtrain = lgb.Dataset(X, label=y, categorical_feature=cat_cols, free_raw_data=False)

for nt in [1, 2, 4, 6, 8, 12]:
    params = dict(params_base, num_threads=nt)
    t0 = time.time()
    lgb.train(params, dtrain, num_boost_round=300)
    dt = time.time() - t0
    print(f"threads={nt:2d}  fixed_300_rounds_time={dt:6.2f}s", flush=True)
