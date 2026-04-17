# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: dspop-us-birth-certificates
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Notes - Features

# %%
import duckdb
import os
import numpy as np
import lightgbm as lgb
import pandas as pd
import matplotlib.pyplot as plt
from dspopulations_us_birth_certificates.variables import Variables as vars

plt.style.use("../notebook.mplstyle")

os.makedirs("./output", exist_ok=True)

RANDOM_SEED = 1673025012
np.random.seed(RANDOM_SEED)

# %%
con = duckdb.connect("../data/us_births.db", read_only=True)

# %%
df = con.execute(
    """
    SELECT
        year,
        CASE
            WHEN COALESCE (ca_down, ca_downs) = 'C' THEN 1
            WHEN COALESCE (ca_down, ca_downs) = 'P' THEN 1
            ELSE 0
        END
        AS ca_down_c_or_p,
        COALESCE(f_ca_down, f_ca_downs) AS f_ca_down_c,
        CASE
            WHEN sex = 'M' THEN 1
            WHEN sex = 'F' THEN 0
            ELSE NULL
        END AS sex,
        CASE
            WHEN dwgt_r >= 100 AND dwgt_r <= 400 THEN dwgt_r
            ELSE NULL
        END AS dwgt_r,
        CASE
            WHEN ca_disor = 'C' THEN 1
            WHEN ca_disor = 'P' THEN 2
            WHEN ca_disor = 'N' THEN 0
            ELSE NULL
        END
        AS ca_disor,
        CASE
            WHEN ca_cchd = 'Y' THEN 1
            WHEN ca_cchd = 'N' THEN 0
            ELSE NULL
        END AS ca_cchd,
        mage_c,
        CASE
            WHEN bmi >= 13.0 AND bmi < 69.9 THEN bmi
            ELSE NULL
        END
        AS bmi,
        CASE
            WHEN  meduc < 9 THEN meduc
            ELSE NULL
        END AS meduc,
        CASE
            WHEN  feduc < 9 THEN feduc
            ELSE NULL
        END AS feduc,
        CASE
            WHEN  mrace6 < 7 THEN mrace6
            ELSE NULL
        END AS mrace6,
        CASE
            WHEN  frace6 < 7 THEN frace6
            ELSE NULL
        END AS frace6,
        CASE
            WHEN  mhisp_r < 6 THEN mhisp_r
            ELSE NULL
        END AS mhisp_r,
        CASE
            WHEN  fhisp_r < 6 THEN fhisp_r
            ELSE NULL
        END AS fhisp_r,
        CASE
            WHEN  pay_rec < 5 THEN pay_rec
            ELSE NULL
        END AS pay_rec
    FROM
        us_births
    WHERE year >= 2018
    ORDER BY
        year, dob_mm
    """
).df()
df

# %%
df["ca_down_c_or_p"] = df["ca_down_c_or_p"].astype("UInt8")
df["sex"] = df["sex"].astype("UInt8")
df["dwgt_r"] = df["dwgt_r"].astype("UInt16")
df["ca_disor"] = df["ca_disor"].astype("UInt8")
df["ca_cchd"] = df["ca_cchd"].astype("UInt8")
df["mage_c"] = df["mage_c"].astype("UInt16")
df["bmi"] = df["bmi"].astype("Float32")
df["meduc"] = df["meduc"].astype("UInt8")
df["feduc"] = df["feduc"].astype("UInt8")
df["mrace6"] = df["mrace6"].astype("UInt8")
df["frace6"] = df["frace6"].astype("UInt8")
df["mhisp_r"] = df["mhisp_r"].astype("UInt8")
df["fhisp_r"] = df["fhisp_r"].astype("UInt8")
df["pay_rec"] = df["pay_rec"].astype("UInt8")
df.dtypes

# %%
df.isna().mean().sort_values(ascending=False)

# %%
df.columns

# %%
features = [
    "year", "f_ca_down_c", "sex", "dwgt_r", "ca_disor", "ca_cchd",
    "mage_c", "bmi", "meduc", "feduc", "mrace6", "frace6", "mhisp_r",
    "fhisp_r", "pay_rec"
]

X = df[features]
y = df["ca_down_c_or_p"]

# %%
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y,
    test_size=0.1,
    random_state=RANDOM_SEED,
    stratify=y
)

# %%
from lightgbm import early_stopping, log_evaluation

categorical = [
    "f_ca_down_c", "sex", "ca_disor", "ca_cchd",
    "meduc", "feduc", "mrace6", "frace6", "mhisp_r",
    "fhisp_r", "pay_rec"
]
numeric = [ "year", "dwgt_r", "mage_c", "bmi" ]

train_data = lgb.Dataset(
    X_train,
    label=y_train,
    categorical_feature=categorical,
    free_raw_data=False
)

valid_data = lgb.Dataset(
    X_valid,
    label=y_valid,
    categorical_feature=categorical,
    reference=train_data,
    free_raw_data=False
)

# Rough imbalance handling
pos_rate = y_train.mean()
scale_pos_weight = (1 - pos_rate) / pos_rate # negative/positive ratio

params = {
    # Core
    "objective": "binary",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,

    # Tree complexity (moderate) + strong regularisation via leaf size
    "num_leaves": 31,
    "max_depth": -1,
    "min_data_in_leaf": 8000,
    "max_bin": 128,

    # Subsampling for variance reduction
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "bagging_freq": 1,

    # Weight regularisation
    "lambda_l1": 0.0,
    "lambda_l2": 5.0,

    # Splitting threshold
    "min_gain_to_split": 0.0,

    # Imbalance handling
    "scale_pos_weight": scale_pos_weight * 0.1,

    # Metrics and infrastructure
    "metric": ["average_precision", "auc"],
    "num_threads": 14,
    "verbose": 1,
}


gbm = lgb.train(
    params,
    train_data,
    num_boost_round=200,
    valid_sets=[train_data, valid_data],
    valid_names=["train", "valid"],
    callbacks=[
        early_stopping(stopping_rounds=20),
        log_evaluation(period=5)
    ]
)


# %%

from datetime import datetime
gbm.save_model(f"./output/features_model_{datetime.now().strftime("%Y%m%d%H%M")}.txt", num_iteration=gbm.best_iteration)

# %%
from sklearn.metrics import roc_auc_score, average_precision_score

y_valid_pred_proba = gbm.predict(X_valid, num_iteration=gbm.best_iteration)

auc = roc_auc_score(y_valid, y_valid_pred_proba)
aupr = average_precision_score(y_valid, y_valid_pred_proba)

print(f"Validation AUC:  {auc:.4f}")
print(f"Validation AUPRC:{aupr:.4f}")


# %%
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

lr_age = LogisticRegression(max_iter=1000)
lr_age.fit(X_train[["mage_c"]], y_train)
p_age = lr_age.predict_proba(X_valid[["mage_c"]])[:, 1]
print("AUC (age only):", roc_auc_score(y_valid, p_age))


# %%
importance_gain = gbm.feature_importance(importance_type="gain")

df_imp_gain = pd.DataFrame({
    "feature": features,
    "importance_gain": importance_gain
}).sort_values("importance_gain", ascending=False)

df_imp_gain


# %%
pos_idx = y_valid[y_valid == 1].index
neg_idx = y_valid[y_valid == 0].index

n_pos_eval = min(10_000, len(pos_idx))
n_neg_eval = min(10 * n_pos_eval, len(neg_idx))  # 10:1 negatives:positives

rng = np.random.default_rng(123)

pos_sample = rng.choice(pos_idx, size=n_pos_eval, replace=False)
neg_sample = rng.choice(neg_idx, size=n_neg_eval, replace=False)

eval_idx = np.concatenate([pos_sample, neg_sample])
eval_idx = rng.permutation(eval_idx)  # shuffle

X_eval = X_valid.loc[eval_idx]
y_eval = y_valid.loc[eval_idx]


# %%
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.inspection import permutation_importance

def ap_scorer(estimator, X, y):
    proba = estimator.predict_proba(X)[:, 1]
    return average_precision_score(y, proba)



class LGBMWrapper:
    def __init__(self, booster):
        self.booster = booster

    def fit(self, X, y=None):
        # Required by sklearn API; we don't actually train here.
        return self

    def predict_proba(self, X):
        # LightGBM Booster.predict gives P(y=1) for binary by default
        p1 = self.booster.predict(X)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X):
        p1 = self.booster.predict(X)
        return (p1 >= 0.5).astype(int)

model_wrapped = LGBMWrapper(gbm)


# %%
result = permutation_importance(
    model_wrapped,
    X_eval,
    y_eval,
    scoring=ap_scorer,
    n_repeats=5,       # small but OK given size
    n_jobs=4,          # adjust for your machine
    random_state=RANDOM_SEED,
)

perm_importance = pd.DataFrame({
    "feature": X_eval.columns,
    "importance_mean": result.importances_mean,
    "importance_std": result.importances_std,
}).sort_values("importance_mean", ascending=False)

perm_importance

# %%
# e.g. take all positives from X_eval and match them 1:2 with negatives
pos_eval_idx = y_eval[y_eval == 1].index
neg_eval_idx = y_eval[y_eval == 0].index

n_pos_shap = min(5_000, len(pos_eval_idx))
n_neg_shap = min(2 * n_pos_shap, len(neg_eval_idx))  # 2:1 neg:pos

pos_shap = rng.choice(pos_eval_idx, size=n_pos_shap, replace=False)
neg_shap = rng.choice(neg_eval_idx, size=n_neg_shap, replace=False)

shap_idx = np.concatenate([pos_shap, neg_shap])
shap_idx = rng.permutation(shap_idx)

X_shap = X_eval.loc[shap_idx]
y_shap = y_eval.loc[shap_idx]


# %%
import shap

# Tell SHAP this is a LightGBM model
explainer = shap.TreeExplainer(gbm)
explanation = explainer(X_shap)
shap_values = explanation.values

# Handle both cases: list or array
if isinstance(shap_values, list):
    shap_pos = shap_values[1]  # SHAP values for positive class
else:
    shap_pos = shap_values     # already positive class

# Global importance: mean |SHAP|
shap_importance = pd.DataFrame({
    "feature": X_shap.columns,
    "mean_abs_shap": np.mean(np.abs(shap_pos), axis=0),
}).sort_values("mean_abs_shap", ascending=False)

shap_importance


# %%
rng = np.random.default_rng(RANDOM_SEED)
shap.summary_plot(shap_values, X_shap, rng=rng)

# %%
X_shap_fp = X_shap.astype("float32")
for name in X_shap_fp.columns:
    print(f"\nFeature: {name}")
    shap.dependence_plot(name, shap_values, X_shap_fp)

# %%
shap.plots.scatter(shap_values[:, "mage_c"])

# %%
shap.plots.scatter(sv[:, "mage_c"], color=sv[:, "ca_cchd"])

# %%
con.close()

# %%
