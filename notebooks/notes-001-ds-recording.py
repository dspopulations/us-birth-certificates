# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: dspop-us-birth-certificates
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Notes - DS births recorded

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

# %% [markdown]
# ## Total records
#
# There are 143 million records, including 68,515 live births with Down syndrome recorded. The status of Down syndrome is unknown in 23.6 million records (16.5%).

# %%
df = con.execute(
    """
    SELECT DISTINCT down_ind,
                    COUNT(*) as count_lb
    FROM us_births
    GROUP BY down_ind
    ORDER BY down_ind
    """
).df()
print(f"Total live births: {df['count_lb'].sum()}")
df

# %% [markdown]
# ### Unknowns
#
# Many unknowns are seen in 2003 and for some years thereafter.
#
# We treat 'Unknown' as 'Not recorded' in our analyses.

# %%
df = con.execute(
    """
    SELECT
        year, SUM (CASE WHEN down_ind IS NULL THEN 1 ELSE 0 END) AS count_NA, SUM (CASE WHEN down_ind = 0 THEN 1 ELSE 0 END) AS count_0, SUM (CASE WHEN down_ind = 1 THEN 1 ELSE 0 END) AS count_1
    FROM us_births
    GROUP BY year
    ORDER BY year;
    """
).df()
df

# %%
df = con.execute(
    """
    SELECT DISTINCT
        year, COALESCE (ca_down, ca_downs) as ds_indication, COUNT (*) as count, AVG (p_ds_lb_nt) as prob_ds_lb_nt, SUM (p_ds_lb_nt) as count_ds_lb_nt, SUM (down_ind) as count_down_ind,
    FROM
        us_births
    WHERE year >= 2004
    GROUP BY
        year, ds_indication
    ORDER BY
        year, ds_indication
    """
).df()
df

# %%
c_df = df[df["ds_indication"] == "C"]
p_df = df[df["ds_indication"] == "P"]
n_df = df[df["ds_indication"] == "N"]
u_df = df[df["ds_indication"] == "U"]
na_df = df[df["ds_indication"].isna()]
plt.figure(figsize=(10, 6))
plt.plot(c_df["year"], c_df["prob_ds_lb_nt"], label="C")
plt.plot(p_df["year"], p_df["prob_ds_lb_nt"], label="P")
plt.plot(n_df["year"], n_df["prob_ds_lb_nt"], label="N")
plt.plot(u_df["year"], u_df["prob_ds_lb_nt"], label="U")
plt.plot(na_df["year"], na_df["prob_ds_lb_nt"], label="NA")
plt.xlim(2003, 2025)
plt.ylim(0, 0.008)
plt.xticks(np.arange(2004, 2025, 2))
plt.xlabel("Year")
plt.ylabel("Probability of live birth with DS (no terminations)")
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

# %% [markdown]
# ## Counts by year

# %%
df = con.execute(
    """
    SELECT DISTINCT
        year, COUNT (*) as count_lb, SUM (
        CASE
        WHEN COALESCE (ca_down, ca_downs) = 'C' THEN 1
        WHEN COALESCE (ca_down, ca_downs) = 'P' THEN 1
        ELSE 0
        END) as count_ds_indication, SUM (down_ind) as count_down_ind, SUM (p_ds_lb_nt) as count_ds_lb_nt, SUM (p_ds_lb_wt_mage) as count_ds_lb_wt_mage, SUM (p_ds_lb_nt_reduc) as count_ds_lb_wt_mage_reduc, SUM (down_ind) / COUNT (*) as prob_ds_rec, SUM (down_ind) / SUM (p_ds_lb_nt) as ratio_nt_recorded, SUM (down_ind) / SUM (p_ds_lb_wt_mage) as ratio_wt_mage_recorded, SUM (down_ind) / SUM (p_ds_lb_nt_reduc) as ratio_wt_mage_reduc_recorded,
    FROM
        us_births
    GROUP BY
        year
    ORDER BY
        year
    """
).df()
df

# %%
plt.figure(figsize=(10, 6))
plt.bar(df["year"], df["count_ds_lb_nt"], label="Estimated live births with DS (no terminations)")
plt.bar(df["year"], df["count_ds_lb_wt_mage_reduc"], label="Estimated live births with DS (after terminations)")
plt.bar(df["year"], df["count_down_ind"], label="Recorded live births with DS (C or P)")
plt.xlabel("Year")
plt.ylabel("Count of live births with DS")
plt.legend()


# %% [markdown]
# ## Proportion of estimated cases recorded

# %%
plt.figure(figsize=(10, 6))
plt.plot(df["year"], df["ratio_nt_recorded"], label="Estimated (absent terminations)")
plt.plot(df["year"], df["ratio_wt_mage_reduc_recorded"], label="Estimated (with terminations)")
plt.xlabel("Year")
plt.ylabel("Ratio recorded / estimated")
plt.title("Ratio of live births with DS recorded to estimated (with/without terminations)", y=1.02)
plt.legend()

# %%
df = con.execute(
    """
    SELECT
        year, mrace15, count (*) as count
    FROM
        us_births
    WHERE year >= 2004
    GROUP BY
        year, mrace15
    ORDER BY
        year, mrace15
    """
).df()
df

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
        as ca_down_c_or_p,
        COALESCE(f_ca_down, f_ca_downs) as f_ca_down_c,
        sex,
        mage_c,
        CASE
             WHEN dwgt_r >= 100 AND dwgt_r <= 400 THEN dwgt_r
             ELSE NULL
        END as dwgt_r,
        CASE
            WHEN bmi >= 13.0 AND bmi < 69.9 THEN bmi
            ELSE NULL
        END
        as bmi,
        CASE
            WHEN  meduc < 9 THEN meduc
            ELSE NULL
        END as meduc,
        CASE
            WHEN  feduc < 9 THEN feduc
            ELSE NULL
        END as feduc,
        CASE
            WHEN  mrace6 < 7 THEN mrace6
            ELSE NULL
        END as mrace6,
        CASE
            WHEN  frace6 < 7 THEN frace6
            ELSE NULL
        END as frace6,
        CASE
            WHEN  mhisp_r < 6 THEN mhisp_r
            ELSE NULL
        END as mhisp_r,
        CASE
            WHEN  fhisp_r < 6 THEN fhisp_r
            ELSE NULL
        END as fhisp_r,
        CASE
            WHEN  pay_rec < 5 THEN pay_rec
            ELSE NULL
        END as pay_rec
    FROM
        us_births
    WHERE year >= 2020
    ORDER BY
        year, dob_mm
    """
).df()
df

# %%
df["ca_down_c_or_p"] = df["ca_down_c_or_p"].astype("UInt8")
df["sex"] = df["sex"].map({"M": 1, "F": 0}).astype("UInt8")
df["mage_c"] = df["mage_c"].astype("UInt16")
df["dwgt_r"] = df["dwgt_r"].astype("UInt16")
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
    "f_ca_down_c", "pay_rec", "sex", "mage_c", "dwgt_r", "bmi", "meduc", "feduc", "mrace6", "frace6", "mhisp_r", "fhisp_r"
]
# , "year"

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

categorical = [ "f_ca_down_c", "pay_rec", "sex", "meduc", "feduc", "mrace6", "frace6", "mhisp_r", "fhisp_r" ]
numeric = ["mage_c", "dwgt_r", "bmi",]

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
scale_pos_weight = (1 - pos_rate) / pos_rate  # negative/positive ratio

params = {
    "objective": "binary",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "metric": ["auc"],
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_data_in_leaf": 5000,
    "max_bin": 255,
    "lambda_l1": 0.0,
    "lambda_l2": 0.0,
    "min_gain_to_split": 0.0,
    "num_threads": 32,
    "scale_pos_weight": scale_pos_weight,
    "verbose": 1,
}

gbm = lgb.train(
    params,
    train_data,
    num_boost_round=2000,
    valid_sets=[train_data, valid_data],
    valid_names=["train", "valid"],
    callbacks=[
        early_stopping(stopping_rounds=100),
        log_evaluation(period=50)
    ]
)


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
bmi_df = con.execute(
    """
    SELECT
        CASE
            WHEN bmi >= 13.0 AND bmi < 69.9 THEN ROUND(bmi,1)
            ELSE NULL
        END
        as bmi,
        COUNT(*) as count_lb,
        SUM(down_ind) as count_ds_rec,
        SUM(down_ind) / COUNT(*) as ratio_ds_rec
    FROM
        us_births
    WHERE year >= 2014
    GROUP BY
        bmi
    ORDER BY
        bmi
    """
).df()
bmi_df

# %%
plt.bar(bmi_df["bmi"], bmi_df["ratio_ds_rec"])

# %%
con.close()

# %%
