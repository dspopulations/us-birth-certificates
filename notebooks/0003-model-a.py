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

# %%
import duckdb, joblib, optuna, os, shap
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
from lightgbm import early_stopping, log_evaluation
from scipy.spatial.distance import squareform
from scipy.cluster import hierarchy
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
)
from sklearn.model_selection import train_test_split

from dspopulations_us_birth_certificates import repl_utils
from dspopulations_us_birth_certificates import stats_utils
from dspopulations_us_birth_certificates import data_utils
from dspopulations_us_birth_certificates import ml_utils
from dspopulations_us_birth_certificates import plot_utils
from dspopulations_us_birth_certificates.variables import Variables as vars

pd.options.mode.copy_on_write = True

plt.style.use("../notebook.mplstyle")

os.makedirs("./output", exist_ok=True)

RANDOM_SEED = repl_utils.RANDOM_SEED
np.random.seed(RANDOM_SEED)
rng = np.random.default_rng(RANDOM_SEED)

N_CORES = joblib.cpu_count(only_physical_cores=True)

START_TIME = datetime.now()

OUTPUT_DIR = f"output/0002-predictors/{START_TIME:%Y%m%d-%H%M%S}"

os.makedirs(OUTPUT_DIR, exist_ok=True)

repl_utils.print_environment_info()

print(f"\n--------------------\nOutput directory: {OUTPUT_DIR}\n--------------------\n")

# %%

# %%
numeric = [
    vars.YEAR,
    vars.DBWT,
    vars.PWGT_R,
    vars.WTGAIN,
    vars.BMI,
    vars.MAGE_C,
    vars.FAGECOMB,
]

categorical = [
    vars.DOB_MM,
    vars.DOB_WK,
    "dob_tt_pm",
    vars.BFACIL3,
    vars.SEX,
    vars.PRECARE,
    vars.GESTREC10,
    vars.RF_PDIAB,
    vars.RF_GDIAB,
    vars.RF_PHYPE,
    vars.RF_GHYPE,
    vars.RF_EHYPE,
    vars.RF_PPTERM,
    vars.RF_INFTR,
    vars.RF_FEDRG,
    vars.RF_ARTEC,
    vars.NO_RISKS,
    vars.LD_INDL,
    vars.LD_AUGM,
    vars.ME_PRES,
    vars.DMETH_REC,
    vars.APGAR5,
    vars.APGAR10,
    vars.AB_AVEN1,
    vars.AB_AVEN6,
    vars.AB_NICU,
    vars.AB_SURF,
    vars.AB_ANTI,
    vars.AB_SEIZ,
    vars.NO_ABNORM,
    vars.CA_ANEN,
    vars.CA_MNSB,
    vars.CA_CCHD,
    vars.CA_CDH,
    vars.CA_OMPH,
    vars.CA_GAST,
    vars.CA_LIMB,
    vars.CA_CLEFT,
    vars.CA_CLPAL,
    vars.CA_HYPO,
    vars.CA_DISOR,
    vars.NO_CONGEN,
    vars.MEDUC,
    vars.MRACEHISP,
    vars.FEDUC,
    vars.FRACEHISP,
    vars.PAY_REC,
    vars.WIC,
]

features = categorical + numeric

# %%
df = data_utils.load_predictors_data(from_year=2005, include_unknown=True)
df[categorical] = df[categorical].astype("category")

# %%
X = df[features]
y = df["ca_down_c_p_n"]

X[categorical] = X[categorical].astype("category")

# %%
gbm = lgb.Booster(model_file="output/0002-predictors/20251223-101917/final_model_2.txt")
calibrator = joblib.load("output/0002-predictors/20251223-101917/final_calibrator.joblib")
meta = joblib.load("output/0002-predictors/20251223-101917/final_model_meta.joblib")
best_iter = meta["best_iteration"] or None

def predict_calibrated_proba(gbm, calibrator, X, best_iter=None):
    s = gbm.predict(X, num_iteration=best_iter, raw_score=True)
    return calibrator.predict_proba(s.reshape(-1, 1))[:, 1]

# Ensure column order matches training
X_full = X[meta["features"]]

p_new = predict_calibrated_proba(gbm, calibrator, X_full, best_iter=best_iter)

# %%
df["p_ds_lb_pred_01"] = p_new

# %%
df[["year", "p_ds_lb_pred_01"]].groupby("year").sum().reset_index()

# %%
import duckdb
con = duckdb.connect("../data/us_births.db")

# %%
con.execute(
    """
    ALTER TABLE us_births ADD COLUMN IF NOT EXISTS p_ds_lb_pred_01 DOUBLE;
    """
)
con.execute("DROP TABLE IF EXISTS ds_lb_pred_01")
con.execute(f"CREATE TABLE ds_lb_pred_01 (id BIGINT, p_ds_lb_pred DOUBLE)")

# %%
prob_results = df[["id", "p_ds_lb_pred_01"]]

con.execute(
    """
    INSERT INTO ds_lb_pred_01 (id, p_ds_lb_pred)
    SELECT id, p_ds_lb_pred_01
    FROM prob_results
    """
)

# %%
imported = con.execute(
    """    
    SELECT *
    FROM ds_lb_pred_01
    """
).df()
imported

# %%
con.execute(
    """
    UPDATE us_births b
    SET p_ds_lb_pred_01 = p.p_ds_lb_pred
    FROM ds_lb_pred_01 p
    WHERE b.id = p.id;
    """
)

# %%
con.execute("DROP TABLE IF EXISTS ds_lb_pred_01")

# %%
con = duckdb.connect("../data/us_births.db")

# %%
df = con.execute(
    """
    SELECT
        b.year,
        ca_down_c,
        count(*) as n_births,
        SUM(b.p_ds_lb_pred_01) as predicted
    FROM us_births b
    WHERE b.year >= 2005
    GROUP BY b.year, ca_down_c
    ORDER BY b.year, ca_down_c;
    """
).df()
df

# %%
df = con.execute(
    """
    SELECT
        ca_down_c,
        count(*) as n_births,
        SUM(b.p_ds_lb_pred_01) as predicted,
        SUM(b.p_ds_lb_pred_01) / count(*) as predicted_rate
    FROM us_births b
    WHERE b.year >= 2020
    GROUP BY ca_down_c
    ORDER BY ca_down_c;
    """
).df()
df

# %%
df = con.execute(
    """
    SELECT
        year,
        AVG(CASE WHEN no_congen IS NULL THEN 1 ELSE 0 END) AS no_congen,
        AVG(CASE WHEN ab_nicu IS NULL THEN 1 ELSE 0 END) AS ab_nicu,
        AVG(CASE WHEN dbwt IS NULL THEN 1 ELSE 0 END) AS dbwt
    FROM us_births
    WHERE year >= 2005
    GROUP BY year
    ORDER BY year;
    """
).df()
df

# %%
plt.plot(df["year"], df["no_congen"], label="no_congen missing")
plt.xticks(df["year"], rotation=45)
plt.ylabel("Proportion missing")
plt.title("Proportion of missing no_congen values")

# %%
df = con.execute(
    """
    SELECT
        ca_down_c,
        count(*) as n_births,
        SUM(b.p_ds_lb_pred_01) as predicted,
        SUM(b.p_ds_lb_pred_01) / count(*) as predicted_rate
    FROM us_births b
    WHERE b.year >= 2005 AND b.year < 2010
    GROUP BY ca_down_c
    ORDER BY ca_down_c;
    """
).df()
df

# %%
con.close()
