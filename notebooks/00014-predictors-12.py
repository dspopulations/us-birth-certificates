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
# # Notes 14 - Predictors of recorded DS live births (iteration 12)
#
# Adapted from `00010-predictors-10-c.py`. The per-model analysis is
# refactored into a helper (`run_model_analysis`) so adding or removing
# feature-reduction iterations is a one-line change.
#

# %% [markdown]
# ## Preparation
#

# %%
import os
from datetime import datetime

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from lightgbm import early_stopping, log_evaluation
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from sklearn.inspection import permutation_importance
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import train_test_split

from dspopulations_us_birth_certificates import (
    data_utils,
    ml_utils,
    plot_utils,
    repl_utils,
    stats_utils,
)
from dspopulations_us_birth_certificates.explain import calibration, shap_analysis
from dspopulations_us_birth_certificates.variables import Variables as vars

pd.options.mode.copy_on_write = True

plt.style.use("../notebook.mplstyle")

RANDOM_SEED = 47
np.random.seed(RANDOM_SEED)
rng = np.random.default_rng(RANDOM_SEED)

N_CORES = joblib.cpu_count(only_physical_cores=True)

START_TIME = datetime.now()

OUTPUT_DIR = f"output/0014-predictors-12/{START_TIME:%Y%m%d-%H%M%S}"

os.makedirs(OUTPUT_DIR, exist_ok=True)

repl_utils.print_environment_info()

print(f"\n--------------------\nOutput directory: {OUTPUT_DIR}\n--------------------\n")

# %% [markdown]
# ### Options
#

# %%
START_YEAR = 2016
END_YEAR = 2024
# LightGBM threads
NUM_THREADS = max(1, N_CORES - 2)
# Train/validation split
TRAINING_SPLIT = 0.8
#
NUM_BOOST_ROUND = 50000
EARLY_STOPPING_ROUNDS = 200
# True to search for hyperparameters; False to use LAST_BEST_PARAMS
SELECT_HYPERPARAMETERS = True
OPTIMIZE_TRIALS = 200
#
SAVE_PLOTS = True
# Top-K cutoffs used for Precision@K / Recall@K curves
KS = (100, 500, 1000, 5000, 10000, 20000, 50000)

# %% [markdown]
# ### Load data
#

# %%
df = data_utils.load_predictors_data(
    from_year=START_YEAR, to_year=END_YEAR, include_unknown=True
)

# %% [markdown]
# ### Define initial feature set
#

# %%
numeric = [
    vars.YEAR,
    vars.DBWT,
    vars.WTGAIN,
    vars.BMI,
    vars.MAGE_C,
    vars.FAGECOMB,
]

categorical = [
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
    vars.MEDUC,
    vars.MRACEHISP,
    vars.FEDUC,
    vars.FRACEHISP,
    vars.PAY_REC,
    vars.WIC,
]

features = categorical + numeric

X = df[features]
y = df["ca_down_c_p_n"].replace({pd.NA: 0, np.nan: 0}).astype(np.int32)

X[categorical] = X[categorical].astype("category")

# %% [markdown]
# ### Split training and validation data
#

# %%
X_train_base, X_valid_base, y_train, y_valid = train_test_split(
    X, y, test_size=(1 - TRAINING_SPLIT), stratify=y, random_state=RANDOM_SEED
)

neg_count_train = int((y_train == 0).sum())
pos_count_train = int((y_train == 1).sum())
neg_count_valid = int((y_valid == 0).sum())
pos_count_valid = int((y_valid == 1).sum())

print(
    f"Training set: {neg_count_train} negatives, {pos_count_train} positives, "
    f"probability positive {pos_count_train / neg_count_train:.8f}"
)
print(
    f"Validation set: {neg_count_valid} negatives, {pos_count_valid} positives, "
    f"probability positive {pos_count_valid / neg_count_valid:.8f}"
)

# %%
base_params = {
    "objective": "binary",
    "metric": ["average_precision", "binary_logloss"],
    "boosting_type": "gbdt",
    "max_bin": 255,  # GPU 63/127; CPU 255
    # for now, we do not scale for better interpretability of outputs
    "scale_pos_weight": 1,
    "force_col_wise": True,
    "seed": RANDOM_SEED,
    "num_threads": NUM_THREADS,
    "verbosity": 1,
}

LAST_BEST_PARAMS = {
    "learning_rate": 0.009461164726049449,
    "num_leaves": 180,
    "min_data_in_leaf": 756,
    "min_gain_to_split": 0.9285634625013361,
    "feature_fraction": 0.9239582799934513,
    "bagging_fraction": 0.9185684081749333,
    "bagging_freq": 2,
    "lambda_l1": 0.0005836073944757167,
    "lambda_l2": 0.6142323696066677,
}

# %% [markdown]
# ## Hyperparameter tuning
#

# %%
# Datasets for tuning use the full initial feature set
train_data_full = lgb.Dataset(
    X_train_base,
    label=y_train,
    categorical_feature=categorical,
    free_raw_data=False,
)
valid_data_full = lgb.Dataset(
    X_valid_base,
    label=y_valid,
    categorical_feature=categorical,
    reference=train_data_full,
    free_raw_data=False,
)


# %%
def objective(trial):
    trial_params = {
        # required to change min_data_in_leaf across trials without rebuilding the Dataset
        "feature_pre_filter": False,
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.75, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 32, 512, log=True),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 500, 10000, log=True),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 1.0),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
    }

    params = {**base_params, **trial_params}

    pruning_cb = optuna.integration.LightGBMPruningCallback(trial, "average_precision")

    gbm = lgb.train(
        params,
        train_data_full,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[valid_data_full],
        callbacks=[
            lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS),
            lgb.log_evaluation(period=0),
            pruning_cb,
        ],
    )

    return gbm.best_score["valid_0"]["average_precision"]


if SELECT_HYPERPARAMETERS:
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.HyperbandPruner(),
    )
    study.optimize(objective, n_trials=OPTIMIZE_TRIALS)
    print(study.best_params, study.best_value)
    best = study.best_params
else:
    best = LAST_BEST_PARAMS

params = {**base_params, **best}
params["feature_pre_filter"] = True  # reset to default for final training

print("Parameters for training:")
for k, v in params.items():
    print(f'  "{k}": {v}')


# %% [markdown]
# ## Per-model analysis helper
#
# Helpers for precision/recall at K, tail calibration, and SHAP plots live in
# `dspopulations_us_birth_certificates.explain` (see
# `explain/calibration.py` and `explain/shap_analysis.py`).
#

# %%
def run_model_analysis(
    model_idx: int,
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    y_train: pd.Series,
    y_valid: pd.Series,
    categorical: list[str],
    params: dict,
    *,
    shap_bar_max: int = 35,
    shap_beeswarm_max: int = 35,
    extra_scatter_pairs: list[tuple[str, str]] | None = None,
    show_pr_at_k: bool = True,
):
    """Fit a model, save artefacts, and produce the full set of diagnostics."""
    features = X_train.columns.to_list()
    print(f"\n{'=' * 70}\nModel {model_idx}: {len(features)} features\n{'=' * 70}\n")
    print(f"Features: {features}")

    train_data = lgb.Dataset(
        X_train, label=y_train, categorical_feature=categorical, free_raw_data=False
    )
    valid_data = lgb.Dataset(
        X_valid,
        label=y_valid,
        categorical_feature=categorical,
        reference=train_data,
        free_raw_data=False,
    )

    gbm = lgb.train(
        params,
        train_data,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[train_data, valid_data],
        valid_names=["train", "valid"],
        callbacks=[
            early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS),
            log_evaluation(period=10),
        ],
    )
    best_iter = gbm.best_iteration
    gbm.save_model(f"{OUTPUT_DIR}/model_{model_idx}.txt", num_iteration=best_iter)

    p_valid = gbm.predict(X_valid, num_iteration=best_iter)

    # Core metrics
    (
        metrics_df,
        p_valid_fpr,
        p_valid_tpr,
        _p_valid_thresholds,
        _tp,
        _fp,
        _n_pos,
    ) = ml_utils.get_metrics(y_valid, p_valid, K=10000, thr=0.01)
    metrics_df.to_csv(
        f"{OUTPUT_DIR}/model_{model_idx}_validation_metrics.csv", index=False
    )
    print(metrics_df)

    plot_utils.plot_roc_curve(
        p_valid_fpr,
        p_valid_tpr,
        model_idx,
        save=SAVE_PLOTS,
        output_dir=OUTPUT_DIR,
        file_name=f"model_{model_idx}_roc_curve",
    )
    plot_utils.plot_precision_recall_curve(
        p_valid_fpr,
        p_valid_tpr,
        model_idx,
        save=SAVE_PLOTS,
        output_dir=OUTPUT_DIR,
        file_name=f"model_{model_idx}_precision_recall_curve",
    )

    if show_pr_at_k:
        pr_valid = calibration.precision_recall_at_k(
            y_valid.to_numpy(), p_valid, ks=KS
        )
        pr_valid.to_csv(
            f"{OUTPUT_DIR}/model_{model_idx}_precision_recall_at_k.csv", index=False
        )
        print(pr_valid)
        calibration.plot_precision_recall_at_k_curve(
            pr_valid,
            title_prefix=f"Model {model_idx} validation",
            save=SAVE_PLOTS,
            output_dir=OUTPUT_DIR,
            file_stem=f"model_{model_idx}_precision_recall_at_k",
        )

    # Built-in gain importance
    df_imp_gain = pd.DataFrame(
        {"feature": features, "importance_gain": gbm.feature_importance(importance_type="gain")}
    ).sort_values("importance_gain", ascending=False)
    df_imp_gain.to_csv(
        f"{OUTPUT_DIR}/model_{model_idx}_feature_importance_gain.csv", index=False
    )
    print(df_imp_gain)

    # Explain set for permutation + SHAP
    X_eval, y_eval = ml_utils.build_explain_set(gbm, X_valid, y_valid, categorical)
    model_wrapped = ml_utils.LGBMEstimator(gbm)

    result = permutation_importance(
        model_wrapped,
        X_eval,
        y_eval,
        scoring=ml_utils.ap_scorer,
        n_repeats=20,
        n_jobs=8,
        random_state=RANDOM_SEED,
    )
    perm_importance = pd.DataFrame(
        {
            "feature": X_eval.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    perm_importance.to_csv(
        f"{OUTPUT_DIR}/model_{model_idx}_permutation_importance.csv", index=False
    )
    print(perm_importance)

    plot_utils.plot_permutation_importances(
        result,
        X_eval,
        model_idx,
        save=SAVE_PLOTS,
        output_dir=OUTPUT_DIR,
        file_name=f"model_{model_idx}_permutation_importances",
    )

    # Correlation dendrogram + heatmap
    distance, corr = stats_utils.distance_corr_dissimilarity(X_eval)
    condensed = squareform(distance, checks=True)
    dist_linkage = hierarchy.linkage(condensed, method="average")
    dendro_labels = X_eval.columns.to_list()
    _, dendro = plot_utils.plot_dendrogram(
        dist_linkage,
        dendro_labels,
        model_idx,
        save=SAVE_PLOTS,
        output_dir=OUTPUT_DIR,
        file_name=f"model_{model_idx}_dendrogram",
    )
    plot_utils.plot_correlation_heatmap(
        corr,
        dendro,
        label_threshold=0.3,
        model_idx=model_idx,
        save=SAVE_PLOTS,
        output_dir=OUTPUT_DIR,
        file_name=f"model_{model_idx}_correlation_heatmap",
    )

    # SHAP
    explanation = shap_analysis.compute_explanation(gbm, X_eval)
    shap_importance = shap_analysis.shap_importance(explanation, X_eval.columns)
    shap_importance.to_csv(
        f"{OUTPUT_DIR}/model_{model_idx}_shap_importance.csv", index=False
    )
    print(shap_importance)

    shap_analysis.plot_bar(
        explanation,
        model_idx=model_idx,
        max_display=shap_bar_max,
        save=SAVE_PLOTS,
        output_dir=OUTPUT_DIR,
        file_stem=f"model_{model_idx}_shap_bar",
    )
    shap_analysis.plot_beeswarm(
        explanation,
        model_idx=model_idx,
        max_display=shap_beeswarm_max,
        save=SAVE_PLOTS,
        output_dir=OUTPUT_DIR,
        file_stem=f"model_{model_idx}_shap_beeswarm",
    )

    if extra_scatter_pairs:
        for target, colour in extra_scatter_pairs:
            if target in X_eval.columns and colour in X_eval.columns:
                shap_analysis.plot_scatter(
                    explanation,
                    x_feature=target,
                    colour_feature=colour,
                    model_idx=model_idx,
                    save=SAVE_PLOTS,
                    output_dir=OUTPUT_DIR,
                    file_stem=f"model_{model_idx}_shap_{target}_vs_{colour}",
                )

    return gbm, p_valid


# %% [markdown]
# ## Model 0 (all initial predictors)
#

# %%
X_train = X_train_base.copy()
X_valid = X_valid_base.copy()
categorical_0 = list(categorical)

gbm_0, p_valid_0 = run_model_analysis(
    model_idx=0,
    X_train=X_train,
    X_valid=X_valid,
    y_train=y_train,
    y_valid=y_valid,
    categorical=categorical_0,
    params=params,
    shap_bar_max=45,
    shap_beeswarm_max=45,
    extra_scatter_pairs=[("year", "mage_c")],
)

# %% [markdown]
# ## Model 1 (drop weakest predictors)
#

# %%
features_to_remove_1 = [
    "ca_cdh",
    "apgar10",
    "ca_cleft",
    "rf_artec",
    "ca_omph",
    "ca_clpal",
    "wic",
    "ca_limb",
    "rf_pdiab",
    "ca_hypo",
    "rf_ppterm",
    "ca_mnsb",
    "rf_ehype",
    "ca_anen",
    "ca_gast",
    "ab_surf",
    "rf_gdiab",
    "ab_seiz",
]

X_train = X_train_base.drop(columns=features_to_remove_1)
X_valid = X_valid_base.drop(columns=features_to_remove_1)
categorical_1 = [c for c in categorical if c not in features_to_remove_1]

print(
    f"Model 1: removed {len(features_to_remove_1)} features; "
    f"remaining {X_train.shape[1]}"
)

gbm_1, p_valid_1 = run_model_analysis(
    model_idx=1,
    X_train=X_train,
    X_valid=X_valid,
    y_train=y_train,
    y_valid=y_valid,
    categorical=categorical_1,
    params=params,
    shap_bar_max=35,
    shap_beeswarm_max=40,
    extra_scatter_pairs=[("year", "mage_c")],
)

# %% [markdown]
# ## Model 2 (further trimming — edit `features_to_remove_2` as needed)
#

# %%
features_to_remove_2: list[str] = []

X_train = X_train.drop(columns=features_to_remove_2) if features_to_remove_2 else X_train
X_valid = X_valid.drop(columns=features_to_remove_2) if features_to_remove_2 else X_valid
categorical_2 = [c for c in categorical_1 if c not in features_to_remove_2]

print(
    f"Model 2: removed {len(features_to_remove_2)} features; "
    f"remaining {X_train.shape[1]}"
)

gbm_2, p_valid_2 = run_model_analysis(
    model_idx=2,
    X_train=X_train,
    X_valid=X_valid,
    y_train=y_train,
    y_valid=y_valid,
    categorical=categorical_2,
    params=params,
    shap_bar_max=35,
    shap_beeswarm_max=30,
    extra_scatter_pairs=[
        ("year", "mage_c"),
        ("year", "fagecomb"),
        ("mage_c", "fagecomb"),
        ("bmi", "mage_c"),
        ("ab_nicu", "ab_aven1"),
        ("ca_cchd", "ab_nicu"),
    ],
)

# %% [markdown]
# ## Calibration + prediction writeback (final model)
#

# %%
gbm = gbm_2
p_valid = gbm.predict(X_valid, num_iteration=gbm.best_iteration, raw_score=False)

print("best_iteration:", gbm.best_iteration)
print("mean raw prob:", float(p_valid.mean()))
print("Raw logloss:", log_loss(y_valid, p_valid, labels=[0, 1]))
print("Raw brier:  ", brier_score_loss(y_valid, p_valid))

# %%
fracs = (1e-2, 1e-3, 1e-4, 1e-5)
calibration_table = calibration.tail_calibration_table(y_valid, p_valid, fracs=fracs)
calibration_table["model"] = "raw"
calibration_table.to_csv(
    f"{OUTPUT_DIR}/model_2_tail_calibration_table.csv", index=False
)
calibration_table

# %%
gbm.save_model(
    f"{OUTPUT_DIR}/final_model_2.txt", num_iteration=gbm.best_iteration
)

# %%
final_features = X_valid.columns.to_list()
final_categorical = [c for c in categorical_2 if c in final_features]
X_full = df[final_features].copy()
X_full[final_categorical] = X_full[final_categorical].astype("category")

p_full = gbm.predict(X_full, num_iteration=gbm.best_iteration)

df["p_ds_lb_pred_01"] = p_full

df[["year", "p_ds_lb_pred_01", "ca_down_c_p_n"]].groupby("year").sum().reset_index()

# %% [markdown]
# ### Write predictions back to DuckDB
#
# Commented out by default — uncomment to persist predictions to `us_births.db`.
#

# %%
# import duckdb
#
# con = duckdb.connect("../data/us_births.db")
# con.execute("ALTER TABLE us_births ADD COLUMN IF NOT EXISTS p_ds_lb_pred_01 DOUBLE;")
# con.execute("DROP TABLE IF EXISTS ds_lb_pred_01")
# con.execute("CREATE TABLE ds_lb_pred_01 (id BIGINT, p_ds_lb_pred DOUBLE)")
# con.execute(
#     "INSERT INTO ds_lb_pred_01 (id, p_ds_lb_pred) SELECT id, p_ds_lb_pred_01 FROM df"
# )
# con.execute(
#     """
#     UPDATE us_births b
#     SET p_ds_lb_pred_01 = p.p_ds_lb_pred
#     FROM ds_lb_pred_01 p
#     WHERE b.id = p.id;
#     """
# )
# con.execute("DROP TABLE IF EXISTS ds_lb_pred_01")
# con.close()
