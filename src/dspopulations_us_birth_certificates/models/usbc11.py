"""USBC11 model family — clinical + maternal-age feature set only.

A diagnostic variant of USBC10 that removes sociodemographic features
(race/ethnicity, education, payer, paternal age) to probe whether the
flat race/payer distributions seen in the usbc10 predicted-missing
pool (see ``docs/analysis/predicted.qmd``) are a training-set artefact
of selection bias on the ``down_ind`` target rather than a genuine
signal that under-ascertainment has become demographically neutral.

Rationale is documented in ``selection_steps`` below and draws on:

- Boulet et al. (2011) — conditional-on-DS birth-certificate sensitivity
  varies systematically with maternal race, education, and hospital
  characteristics.
- Egan et al. (2004, 2011) — percentage of expected DS live births
  reported varies by maternal age, region, education, race.
- Salemi et al. (2017) — no sensitivity improvement after the 2003 form
  redesign; the bias is structural.

The experiment: train on the same ``down_ind = 1`` target with the same
year×month ``ceil(1.5 × recorded)`` quota, but on a feature set that
excludes variables whose predictive signal routes through the recording
process rather than through DS biology. The resulting predicted-missing
cohort's distribution across the held-out sociodemographic variables
is a less-contaminated estimate of who is being under-ascertained.

``USBC11_M0`` is re-derived from ``USBC10_M0``'s full 48-feature set
minus the 6 sociodemographic features (see PR #26 and
``notes/20260422-1132-retune-afterapgar-fix.md``). Earlier versions of
the class inherited from ``USBC10_M1`` and therefore silently adopted
its pre-apgar-fix prune; that inheritance is severed here so the USBC11
family is independent of USBC10_M1's bugged-data prune decisions.

``USBC11_M1`` drops predictors whose permutation importance was below
the AP-loss < 1e-4 threshold in the post-apgar-fix ``USBC11_M0`` test-
profile fit.
"""

from __future__ import annotations

from datetime import date

from dspopulations_us_birth_certificates.models.common import (
    SelectionStep,
    ShapScatterSpec,
)
from dspopulations_us_birth_certificates.models.usbc10 import USBC10_M0

_USBC11_BASE_PARAMS: dict = {
    "objective": "binary",
    "metric": ["average_precision", "binary_logloss"],
    "boosting_type": "gbdt",
    "max_bin": 255,
    "scale_pos_weight": 1,
    "force_col_wise": True,
}

# Hyperparameters from output/tuning/usbc11_m0_base/best_params.json —
# Optuna test-profile search on the post-PR-#26 USBC11_M0 42-feature set.
# Intended as a starting point; the reporting-profile M1 retune replaces
# these via `_USBC11_M1_PARAMS` on ``USBC11_M1``.
_USBC11_PARAMS: dict = {
    "learning_rate": 0.01720685773741596,
    "num_leaves": 64,
    "min_data_in_leaf": 537,
    "min_gain_to_split": 0.09872595484881053,
    "feature_fraction": 0.7201745741445154,
    "bagging_fraction": 0.8563422711241703,
    "bagging_freq": 4,
    "lambda_l1": 4.6714020036883115e-07,
    "lambda_l2": 1.7978023398871819,
}

# Hyperparameters from output/tuning/usbc11_m1/best_params.json — Optuna
# reporting-profile search (200 trials, 50K boost rounds, 9 yr) on the
# post-PR-#26 USBC11_M1 23-feature set. Re-tune by
# `python scripts/tune_model.py usbc11_m1 --profile reporting`.
_USBC11_M1_PARAMS: dict = {
    "learning_rate": 0.009887010227021193,
    "num_leaves": 82,
    "min_data_in_leaf": 596,
    "min_gain_to_split": 0.28656268185372236,
    "feature_fraction": 0.7086363648514096,
    "bagging_fraction": 0.8650825279896867,
    "bagging_freq": 4,
    "lambda_l1": 0.00027479065008141645,
    "lambda_l2": 1.5712030842124038,
}

_USBC11_TRAIN_CONFIG: dict = {
    "training_split": 0.8,
    "verbosity": 1,
    "log_period": 10,
}


# Sociodemographic features excluded vs. USBC10_M0.
_SOCIODEMO_FEATURES_REMOVED: tuple[str, ...] = (
    "mracehisp",
    "meduc",
    "fracehisp",
    "feduc",
    "pay_rec",
    "fagecomb",
)

_M0_NUMERIC: tuple[str, ...] = tuple(
    f for f in USBC10_M0.numeric_features if f not in _SOCIODEMO_FEATURES_REMOVED
)
_M0_CATEGORICAL: tuple[str, ...] = tuple(
    f for f in USBC10_M0.categorical_features if f not in _SOCIODEMO_FEATURES_REMOVED
)


# Features dropped between USBC11_M0 and USBC11_M1. Re-derived under
# PR #26's apgar5/apgar10 fix on 2026-04-24 against the post-fix
# USBC11_M0 test-profile fit (output/fit_model_test/20260424-200531/).
# 19 categorical features fall below the AP-loss < 1e-4 threshold; no
# numeric features drop. The apgar5 ~ apgar10 and rf_fedrg ~ rf_artec
# correlation pairs are both resolved by the importance threshold alone
# (the weaker side of each pair is already sub-threshold).
_M1_FEATURES_REMOVED: tuple[str, ...] = (
    "rf_pdiab",
    "rf_gdiab",
    "rf_ehype",
    "rf_ppterm",
    "rf_fedrg",
    "rf_artec",
    "ld_augm",
    "apgar10",
    "ab_surf",
    "ab_seiz",
    "ca_anen",
    "ca_mnsb",
    "ca_cdh",
    "ca_omph",
    "ca_gast",
    "ca_limb",
    "ca_cleft",
    "ca_hypo",
    "wic",
)

_M1_NUMERIC: tuple[str, ...] = tuple(
    f for f in _M0_NUMERIC if f not in _M1_FEATURES_REMOVED
)
_M1_CATEGORICAL: tuple[str, ...] = tuple(
    f for f in _M0_CATEGORICAL if f not in _M1_FEATURES_REMOVED
)


class USBC11_M0(USBC10_M0):
    """Clinical + maternal age only — diagnostic variant of USBC10_M0.

    Inherits from ``USBC10_M0`` (not ``USBC10_M1``) so USBC11 sits on
    the pre-pruning baseline and its post-apgar-fix M1 prune is decided
    against the usbc11-specific feature distribution, independent of
    the USBC10 C+P prune.
    """

    model_id = "usbc11_m0"
    variant_of = "usbc10_m0"
    numeric_features = _M0_NUMERIC
    categorical_features = _M0_CATEGORICAL
    base_params = _USBC11_BASE_PARAMS
    params = _USBC11_PARAMS
    train_config = _USBC11_TRAIN_CONFIG
    predictions_column = "p_ds_lb_pred_02"
    missing_flag_column = "ds_pred_missing_02"
    selection_steps = (
        SelectionStep(
            step_date=date(2026, 4, 24),
            rationale=(
                "Remove sociodemographic features (race/ethnicity, education, "
                "payer) and paternal age from USBC10_M0's 48-feature set. "
                "Training on down_ind = 1 conflates P(DS | X) with "
                "P(recorded | DS, X); demographic features are informative "
                "about recording sensitivity (Boulet 2011; Egan 2011), so "
                "including them entrenches the selection bias. Clinical "
                "features are causally downstream of DS and less entangled "
                "with recording, so a clinical+maternal-age model gives a "
                "less-contaminated estimate of P(DS | X). The predicted-"
                "missing cohort's distribution across held-out demographic "
                "variables is then interpretable as a diagnostic of who is "
                "being under-ascertained at national scale. Re-derived on "
                "2026-04-24 from USBC10_M0 directly, severing the earlier "
                "inheritance from USBC10_M1 so USBC11's prune is independent "
                "of USBC10's pre-PR-#26 (bugged-apgar) prune decisions."
            ),
            features_removed=_SOCIODEMO_FEATURES_REMOVED,
        ),
    )
    shap_scatter_specs = (
        ShapScatterSpec(
            x_feature="mage_c",
            colour_by_feature="year",
            description="Maternal age coloured by year.",
        ),
        ShapScatterSpec(
            x_feature="ca_cchd",
            colour_by_feature="ab_nicu",
            description="CCHD coloured by NICU admission.",
        ),
        ShapScatterSpec(
            x_feature="ab_nicu",
            colour_by_feature="ab_aven1",
            description="NICU coloured by assisted ventilation.",
        ),
        ShapScatterSpec(
            x_feature="dbwt",
            colour_by_feature="gestrec10",
            description="Birth weight coloured by gestational age.",
        ),
    )
    notes = (
        "Clinical + maternal age only. Diagnostic companion to usbc10_m1 for "
        "probing whether flat race/payer distributions in the predicted-"
        "missing pool reflect genuine demographic neutrality or selection "
        "bias on the training target."
    )


class USBC11_M1(USBC11_M0):
    """``USBC11_M0`` minus 19 predictors with near-zero permutation importance."""

    model_id = "usbc11_m1"
    variant_of = "usbc11_m0"
    numeric_features = _M1_NUMERIC
    categorical_features = _M1_CATEGORICAL
    params = _USBC11_M1_PARAMS
    selection_steps = (
        SelectionStep(
            step_date=date(2026, 4, 24),
            rationale=(
                "Drop 19 categorical predictors whose permutation AP loss was "
                "below 1e-4 in the post-PR-#26 USBC11_M0 test-profile fit "
                "(output/fit_model_test/20260424-200531/). apgar10 sits in "
                "this list both on its 1.63e-5 importance and on its dcor "
                "0.86 with apgar5 (which stays); rf_artec and rf_fedrg are "
                "both sub-threshold (5.63e-5 and 2.67e-5) and dcor 0.80, so "
                "the pair falls out entirely. No numeric features drop — "
                "mage_c, dbwt, wtgain, bmi, and year all clear the threshold. "
                "See notes/20260422-1132-retune-afterapgar-fix.md for "
                "per-feature importances and the correlation-pair analysis."
            ),
            features_removed=_M1_FEATURES_REMOVED,
        ),
    )
    notes = (
        "Post-apgar-fix USBC11_M1: 23 predictors after the PR #26 re-prune "
        "(5 numeric + 18 categorical)."
    )
