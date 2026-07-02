"""USBC10 model family — the three variants explored in the notebook.

Each variant records its own ``selection_steps``; ``selection_history()``
on the base class walks the MRO to give the full decision chain.

- ``USBC10_M0`` — all initial predictors from 2016–2024 NVSS natality.
- ``USBC10_M1`` — ``M0`` minus 18 features whose permutation importance
  was near zero in the 00010-predictors-10-c notebook run.
- ``USBC10_M2`` — ``M1`` held stable (no further removals yet) for a
  second-look evaluation with more boosting rounds.

These are project-local definitions. The pipeline and registry machinery
live in ``base_model.py`` and ``base_pipeline.py``.
"""

from __future__ import annotations

from datetime import date

from dspopulations_us_birth_certificates.models.base_model import ModelDefinition
from dspopulations_us_birth_certificates.models.common import (
    SelectionStep,
    ShapScatterSpec,
    prune_features,
)
from dspopulations_us_birth_certificates.variables import Variables as vars

# Shared with usbc11.py: both families are binary LightGBM classifiers with
# the same objective-level configuration; only the tuned params differ.
DEFAULT_BASE_PARAMS: dict = {
    "objective": "binary",
    "metric": ["average_precision", "binary_logloss"],
    "boosting_type": "gbdt",
    "max_bin": 255,
    "scale_pos_weight": 1,
    "force_col_wise": True,
}

# Hyperparameters from output/tuning/usbc10_m1/best_params.json — Optuna
# reporting-profile search (200 trials, 50K boost rounds, 9 yr) on the
# post-PR-#26 USBC10_M1 feature set. The earlier values predated the apgar
# fix and were tuned against a feature set that included a near-constant
# apgar5 column. Re-tune by `python scripts/tune_model.py usbc10_m1
# --profile reporting`.
_USBC10_PARAMS: dict = {
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


# Shared with usbc11.py — same training-loop knobs across both families.
DEFAULT_TRAIN_CONFIG: dict = {
    "training_split": 0.8,
    "verbosity": 1,
    "log_period": 10,
}


_M0_NUMERIC: tuple[str, ...] = (
    vars.YEAR,
    vars.DBWT,
    vars.WTGAIN,
    vars.BMI,
    vars.MAGE_C,
    vars.FAGECOMB,
)

_M0_CATEGORICAL: tuple[str, ...] = (
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
)


# Features dropped between M0 and M1. Re-derived under PR #26's apgar5/apgar10
# fix on 2026-04-23: 23 features fall below the AP-loss < 1e-4 threshold and
# `fagecomb` is added on correlation grounds (dcor 0.75 with `mage_c`, which
# absorbs its signal). See notes/20260422-1132-retune-afterapgar-fix.md and
# the per-class SelectionStep below for the full per-feature reasoning.
_M1_FEATURES_REMOVED: tuple[str, ...] = (
    # 23 importance-based drops (permutation AP loss < 1e-4):
    "sex",
    "ca_mnsb",
    "ld_augm",
    "rf_ppterm",
    "wic",
    "ca_anen",
    "ld_indl",
    "rf_ehype",
    "rf_pdiab",
    "ca_gast",
    "ca_limb",
    "ab_surf",
    "ca_cleft",
    "ca_hypo",
    "ca_cdh",
    "rf_gdiab",
    "ab_seiz",
    "fracehisp",
    "ca_omph",
    "rf_artec",
    "rf_fedrg",
    "ca_clpal",
    "apgar10",
    # Correlation-based drop: dcor 0.75 with mage_c, which has 142x its
    # permutation importance — fagecomb is essentially a redundant signal.
    "fagecomb",
)

_M1_CATEGORICAL: tuple[str, ...] = prune_features(_M0_CATEGORICAL, _M1_FEATURES_REMOVED)
_M1_NUMERIC: tuple[str, ...] = prune_features(_M0_NUMERIC, _M1_FEATURES_REMOVED)


class USBC10_M0(ModelDefinition):
    """All initial predictors from 2016–2024 NVSS natality data."""

    model_id = "usbc10_m0"
    variant_of = None
    target_var = "ca_down_c_p_n"
    numeric_features = _M0_NUMERIC
    categorical_features = _M0_CATEGORICAL
    base_params = DEFAULT_BASE_PARAMS
    params = _USBC10_PARAMS
    train_config = DEFAULT_TRAIN_CONFIG
    year_range = (2016, 2024)
    include_unknown = True
    selection_steps = (
        SelectionStep(
            step_date=date(2026, 4, 17),
            rationale=(
                "Initial baseline: all 2016+ NVSS fields considered plausible "
                "predictors of recorded DS live births."
            ),
            features_added=_M0_NUMERIC + _M0_CATEGORICAL,
        ),
    )
    shap_scatter_specs = (
        ShapScatterSpec(
            x_feature="year",
            colour_by_feature="mage_c",
            description="Year vs maternal age — tracks the trend over time.",
        ),
    )
    notes = "All initial predictors; baseline for the USBC10 family."


class USBC10_M1(USBC10_M0):
    """``M0`` minus 24 predictors (post-PR-#26 re-derivation, 2026-04-23)."""

    model_id = "usbc10_m1"
    variant_of = "usbc10_m0"
    numeric_features = _M1_NUMERIC
    categorical_features = _M1_CATEGORICAL
    selection_steps = (
        SelectionStep(
            step_date=date(2026, 4, 23),
            rationale=(
                "Re-derived under PR #26's apgar5/apgar10 fix. Drops 23 "
                "features whose permutation AP loss is < 1e-4 in the post-fix "
                "USBC10_M0 test-profile fit, plus `fagecomb` (numeric) on "
                "correlation grounds: dcor 0.75 with `mage_c`, which has "
                "~140x its permutation importance, so fagecomb's signal is "
                "redundant once mage_c is in. Notable changes vs the original "
                "2026-04-17 prune: `apgar5` is newly retained (its variance "
                "was destroyed by the pre-PR-#26 SQL bug); `apgar10` stays "
                "dropped with stronger evidence (now actively harmful at "
                "permutation importance -1.33e-4); 6 features (`sex`, "
                "`ld_augm`, `ld_indl`, `fracehisp`, `rf_fedrg`, `fagecomb`) "
                "are newly dropped because they no longer act as surrogates "
                "for the apgar5 variance the bugged data lacked. See "
                "notes/20260422-1132-retune-afterapgar-fix.md for the full "
                "rationale and per-feature numbers."
            ),
            features_removed=_M1_FEATURES_REMOVED,
        ),
    )
    notes = "Post-apgar-fix USBC10_M1: 24 predictors after the PR #26 re-prune."


class USBC10_M2(USBC10_M1):
    """Stable feature set from M1 — a second-look evaluation."""

    model_id = "usbc10_m2"
    variant_of = "usbc10_m1"
    # Same features as M1 for now; the selection_steps records that this
    # was a deliberate "hold" rather than a forgotten edit.
    selection_steps = (
        SelectionStep(
            step_date=date(2026, 4, 17),
            rationale=(
                "Re-evaluate M1's feature set under longer boosting (step 4's "
                "reporting preset). No further removals until paired per-fold "
                "comparison (step 6) says otherwise."
            ),
        ),
    )
    # fagecomb-coloured scatters were dropped here when USBC10_M1's prune
    # removed fagecomb itself — the SHAP explanation has no such column to
    # index, and the scatter helpers would raise on access.
    shap_scatter_specs = USBC10_M1.shap_scatter_specs + (
        ShapScatterSpec(
            x_feature="bmi",
            colour_by_feature="mage_c",
            description="BMI coloured by maternal age.",
        ),
        ShapScatterSpec(
            x_feature="ab_nicu",
            colour_by_feature="ab_aven1",
            description="NICU coloured by assisted ventilation.",
        ),
        ShapScatterSpec(
            x_feature="ca_cchd",
            colour_by_feature="ab_nicu",
            description="CCHD coloured by NICU admission.",
        ),
    )
    notes = "Held feature set from M1; longer boosting for a publication-grade run."
