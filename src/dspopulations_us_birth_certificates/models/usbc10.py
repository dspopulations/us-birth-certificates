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
)
from dspopulations_us_birth_certificates.variables import Variables as vars

_USBC10_BASE_PARAMS: dict = {
    "objective": "binary",
    "metric": ["average_precision", "binary_logloss"],
    "boosting_type": "gbdt",
    "max_bin": 255,
    "scale_pos_weight": 1,
    "force_col_wise": True,
}

# Last-known-good hyperparameters from the 00010-predictors-10-c notebook.
# Updates are a deliberate reviewable commit — rerun tune_model.py (step 6)
# to regenerate.
_USBC10_PARAMS: dict = {
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


_USBC10_TRAIN_CONFIG: dict = {
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


# Features dropped between M0 and M1. Lifted from the 00010-predictors-10-c
# notebook (features_to_remove_0 list). These had near-zero permutation
# importance in the M0 run.
_M1_FEATURES_REMOVED: tuple[str, ...] = (
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
)

_M1_CATEGORICAL: tuple[str, ...] = tuple(
    f for f in _M0_CATEGORICAL if f not in _M1_FEATURES_REMOVED
)


class USBC10_M0(ModelDefinition):
    """All initial predictors from 2016–2024 NVSS natality data."""

    model_id = "usbc10_m0"
    variant_of = None
    target_var = "ca_down_c_p_n"
    numeric_features = _M0_NUMERIC
    categorical_features = _M0_CATEGORICAL
    base_params = _USBC10_BASE_PARAMS
    params = _USBC10_PARAMS
    train_config = _USBC10_TRAIN_CONFIG
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
    """``M0`` minus 18 predictors with near-zero permutation importance."""

    model_id = "usbc10_m1"
    variant_of = "usbc10_m0"
    categorical_features = _M1_CATEGORICAL
    selection_steps = (
        SelectionStep(
            step_date=date(2026, 4, 17),
            rationale=(
                "Drop predictors whose permutation importance was near zero "
                "in the M0 run (AP loss < 1e-4). Includes rare-disorder flags "
                "(CA_CDH, CA_ANEN, CA_MNSB, ...) whose low prevalence makes "
                "them more noise than signal here."
            ),
            features_removed=_M1_FEATURES_REMOVED,
        ),
    )
    notes = "After removing 18 low-importance predictors from M0."


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
    shap_scatter_specs = USBC10_M1.shap_scatter_specs + (
        ShapScatterSpec(
            x_feature="year",
            colour_by_feature="fagecomb",
            description="Year vs paternal age.",
        ),
        ShapScatterSpec(
            x_feature="mage_c",
            colour_by_feature="fagecomb",
            description="Maternal vs paternal age interaction.",
        ),
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
