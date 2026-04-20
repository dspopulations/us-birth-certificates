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

Features kept (from ``USBC10_M1``):

- Maternal age (``mage_c``) — strongest biological predictor of DS.
- Numeric pregnancy/birth clinical: ``dbwt``, ``wtgain``, ``bmi``.
- Temporal: ``year``.
- Delivery / neonatal clinical: ``sex``, ``precare``, ``gestrec10``,
  ``bfacil3``, ``dmeth_rec``, ``me_pres``, ``ld_indl``, ``ld_augm``,
  ``apgar5``.
- Pregnancy risk flags: ``rf_phype``, ``rf_ghype``, ``rf_inftr``,
  ``rf_fedrg``.
- Post-delivery baby clinical: ``ab_aven1``, ``ab_aven6``, ``ab_nicu``,
  ``ab_anti``.
- Congenital anomaly flags: ``ca_cchd``, ``ca_disor``.

Features removed:

- ``mracehisp``, ``fracehisp`` — maternal / paternal race/ethnicity.
- ``meduc``, ``feduc`` — maternal / paternal education.
- ``pay_rec`` — principal source of payment (SES proxy).
- ``fagecomb`` — paternal age (correlates with maternal age but
  introduces heavy SES-correlated missingness; the user asked for
  "maternal age" specifically).

Hyperparameters start from the USBC10 tuned values. Re-running
``scripts/tune_model.py --model-id usbc11_m0`` is expected to give
slightly different optima on the reduced feature set; the initial
inherited values are a deliberate, reviewable starting point.
"""

from __future__ import annotations

from datetime import date

from dspopulations_us_birth_certificates.models.common import (
    SelectionStep,
    ShapScatterSpec,
)
from dspopulations_us_birth_certificates.models.usbc10 import USBC10_M1
from dspopulations_us_birth_certificates.variables import Variables as vars

_USBC11_BASE_PARAMS: dict = {
    "objective": "binary",
    "metric": ["average_precision", "binary_logloss"],
    "boosting_type": "gbdt",
    "max_bin": 255,
    "scale_pos_weight": 1,
    "force_col_wise": True,
}

# Starting point: USBC10_M1 tuned params. Re-tuning on the usbc11
# feature set is the expected next step.
_USBC11_PARAMS: dict = {
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

_USBC11_TRAIN_CONFIG: dict = {
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
)

_M0_CATEGORICAL: tuple[str, ...] = (
    vars.BFACIL3,
    vars.SEX,
    vars.PRECARE,
    vars.GESTREC10,
    vars.RF_PHYPE,
    vars.RF_GHYPE,
    vars.RF_INFTR,
    vars.RF_FEDRG,
    vars.LD_INDL,
    vars.LD_AUGM,
    vars.ME_PRES,
    vars.DMETH_REC,
    vars.APGAR5,
    vars.AB_AVEN1,
    vars.AB_AVEN6,
    vars.AB_NICU,
    vars.AB_ANTI,
    vars.CA_CCHD,
    vars.CA_DISOR,
)


# Sociodemographic / non-clinical features excluded vs. usbc10_m1.
_SOCIODEMO_FEATURES_REMOVED: tuple[str, ...] = (
    "mracehisp",
    "meduc",
    "fracehisp",
    "feduc",
    "pay_rec",
    "fagecomb",
)


class USBC11_M0(USBC10_M1):
    """Clinical + maternal age only — diagnostic variant of usbc10_m1.

    Inherits from ``USBC10_M1`` so ``selection_history()`` walks the
    full decision chain (M0 baseline → M1 low-importance drops →
    M11-M0 sociodemographic drops) via the standard Python MRO.
    """

    model_id = "usbc11_m0"
    variant_of = "usbc10_m1"
    numeric_features = _M0_NUMERIC
    categorical_features = _M0_CATEGORICAL
    base_params = _USBC11_BASE_PARAMS
    params = _USBC11_PARAMS
    train_config = _USBC11_TRAIN_CONFIG
    # Separate prediction + flag columns so a usbc11 run does not
    # overwrite usbc10's predictions — both can coexist for
    # side-by-side diagnostic comparison.
    predictions_column = "p_ds_lb_pred_02"
    missing_flag_column = "ds_pred_missing_02"
    selection_steps = (
        SelectionStep(
            step_date=date(2026, 4, 20),
            rationale=(
                "Remove sociodemographic features (race/ethnicity, education, "
                "payer) and paternal age from the usbc10_m1 feature set. "
                "Training on down_ind = 1 conflates P(DS | X) with "
                "P(recorded | DS, X); demographic features are informative "
                "about recording sensitivity (Boulet 2011; Egan 2011), so "
                "including them entrenches the selection bias. Clinical "
                "features are causally downstream of DS and less entangled "
                "with recording, so a clinical+maternal-age model gives a "
                "less-contaminated estimate of P(DS | X). The predicted-"
                "missing cohort's distribution across held-out demographic "
                "variables is then interpretable as a diagnostic of who is "
                "being under-ascertained at national scale."
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
