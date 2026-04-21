"""USBC11 C-only variants — clinical feature set, confirmed-only label.

``USBC11_M0_CN`` mirrors ``USBC11_M0``'s clinical + maternal-age feature
set but trains on ``ca_down = 'C'`` (confirmed) only, dropping ``'P'``
(pending) rows.

``USBC11_M1_CN`` drops features whose permutation importance was near
zero in the ``USBC11_M0_CN`` test-profile run — a fresh prune against
the C-only label on the already-pruned clinical feature set.

Pairs with the USBC10 C-only family to expose two effects side by side:

1. C-only vs C+P under a fixed feature set.
2. With vs without non-clinical predictors under the cleaner label
   (``USBC10_M1_CN`` vs ``USBC11_M1_CN`` at matched prune level).

If the second comparison narrows relative to ``USBC10_M1`` vs
``USBC11_M0``, that's a signal that the non-clinical predictors were
picking up pending-code recording noise rather than underlying DS
signal.
"""

from __future__ import annotations

from datetime import date

from dspopulations_us_birth_certificates.models.common import SelectionStep
from dspopulations_us_birth_certificates.models.usbc11 import USBC11_M0

# Hyperparameters from output/fit_model_test/usbc11_m0_cn/best_params.json
# (Optuna test-profile search, 50 trials). Less drastic simplification
# than the USBC10 C-only tune — the clinical-only feature set was
# already closer to a well-conditioned problem so the optimum is closer
# to the family-wide C+P starting point. Re-tuning under M1_CN is the
# expected next step before the reporting fit.
_USBC11_CN_PARAMS: dict = {
    "learning_rate": 0.005799269440315334,
    "num_leaves": 173,
    "min_data_in_leaf": 1542,
    "min_gain_to_split": 0.44215350713342805,
    "feature_fraction": 0.9369179058153728,
    "bagging_fraction": 0.9682087031934399,
    "bagging_freq": 5,
    "lambda_l1": 1.9547805855762397e-08,
    "lambda_l2": 0.1618363093741931,
}


# Features dropped between M0_CN and M1_CN on the clinical set. Same
# threshold as USBC10_M1 / USBC10_M1_CN: permutation AP loss < 1e-4.
# One feature (rf_phype) has strongly negative importance (-2.5e-4) —
# treated as noise under the C-only label, likely because its DS
# correlation is fully captured by maternal age, leaving only noise once
# age is in the model.
_M1_CN_FEATURES_REMOVED: tuple[str, ...] = (
    "sex",
    "rf_phype",
    "rf_ghype",
    "me_pres",
    "apgar5",
    "ab_anti",
)

_M1_CN_CATEGORICAL: tuple[str, ...] = tuple(
    f for f in USBC11_M0.categorical_features if f not in _M1_CN_FEATURES_REMOVED
)


class USBC11_M0_CN(USBC11_M0):
    """C-only variant of USBC11_M0 — pending rows dropped from training."""

    model_id = "usbc11_m0_cn"
    variant_of = "usbc11_m0"
    target_var = "ca_down_c_n"
    confirmed_only = True
    params = _USBC11_CN_PARAMS
    # Separate prediction + flag columns so the C-only run does not
    # overwrite the C+P usbc11 predictions.
    predictions_column = "p_ds_lb_pred_12"
    missing_flag_column = "ds_pred_missing_12"
    selection_steps = (
        SelectionStep(
            step_date=date(2026, 4, 21),
            rationale=(
                "Restrict the training positive class to ca_down = 'C' "
                "(confirmed); drop 'P' (pending) rows. Same rationale as "
                "USBC10_M0_CN but with USBC11's clinical + maternal-age "
                "feature set — the paired comparison with USBC10_M0_CN "
                "isolates the non-clinical-predictor contribution under a "
                "cleaner label."
            ),
        ),
    )
    notes = (
        "C-only training variant of USBC11_M0 (clinical + maternal age). "
        "Paired with USBC10_M0_CN for the four-way comparison across "
        "{C+P, C-only} × {full, clinical-only} feature sets."
    )


class USBC11_M1_CN(USBC11_M0_CN):
    """``M0_CN`` minus features with near-zero permutation importance under C-only."""

    model_id = "usbc11_m1_cn"
    variant_of = "usbc11_m0_cn"
    categorical_features = _M1_CN_CATEGORICAL
    # Separate writeback columns from M0_CN.
    predictions_column = "p_ds_lb_pred_14"
    missing_flag_column = "ds_pred_missing_14"
    selection_steps = (
        SelectionStep(
            step_date=date(2026, 4, 21),
            rationale=(
                "Drop predictors whose permutation importance was below the "
                "AP-loss < 1e-4 threshold in the USBC11_M0_CN test-profile "
                "fit. Notable drops: rf_phype (pre-pregnancy hypertension, "
                "-2.5e-4 — strongly negative, meaning permuting it improves "
                "AP; likely a co-linearity artefact with maternal age); sex "
                "(near zero, indicating the modest M>F skew in DS is either "
                "absent at this sample size or fully captured by birth "
                "weight and clinical correlates); rf_ghype, me_pres, "
                "apgar5, ab_anti (all sub-threshold with narrower confidence "
                "than under the full feature set, as expected given the "
                "clinical-only model has less signal to distribute across "
                "predictors)."
            ),
            features_removed=_M1_CN_FEATURES_REMOVED,
        ),
    )
    notes = (
        "C-only and pruned variant of USBC11_M0. Target of the matched-prune "
        "comparisons: USBC11_M0 vs USBC11_M1_CN isolates the C/P-label "
        "effect under clinical features; USBC10_M1_CN vs USBC11_M1_CN "
        "isolates the non-clinical-predictor contribution under the cleaner "
        "label."
    )
