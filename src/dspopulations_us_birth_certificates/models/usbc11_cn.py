"""USBC11 C-only variants — clinical feature set, confirmed-only label.

``USBC11_M0_CN`` mirrors ``USBC11_M0``'s clinical + maternal-age feature
set but trains on ``ca_down = 'C'`` (confirmed) only, dropping ``'P'``
(pending) rows.

``USBC11_M1_CN`` drops features whose permutation importance was below
the AP-loss < 1e-4 threshold in the post-apgar-fix ``USBC11_M0_CN``
test-profile fit. Pairs with ``USBC10_M1_CN`` for the four-way
{C+P, C-only} × {full, clinical-only} comparison.
"""

from __future__ import annotations

from datetime import date

from dspopulations_us_birth_certificates.models.common import (
    SelectionStep,
    prune_features,
)
from dspopulations_us_birth_certificates.models.usbc11 import USBC11_M0

# Hyperparameters from output/tuning/usbc11_m0_base_cn/best_params.json —
# Optuna test-profile search (50 trials × 10K boost rounds, 9 yr) on the
# post-PR-#26 USBC11_M0_CN 42-feature set. Replaced on ``USBC11_M1_CN``
# by the reporting-profile retune under the pruned feature set.
_USBC11_CN_PARAMS: dict = {
    "learning_rate": 0.015064508020214277,
    "num_leaves": 87,
    "min_data_in_leaf": 3145,
    "min_gain_to_split": 0.0043019588557071786,
    "feature_fraction": 0.677530739501765,
    "bagging_fraction": 0.9799712169355064,
    "bagging_freq": 5,
    "lambda_l1": 1.0208063974043875e-08,
    "lambda_l2": 3.480596257324116e-06,
}

# Hyperparameters from output/tuning/usbc11_m1_cn/best_params.json —
# Optuna reporting-profile search (200 trials, 50K boost rounds, 9 yr)
# on the post-PR-#26 USBC11_M1_CN 20-feature set. The high learning
# rate + large num_leaves trade depth for breadth on a sparse target
# (1,597 positives) — best_iteration=99 in the final fit.
_USBC11_M1_CN_PARAMS: dict = {
    "learning_rate": 0.05025496780779066,
    "num_leaves": 205,
    "min_data_in_leaf": 3253,
    "min_gain_to_split": 0.1923723563054148,
    "feature_fraction": 0.7558455476716968,
    "bagging_fraction": 0.8752248180085902,
    "bagging_freq": 3,
    "lambda_l1": 7.582388918754272e-07,
    "lambda_l2": 0.0016822348057918435,
}


# Features dropped between USBC11_M0_CN and USBC11_M1_CN. Re-derived
# under PR #26's apgar5/apgar10 fix on 2026-04-24 against the post-fix
# USBC11_M0_CN test-profile fit
# (output/fit_model_test/20260424-220729/). 22 categorical features fall
# below the AP-loss < 1e-4 threshold; no numeric features drop. The
# apgar5 ~ apgar10 and rf_fedrg ~ rf_artec correlation pairs are both
# resolved by importance alone (the weaker side of each pair is already
# sub-threshold). Eight features have strongly-negative permutation
# importance under the C-only label (rf_phype, rf_gdiab, rf_ppterm,
# rf_fedrg, rf_artec, ld_augm, ca_cleft, ca_clpal), i.e. permuting them
# improves AP. The rf_phype signal is the same co-linearity-with-mage_c
# artefact already flagged in compare-confirmed-only.md §2 and
# re-confirmed under the USBC10 C-only retune.
_M1_CN_FEATURES_REMOVED: tuple[str, ...] = (
    "sex",
    "rf_gdiab",
    "rf_phype",
    "rf_ehype",
    "rf_ppterm",
    "rf_inftr",
    "rf_fedrg",
    "rf_artec",
    "ld_augm",
    "me_pres",
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
    "ca_clpal",
    "ca_hypo",
)

_M1_CN_NUMERIC: tuple[str, ...] = prune_features(
    USBC11_M0.numeric_features, _M1_CN_FEATURES_REMOVED
)
_M1_CN_CATEGORICAL: tuple[str, ...] = prune_features(
    USBC11_M0.categorical_features, _M1_CN_FEATURES_REMOVED
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
    """``USBC11_M0_CN`` minus 22 predictors (post-PR-#26 re-derivation)."""

    model_id = "usbc11_m1_cn"
    variant_of = "usbc11_m0_cn"
    numeric_features = _M1_CN_NUMERIC
    categorical_features = _M1_CN_CATEGORICAL
    params = _USBC11_M1_CN_PARAMS
    # Separate writeback columns from M0_CN.
    predictions_column = "p_ds_lb_pred_14"
    missing_flag_column = "ds_pred_missing_14"
    selection_steps = (
        SelectionStep(
            step_date=date(2026, 4, 24),
            rationale=(
                "Re-derived under PR #26's apgar5/apgar10 fix. Drop 22 "
                "categorical predictors whose permutation AP loss was "
                "below 1e-4 in the post-fix USBC11_M0_CN test-profile fit "
                "(output/fit_model_test/20260424-220729/). No numeric "
                "features drop. Notable drops: rf_phype, rf_gdiab, "
                "rf_ppterm, rf_fedrg, rf_artec, ld_augm, ca_cleft, "
                "ca_clpal have strongly-negative importance (permuting "
                "them improves AP) — rf_phype in particular replicates "
                "the co-linearity-with-mage_c artefact flagged in "
                "compare-confirmed-only.md §2 and the USBC10_M1_CN post-"
                "fix retune. rf_pdiab stays above the threshold under "
                "C-only here (1.86e-4) while it dropped under USBC11_M1; "
                "`wic` likewise flips from drop under C+P to keep under "
                "C-only (3.68e-4). See "
                "notes/20260422-1132-retune-afterapgar-fix.md for the "
                "full per-feature numbers."
            ),
            features_removed=_M1_CN_FEATURES_REMOVED,
        ),
    )
    notes = (
        "Post-apgar-fix USBC11_M1_CN: 20 predictors after the PR #26 "
        "re-prune (5 numeric + 15 categorical)."
    )
