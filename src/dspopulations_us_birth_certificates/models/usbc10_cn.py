"""USBC10 C-only variants — train on ``ca_down = 'C'`` only.

``USBC10_M0_CN`` mirrors ``USBC10_M0``'s pre-pruning feature set but
restricts the training label to confirmed Down syndrome codes, dropping
pending (``'P'``) rows from the training set. Wallace et al. (2024)
report ~21% of birth-certificate Down syndrome codes fail chart review;
pending codes are a plausible home for most of that false-positive mass,
so this variant tests whether a cleaner label materially changes which
non-recorded births the model flags as likely missed.

``USBC10_M1_CN`` drops features whose permutation importance was near
zero in the ``USBC10_M0_CN`` test-profile run — a fresh prune against
the new label (the ``USBC10_M1`` prune was chosen under C+P, which is a
different decision surface).

Inherits from ``USBC10_M0`` (not ``_M1``) deliberately: feature pruning
under the C+P label may not be appropriate for the C-only label. Letting
feature importance be rediscovered on the new target is the cleaner
comparison.
"""

from __future__ import annotations

from datetime import date

from dspopulations_us_birth_certificates.models.common import SelectionStep
from dspopulations_us_birth_certificates.models.usbc10 import USBC10_M0

# Hyperparameters from output/fit_model_test/usbc10_m0_cn/best_params.json
# (Optuna test-profile search, 50 trials). Notably simpler than the C+P
# tune (num_leaves 180→38, min_data_in_leaf 756→2590) — fewer positives
# under C-only push the model toward more conservative regularisation.
# Re-tuning under the M1-CN feature set is the expected next step before
# the reporting fit; these inherited values are a reviewable starting
# point.
_USBC10_CN_PARAMS: dict = {
    "learning_rate": 0.00985764771940847,
    "num_leaves": 38,
    "min_data_in_leaf": 2590,
    "min_gain_to_split": 0.5228409298411366,
    "feature_fraction": 0.842269144947652,
    "bagging_fraction": 0.8547221230128124,
    "bagging_freq": 4,
    "lambda_l1": 0.0005876364919646983,
    "lambda_l2": 0.00010450639051902668,
}


# Features dropped between M0_CN and M1_CN. Selected from the
# permutation_importance.csv of the test-profile USBC10_M0_CN fit using
# the same threshold USBC10_M1 applied to its C+P baseline: drop any
# feature whose mean AP loss from permutation < 1e-4 (including
# negative-valued features, where permuting them improves AP — i.e.,
# pure noise). Four more features drop than under C+P (see notes below).
_M1_CN_FEATURES_REMOVED: tuple[str, ...] = (
    # Present in USBC10_M1's drops AND still near-zero under C-only:
    "ca_cdh",
    "apgar10",
    "ca_cleft",
    "rf_artec",
    "ca_omph",
    "ca_clpal",
    "ca_limb",
    "ca_hypo",
    "rf_ppterm",
    "ca_mnsb",
    "rf_ehype",
    "ca_anen",
    "ca_gast",
    "ab_surf",
    "rf_gdiab",
    "ab_seiz",
    # Additional drops under C-only:
    "ld_indl",
    "rf_inftr",
    "apgar5",
    "ld_augm",
    "rf_fedrg",
    "bfacil3",
)

_M1_CN_CATEGORICAL: tuple[str, ...] = tuple(
    f for f in USBC10_M0.categorical_features if f not in _M1_CN_FEATURES_REMOVED
)


class USBC10_M0_CN(USBC10_M0):
    """C-only variant of USBC10_M0 — pending rows dropped from training."""

    model_id = "usbc10_m0_cn"
    variant_of = "usbc10_m0"
    target_var = "ca_down_c_n"
    confirmed_only = True
    params = _USBC10_CN_PARAMS
    # Separate prediction + flag columns so the C-only run does not
    # overwrite the C+P usbc10 predictions.
    predictions_column = "p_ds_lb_pred_11"
    missing_flag_column = "ds_pred_missing_11"
    selection_steps = (
        SelectionStep(
            step_date=date(2026, 4, 21),
            rationale=(
                "Restrict the training positive class to ca_down = 'C' "
                "(confirmed); drop 'P' (pending) rows entirely rather than "
                "treating them as negative. Pending codes carry most of the "
                "~21% false-positive mass reported by Wallace et al. (2024), "
                "so pooling C+P as positives injects label noise. The "
                "feature set is held at the USBC10_M0 baseline so feature "
                "importance can be re-examined under the cleaner label "
                "rather than inheriting the M1 prune which was decided "
                "under C+P."
            ),
        ),
    )
    notes = (
        "C-only training variant of USBC10_M0. Diagnostic companion to the "
        "C+P baseline: side-by-side comparison exposes how much of the "
        "predicted-missing cohort shape is driven by pending-code label "
        "noise versus underlying DS signal."
    )


class USBC10_M1_CN(USBC10_M0_CN):
    """``M0_CN`` minus features with near-zero permutation importance under C-only."""

    model_id = "usbc10_m1_cn"
    variant_of = "usbc10_m0_cn"
    categorical_features = _M1_CN_CATEGORICAL
    # Separate writeback columns from M0_CN so both variants can coexist
    # in DuckDB for direct comparison.
    predictions_column = "p_ds_lb_pred_13"
    missing_flag_column = "ds_pred_missing_13"
    selection_steps = (
        SelectionStep(
            step_date=date(2026, 4, 21),
            rationale=(
                "Drop predictors whose permutation importance was below the "
                "AP-loss < 1e-4 threshold in the USBC10_M0_CN test-profile "
                "fit. Four more features drop than under C+P (ld_indl, "
                "rf_inftr, apgar5, ld_augm, rf_fedrg, bfacil3 added; wic and "
                "rf_pdiab cross back above the threshold under the cleaner "
                "label). Two notes on the added drops: apgar5 is stripped "
                "by the existing data_utils filter to only values exactly "
                "equal to 10, so it has no remaining variance; bfacil3's "
                "importance turns negative under C-only (permuting it "
                "improves AP), suggesting it encodes recording variation "
                "across facility types that is tangled with the C/P split."
            ),
            features_removed=_M1_CN_FEATURES_REMOVED,
        ),
    )
    notes = (
        "C-only equivalent of USBC10_M1. Pruned against the ca_down = 'C' "
        "label to get a cleanly-matched comparison against USBC10_M1 "
        "(rather than comparing a post-prune C+P model to a pre-prune "
        "C-only model)."
    )
