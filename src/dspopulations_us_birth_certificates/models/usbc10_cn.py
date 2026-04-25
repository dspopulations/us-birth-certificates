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

# Hyperparameters from output/tuning/usbc10_m1_cn/best_params.json — Optuna
# reporting-profile search (200 trials, 50K boost rounds, 9 yr) on the
# post-PR-#26 USBC10_M1_CN feature set. Re-tune by
# `python scripts/tune_model.py usbc10_m1_cn --profile reporting`.
_USBC10_CN_PARAMS: dict = {
    "learning_rate": 0.017611067651380975,
    "num_leaves": 78,
    "min_data_in_leaf": 507,
    "min_gain_to_split": 0.18220994215266872,
    "feature_fraction": 0.7488234311216709,
    "bagging_fraction": 0.8570140617352552,
    "bagging_freq": 4,
    "lambda_l1": 4.3675423264507757e-07,
    "lambda_l2": 0.12848472888514878,
}


# Features dropped between M0_CN and M1_CN. Re-derived under PR #26's
# apgar5/apgar10 fix on 2026-04-24 against the post-fix
# USBC10_M0_CN test-profile fit (output/fit_model_test/20260424-105543/).
# 30 features fall below the AP-loss < 1e-4 threshold and `fagecomb` is
# added on correlation grounds (dcor 0.75 with `mage_c`). Two notable
# features cross zero under C-only: `bmi` (-4.3e-5) and `rf_phype`
# (-2.6e-4) — both register as actively harmful, i.e. permuting them
# improves AP. `rf_phype` was already flagged as a co-linearity artefact
# with maternal age in notes/20260422-compare-confirmed-only.md §1
# Caveats; the apgar fix didn't rescue it. `apgar5` is newly retained
# (importance 2.65e-3 vs 1.37e-5 pre-fix). `pay_rec` (-1.55e-4) drops
# under C-only despite being SES-proxy — consistent with the C+P→C-only
# pattern documented in the prior comparison.
_M1_CN_FEATURES_REMOVED: tuple[str, ...] = (
    # 1 numeric drop on importance:
    "bmi",
    # Correlation-only drop (numeric): dcor 0.75 with mage_c.
    "fagecomb",
    # 29 categorical drops on importance < 1e-4:
    "mracehisp",
    "bfacil3",
    "ab_anti",
    "ld_indl",
    "rf_ehype",
    "ca_omph",
    "rf_pdiab",
    "sex",
    "me_pres",
    "ca_clpal",
    "apgar10",
    "ab_surf",
    "ca_anen",
    "ca_gast",
    "ab_seiz",
    "ca_cdh",
    "rf_fedrg",
    "ca_cleft",
    "rf_gdiab",
    "ca_hypo",
    "rf_ppterm",
    "rf_inftr",
    "rf_artec",
    "ca_limb",
    "ca_mnsb",
    "fracehisp",
    "wic",
    "pay_rec",
    "rf_phype",
)

_M1_CN_CATEGORICAL: tuple[str, ...] = tuple(
    f for f in USBC10_M0.categorical_features if f not in _M1_CN_FEATURES_REMOVED
)
_M1_CN_NUMERIC: tuple[str, ...] = tuple(
    f for f in USBC10_M0.numeric_features if f not in _M1_CN_FEATURES_REMOVED
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
    """``M0_CN`` minus 31 predictors (post-PR-#26 re-derivation, 2026-04-24)."""

    model_id = "usbc10_m1_cn"
    variant_of = "usbc10_m0_cn"
    numeric_features = _M1_CN_NUMERIC
    categorical_features = _M1_CN_CATEGORICAL
    # Separate writeback columns from M0_CN so both variants can coexist
    # in DuckDB for direct comparison.
    predictions_column = "p_ds_lb_pred_13"
    missing_flag_column = "ds_pred_missing_13"
    selection_steps = (
        SelectionStep(
            step_date=date(2026, 4, 24),
            rationale=(
                "Re-derived under PR #26's apgar5/apgar10 fix. Drops 30 "
                "features below the AP-loss < 1e-4 threshold in the post-fix "
                "USBC10_M0_CN test-profile fit, plus `fagecomb` (numeric) "
                "on correlation grounds (dcor 0.75 with `mage_c`). "
                "Notable changes vs the original 2026-04-21 prune: `apgar5` "
                "is newly retained (importance jumped from 1.37e-5 to "
                "2.65e-3 under the fix); `bmi` drops (importance -4.3e-5 "
                "— now actively harmful, was a default-kept numeric); "
                "`pay_rec` (-1.55e-4) and `rf_phype` (-2.61e-4) drop with "
                "strongly negative importance, the `rf_phype` finding "
                "consistent with the prior C-only co-linearity-with-mage "
                "observation in compare-confirmed-only.md §2 Caveats. "
                "`ld_augm` is re-instated above the threshold under the "
                "fixed apgar inputs. See "
                "notes/20260422-1132-retune-afterapgar-fix.md for full "
                "per-feature numbers and the C+P↔C-only contrast."
            ),
            features_removed=_M1_CN_FEATURES_REMOVED,
        ),
    )
    notes = (
        "Post-apgar-fix USBC10_M1_CN: 17 predictors after the PR #26 "
        "re-prune (4 numeric + 13 categorical)."
    )
