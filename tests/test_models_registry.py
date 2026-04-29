"""Structural invariants of the ``MODELS`` registry.

Ensures every ``ModelDefinition`` in the registry is well-formed:

- model_ids are unique
- every ``variant_of`` resolves to a registered model
- every SelectionStep carries a rationale and a date
- ``selection_history()`` chains through the MRO, so a descendant's
  history includes every ancestor's steps in order.
"""

from __future__ import annotations

from datetime import date

import pytest

from dspopulations_us_birth_certificates.models import MODELS, ModelDefinition


def test_registry_non_empty() -> None:
    assert MODELS, "MODELS registry should contain at least one definition"


def test_every_model_id_is_unique() -> None:
    ids = [cls.model_id for cls in MODELS.values()]
    assert len(ids) == len(set(ids))


def test_every_variant_of_resolves_in_registry() -> None:
    for cls in MODELS.values():
        if cls.variant_of is not None:
            assert cls.variant_of in MODELS, (
                f"{cls.model_id!r} declares variant_of={cls.variant_of!r} "
                f"but that id is not in MODELS"
            )


def test_every_selection_step_has_rationale_and_date() -> None:
    for cls in MODELS.values():
        for step in cls.selection_steps:
            assert step.rationale.strip(), (
                f"{cls.model_id}: SelectionStep missing rationale"
            )
            assert isinstance(step.step_date, date), (
                f"{cls.model_id}: SelectionStep.step_date must be a date"
            )


def test_selection_history_chains_through_mro() -> None:
    # USBC10_M2 inherits from USBC10_M1 inherits from USBC10_M0.
    # selection_history() walks oldest ancestor first.
    m0 = MODELS["usbc10_m0"]
    m1 = MODELS["usbc10_m1"]
    m2 = MODELS["usbc10_m2"]

    assert len(m0.selection_history()) == 1
    assert len(m1.selection_history()) == 2
    assert len(m2.selection_history()) == 3

    m2_history = m2.selection_history()
    assert m2_history[0].rationale == m0.selection_steps[0].rationale
    assert m2_history[1].rationale == m1.selection_steps[0].rationale
    assert m2_history[2].rationale == m2.selection_steps[0].rationale


def test_to_config_reflects_variant_state() -> None:
    m1 = MODELS["usbc10_m1"]
    m0 = MODELS["usbc10_m0"]

    m1_cfg = m1.to_config()
    m0_cfg = m0.to_config()

    # M1's numeric set is a subset of M0's (post-PR-#26 prune drops
    # ``fagecomb`` on correlation grounds with ``mage_c``).
    assert set(m1_cfg.numeric_features).issubset(set(m0_cfg.numeric_features))
    assert "fagecomb" not in set(m1_cfg.numeric_features)
    # M1's categorical set is strictly smaller than M0's.
    assert len(m1_cfg.categorical_features) < len(m0_cfg.categorical_features)
    removed = set(m0_cfg.categorical_features) - set(m1_cfg.categorical_features)
    assert removed  # non-empty


def test_abstract_subclass_without_model_id_is_not_registered() -> None:
    """A subclass without model_id doesn't register (used for abstract helpers)."""
    before = dict(MODELS)

    class _AbstractHelper(ModelDefinition):
        """No model_id — not registered."""

    assert dict(MODELS) == before


def test_duplicate_model_id_raises() -> None:
    with pytest.raises(ValueError, match="Duplicate model_id"):

        class _ConflictingM0(ModelDefinition):
            model_id = "usbc10_m0"  # already taken


def test_confirmed_only_variants_expose_c_only_target() -> None:
    """The C-only variants declare the right target + writeback columns."""
    for parent_id, cn_id, pred_col in [
        ("usbc10_m0", "usbc10_m0_cn", "p_ds_lb_pred_11"),
        ("usbc11_m0", "usbc11_m0_cn", "p_ds_lb_pred_12"),
    ]:
        parent_cfg = MODELS[parent_id].to_config()
        cn_cfg = MODELS[cn_id].to_config()

        assert parent_cfg.confirmed_only is False
        assert parent_cfg.target_var == "ca_down_c_p_n"

        assert cn_cfg.confirmed_only is True
        assert cn_cfg.target_var == "ca_down_c_n"
        assert cn_cfg.predictions_column == pred_col
        # Parent and C-only must not share writeback columns.
        assert cn_cfg.predictions_column != parent_cfg.predictions_column
        assert cn_cfg.missing_flag_column != parent_cfg.missing_flag_column

        # C-only variant inherits the parent's feature set and appends one
        # selection step documenting the label change.
        assert cn_cfg.numeric_features == parent_cfg.numeric_features
        assert cn_cfg.categorical_features == parent_cfg.categorical_features
        assert len(cn_cfg.selection_history) == len(parent_cfg.selection_history) + 1


def test_usbc10_m1_cn_prunes_under_c_only_importance() -> None:
    """M1_CN drops additional features beyond M1's prune under C-only importance."""
    m0_cn = MODELS["usbc10_m0_cn"].to_config()
    m1_cn = MODELS["usbc10_m1_cn"].to_config()

    # Confirmed-only flag propagates to the pruned variant.
    assert m1_cn.confirmed_only is True
    assert m1_cn.target_var == "ca_down_c_n"

    # M1_CN has fewer categorical features than M0_CN.
    assert len(m1_cn.categorical_features) < len(m0_cn.categorical_features)
    # M1_CN has its own writeback columns distinct from M0_CN.
    assert m1_cn.predictions_column != m0_cn.predictions_column
    assert m1_cn.missing_flag_column != m0_cn.missing_flag_column

    cn_cats = set(m1_cn.categorical_features)
    # Under the post-PR-#26 C-only re-derivation, ``ld_augm`` is re-instated
    # above the AP-loss < 1e-4 threshold whereas ``wic``, ``rf_pdiab``,
    # ``mracehisp``, ``ab_anti``, ``rf_phype``, ``pay_rec``, ``me_pres``,
    # ``sex``, ``bfacil3`` all cross BELOW it. Guard the asymmetry.
    assert "ld_augm" in cn_cats
    assert not cn_cats & {
        "wic",
        "rf_pdiab",
        "mracehisp",
        "ab_anti",
        "rf_phype",
        "pay_rec",
        "me_pres",
        "sex",
        "bfacil3",
    }
    # Numeric drop under C-only: ``bmi`` registers as actively harmful
    # (-4.3e-5 in the M0_CN test-profile fit) and falls out.
    assert "bmi" not in set(m1_cn.numeric_features)
    # ``apgar5`` is retained under C-only after the PR #26 fix.
    assert "apgar5" in cn_cats


def test_usbc11_m1_cn_prunes_under_c_only_importance() -> None:
    """usbc11_m1_cn drops 22 clinical features that fall below threshold under C-only."""
    m0_cn = MODELS["usbc11_m0_cn"].to_config()
    m1_cn = MODELS["usbc11_m1_cn"].to_config()

    assert m1_cn.confirmed_only is True
    assert m1_cn.target_var == "ca_down_c_n"
    assert m1_cn.predictions_column != m0_cn.predictions_column
    assert m1_cn.missing_flag_column != m0_cn.missing_flag_column

    cn_cats = set(m1_cn.categorical_features)
    # Strongly-negative-importance features under C-only should be gone:
    # rf_phype (-1.51e-4), rf_gdiab (-1.37e-4), rf_artec (-5.68e-5),
    # rf_fedrg (-6.74e-5), rf_ppterm (-8.60e-5), ld_augm (-5.15e-5),
    # ca_cleft (-2.13e-4), ca_clpal (-6.32e-5).
    assert not cn_cats & {
        "rf_phype",
        "rf_gdiab",
        "rf_artec",
        "rf_fedrg",
        "rf_ppterm",
        "ld_augm",
        "ca_cleft",
        "ca_clpal",
    }
    # ``apgar5`` is retained under the post-PR-#26 C-only fit (importance
    # 3.05e-3 in the M0_CN test fit, the largest of the four families).
    assert "apgar5" in cn_cats
    # ``rf_pdiab`` and ``wic`` cross BACK ABOVE threshold under C-only on
    # the clinical-only feature set, opposite of the USBC11_M1 (C+P) drop.
    assert "rf_pdiab" in cn_cats
    assert "wic" in cn_cats
    # The clinical predictors that survive at every prune step should all
    # still be present — they carry the bulk of the signal under both
    # label definitions.
    for core_feature in ("ca_cchd", "ca_disor", "ab_nicu", "gestrec10", "dmeth_rec"):
        assert core_feature in cn_cats
