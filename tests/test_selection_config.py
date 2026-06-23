"""Tests for ``selection.config``."""

from __future__ import annotations

import json

import numpy as np
import pytest

from dspopulations_us_birth_certificates.selection import (
    MODEL_ID,
    SelectionModelConfig,
    preset_names,
    selection_run_config,
)


def test_preset_names() -> None:
    assert set(preset_names()) == {"dev", "reporting"}


def test_run_config_dev_defaults() -> None:
    rc = selection_run_config("dev")
    assert rc.name == "dev"
    assert rc.draws == 1000
    assert rc.tune == 1000
    assert rc.chains == 2
    assert rc.target_accept == 0.9
    assert rc.nuts_sampler == "nutpie"
    assert rc.posterior_predictive is True


def test_run_config_reporting_tighter_than_dev() -> None:
    """Reporting must draw more samples and step at higher target_accept."""
    dev = selection_run_config("dev")
    rpt = selection_run_config("reporting")
    assert rpt.draws > dev.draws
    assert rpt.chains > dev.chains
    assert rpt.target_accept > dev.target_accept


def test_run_config_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unknown selection run profile"):
        selection_run_config("bogus")  # type: ignore[arg-type]


def test_run_config_respects_random_seed() -> None:
    rc = selection_run_config("dev", random_seed=123)
    assert rc.random_seed == 123


def test_model_config_from_priors_serialises() -> None:
    """``from_priors`` serialises priors to a JSON-safe dict."""
    cfg = SelectionModelConfig.from_priors(
        variant="C",
        spec="full",
        year_range=(2016, 2024),
        notes="smoke test",
    )
    d = cfg.to_dict()
    assert d["model_id"] == MODEL_ID
    assert d["variant"] == "C"
    assert d["spec"] == "full"
    assert d["year_range"] == [2016, 2024]
    assert d["notes"] == "smoke test"
    # Priors dict carries the expected scalar and array-valued keys.
    priors = d["priors"]
    assert isinstance(priors["eta_detect_sigma"], float)
    assert isinstance(priors["eta_detect_race"], list)
    # Full JSON round-trip.
    restored = json.loads(json.dumps(d))
    assert restored["priors"]["eta_detect_race"] == priors["eta_detect_race"]


def test_model_config_variant_selection_covers_A_through_C() -> None:
    """Each variant letter resolves to a distinct priors snapshot."""
    snapshots = {
        v: SelectionModelConfig.from_priors(
            variant=v,
            spec="full",
            year_range=(2016, 2024),
        ).to_dict()["priors"]
        for v in ("A", "B", "C")
    }
    # A tightens s_race_year_sigma relative to C; B loosens it (elementwise).
    a = np.asarray(snapshots["A"]["s_race_year_sigma"])
    b = np.asarray(snapshots["B"]["s_race_year_sigma"])
    c = np.asarray(snapshots["C"]["s_race_year_sigma"])
    assert (a < c).all()
    assert (b > c).all()
