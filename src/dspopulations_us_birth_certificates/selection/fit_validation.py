"""Persisted numerical validation for DSP fits, separate from fit completion."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dspopulations_us_birth_certificates.selection.diagnostics import (
    convergence_health,
    summary_table,
)


def validate_fit(idata, model, *, margin_tolerance: float = 1e-9):
    """Check every free parameter, reported quantities and sampler diagnostics.

    A pass is numerical validation only. It does not validate identification or
    external-data assumptions. Undefined diagnostics are exempted only for
    known deterministic constants, never for free parameters.
    """
    import arviz as az

    free = [variable.name for variable in model.free_RVs]
    focal = [
        name
        for name in (
            "rho_year",
            "eta_year",
            "recording_s",
            "recording_s_year",
            "expected_ds_count_year",
            "expected_ds_count_total",
            "prevalence_year",
            "panel_loading_ds",
            "recording_s_panel_ratio",
            "recording_s_drift_ratio",
        )
        if name in idata.posterior
    ]
    missing = set(free) - set(idata.posterior)
    if missing:
        raise ValueError(f"posterior missing free parameters: {sorted(missing)}")
    names = tuple(dict.fromkeys(free + focal))
    constants = tuple(
        name
        for name in focal
        if name not in free and np.ptp(idata.posterior[name].values) == 0
    )
    summary = summary_table(idata, var_names=names)
    health = convergence_health(summary, constant_names=constants)
    failures = []
    if not health["all_ok"]:
        failures.append(
            "R-hat, effective sample size or finite-diagnostic check failed"
        )
    posterior_finite = all(
        np.isfinite(idata.posterior[name].values).all() for name in names
    )
    if not posterior_finite:
        failures.append("posterior contains non-finite values")
    stats = getattr(idata, "sample_stats", None)
    divergences = None
    bfmi = []
    max_depth_hits = None
    if stats is None or "diverging" not in stats:
        failures.append("divergence diagnostics unavailable")
    else:
        divergences = int(stats["diverging"].sum())
        if divergences:
            failures.append("divergent transitions after warmup")
    if stats is not None and "energy" in stats:
        bfmi = np.atleast_1d(az.bfmi(stats["energy"])).astype(float).tolist()
        if not np.all(np.isfinite(bfmi)) or np.any(np.asarray(bfmi) < 0.3):
            failures.append("energy diagnostic BFMI below 0.3 or undefined")
    else:
        failures.append("energy diagnostic unavailable")
    if stats is not None and "reached_max_treedepth" in stats:
        max_depth_hits = int(stats["reached_max_treedepth"].sum())
    margin_error = 0.0
    if "rho_year_margin_error" in idata.posterior:
        margin_error = float(
            np.max(np.abs(idata.posterior["rho_year_margin_error"].values))
        )
        if not np.isfinite(margin_error) or margin_error > margin_tolerance:
            failures.append(
                "age-specific reduction does not preserve the national margin"
            )
    if "p_ds_lb_overshoot" in idata.posterior:
        overshoot = float(idata.posterior["p_ds_lb_overshoot"].max())
        if not np.isfinite(overshoot) or overshoot > 1e-9:
            failures.append("probability clipping/barrier active in posterior draws")
    return summary, {
        "status": "passed" if not failures else "failed",
        "scope": "numerical_only_not_scientific_validation",
        "failures": failures,
        "convergence": health,
        "checked_variables": list(names),
        "constant_exemptions": list(constants),
        "divergences": divergences,
        "bfmi": bfmi,
        "max_treedepth_hits": max_depth_hits,
        "warnings": ["maximum tree depth reached; inspect efficiency"]
        if max_depth_hits
        else [],
        "max_margin_error": margin_error,
    }


def write_validation(directory: Path, result: dict) -> None:
    """Write a stable status that remains visible when a report is regenerated."""

    def clean(value):
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        if isinstance(value, list):
            return [clean(v) for v in value]
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    directory.mkdir(parents=True, exist_ok=True)
    (directory / "validation.json").write_text(
        json.dumps(clean(result), indent=2, allow_nan=False), encoding="utf-8"
    )
