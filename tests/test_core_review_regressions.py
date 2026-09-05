"""Regression tests for the DSP statistical-code review."""

import importlib.util
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scipy.special import expit, logit

from dspopulations_us_birth_certificates.selection.anomaly_panel import AnomalyPanel
from dspopulations_us_birth_certificates.selection.core_math import (
    calibrated_age_intercept,
    window_design,
    window_error_correlation,
)
from dspopulations_us_birth_certificates.selection.core_models import (
    get_core_model_definition,
)
from dspopulations_us_birth_certificates.selection.core_reduction import (
    CoreReductionPriors,
    SurveillanceAnchor,
    build_core_reduction_model,
)
from dspopulations_us_birth_certificates.selection.core_reporting import (
    accounting_by_year_table,
    recording_s_by_year_table,
    render_core_all,
)
from dspopulations_us_birth_certificates.selection.core_specification import (
    AnchorSettings,
    CoreFitSpecification,
    PanelSettings,
)
from dspopulations_us_birth_certificates.selection.diagnostics import convergence_health
from dspopulations_us_birth_certificates.selection.fit_validation import validate_fit
from dspopulations_us_birth_certificates.selection.sampling import sample_model_prior


def cells():
    frame = pd.DataFrame(
        {
            "year_idx": np.repeat(np.arange(3), 3),
            "age_idx": np.tile(np.arange(3), 3),
            "maternal_age": np.tile([20, 30, 40], 3),
            "N_cell": np.tile([1000, 800, 200], 3),
            "R_cell": np.tile([1, 2, 3], 3),
            "revised": [1] * 9,
        }
    )
    frame.attrs.update(n_year=3, year_range=(2020, 2022))
    return frame


def priors():
    return CoreReductionPriors(
        reduction_mean=np.full(3, 0.35),
        reduction_logit=np.full(3, logit(0.35)),
        reduction_sigma=np.full(3, 0.35),
    )


def anchor():
    return SurveillanceAnchor(
        np.array([0]), np.log([0.0013]), half_width=0, mid_years=(2020,)
    )


def panel():
    return AnomalyPanel(
        condition=("one", "two"),
        year_idx=np.arange(3),
        years=(2020, 2021, 2022),
        flags=np.array([[3, 2], [3, 2], [2, 2]]),
        births=np.full(3, 2000),
        expected_share=np.full((3, 2), 0.001),
        true_trend_log_per_year=np.zeros(2),
        reference_year_idx=0,
    )


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("year_idx", -1),
        ("year_idx", 0.5),
        ("year_idx", 3),
        ("age_idx", -1),
        ("N_cell", 1000.9),
        ("R_cell", 1.9),
        ("N_cell", -1),
        ("R_cell", -1),
        ("R_cell", 1001),
        ("N_cell", np.nan),
        ("R_cell", np.inf),
    ],
)
def test_builder_rejects_invalid_cells_before_casting(column, value):
    frame = cells().astype({column: float})
    frame.loc[0, column] = value
    with pytest.raises(ValueError):
        build_core_reduction_model(frame, priors())


def test_builder_rejects_unsorted_ages_and_invalid_theta():
    frame = cells()
    frame["maternal_age"] = frame.maternal_age.map({20: 40, 30: 30, 40: 20})
    with pytest.raises(ValueError, match="increasing"):
        build_core_reduction_model(frame, priors(), reduction_model="year_age")
    with pytest.raises(ValueError, match="probabilities"):
        build_core_reduction_model(
            cells().drop(columns="maternal_age"),
            replace(priors(), theta_lb_age=np.full(7, np.nan)),
        )


@pytest.mark.parametrize(
    "indices",
    [np.array([0, 0]), np.array([1, 0]), np.array([0, 0.5]), np.array([-1, 1])],
)
def test_anchor_rejects_duplicate_unsorted_or_noninteger_indices(indices):
    with pytest.raises(ValueError):
        SurveillanceAnchor(indices, np.log([0.001, 0.001]))


@pytest.mark.parametrize(
    "change",
    [
        {"flags": np.full((3, 2), 0.5)},
        {"births": np.array([2000, 0, 2000])},
        {"expected_share": np.full((3, 2), np.nan)},
        {"condition": ("one", "one")},
        {"years": (2020, 2022, 2023)},
    ],
)
def test_panel_validates_actual_likelihood_inputs(change):
    with pytest.raises(ValueError):
        replace(panel(), **change)


def test_dsp003_wide_prior_preserves_margin_in_actual_graph():
    model = build_core_reduction_model(
        cells(),
        replace(priors(), reduction_age_step_sigma=0.5),
        reduction_model="year_age",
    )
    point = model.initial_point()
    point["rho_age_step"] = np.array([2.4222068688678102, 6.559545021995628])
    point["rho_logit_year"] = np.full(3, logit(0.3500161523448277))
    function = model.compile_fn(
        model.replace_rvs_by_values([model["rho_year_margin_error"]]),
        inputs=model.value_vars,
        on_unused_input="ignore",
    )
    assert np.max(np.abs(function(point)[0])) < 1e-10
    assert np.isfinite(model.compile_dlogp()(point)).all()


def test_margin_solver_values_and_gradients_at_zero_and_wide_offsets():
    import pytensor
    import pytensor.tensor as pt
    from scipy.optimize import brentq

    target, offset, weight = pt.dvector(), pt.dvector(), pt.dmatrix()
    result = calibrated_age_intercept(target, offset, weight)
    fn = pytensor.function(
        [target, offset, weight], [result, pt.grad(result.sum(), offset)]
    )
    weights = np.array([[0.15, 0.28, 0.57], [0.01, 0.98, 0.01]])
    targets = logit([0.35, 0.999])
    for offsets in (
        np.zeros(3),
        np.array([-8.0, -2.0, 10.0]),
        np.array([20.0, 0.0, -20.0]),
    ):
        intercept, gradient = fn(targets, offsets, weights)
        truth = np.array(
            [
                brentq(
                    lambda x, w=w, offsets=offsets, t=t: (
                        np.dot(w, expit(x + offsets)) - expit(t)
                    ),
                    t - max(offsets) - 1,
                    t - min(offsets) + 1,
                )
                for w, t in zip(weights, targets, strict=True)
            ]
        )
        assert intercept == pytest.approx(truth, abs=1e-9)
        r = expit(intercept[:, None] + offsets)
        derivative = weights * r * (1 - r)
        expected = -(derivative / derivative.sum(axis=1, keepdims=True)).sum(axis=0)
        assert gradient == pytest.approx(expected, abs=1e-8)


def test_overlap_correlation_and_birth_weighted_means():
    design = window_design(np.array([0, 1]), 2)
    correlation = window_error_correlation(design)
    assert correlation[0, 1] == pytest.approx(0.8)
    assert np.diag(correlation) == pytest.approx([1, 1])
    assert window_error_correlation(design, 0) == pytest.approx(np.eye(2))
    weighted = window_design(np.array([0]), 1, np.array([[1, 2, 7]]))
    assert weighted @ [0.001, 0.002, 0.003] == pytest.approx([0.0026])


def test_specification_rejects_calendar_mismatch():
    specification = CoreFitSpecification(
        get_core_model_definition("DSP008"),
        CoreReductionPriors(),
        (2020, 2022),
        anchor=anchor(),
    )
    shifted = replace(specification, year_range=(2021, 2023))
    with pytest.raises(ValueError, match="calendar"):
        shifted.build(cells())
    shifted = replace(specification, anchor=replace(anchor(), mid_years=(2021,)))
    with pytest.raises(ValueError, match="anchor calendar"):
        shifted.build(cells())


def test_reduction_plot_does_not_hide_prior_tails():
    import matplotlib.pyplot as plt

    from dspopulations_us_birth_certificates.selection.core_reporting import (
        _reduction_plot,
    )

    table = pd.DataFrame(
        {
            "year": [2020],
            "rho_prior_mean": [0.3],
            "rho_prior_lo": [-0.2],
            "rho_prior_hi": [0.98],
            "rho_year_mean": [0.3],
            "rho_year_lo": [0.2],
            "rho_year_hi": [0.4],
        }
    )
    figure = _reduction_plot(table)
    lower, upper = figure.axes[0].get_ylim()
    assert lower < -0.2 and upper > 0.98
    plt.close(figure)


def test_anchor_loader_weights_and_nonoverlapping_subset(tmp_path):
    path = tmp_path / "anchor.csv"
    pd.DataFrame(
        {"mid_year": range(2000, 2007), "prevalence_per10k": [13.0] * 7}
    ).to_csv(path, index=False)
    counts = {year: 1000 + year for year in range(1998, 2009)}
    value = SurveillanceAnchor.from_csv(
        path=path, year_range=(2000, 2006), births_by_year=counts, non_overlapping=True
    )
    assert value.mid_years == (2000, 2005)
    assert value.to_dict()["window_weighting"] == "births"
    assert value.window_births[0].tolist() == [2998, 2999, 3000, 3001, 3002]
    with pytest.raises(ValueError, match="weights"):
        SurveillanceAnchor.from_csv(
            path=path, year_range=(2000, 2006), births_by_year={}
        )


def test_partial_nan_diagnostics_fail_unless_known_constant():
    summary = pd.DataFrame(
        {
            "r_hat": [1.0, np.nan],
            "ess_bulk": [1000.0, np.nan],
            "ess_tail": [1000.0, np.nan],
        },
        index=["x", "fixed"],
    )
    assert not convergence_health(summary)["all_ok"]
    assert convergence_health(summary, constant_names=("fixed",))["all_ok"]
    assert not convergence_health(summary.drop(columns="ess_tail"))["all_ok"]


def test_validation_checks_free_variables_divergences_and_energy():
    rng = np.random.default_rng(42)
    shape = (4, 500)
    idata = xr.DataTree.from_dict(
        {
            "posterior": xr.Dataset({"x": (("chain", "draw"), rng.normal(size=shape))}),
            "sample_stats": xr.Dataset(
                {
                    "diverging": (("chain", "draw"), np.zeros(shape, dtype=bool)),
                    "energy": (("chain", "draw"), rng.normal(size=shape)),
                }
            ),
        }
    )
    model = SimpleNamespace(free_RVs=[SimpleNamespace(name="x")])
    _, health = validate_fit(idata, model)
    assert health["status"] == "passed"
    idata.sample_stats["diverging"].values[0, 0] = True
    assert validate_fit(idata, model)[1]["status"] == "failed"
    idata.sample_stats["diverging"].values[0, 0] = False
    idata.sample_stats["energy"] = xr.zeros_like(idata.sample_stats["energy"])
    assert validate_fit(idata, model)[1]["status"] == "failed"


def test_prior_rejection_includes_potential_in_distribution():
    import pymc as pm

    with pm.Model() as model:
        x = pm.Normal("x")
        pm.Potential("penalty", -3 * x**2)
        model.dsp_prior_log_weight = "penalty"
        result = sample_model_prior(model, draws=1800, random_seed=4)
    assert result.attrs["dsp_prior_sampling"] == "rejection_weighted"
    assert float(result.prior.x.mean()) == pytest.approx(0, abs=0.035)
    assert float(result.prior.x.var()) == pytest.approx(1 / 7, abs=0.02)
    assert result.prior.sizes["draw"] == 1800


@pytest.mark.parametrize("model_id", ["DSP006", "DSP007", "DSP008", "DSP009", "DSP010"])
def test_family_specification_and_report_use_actual_priors(model_id, tmp_path):
    definition = get_core_model_definition(model_id)
    anchored = definition.reduction_model == "anchor"
    spec = CoreFitSpecification(
        definition,
        CoreReductionPriors() if anchored else priors(),
        (2020, 2022),
        anchor=anchor() if anchored else None,
        panel=panel() if model_id == "DSP010" else None,
        anchor_settings=AnchorSettings(overlap_share=0.5),
        panel_settings=PanelSettings(loading_fixed=1.0),
    )
    model = spec.build(cells())
    result = sample_model_prior(model, draws=30, random_seed=7)
    # Reporting is a transformation test, not a claim that these are fitted draws.
    result["posterior"] = result["prior"].to_dataset()
    result["posterior_predictive"] = result["prior_predictive"].to_dataset()
    config = spec.to_config().to_dict()
    table = accounting_by_year_table(
        result, cells(), config["priors"], year_range=(2020, 2022), model_config=config
    )
    assert set(table.rho_prior_source) == {"fitted_model_prior_draws"}
    assert table.rho_prior_mean.to_numpy() == pytest.approx(
        np.asarray(result.prior.rho_year.mean(dim=("chain", "draw")))
    )
    recording = recording_s_by_year_table(
        result, config["priors"], years=[2020, 2021, 2022], model_config=config
    )
    assert recording.prior_mean.to_numpy() == pytest.approx(
        np.asarray(result.prior.recording_s_year.mean(dim=("chain", "draw")))
    )
    tables = render_core_all(
        result,
        cells(),
        tmp_path,
        priors_config=config["priors"],
        year_range=(2020, 2022),
        recording_model=definition.recording_model,
        model_config=config,
    )
    assert "expected_ds_livebirths" in set(tables["core_headlines"].metric)
    assert (tmp_path / "report_metadata.json").is_file()
    if anchored:
        assert config["priors"]["reduction_prior_enters_likelihood"] is False
        assert config["surveillance_anchor"]["hyperpriors"]["overlap_share"] == 0.5


@pytest.mark.parametrize("model_id", ["DSP007", "DSP009", "DSP010"])
def test_cli_retains_but_marks_failed_fits(model_id, tmp_path, monkeypatch):
    import duckdb

    path = Path(__file__).resolve().parents[1] / "scripts/fit_core_reduction_model.py"
    spec = importlib.util.spec_from_file_location("dsp_cli", path)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    db = tmp_path / "synthetic.db"
    with duckdb.connect(str(db)) as con:
        con.execute("CREATE TABLE us_births AS SELECT 2020 AS year")
    monkeypatch.setattr(cli, "prepare_core_age_year_cells", lambda *a, **k: cells())
    monkeypatch.setattr(cli, "prepare_anomaly_panel", lambda *a, **k: panel())
    monkeypatch.setattr(
        cli.package_metadata, "report_package_versions", lambda *a: None
    )
    anchor_path = tmp_path / "anchor.csv"
    pd.DataFrame({"mid_year": [2020], "prevalence_per10k": [13.0]}).to_csv(
        anchor_path, index=False
    )

    def insufficient_sample(model, **kwargs):
        result = sample_model_prior(model, draws=12, random_seed=8)
        # One short chain and no sampler diagnostics deliberately fail the gate.
        result["posterior"] = result["prior"].to_dataset()
        result["posterior_predictive"] = result["prior_predictive"].to_dataset()
        return result

    monkeypatch.setattr(cli, "sample", insufficient_sample)
    directory = tmp_path / "fit"
    result = cli.main(
        [
            model_id,
            "--years",
            "2020-2022",
            "--duckdb-path",
            str(db),
            "--anchor-csv",
            str(anchor_path),
            "--anchor-half-width",
            "0",
            "--output-dir",
            str(directory),
        ]
    )
    assert result == 2
    assert json.loads((directory / "validation.json").read_text())["status"] == "failed"
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["validation_status"] == "failed"
    assert manifest["artefact_sha256"]["cells.parquet"]
    assert manifest["source_sha256"]
    assert (directory / "idata.nc").is_file()
    report = json.loads((directory / "report_metadata.json").read_text())
    assert report["model"]["model_id"] == model_id
    assert "UNVALIDATED FIT" in (directory / "index.qmd").read_text()


def test_latest_fit_excludes_failed_and_incomplete_runs(tmp_path):
    from dspopulations_us_birth_certificates.selection.io import latest_fit_dir

    root = tmp_path / "A/full"
    legacy = root / "legacy"
    legacy.mkdir(parents=True)
    (legacy / "idata.nc").touch()
    assert latest_fit_dir("A", root=tmp_path) == legacy
    with pytest.raises(FileNotFoundError):
        latest_fit_dir("A", root=tmp_path, require_validated=True)
    good = root / "good"
    good.mkdir()
    (good / "idata.nc").touch()
    (good / "validation.json").write_text(json.dumps({"status": "passed"}))
    for status in ("failed", "sampling", "prior_only"):
        bad = root / status
        bad.mkdir()
        (bad / "idata.nc").touch()
        (bad / "validation.json").write_text(json.dumps({"status": status}))
    assert latest_fit_dir("A", root=tmp_path, require_validated=True) == good


@pytest.mark.slow
@pytest.mark.parametrize("model_id", ["DSP003", "DSP008", "DSP009", "DSP010"])
def test_joint_simulation_fit_numerical_health(model_id):
    from dspopulations_us_birth_certificates.selection.config import RunConfig
    from dspopulations_us_birth_certificates.selection.core_simulation import (
        calibration_table,
        simulate_core,
        simulation_design,
    )
    from dspopulations_us_birth_certificates.selection.sampling import sample

    frame, specification = simulation_design(model_id)
    frame, specification, truth = simulate_core(frame, specification, seed=47)
    model = specification.build(frame)
    result = sample(
        model,
        config=RunConfig(
            "reporting",
            draws=1000,
            tune=1000,
            chains=4,
            target_accept=0.98,
            nuts_sampler="nutpie",
            random_seed=48,
            prior_predictive_samples=10,
            posterior_predictive=True,
        ),
    )
    _, health = validate_fit(result, model)
    assert health["status"] == "passed", health
    ranks = calibration_table(result, truth, seed=47, model_id=model_id)
    assert ranks.rank_fraction.between(0, 1).all()
    assert "expected_ds_count_total" in set(ranks.variable)
    # Coverage for one data set is not a calibration test. The repeated-run CLI
    # retains both covered and uncovered truths for a separate statistical check.
