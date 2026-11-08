"""Exercise summary scripts with saved synthetic draws for all seven races."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from dspopulations_us_birth_certificates.selection import (
    AGE_LEVELS,
    EDU_LEVELS,
    PAYER_LEVELS,
    RACE_LEVELS,
    SelectionModelConfig,
)


@pytest.fixture(scope="module")
def summary_scripts():
    modules = {}
    root = Path(__file__).resolve().parents[1]
    for name in ("year_trends", "summary_figures"):
        spec = importlib.util.spec_from_file_location(
            f"selection_summary_{name}", root / "scripts" / f"{name}.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        modules[name] = module
    return modules


@pytest.fixture
def saved_variants(tmp_path, monkeypatch):
    root = tmp_path / "fits"
    monkeypatch.setenv("DSPOP_OUTPUT_ROOT", str(root))
    n_chain, n_draw, n_year = 2, 8, 9
    cells = pd.MultiIndex.from_product(
        [range(n_year), range(len(AGE_LEVELS)), range(len(RACE_LEVELS))],
        names=["year_idx", "age_idx", "race_idx"],
    ).to_frame(index=False)
    cells["edu_idx"] = np.arange(len(cells)) % len(EDU_LEVELS)
    cells["payer_idx"] = np.arange(len(cells)) % len(PAYER_LEVELS)
    cells["N_cell"] = 20_000 + 100 * np.arange(len(cells))
    cells["R_cell"] = 1 + cells["age_idx"] + cells["race_idx"]
    coords = {
        "chain": np.arange(n_chain),
        "draw": np.arange(n_draw),
        "year": np.arange(2016, 2025),
        "age": AGE_LEVELS,
        "race": RACE_LEVELS,
        "edu": EDU_LEVELS,
        "payer": PAYER_LEVELS,
        "cell": np.arange(len(cells)),
    }
    rng = np.random.default_rng(67)
    values = {}
    for name, dimensions, centre, scale in (
        ("theta_lb_age", ("age",), -6.0, 0.25),
        ("eta_detect_int", (), 0.6, 0.1),
        ("eta_term_int", (), 0.3, 0.1),
        ("eta_detect_year_age", ("year", "age"), 0.0, 0.6),
        *[
            (f"eta_detect_{d}", (d,), 0.0, 0.15)
            for d in ("year", "age", "race", "edu", "payer")
        ],
        *[(f"eta_term_{d}", (d,), 0.0, 0.15) for d in ("year", "age", "race", "edu")],
    ):
        shape = (n_chain, n_draw, *(len(coords[d]) for d in dimensions))
        values[name] = (
            ("chain", "draw", *dimensions),
            rng.normal(centre, scale, shape),
        )
    posterior = xr.Dataset(values, coords=coords)

    def probability(x):
        return 1.0 / (1.0 + np.exp(-x))

    def cell_offsets(post, stage):
        total = post[f"eta_{stage}_int"].values[..., None].copy()
        for dimension in ("year", "age", "race", "edu"):
            total = (
                total
                + post[f"eta_{stage}_{dimension}"].values[
                    ..., cells[f"{dimension}_idx"]
                ]
            )
        if stage == "detect":
            total = total + post["eta_detect_payer"].values[..., cells["payer_idx"]]
        return total

    variants = {}
    for variant in ("B", "C"):
        post = posterior.copy(deep=True)
        if variant == "B":
            post["eta_term_int"] -= 0.2
        theta = probability(post["theta_lb_age"].values[..., cells["age_idx"]])
        detect = cell_offsets(post, "detect")
        interaction = post["eta_detect_year_age"].values[
            ..., cells["year_idx"], cells["age_idx"]
        ]
        term = probability(cell_offsets(post, "term"))
        p = theta * (1.0 - probability(detect + interaction) * term)
        without_interaction = theta * (1.0 - probability(detect) * term)
        assert np.max(np.abs(p - without_interaction)) > 1e-4
        post["p_ds_lb"] = (("chain", "draw", "cell"), p)
        run = root / "selection" / variant / "full" / "20261108-120000"
        run.mkdir(parents=True)
        xr.DataTree.from_dict({"posterior": post}).to_netcdf(run / "idata.nc")
        cells.to_parquet(run / "cells.parquet", index=False)
        config = SelectionModelConfig.from_priors(
            variant=variant,
            spec="full",
            year_range=(2016, 2024),
        )
        (run / "config.json").write_text(json.dumps(config.to_dict()))
        variants[variant] = {"run": run, "post": post, "p": p, "theta": theta}
    return cells, variants


def test_year_trends_reconstructs_each_draw_with_year_age_interaction(
    summary_scripts, saved_variants
):
    cells, variants = saved_variants
    result = summary_scripts["year_trends"].load_variant("C", with_ci=True)
    assert result["recon_max_err"] < 1e-15
    for y in range(9):
        mask = cells["year_idx"].to_numpy() == y
        counts = cells.loc[mask, "N_cell"].to_numpy()
        expected = (variants["C"]["p"][..., mask] * counts).sum(axis=-1)
        natural = (variants["C"]["theta"][..., mask] * counts).sum(axis=-1)
        lo, hi = np.quantile(1.0 - expected / natural, [0.055, 0.945])
        assert result["red_lo"][y] == pytest.approx(lo)
        assert result["red_hi"][y] == pytest.approx(hi)
        assert result["df"].loc[2016 + y, "true_per10k"] == pytest.approx(
            expected.mean() / counts.sum() * 1e4
        )


@pytest.mark.parametrize("variant", ["B", "C"])
def test_year_trends_rejects_inconsistent_saved_draws(
    summary_scripts, saved_variants, monkeypatch, tmp_path, variant
):
    _, variants = saved_variants
    fit = variants[variant]
    inconsistent = fit["post"].copy(deep=True)
    inconsistent["p_ds_lb"] += 0.001
    xr.DataTree.from_dict({"posterior": inconsistent}).to_netcdf(
        fit["run"] / "idata.nc"
    )
    script = summary_scripts["year_trends"]
    monkeypatch.setattr(script.setup, "init_script", lambda: None)
    output = tmp_path / "figures"
    monkeypatch.setattr(script, "OUTPUT_DIR", str(output))
    with pytest.raises(
        RuntimeError, match=f"Variant {variant} reconstruction mismatch"
    ):
        script.main()
    assert list(output.iterdir()) == []


def test_summary_figures_render_all_seven_races(
    summary_scripts, saved_variants, monkeypatch, tmp_path
):
    cells, variants = saved_variants
    script = summary_scripts["summary_figures"]
    monkeypatch.setattr(script.setup, "init_script", lambda: None)
    output = tmp_path / "figures"
    monkeypatch.setattr(script, "OUTPUT_DIR", str(output))
    original_save = script.save_fig
    labels = {}

    def save_with_labels(fig, directory, stem, **kwargs):
        labels[stem] = [tick.get_text() for tick in fig.axes[0].get_xticklabels()]
        return original_save(fig, directory, stem, **kwargs)

    monkeypatch.setattr(script, "save_fig", save_with_labels)
    assert script.main() == 0
    for stem in (
        "ascertainment_funnel",
        "age_termination_reduction",
        "ds_rate_by_ethnicity",
        "education_termination_gradient",
    ):
        for extension in ("png", "svg", "csv"):
            assert (output / f"{stem}.{extension}").stat().st_size > 0
    assert labels["ds_rate_by_ethnicity"] == [
        "White",
        "Black",
        "AIAN",
        "Asian/PI",
        "Hispanic",
        "Unknown",
        "Multi-race",
    ]
    table = pd.read_csv(output / "ds_rate_by_ethnicity.csv")
    assert table["ethnicity"].tolist() == RACE_LEVELS
    assert len(table) == 7
    for race in range(7):
        mask = cells["race_idx"].to_numpy() == race
        n = cells.loc[mask, "N_cell"].to_numpy()
        assert table.loc[race, "recorded_per10k"] == pytest.approx(
            cells.loc[mask, "R_cell"].sum() / n.sum() * 1e4
        )
        for variant in ("B", "C"):
            expected = (variants[variant]["p"][..., mask] * n).sum(axis=-1).mean()
            assert table.loc[race, f"true_per10k_{variant}"] == pytest.approx(
                expected / n.sum() * 1e4
            )
    result = script.load_variant("C")
    n = cells.N_cell.to_numpy()
    assert result["natural"] == pytest.approx(
        (variants["C"]["theta"] * n).sum(axis=-1).mean()
    )
    for age in range(len(AGE_LEVELS)):
        mask = cells.age_idx.to_numpy() == age
        expected = (variants["C"]["p"][..., mask] * n[mask]).sum(axis=-1)
        natural = (variants["C"]["theta"][..., mask] * n[mask]).sum(axis=-1)
        assert result["age_reduction"][age] == pytest.approx(
            (1 - expected / natural).mean()
        )
