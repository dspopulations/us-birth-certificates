"""End-to-end test for ``scripts/render_selection_diagnostics.py``.

Reuses the fitted tiny-sample InferenceData from ``test_selection_diagnostics``
would be ideal, but keeping these modules independent is simpler (pytest
module-scoped fixtures don't cross files). We fit a smaller model here
and invoke the CLI through its ``main`` function.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

pytest.importorskip("pymc")


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "render_selection_diagnostics.py"


def _load_cli_module():
    spec = importlib.util.spec_from_file_location(
        "render_selection_diagnostics_cli", SCRIPT_PATH
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["render_selection_diagnostics_cli"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def fit_dir(tmp_path_factory) -> Path:
    import pymc as pm

    from dspopulations_us_birth_certificates.selection import (
        TrueParams,
        build_model,
        simulate_cells,
        variant_C_default,
    )

    out = tmp_path_factory.mktemp("selection_fit")
    truth = TrueParams.from_priors(
        variant_C_default(),
        n_year=9,
        post_dobbs_year_start=6,
        seed=0,
    )
    cells = simulate_cells(
        truth,
        n_cells_per_month=2,
        n_year=9,
        post_dobbs_year_start=6,
        n_cells_mean=1000,
        seed=0,
    )
    cells.attrs["post_dobbs_year_start"] = 6
    cells.attrs["n_year"] = 9

    model = build_model(
        cells,
        variant_C_default(),
        spec="full",
        n_year=9,
        post_dobbs_year_start=6,
    )
    with model:
        idata = pm.sample(
            draws=60,
            tune=60,
            chains=2,
            target_accept=0.9,
            random_seed=0,
            progressbar=False,
            nuts_sampler="pymc",
        )
        idata.extend(pm.sample_posterior_predictive(idata, random_seed=0))

    idata.to_netcdf(str(out / "idata.nc"))
    # Parquet drops attrs; write them into a sidecar so the CLI can still read them.
    cells.to_parquet(out / "cells.parquet", index=False)
    return out


def test_cli_renders_all_figures(fit_dir: Path, tmp_path: Path) -> None:
    mod = _load_cli_module()
    out_dir = tmp_path / "out"
    exit_code = mod.main(
        [
            "--idata",
            str(fit_dir / "idata.nc"),
            "--cells",
            str(fit_dir / "cells.parquet"),
            "--out-dir",
            str(out_dir),
            "--post-dobbs-year-start",
            "6",
            "--strata",
            "year_idx",
            "race_idx",
        ]
    )
    assert exit_code == 0

    plots = out_dir / "plots"
    tables = out_dir / "tables"
    expected = {
        "identifiability",
        "dobbs_year_trajectory",
        "cchd_consistency",
        "age_curve",
        "decomposition_by_race",
        "ppc_year_idx",
        "ppc_race_idx",
    }
    for stem in expected:
        assert (plots / f"{stem}.png").is_file(), f"missing {stem}.png"
        assert (plots / f"{stem}.svg").is_file(), f"missing {stem}.svg"

    # The non-PPC diagnostics all produce a CSV companion.
    for stem in (
        "identifiability",
        "dobbs_year_trajectory",
        "cchd_consistency",
        "age_curve",
        "decomposition_by_race",
    ):
        csv = tables / f"{stem}.csv"
        assert csv.is_file(), f"missing {stem}.csv"
        assert len(pd.read_csv(csv)) >= 1

    assert (out_dir / "convergence_summary.csv").is_file()


def test_cli_rejects_missing_paths(tmp_path: Path) -> None:
    mod = _load_cli_module()
    with pytest.raises(SystemExit):
        mod.main(
            [
                "--idata",
                str(tmp_path / "missing.nc"),
                "--cells",
                str(tmp_path / "missing.parquet"),
                "--out-dir",
                str(tmp_path / "out"),
            ]
        )


def test_cli_fit_dir_discovery(fit_dir: Path, tmp_path: Path) -> None:
    mod = _load_cli_module()
    exit_code = mod.main(
        [
            "--fit-dir",
            str(fit_dir),
            "--out-dir",
            str(tmp_path / "out2"),
            "--post-dobbs-year-start",
            "6",
            "--strata",
            "year_idx",
        ]
    )
    assert exit_code == 0
    assert (tmp_path / "out2" / "plots" / "identifiability.png").is_file()
