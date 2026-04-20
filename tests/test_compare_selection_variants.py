"""Test ``scripts/compare_selection_variants.py`` against synthetic fit dirs.

Builds four minimal ``idata.nc`` + ``cells.parquet`` + ``summary.csv`` +
``tables/*.csv`` trees (one per variant), invokes the script, and
verifies the aggregated CSV + forest plot appear with the expected
rows.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

pytest.importorskip("arviz")

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "compare_selection_variants.py"


def _load_cli_module():
    spec = importlib.util.spec_from_file_location(
        "compare_selection_variants_cli", SCRIPT_PATH
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["compare_selection_variants_cli"] = mod
    spec.loader.exec_module(mod)
    return mod


def _build_minimal_fit(fit_dir: Path, variant: str, seed: int) -> None:
    """Write a minimum set of artefacts that compare_selection_variants reads."""
    import arviz as az
    import xarray as xr

    fit_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # Minimal InferenceData with p_ds_lb (needed for total-true computation).
    n_chain, n_draw, n_cell = 2, 50, 12
    p_ds_lb = rng.uniform(1e-4, 1e-3, size=(n_chain, n_draw, n_cell))
    posterior = xr.Dataset(
        {
            "p_ds_lb": (("chain", "draw", "cell"), p_ds_lb),
        },
        coords={
            "chain": np.arange(n_chain),
            "draw": np.arange(n_draw),
            "cell": np.arange(n_cell),
        },
    )
    idata = az.InferenceData(posterior=posterior)
    idata.to_netcdf(str(fit_dir / "idata.nc"))

    # Cells frame — enough structure to let compare_build_comparison run.
    cells = pd.DataFrame(
        {
            "N_cell": rng.integers(1000, 5000, size=n_cell),
            "R_cell": rng.integers(0, 5, size=n_cell),
            "race_idx": rng.integers(0, 6, size=n_cell),
            "cchd": rng.integers(0, 2, size=n_cell),
        }
    )
    cells.to_parquet(fit_dir / "cells.parquet", index=False)

    # config.json carries the variant label.
    (fit_dir / "config.json").write_text(
        json.dumps({"model_id": "selection", "variant": variant, "spec": "full"})
    )

    # summary.csv with eta_term_race[0..5] + s_race[0..5].
    # lo/hi are derived from mean ± half-width so forest-plot xerr stays
    # non-negative (matches real az.summary output).
    index = []
    rows = []
    for prefix in ("eta_term_race", "s_race"):
        for i in range(6):
            mean = rng.normal(0, 0.3)
            half_width = rng.uniform(0.1, 0.3)
            index.append(f"{prefix}[{i}]")
            rows.append(
                {
                    "mean": mean,
                    "sd": half_width / 2,
                    "hdi_3%": mean - half_width,
                    "hdi_97%": mean + half_width,
                    "ess_bulk": 1500,
                    "ess_tail": 1000,
                    "r_hat": 1.00,
                }
            )
    summary = pd.DataFrame(rows, index=index)
    summary.to_csv(fit_dir / "summary.csv")

    # tables/identifiability.csv + tables/dobbs_year_trajectory.csv.
    tables = fit_dir / "tables"
    tables.mkdir(exist_ok=True)
    pd.DataFrame(
        {
            "race_idx": range(6),
            "race": [
                "NH White",
                "NH Black",
                "NH AIAN_NHOPI_Other",
                "NH Asian",
                "Hispanic",
                "Unknown",
            ],
            "correlation": rng.uniform(-0.9, 0.3, size=6),
            "abs_correlation": rng.uniform(0.1, 0.9, size=6),
            "interpretation": ["data-informed"] * 6,
        }
    ).to_csv(tables / "identifiability.csv", index=False)

    dobbs_rows = [
        {
            "year_idx": i,
            "is_post_dobbs": i >= 6,
            "posterior_mean": rng.normal(0, 0.2),
            "lo": -0.3,
            "hi": 0.3,
            "hdi_prob": 0.94,
        }
        for i in range(9)
    ]
    dobbs_rows.append(
        {
            "year_idx": -1,
            "is_post_dobbs": True,
            "posterior_mean": rng.normal(0, 0.1),
            "lo": -0.2,
            "hi": 0.2,
            "hdi_prob": 0.94,
        }
    )
    pd.DataFrame(dobbs_rows).to_csv(
        tables / "dobbs_year_trajectory.csv", index=False
    )


@pytest.fixture
def four_variant_fits(tmp_path: Path) -> Path:
    """Build synthetic output/selection/{A,B,C,D}/full/<ts>/ trees."""
    root = tmp_path / "selection_root"
    for i, v in enumerate(("A", "B", "C", "D")):
        fit_dir = root / v / "full" / f"20260420-0000{i}0"
        _build_minimal_fit(fit_dir, v, seed=i)
    return root


def test_compare_autodiscovers_and_aggregates(
    four_variant_fits: Path, tmp_path: Path
) -> None:
    mod = _load_cli_module()
    out = tmp_path / "compare_out"
    exit_code = mod.main(
        [
            "--root",
            str(four_variant_fits),
            "--output-dir",
            str(out),
        ]
    )
    assert exit_code == 0
    csv = pd.read_csv(out / "comparison.csv")
    # All four variants × at least three metrics.
    assert set(csv["variant"].unique()) == {"A", "B", "C", "D"}
    assert {"total_true", "eta_term_race", "s_race", "dobbs_effect"}.issubset(
        csv["metric"].unique()
    )
    # Forest-plot image written.
    assert (out / "comparison_forest.png").is_file()
    assert (out / "comparison_forest.svg").is_file()


def test_compare_explicit_fit_dirs(four_variant_fits: Path, tmp_path: Path) -> None:
    mod = _load_cli_module()
    fit_dirs = [
        next((four_variant_fits / v / "full").iterdir())
        for v in ("A", "B", "C", "D")
    ]
    out = tmp_path / "compare_explicit"
    exit_code = mod.main(
        [
            "--fit-dirs",
            *[str(fd) for fd in fit_dirs],
            "--output-dir",
            str(out),
        ]
    )
    assert exit_code == 0
    assert (out / "comparison.csv").is_file()


def test_compare_rejects_missing_config(tmp_path: Path) -> None:
    mod = _load_cli_module()
    bad = tmp_path / "bad"
    bad.mkdir()
    with pytest.raises(SystemExit, match="Missing config"):
        mod.main(
            [
                "--fit-dirs",
                str(bad),
                "--output-dir",
                str(tmp_path / "o"),
            ]
        )


def test_compare_rejects_empty_root(tmp_path: Path) -> None:
    mod = _load_cli_module()
    with pytest.raises(SystemExit, match="No variant fits found"):
        mod.main(
            [
                "--root",
                str(tmp_path / "empty_root"),
                "--output-dir",
                str(tmp_path / "o"),
            ]
        )
