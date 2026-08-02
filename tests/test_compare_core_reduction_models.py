"""Tests for ``scripts/compare_core_reduction_models.py``."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import compare_core_reduction_models  # noqa: E402


def _write_fake_core_run(
    run_dir: Path,
    *,
    model_id: str,
    true_total: float,
    recording_s: float,
) -> None:
    tables = run_dir / "tables"
    tables.mkdir(parents=True)
    (run_dir / "config.json").write_text(
        json.dumps({"model_id": model_id}),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "metric": "true_ds_livebirths",
                "mean": true_total,
                "lo": true_total - 10,
                "hi": true_total + 10,
                "notes": "fake total",
            },
            {
                "metric": "aggregate_reduction",
                "mean": 0.4,
                "lo": float("nan"),
                "hi": float("nan"),
                "notes": "fake reduction",
            },
            {
                "metric": "recording_s",
                "mean": recording_s,
                "lo": recording_s - 0.01,
                "hi": recording_s + 0.01,
                "notes": "fake recording",
            },
        ]
    ).to_csv(tables / "core_headlines.csv", index=False)
    pd.DataFrame(
        [
            {
                "year": 2020,
                "posterior_mean": recording_s,
                "posterior_lo": recording_s - 0.01,
                "posterior_hi": recording_s + 0.01,
            },
            {
                "year": 2021,
                "posterior_mean": recording_s + 0.02,
                "posterior_lo": recording_s + 0.01,
                "posterior_hi": recording_s + 0.03,
            },
        ]
    ).to_csv(tables / "core_recording_s_by_year.csv", index=False)


def test_compare_core_model_outputs_writes_tables_and_plot(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    extension = tmp_path / "extension"
    output = tmp_path / "comparison"
    _write_fake_core_run(
        baseline,
        model_id="DSP001",
        true_total=100.0,
        recording_s=0.40,
    )
    _write_fake_core_run(
        extension,
        model_id="DSP002",
        true_total=105.0,
        recording_s=0.42,
    )

    paths = compare_core_reduction_models.compare_core_model_outputs(
        baseline,
        extension,
        output,
    )

    assert paths["headline"].is_file()
    assert paths["recording"].is_file()
    assert paths["recording_plot_png"].is_file()
    assert paths["recording_plot_svg"].is_file()
    headlines = pd.read_csv(paths["headline"])
    recording = pd.read_csv(paths["recording"])
    assert list(headlines["metric"]) == [
        "true_ds_livebirths",
        "aggregate_reduction",
        "recording_s",
    ]
    assert headlines.loc[0, "extension_minus_baseline_mean"] == 5.0
    assert list(recording["year"]) == [2020, 2021]
    assert recording.loc[0, "extension_minus_baseline_mean"] == pytest.approx(0.02)
