"""Compare two fitted core reduction-recording model outputs."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import dse_research_utils.environment.setup as setup
import numpy as np
import pandas as pd

from dspopulations_us_birth_certificates import cli_output
from dspopulations_us_birth_certificates.selection.core_reduction import (
    CORE_REDUCTION_MODEL_ID,
)

HEADLINE_METRICS = (
    "true_ds_livebirths",
    "aggregate_reduction",
    "recording_s",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _run_label(run_dir: Path) -> str:
    config = _read_json(_require_file(run_dir / "config.json"))
    return str(config.get("model_id", run_dir.name))


def _headlines(run_dir: Path) -> pd.DataFrame:
    path = _require_file(run_dir / "tables" / "core_headlines.csv")
    return pd.read_csv(path).set_index("metric")


def _recording_by_year(run_dir: Path) -> pd.DataFrame:
    path = _require_file(run_dir / "tables" / "core_recording_s_by_year.csv")
    return pd.read_csv(path)


def headline_comparison_table(
    baseline_dir: Path,
    extension_dir: Path,
    *,
    metrics: tuple[str, ...] = HEADLINE_METRICS,
) -> pd.DataFrame:
    """Return a direct headline comparison for two fitted runs."""
    baseline_id = _run_label(baseline_dir)
    extension_id = _run_label(extension_dir)
    baseline = _headlines(baseline_dir)
    extension = _headlines(extension_dir)
    rows = []
    for metric in metrics:
        if metric not in baseline.index or metric not in extension.index:
            continue
        base_row = baseline.loc[metric]
        ext_row = extension.loc[metric]
        rows.append(
            {
                "metric": metric,
                "baseline_model_id": baseline_id,
                "extension_model_id": extension_id,
                "baseline_mean": base_row["mean"],
                "baseline_lo": base_row["lo"],
                "baseline_hi": base_row["hi"],
                "extension_mean": ext_row["mean"],
                "extension_lo": ext_row["lo"],
                "extension_hi": ext_row["hi"],
                "extension_minus_baseline_mean": (ext_row["mean"] - base_row["mean"]),
                "baseline_notes": base_row.get("notes", ""),
                "extension_notes": ext_row.get("notes", ""),
            }
        )
    return pd.DataFrame(rows)


def recording_s_year_comparison_table(
    baseline_dir: Path,
    extension_dir: Path,
) -> pd.DataFrame:
    """Return a by-year comparison of recording sensitivity summaries."""
    baseline_id = _run_label(baseline_dir)
    extension_id = _run_label(extension_dir)
    baseline = _recording_by_year(baseline_dir)
    extension = _recording_by_year(extension_dir)

    keep = ["year", "posterior_mean", "posterior_lo", "posterior_hi"]
    merged = baseline[keep].merge(
        extension[keep],
        on="year",
        how="inner",
        suffixes=("_baseline", "_extension"),
    )
    merged.insert(1, "baseline_model_id", baseline_id)
    merged.insert(2, "extension_model_id", extension_id)
    merged["extension_minus_baseline_mean"] = (
        merged["posterior_mean_extension"] - merged["posterior_mean_baseline"]
    )
    return merged


def _recording_comparison_plot(df: pd.DataFrame):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(df))
    for label, suffix, marker in (
        ("baseline", "baseline", "o"),
        ("extension", "extension", "s"),
    ):
        mean = df[f"posterior_mean_{suffix}"].to_numpy(dtype=float)
        lo = df[f"posterior_lo_{suffix}"].to_numpy(dtype=float)
        hi = df[f"posterior_hi_{suffix}"].to_numpy(dtype=float)
        ax.errorbar(
            x,
            mean,
            yerr=np.vstack((np.maximum(mean - lo, 0.0), np.maximum(hi - mean, 0.0))),
            fmt=f"{marker}-",
            capsize=4,
            label=label,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(df["year"].astype(str), rotation=35, ha="right")
    ax.set_ylabel("certificate recording sensitivity")
    ax.set_title("Recording sensitivity: constant s vs s_year")
    ax.legend()
    fig.tight_layout()
    return fig


def compare_core_model_outputs(
    baseline_dir: Path,
    extension_dir: Path,
    output_dir: Path,
) -> dict[str, Path]:
    """Write direct comparison tables and plots for two fitted core models."""
    baseline_dir = Path(baseline_dir)
    extension_dir = Path(extension_dir)
    output_dir = Path(output_dir)
    tables_dir = output_dir / "tables"
    plots_dir = output_dir / "plots"
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    headline = headline_comparison_table(baseline_dir, extension_dir)
    recording = recording_s_year_comparison_table(baseline_dir, extension_dir)

    headline_path = tables_dir / "core_model_headline_comparison.csv"
    recording_path = tables_dir / "core_model_recording_s_year_comparison.csv"
    headline.to_csv(headline_path, index=False)
    recording.to_csv(recording_path, index=False)

    fig = _recording_comparison_plot(recording)
    png_path = plots_dir / "core_model_recording_s_year_comparison.png"
    svg_path = plots_dir / "core_model_recording_s_year_comparison.svg"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)

    config = {
        "baseline_dir": str(baseline_dir),
        "extension_dir": str(extension_dir),
        "baseline_model_id": _run_label(baseline_dir),
        "extension_model_id": _run_label(extension_dir),
    }
    config_path = output_dir / "comparison_config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    return {
        "headline": headline_path,
        "recording": recording_path,
        "recording_plot_png": png_path,
        "recording_plot_svg": svg_path,
        "config": config_path,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare two core reduction-recording model fit outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("baseline_dir", type=Path)
    p.add_argument("extension_dir", type=Path)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory (default: output/selection_core_reduction/"
            "comparisons/<baseline>-vs-<extension>/<timestamp>)."
        ),
    )
    ns = p.parse_args(argv)
    if ns.output_dir is None:
        baseline_id = _run_label(ns.baseline_dir)
        extension_id = _run_label(ns.extension_dir)
        ns.output_dir = (
            Path("output")
            / CORE_REDUCTION_MODEL_ID
            / "comparisons"
            / f"{baseline_id}-vs-{extension_id}"
            / datetime.now().strftime("%Y%m%d-%H%M%S")
        )
    return ns


def main(argv: list[str] | None = None) -> int:
    ns = parse_args(argv)
    setup.init_script()

    cli_output.banner(
        "compare_core_reduction_models",
        f"{_run_label(ns.baseline_dir)} vs {_run_label(ns.extension_dir)}",
    )
    paths = compare_core_model_outputs(
        ns.baseline_dir,
        ns.extension_dir,
        ns.output_dir,
    )
    cli_output.success(f"comparison tables and plots -> {ns.output_dir}")
    cli_output.print_kv("Outputs", paths.items())
    return 0


if __name__ == "__main__":
    sys.exit(main())
