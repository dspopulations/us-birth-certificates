"""Compare validation metrics across model runs.

Reads ``metrics.json`` from each supplied run directory and prints a
side-by-side comparison table. Also emits ``comparison.csv`` in the
current working directory (or ``--output`` path) for downstream use.

Currently compares single-split metrics only (matches step 4's
pipeline). Paired per-fold comparison will arrive when cross_validate
lands.

Examples
--------
    # Compare three runs side-by-side
    python scripts/compare_variants.py \\
        output/fit_model_reporting/20260418-120000 \\
        output/fit_model_reporting/20260418-130000 \\
        output/fit_model_reporting/20260418-140000

    # Or by model id, picking each variant's most recent run under output/models/
    python scripts/compare_variants.py --by-model-id usbc10_m0 usbc10_m1 usbc10_m2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

METRIC_COLUMNS: tuple[str, ...] = (
    "average_precision",
    "roc_auc",
    "log_loss",
    "brier_score",
    "mean_predicted_prob",
    "best_iteration",
    "n_valid",
    "n_positive_valid",
)


def _latest_run_for_model(models_root: Path, model_id: str) -> Path | None:
    """Pick the most recent run under ``models_root`` for ``model_id``.

    Matches any subdirectory whose name starts with ``model_id-``
    (e.g. ``usbc10_m0-dev``, ``usbc10_m0-reporting``). Within those,
    picks the one whose nested timestamp subdirectory is newest.
    """
    if not models_root.exists():
        return None
    candidates: list[Path] = []
    for dir_ in models_root.iterdir():
        if dir_.is_dir() and dir_.name.startswith(f"{model_id}-"):
            candidates.extend(p for p in dir_.iterdir() if p.is_dir())
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _read_run(run_dir: Path) -> dict:
    metrics_path = run_dir / "metrics.json"
    config_path = run_dir / "config.json"
    manifest_path = run_dir / "manifest.json"
    row: dict = {"run_dir": str(run_dir)}
    if metrics_path.is_file():
        row.update(json.loads(metrics_path.read_text()))
    if config_path.is_file():
        cfg = json.loads(config_path.read_text())
        row["model_id"] = cfg.get("model_id")
        row["year_range"] = cfg.get("year_range")
        row["n_features"] = len(cfg.get("numeric_features", [])) + len(
            cfg.get("categorical_features", [])
        )
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        git = manifest.get("git", {}) or {}
        row["git_sha"] = git.get("sha", "")[:8] if git.get("sha") else ""
        row["git_dirty"] = git.get("dirty")
    return row


def compare_runs(run_dirs: list[Path]) -> pd.DataFrame:
    rows = [_read_run(d) for d in run_dirs]
    df = pd.DataFrame(rows)

    priority = ["run_dir", "model_id", "n_features", "year_range", "git_sha"]
    priority += [c for c in METRIC_COLUMNS if c in df.columns]
    extras = [c for c in df.columns if c not in priority]
    df = df[[c for c in priority if c in df.columns] + extras]
    return df


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "run_dirs",
        nargs="*",
        type=Path,
        help="Run directories to compare.",
    )
    p.add_argument(
        "--by-model-id",
        nargs="+",
        default=None,
        help=(
            "Instead of explicit run dirs, pick the most recent run for each "
            "model id under --models-root."
        ),
    )
    p.add_argument(
        "--models-root",
        type=Path,
        default=Path("output/models"),
        help="Root where per-model runs live (used with --by-model-id).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("comparison.csv"),
        help="Destination CSV.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    ns = parse_args(argv)

    if ns.by_model_id:
        run_dirs: list[Path] = []
        for model_id in ns.by_model_id:
            run = _latest_run_for_model(ns.models_root, model_id)
            if run is None:
                print(
                    f"No runs found for model_id={model_id!r} under {ns.models_root}",
                    file=sys.stderr,
                )
                return 1
            run_dirs.append(run)
    else:
        run_dirs = list(ns.run_dirs)

    if not run_dirs:
        print(
            "No run directories supplied. Pass paths or use --by-model-id.",
            file=sys.stderr,
        )
        return 1

    df = compare_runs(run_dirs)
    df.to_csv(ns.output, index=False)
    print(df.to_string(index=False))
    print(f"\nWritten to {ns.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
