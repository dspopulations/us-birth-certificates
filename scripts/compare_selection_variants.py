"""Aggregate posterior summaries across selection-model variants.

Reads the three variant fit directories (A/B/C) produced by
``scripts/fit_selection_model.py --spec full --profile reporting`` and
builds a side-by-side comparison CSV + forest-plot figure of the
headline posterior quantities:

- Total true DS livebirths 2016–2024 (posterior mean + 95% CI)
- Per-race ``eta_term_race`` and ``s_race`` posterior means + CIs
- Per-race identifiability correlation ``|r|`` between ``eta_term_race``
  and ``s_race`` (from each fit's ``tables/identifiability.csv``)

A material spread across variants on any row indicates prior-driven
decomposition for that quantity; tight agreement indicates
data-identified structure (plan §4.4).

Examples
--------
    # Default: pick the most recent variant run from each of
    # output/selection/{A,B,C}/full/
    python scripts/compare_selection_variants.py \\
        --output-dir output/selection/_compare_$(date +%Y%m%d-%H%M%S)

    # Explicit fit directories (e.g. specific runs, not the latest)
    python scripts/compare_selection_variants.py \\
        --fit-dirs output/selection/A/full/20260421-020000 \\
                   output/selection/B/full/20260421-030000 \\
                   output/selection/C/full/20260421-040000 \\
        --output-dir output/selection/_compare
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import dse_research_utils.environment.setup as setup
import numpy as np
import pandas as pd

from dspopulations_us_birth_certificates import cli_output
from dspopulations_us_birth_certificates.selection import RACE_LEVELS

DEFAULT_ROOT = Path("output/selection")
VARIANTS: tuple[str, ...] = ("A", "B", "C")


@dataclass
class CompareCliConfig:
    fit_dirs: dict[str, Path]  # variant letter -> fit dir
    output_dir: Path
    root: Path


def _latest_full_run(root: Path, variant: str) -> Path | None:
    """Return the most recent ``<root>/<variant>/full/<ts>/`` directory."""
    parent = root / variant / "full"
    if not parent.exists():
        return None
    candidates = [p for p in parent.iterdir() if p.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _parse_args(argv: list[str] | None) -> CompareCliConfig:
    p = argparse.ArgumentParser(
        description=(
            "Side-by-side comparison of selection-model variants (A/B/C/D) "
            "from their saved fit directories."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--fit-dirs",
        nargs="+",
        type=Path,
        default=None,
        help=(
            "Explicit variant fit directories (one per variant). If omitted, "
            "the most recent run under <root>/{A,B,C}/full/ is used."
        ),
    )
    p.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Root for auto-discovery when --fit-dirs is omitted.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write comparison.csv and comparison_forest.png.",
    )
    ns = p.parse_args(argv)

    fit_dirs: dict[str, Path] = {}
    if ns.fit_dirs:
        # Pair each explicit dir to a variant by reading config.json.
        for fd in ns.fit_dirs:
            cfg_path = fd / "config.json"
            if not cfg_path.is_file():
                raise SystemExit(f"Missing config.json in {fd}")
            variant = json.loads(cfg_path.read_text()).get("variant")
            if variant not in VARIANTS:
                raise SystemExit(
                    f"{fd}/config.json has unexpected variant={variant!r}"
                )
            fit_dirs[variant] = fd
    else:
        for v in VARIANTS:
            latest = _latest_full_run(ns.root, v)
            if latest is not None:
                fit_dirs[v] = latest
    if not fit_dirs:
        raise SystemExit("No variant fits found — pass --fit-dirs explicitly.")
    output_dir = ns.output_dir or (
        ns.root
        / f"_compare_{pd.Timestamp.now(tz='UTC').strftime('%Y%m%d-%H%M%S')}"
    )
    return CompareCliConfig(
        fit_dirs=fit_dirs,
        output_dir=output_dir,
        root=ns.root,
    )


# --------------------------------------------------------------------------- #
# Per-fit extraction                                                           #
# --------------------------------------------------------------------------- #


def _extract_total_true(fit_dir: Path) -> dict[str, float]:
    """Posterior mean + 95% CI of total true DS livebirths."""
    import arviz as az

    idata = az.from_netcdf(str(fit_dir / "idata.nc"))
    cells = pd.read_parquet(fit_dir / "cells.parquet")
    p_ds_lb = np.asarray(idata.posterior["p_ds_lb"].values)
    N = cells["N_cell"].to_numpy(dtype=float)
    totals = (p_ds_lb * N[None, None, :]).sum(axis=-1)
    return {
        "mean": float(totals.mean()),
        "lo": float(np.quantile(totals, 0.025)),
        "hi": float(np.quantile(totals, 0.975)),
    }


def _extract_summary_rows(summary: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Pull mean + hdi_3%/hdi_97% columns for rows matching a name prefix."""
    rows = summary[summary.index.str.startswith(prefix)].copy()
    rows.index = rows.index.str.extract(r"\[(\d+)\]", expand=False).astype(int)
    rows = rows.sort_index()
    keep = {}
    if "mean" in rows.columns:
        keep["mean"] = rows["mean"]
    if "hdi_3%" in rows.columns and "hdi_97%" in rows.columns:
        keep["lo"] = rows["hdi_3%"]
        keep["hi"] = rows["hdi_97%"]
    return pd.DataFrame(keep)


def _load_identifiability(fit_dir: Path) -> pd.DataFrame:
    path = fit_dir / "tables" / "identifiability.csv"
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path)


# --------------------------------------------------------------------------- #
# Aggregation                                                                  #
# --------------------------------------------------------------------------- #


def build_comparison(fit_dirs: dict[str, Path]) -> pd.DataFrame:
    """Build a long-format comparison frame across variants.

    Columns: ``variant``, ``metric``, ``level``, ``mean``, ``lo``, ``hi``.

    ``metric`` ∈ {``total_true``, ``eta_term_race``, ``s_race``,
    ``identifiability_abs_r``}; ``level`` is the race label (where
    applicable), or "(all)" for scalars.
    """
    rows: list[dict] = []
    for variant, fit_dir in sorted(fit_dirs.items()):
        cli_output.info(f"Variant {variant}: reading {fit_dir}")

        # Total true DS livebirths.
        total = _extract_total_true(fit_dir)
        rows.append(
            {
                "variant": variant,
                "metric": "total_true",
                "level": "(all)",
                **total,
            }
        )

        # Per-race eta_term_race, s_race from summary.csv.
        summary_path = fit_dir / "summary.csv"
        if summary_path.is_file():
            summary = pd.read_csv(summary_path, index_col=0)
            for metric, prefix in (
                ("eta_term_race", "eta_term_race["),
                ("s_race", "s_race["),
            ):
                df = _extract_summary_rows(summary, prefix)
                for idx, row in df.iterrows():
                    rows.append(
                        {
                            "variant": variant,
                            "metric": metric,
                            "level": (
                                RACE_LEVELS[idx] if idx < len(RACE_LEVELS)
                                else f"race_{idx}"
                            ),
                            **row.to_dict(),
                        }
                    )

        # Identifiability |r| per race.
        ident = _load_identifiability(fit_dir)
        for _, r in ident.iterrows():
            rows.append(
                {
                    "variant": variant,
                    "metric": "identifiability_abs_r",
                    "level": r.get("race", f"race_{int(r['race_idx'])}"),
                    "mean": float(r["abs_correlation"]),
                    "lo": float("nan"),
                    "hi": float("nan"),
                }
            )

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Forest-plot figure                                                           #
# --------------------------------------------------------------------------- #


def render_forest(comparison: pd.DataFrame, output_path: Path) -> None:
    """Forest plot of the per-race eta_term_race and s_race posteriors."""
    import dse_research_utils.plot.styles as plot_styles
    import matplotlib.pyplot as plt

    variants = sorted(comparison["variant"].unique())
    colours = [
        plot_styles.COLOUR_BLUE,
        plot_styles.COLOUR_ORANGE,
        plot_styles.COLOUR_GREEN,
        plot_styles.COLOUR_PURPLE,
    ]
    fig, axes = plt.subplots(1, 2, figsize=plot_styles.FIGSIZE_XL, sharey=True)
    for ax, metric, title in zip(
        axes,
        ("eta_term_race", "s_race"),
        (r"$\eta_\mathrm{term}$ race effect", "s race effect"),
        strict=True,
    ):
        sub = comparison[comparison["metric"] == metric]
        levels = list(sub["level"].unique())
        y = np.arange(len(levels))
        for i, v in enumerate(variants):
            ssub = sub[sub["variant"] == v].set_index("level").reindex(levels)
            offset = (i - (len(variants) - 1) / 2) * 0.15
            ax.errorbar(
                ssub["mean"].values,
                y + offset,
                xerr=[
                    (ssub["mean"] - ssub["lo"]).values,
                    (ssub["hi"] - ssub["mean"]).values,
                ],
                fmt="o",
                color=colours[i % len(colours)],
                ecolor=plot_styles.TEXT_COLOUR,
                capsize=2,
                label=f"Variant {v}",
            )
        ax.axvline(0, color=plot_styles.TEXT_COLOUR, lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(levels)
        ax.set_xlabel(title)
    axes[0].set_title("Termination race effects across variants")
    axes[1].set_title("BC sensitivity race effects across variants")
    axes[1].legend(loc="best", fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path.with_suffix(".png"),
        dpi=plot_styles.DPI_FILE,
        bbox_inches="tight",
    )
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# CLI main                                                                     #
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    cli = _parse_args(argv)
    setup.init_script()

    cli_output.banner(
        "compare_selection_variants",
        f"root={cli.root}",
    )
    cli_output.section("Variants")
    cli_output.print_kv(
        "Fit directories",
        [(v, p) for v, p in sorted(cli.fit_dirs.items())],
    )

    cli_output.section("Build comparison")
    comparison = build_comparison(cli.fit_dirs)
    cli.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = cli.output_dir / "comparison.csv"
    comparison.to_csv(csv_path, index=False)
    cli_output.success(f"comparison.csv -> {csv_path}")

    cli_output.section("Render forest plot")
    try:
        render_forest(comparison, cli.output_dir / "comparison_forest")
        cli_output.success(
            f"comparison_forest.png/.svg -> {cli.output_dir}"
        )
    except Exception as exc:  # noqa: BLE001
        cli_output.warning(
            f"Forest plot failed: {type(exc).__name__}: {exc}"
        )

    cli_output.section("Headline")
    totals = comparison[comparison["metric"] == "total_true"].set_index("variant")
    cli_output.print_kv(
        "Total true DS livebirths (posterior mean [95% CI])",
        [
            (v, f"{r['mean']:,.0f}  [{r['lo']:,.0f}, {r['hi']:,.0f}]")
            for v, r in totals.iterrows()
        ],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
