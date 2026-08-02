"""Aggregate posterior summaries across selection-model variants.

Reads the variant fit directories (A/B/C/D) produced by
``scripts/fit_selection_model.py --spec full --profile reporting`` and
builds a side-by-side comparison CSV + forest-plot figure of the
headline posterior quantities:

- Total true DS livebirths 2016–2024 (posterior mean + 95% CI)
- Per-race ``eta_term_race`` and ``s_race`` posterior means + CIs
- Per-race eta/s ridge correlation ``|r|`` between ``eta_term_race``
  and ``s_race`` (from each fit's ``tables/identifiability.csv``)

A material spread across variants on any row indicates prior-driven
decomposition for that quantity. Tight agreement is useful but is not,
by itself, proof that the decomposition is data-identified.

Examples
--------
    # Default: pick the most recent variant run from each of
    # output/selection/{A,B,C,D}/full/ that has a completed fit
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
from dspopulations_us_birth_certificates.selection import (
    AGE_LEVELS,
    RACE_LEVELS,
    inv_logit,
    latest_fit_dir,
)

DEFAULT_ROOT = Path("output/selection")
VARIANTS: tuple[str, ...] = ("A", "B", "C", "D")


@dataclass
class CompareCliConfig:
    fit_dirs: dict[str, Path]  # variant letter -> fit dir
    output_dir: Path
    root: Path


def _latest_full_run(root: Path, variant: str) -> Path | None:
    """Return the most recent completed ``<root>/<variant>/full/<ts>/`` directory."""
    try:
        return latest_fit_dir(variant, root=root)
    except FileNotFoundError:
        return None


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
            "the most recent completed run under <root>/{A,B,C,D}/full/ is "
            "used for whichever variants have one."
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
                raise SystemExit(f"{fd}/config.json has unexpected variant={variant!r}")
            fit_dirs[variant] = fd
    else:
        for v in VARIANTS:
            latest = _latest_full_run(ns.root, v)
            if latest is not None:
                fit_dirs[v] = latest
    if not fit_dirs:
        raise SystemExit("No variant fits found — pass --fit-dirs explicitly.")
    output_dir = ns.output_dir or (
        ns.root / f"_compare_{pd.Timestamp.now(tz='UTC').strftime('%Y%m%d-%H%M%S')}"
    )
    return CompareCliConfig(
        fit_dirs=fit_dirs,
        output_dir=output_dir,
        root=ns.root,
    )


# --------------------------------------------------------------------------- #
# Per-fit extraction                                                           #
# --------------------------------------------------------------------------- #


def _load_fit_arrays(
    fit_dir: Path,
) -> tuple[dict[str, np.ndarray], pd.DataFrame, float]:
    """Load one fit's posterior draws + cells needed for aggregation.

    Returns ``(arrays, cells, fpr)``. ``arrays`` holds ``p`` (``p_ds_lb``),
    ``theta_cell`` (per-cell ``theta_lb``), and ``p_rec`` (``p_recorded``) —
    each reshaped to ``(draws, cell)`` (chain and draw flattened together).
    """
    import arviz as az

    idata = az.from_netcdf(str(fit_dir / "idata.nc"))
    cells = pd.read_parquet(fit_dir / "cells.parquet")
    fpr = float(
        json.loads((fit_dir / "config.json").read_text())["priors"][
            "false_positive_rate"
        ]
    )
    post = idata.posterior
    p = np.asarray(post["p_ds_lb"].values)
    p = p.reshape(-1, p.shape[-1])  # (draws, cell)
    tla = np.asarray(post["theta_lb_age"].values).reshape(p.shape[0], -1)
    age = cells["age_idx"].to_numpy()
    theta_cell = inv_logit(tla[:, age])  # (draws, cell)
    p_rec = np.asarray(post["p_recorded"].values).reshape(p.shape[0], -1)
    return {"p": p, "theta_cell": theta_cell, "p_rec": p_rec}, cells, fpr


def _mean_ci(a: np.ndarray) -> dict[str, float]:
    """Posterior mean + 95% CI as a ``{mean, lo, hi}`` dict."""
    return {
        "mean": float(a.mean()),
        "lo": float(np.quantile(a, 0.025)),
        "hi": float(np.quantile(a, 0.975)),
    }


def _scalar_aggregates(
    arrays: dict[str, np.ndarray], cells: pd.DataFrame, fpr: float
) -> dict[str, dict[str, float]]:
    """Total true DS, eta, reduction, and s — each a posterior mean + 95% CI."""
    p, theta_cell = arrays["p"], arrays["theta_cell"]
    N = cells["N_cell"].to_numpy(dtype=float)
    R = cells["R_cell"].to_numpy(dtype=float)

    total = (p * N).sum(axis=1)
    natural = (theta_cell * N).sum(axis=1)
    eta = total / natural
    fp = fpr * ((1.0 - p) * N).sum(axis=1)
    agg_s = (R.sum() - fp) / total

    return {
        "total_true": _mean_ci(total),
        "agg_eta": _mean_ci(eta),
        "reduction": _mean_ci(1.0 - eta),
        "agg_s": _mean_ci(agg_s),
    }


def _age_decomposition(
    arrays: dict[str, np.ndarray], cells: pd.DataFrame, fpr: float
) -> pd.DataFrame:
    """Per-maternal-age posterior-predictive table.

    ``obs_rate`` vs ``pred_rate`` (close = the model fits that band),
    N-weighted ``eta``/``reduction``, and implied ``s``.
    """
    p, theta_cell, p_rec = arrays["p"], arrays["theta_cell"], arrays["p_rec"]
    N = cells["N_cell"].to_numpy(dtype=float)
    R = cells["R_cell"].to_numpy(dtype=float)
    age = cells["age_idx"].to_numpy()

    p_mean = p.mean(axis=0)
    theta_mean = theta_cell.mean(axis=0)
    p_rec_mean = p_rec.mean(axis=0)

    rows: list[dict] = []
    for a in range(len(AGE_LEVELS)):
        m = age == a
        if not m.any():
            continue
        Na = float(N[m].sum())
        Ra = float(R[m].sum())
        true_ct = float((p_mean[m] * N[m]).sum())
        tlb = float(theta_mean[m][0])
        fp_a = fpr * float(((1.0 - p_mean[m]) * N[m]).sum())
        rows.append(
            {
                "age": AGE_LEVELS[a],
                "N_frac": Na / float(N.sum()),
                "R": int(Ra),
                "obs_rate": Ra / Na,
                "pred_rate": float((p_rec_mean[m] * N[m]).sum()) / Na,
                "theta_lb": tlb,
                "eta": true_ct / (tlb * Na),
                "reduction": 1.0 - true_ct / (tlb * Na),
                "s": (Ra - fp_a) / true_ct,
            }
        )
    return pd.DataFrame(rows)


def _extract_aggregates(
    fit_dir: Path,
) -> tuple[dict[str, dict[str, float]], pd.DataFrame]:
    """Posterior aggregates + per-age decomposition from one idata load.

    Returns ``(aggregates, age_df)``. ``aggregates`` maps ``total_true`` /
    ``agg_eta`` / ``reduction`` / ``agg_s`` to ``{mean, lo, hi}`` (95% CI).
    ``age_df`` is the per-maternal-age decomposition used as an age
    posterior-predictive check: ``obs_rate`` vs ``pred_rate`` (close = the
    model fits that band), N-weighted ``eta``/``reduction``, and implied ``s``.
    """
    arrays, cells, fpr = _load_fit_arrays(fit_dir)
    aggregates = _scalar_aggregates(arrays, cells, fpr)
    age_df = _age_decomposition(arrays, cells, fpr)
    return aggregates, age_df


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


def build_comparison(
    fit_dirs: dict[str, Path],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the variant comparison: a long-format metric frame + per-age table.

    Returns ``(comparison, age_decomposition)``.

    ``comparison`` columns: ``variant``, ``metric``, ``level``, ``mean``,
    ``lo``, ``hi``. ``metric`` ∈ {``total_true``, ``agg_eta``, ``reduction``,
    ``agg_s``, ``eta_term_race``, ``s_race``, ``eta_s_corr_abs``};
    ``level`` is the race label (where applicable) or "(all)" for scalars.

    ``age_decomposition`` is the per-maternal-age posterior-predictive table
    (one block per variant): observed vs predicted recorded-DS rate, the
    N-weighted ``eta``/``reduction``, and implied ``s``.
    """
    rows: list[dict] = []
    age_frames: list[pd.DataFrame] = []
    for variant, fit_dir in sorted(fit_dirs.items()):
        cli_output.info(f"Variant {variant}: reading {fit_dir}")

        # Aggregates: total true DS, eta, reduction, s (+95% CI) + per-age table.
        aggregates, age_df = _extract_aggregates(fit_dir)
        for metric, vals in aggregates.items():
            rows.append(
                {"variant": variant, "metric": metric, "level": "(all)", **vals}
            )
        age_df.insert(0, "variant", variant)
        age_frames.append(age_df)

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
                                RACE_LEVELS[idx]
                                if idx < len(RACE_LEVELS)
                                else f"race_{idx}"
                            ),
                            **row.to_dict(),
                        }
                    )

        # Eta/s ridge-correlation |r| per race. Under anchored s, this is
        # not a stand-alone identification metric.
        ident = _load_identifiability(fit_dir)
        for _, r in ident.iterrows():
            rows.append(
                {
                    "variant": variant,
                    "metric": "eta_s_corr_abs",
                    "level": r.get("race", f"race_{int(r['race_idx'])}"),
                    "mean": float(r["abs_correlation"]),
                    "lo": float("nan"),
                    "hi": float("nan"),
                }
            )

    age_decomp = (
        pd.concat(age_frames, ignore_index=True) if age_frames else pd.DataFrame()
    )
    return pd.DataFrame(rows), age_decomp


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
    comparison, age_decomp = build_comparison(cli.fit_dirs)
    cli.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = cli.output_dir / "comparison.csv"
    comparison.to_csv(csv_path, index=False)
    cli_output.success(f"comparison.csv -> {csv_path}")
    if not age_decomp.empty:
        age_path = cli.output_dir / "age_decomposition.csv"
        age_decomp.to_csv(age_path, index=False)
        cli_output.success(f"age_decomposition.csv -> {age_path}")

    cli_output.section("Render forest plot")
    try:
        render_forest(comparison, cli.output_dir / "comparison_forest")
        cli_output.success(f"comparison_forest.png/.svg -> {cli.output_dir}")
    except Exception as exc:  # noqa: BLE001
        cli_output.warning(f"Forest plot failed: {type(exc).__name__}: {exc}")

    cli_output.section("Headline")
    totals = comparison[comparison["metric"] == "total_true"].set_index("variant")
    cli_output.print_kv(
        "Total true DS livebirths (posterior mean [95% CI])",
        [
            (v, f"{r['mean']:,.0f}  [{r['lo']:,.0f}, {r['hi']:,.0f}]")
            for v, r in totals.iterrows()
        ],
    )
    red = comparison[comparison["metric"] == "reduction"].set_index("variant")
    s_agg = comparison[comparison["metric"] == "agg_s"].set_index("variant")
    cli_output.print_kv(
        "Elective-termination reduction (1 - eta) [95% CI]",
        [
            (v, f"{r['mean']:.3f}  [{r['lo']:.3f}, {r['hi']:.3f}]")
            for v, r in red.iterrows()
        ],
    )
    cli_output.print_kv(
        "Aggregate BC sensitivity s [95% CI]",
        [
            (v, f"{r['mean']:.3f}  [{r['lo']:.3f}, {r['hi']:.3f}]")
            for v, r in s_agg.iterrows()
        ],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
