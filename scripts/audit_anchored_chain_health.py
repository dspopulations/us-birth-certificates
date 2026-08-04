"""Audit surveillance-anchored fits per chain for the anchor-off degenerate mode.

The anchored likelihood (``reduction_model='anchor'``, so ``DSP007`` onward)
admits a second mode that pooled convergence statistics hide.  When
``anchor_obs_sigma`` is estimated rather than fixed, nothing stops it inflating
far past its ``HalfNormal`` prior scale; at a large enough value the surveillance
observation equation contributes almost nothing to the log-probability and the
anchor effectively switches off.  Latent prevalence is then unconstrained and
runs up until ``theta * eta`` exceeds one for every maternal age, where
``build_core_reduction_model`` clips ``p_ds_lb``.  A clip is a flat region with no
gradient, so there is nothing to push a chain back out, and recording sensitivity
collapses towards zero to keep the product near the observed recorded rate.

This was found in a ``DSP009`` fit where one chain in four escaped.  It is not a
``DSP009`` defect: the escape needs only an inflated ``anchor_obs_sigma``, which
every anchored model with a free observation SD permits.  A drifted fit merely
reaches it sooner.

Pooled summaries are the wrong instrument.  Three healthy chains out of four
still produced a max R-hat of ``1.0111``, which reads as "needs a slightly longer
run" rather than "one chain is in a different mode".  Per-chain means make it
obvious in one line, so this audit works per chain and reports the worst chain
rather than the average.

Read-only.  Writes ``anchored_chain_health.csv`` and, unless every run is clean,
``anchored_chain_detail.csv`` with one row per chain.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dse_research_utils.environment.setup import init_script

DEFAULT_SEARCH_ROOTS = (Path("output"),)
DEFAULT_OUTPUT_ROOT = Path("output/anchored_chain_audit")

# eta is prevalence over the Morris no-reduction expectation. Values slightly
# above one are a real diagnostic rather than an error -- they mean the anchored
# prevalence exceeds the natural expectation -- so the code deliberately does not
# exclude them. The degenerate mode is nothing like slight: it reached eta of 39.
ETA_SUSPECT = 1.0
ETA_DEGENERATE = 1.5
# Chains in one mode agree on the recording level to a few hundredths of a
# percent in every healthy run inspected. A percent is already far outside that.
S_SPREAD_SUSPECT_PCT = 1.0
S_SPREAD_DEGENERATE_PCT = 10.0
# Share of a chain's draws that must sit above ETA_DEGENERATE before the chain is
# called degenerate rather than merely excursive.
DEGENERATE_DRAW_SHARE = 0.01


def _model_section(config: dict[str, Any]) -> dict[str, Any]:
    """Core-model config is stored either at the top level or under 'model'."""
    section = config.get("model", config)
    return section if isinstance(section, dict) else config


def find_anchored_runs(roots: tuple[Path, ...]) -> list[Path]:
    """Return directories holding an anchored fit, newest path order last."""
    found: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        for config_path in sorted(root.rglob("config.json")):
            try:
                config = json.loads(config_path.read_text())
            except OSError, json.JSONDecodeError:
                continue
            section = _model_section(config)
            if section.get("reduction_model") != "anchor":
                continue
            if not (config_path.parent / "idata.nc").is_file():
                continue
            found.append(config_path.parent)
    return found


def audit_run(run_dir: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    """Return a one-row verdict for ``run_dir`` and its per-chain detail."""
    import arviz as az

    idata = az.from_netcdf(run_dir / "idata.nc")
    posterior = idata.posterior
    section = _model_section(json.loads((run_dir / "config.json").read_text()))
    hyperpriors = (section.get("surveillance_anchor") or {}).get("hyperpriors", {})

    eta = np.asarray(posterior["eta_year"].values)  # (chain, draw, year)
    recording_s = np.asarray(posterior["recording_s"].values)  # (chain, draw)
    obs_sigma = np.asarray(posterior["anchor_obs_sigma"].values)
    if obs_sigma.ndim == 0:
        obs_sigma = np.full(recording_s.shape, float(obs_sigma))

    rows = []
    for chain in range(eta.shape[0]):
        eta_chain = eta[chain]
        rows.append(
            {
                "run": str(run_dir),
                "chain": chain,
                "recording_s_mean": float(recording_s[chain].mean()),
                "anchor_obs_sigma_mean": float(np.atleast_1d(obs_sigma[chain]).mean()),
                "eta_max": float(eta_chain.max()),
                "draw_share_eta_above_1": float(
                    (eta_chain > ETA_SUSPECT).any(axis=-1).mean()
                ),
                "draw_share_eta_above_1_5": float(
                    (eta_chain > ETA_DEGENERATE).any(axis=-1).mean()
                ),
            }
        )
    detail = pd.DataFrame(rows)

    chain_means = detail["recording_s_mean"].to_numpy()
    spread_pct = float((chain_means.max() / chain_means.min() - 1.0) * 100.0)
    worst_share = float(detail["draw_share_eta_above_1_5"].max())
    any_excursion = float(detail["draw_share_eta_above_1"].max()) > 0.0

    if worst_share > DEGENERATE_DRAW_SHARE or spread_pct > S_SPREAD_DEGENERATE_PCT:
        verdict = "DEGENERATE"
    elif any_excursion or spread_pct > S_SPREAD_SUSPECT_PCT:
        verdict = "SUSPECT"
    else:
        verdict = "CLEAN"

    summary_path = run_dir / "summary.csv"
    max_rhat = min_ess = float("nan")
    if summary_path.is_file():
        summary = pd.read_csv(summary_path, index_col=0)
        if "r_hat" in summary:
            max_rhat = float(summary["r_hat"].max())
        if "ess_bulk" in summary:
            min_ess = float(summary["ess_bulk"].min())

    verdict_row = {
        "run": str(run_dir),
        "model_id": section.get("model_id"),
        "recording_drift": section.get("recording_drift") or "none",
        "obs_sigma_fixed": hyperpriors.get("obs_sigma_fixed"),
        "forecast_flat": bool(hyperpriors.get("forecast_flat") or False),
        "chains": int(eta.shape[0]),
        "draws": int(eta.shape[1]),
        "recording_s_spread_pct": round(spread_pct, 4),
        "worst_chain_eta_max": round(float(detail["eta_max"].max()), 4),
        "worst_chain_draw_share_eta_above_1_5": round(worst_share, 4),
        "max_rhat": None if np.isnan(max_rhat) else round(max_rhat, 4),
        "min_ess_bulk": None if np.isnan(min_ess) else round(min_ess),
        "verdict": verdict,
    }
    return verdict_row, detail


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "run_dirs",
        nargs="*",
        type=Path,
        help=(
            "Fit directories to audit. Defaults to every anchored fit discovered "
            "under output/."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory for the audit CSVs.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any audited run is not CLEAN.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    ns = parse_args(argv)
    init_script()

    run_dirs = list(ns.run_dirs) or find_anchored_runs(DEFAULT_SEARCH_ROOTS)
    if not run_dirs:
        print("No anchored fits found. Nothing to audit.")
        return 0

    verdicts, details = [], []
    for run_dir in run_dirs:
        if not (run_dir / "idata.nc").is_file():
            print(f"[skip] {run_dir}: no idata.nc")
            continue
        verdict, detail = audit_run(run_dir)
        verdicts.append(verdict)
        details.append(detail)

    if not verdicts:
        print("No audited runs carried InferenceData.")
        return 0

    summary = pd.DataFrame(verdicts)
    detail = pd.concat(details, ignore_index=True)

    ns.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(ns.output_dir / "anchored_chain_health.csv", index=False)

    columns = [
        "model_id",
        "recording_drift",
        "obs_sigma_fixed",
        "chains",
        "draws",
        "recording_s_spread_pct",
        "worst_chain_eta_max",
        "max_rhat",
        "verdict",
        "run",
    ]
    print(summary[columns].to_string(index=False))

    unclean = summary[summary["verdict"] != "CLEAN"]
    if not unclean.empty:
        detail.to_csv(ns.output_dir / "anchored_chain_detail.csv", index=False)
        print("\nPer-chain detail for runs that are not CLEAN:")
        affected = detail[detail["run"].isin(unclean["run"])]
        print(affected.to_string(index=False))
        print(
            f"\n{len(unclean)} of {len(summary)} runs are not CLEAN. A DEGENERATE "
            "verdict means at least one chain sat in the anchor-off mode; those "
            "draws must not be pooled into any reported quantity. Re-fit with "
            "--anchor-obs-sigma-fixed, which closes the escape route and is "
            "preferred on reporting grounds anyway."
        )
    else:
        print(f"\nAll {len(summary)} audited runs are CLEAN.")

    # A short run can miss the mode entirely: the DSP009 fit that exposed it read
    # max R-hat 1.0111 at 1,500 draws and only became unmistakable at 4,000. So a
    # clean verdict on a short run is weak evidence, and saying so is the point.
    short = summary[(summary["verdict"] == "CLEAN") & (summary["draws"] < 4000)]
    if not short.empty:
        print(
            f"\nNote: {len(short)} clean run(s) have fewer than 4,000 draws per "
            "chain. The mode was invisible at 1,500 draws in the fit that first "
            "exposed it, so absence here is not evidence that the run's "
            "specification excludes it."
        )

    print(f"\nWrote {ns.output_dir}/anchored_chain_health.csv")
    if ns.strict and not unclean.empty:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
