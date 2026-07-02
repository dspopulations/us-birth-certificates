"""Quantify the eta_detect year-by-age interaction: did screening reach older
mothers first?

Reads a selection fit that includes ``eta_detect_year_age`` (the zero-sum
year-by-age interaction on the screening stage) and reports, per maternal-age band,
the *extra* change in screening detection between the early (first 3 years) and late
(last 3 years) periods attributable to the interaction — i.e. the rise beyond the
shared, age-averaged rollout. A positive gradient with age means screening expanded
faster in older mothers.

Outputs: a printed table (extra log-odds rise + 95% CI per age band, and the
age-gradient slope with CI), and a figure (year_age_interaction) to notes/figures/.

The interaction is data-identified: recording ``s`` has no year term, so year-to-year
movement in recorded rates maps onto screening/termination, and only one channel
(eta_detect) carries a year-by-age interaction term.

Usage:
    python scripts/year_age_interaction.py [FIT_DIR]
"""

from __future__ import annotations  # noqa: I001

import dspopulations_us_birth_certificates.env_guard  # noqa: F401

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402
from dse_research_utils.environment import setup  # noqa: E402
from dse_research_utils.plot import styles  # noqa: E402

from dspopulations_us_birth_certificates.plot_utils import save_fig  # noqa: E402
from dspopulations_us_birth_certificates.selection import (  # noqa: E402
    AGE_LEVELS,
    latest_fit_dir,
)

OUTPUT_DIR = "notes/figures"


def _q(a: np.ndarray, lo: float, axis=0) -> np.ndarray:
    return np.quantile(a, lo, axis=axis)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("fit_dir", nargs="?", default=None)
    ns = ap.parse_args(argv)
    setup.init_script()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fit_dir = Path(ns.fit_dir) if ns.fit_dir else latest_fit_dir("C")
    with open(fit_dir / "config.json") as fh:
        cfg = json.load(fh)
    y0 = int(cfg["year_range"][0])
    with xr.open_dataset(fit_dir / "idata.nc", group="posterior") as post:
        if "eta_detect_year_age" not in post.data_vars:
            raise SystemExit(
                f"{fit_dir} has no eta_detect_year_age (re-fit with the interaction)"
            )
        ixn = post["eta_detect_year_age"].values  # (chain, draw, year, age)
        ixn = ixn.reshape(-1, ixn.shape[-2], ixn.shape[-1])  # (draws, year, age)

    n_year, n_age = ixn.shape[1], ixn.shape[2]
    years = y0 + np.arange(n_year)
    k = min(3, n_year // 2)
    # Extra screening rise (late minus early period), per draw, per age band.
    delta = ixn[:, -k:, :].mean(1) - ixn[:, :k, :].mean(1)  # (draws, age)

    # Age-gradient of the extra rise: slope of delta vs centred age-band index.
    a_idx = np.arange(n_age) - (n_age - 1) / 2.0
    slope = (delta * a_idx).sum(1) / (a_idx**2).sum()  # (draws,)

    tab = pd.DataFrame(
        {
            "age_band": AGE_LEVELS,
            "extra_rise": delta.mean(0),
            "lo95": _q(delta, 0.025),
            "hi95": _q(delta, 0.975),
        }
    )
    pd.set_option("display.width", 160)
    print(f"fit: {fit_dir}  years {years[0]}-{years[-1]}\n")
    print("=== Extra screening rise by age (late vs early period; eta_detect log-odds) ===")
    print(tab.to_string(index=False, float_format="{:+.3f}".format))
    print(
        f"\nAge gradient of the extra rise: {slope.mean():+.3f} "
        f"[{np.quantile(slope, 0.025):+.3f}, {np.quantile(slope, 0.975):+.3f}] "
        "log-odds per age band"
    )
    p_pos = float((slope > 0).mean())
    print(f"P(gradient > 0 | data) = {p_pos:.3f}  (older-mothers-first hypothesis)")

    fig, ax = plt.subplots(figsize=styles.FIGSIZE_MD)
    x = np.arange(n_age)
    err = np.vstack([tab["extra_rise"] - tab["lo95"], tab["hi95"] - tab["extra_rise"]])
    colors = [styles.COLOUR_BLUE if v >= 0 else styles.COLOUR_RED for v in tab["extra_rise"]]
    ax.bar(x, tab["extra_rise"], color=colors, alpha=0.85)
    ax.errorbar(x, tab["extra_rise"], yerr=err, fmt="none", ecolor=styles.TEXT_COLOUR, capsize=3)
    ax.axhline(0, color=styles.TEXT_COLOUR, lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(AGE_LEVELS, rotation=45, ha="right")
    ax.set_ylabel("Extra screening rise (log-odds)\nlate vs early period")
    ax.set_xlabel("Maternal age band")
    ax.set_title("Year-by-age interaction in screening detection (extra rise by age)")
    save_fig(fig, OUTPUT_DIR, "year_age_interaction", data=tab)
    plt.close(fig)
    print(f"\nwrote year_age_interaction (png/svg/csv) to {OUTPUT_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
