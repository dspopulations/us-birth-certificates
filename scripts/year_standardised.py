"""Composition-standardised screening / termination / reduction by year.

The by-year curves in ``year_trends.py`` are population-weighted: each year's average
mixes the fitted year effect with that year's demographic composition (shifting age,
ethnicity, education, payer mix). This script holds the composition FIXED at the
pooled (all-years) distribution and varies only the fitted year terms (the year main
effect and the year-by-age interaction in screening; the year main effect in
termination), so the resulting trend is the year effect *net of who is giving birth*.

For variant C it reports and plots both the standardised trend (solid) and the
as-observed population-weighted trend (dashed); the gap between them is the
compositional contribution. The screening-vs-termination split is still prior-driven
(only the combined reduction on eta is data-identified). Figure ->
notes/figures/year_standardised (png/svg/csv).

Usage:
    python scripts/year_standardised.py [--variant C]
"""

from __future__ import annotations  # noqa: I001

import dspopulations_us_birth_certificates.env_guard  # noqa: F401

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402
from dse_research_utils.environment import setup  # noqa: E402
from dse_research_utils.plot import styles  # noqa: E402

from dspopulations_us_birth_certificates.plot_utils import save_fig  # noqa: E402
from dspopulations_us_birth_certificates.selection import inv_logit, latest_fit_dir  # noqa: E402

OUTPUT_DIR = "notes/figures"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", default="C")
    ns = ap.parse_args(argv)
    setup.init_script()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fit = latest_fit_dir(ns.variant)
    cells = pd.read_parquet(fit / "cells.parquet")
    with open(fit / "config.json") as fh:
        y0 = int(json.load(fh)["year_range"][0])
    n = cells["N_cell"].to_numpy(float)
    idx = {k: cells[f"{k}_idx"].to_numpy() for k in ("year", "age", "race", "edu", "payer")}

    with xr.open_dataset(fit / "idata.nc", group="posterior") as post:
        n_year = post.sizes["year"]

        def cm(name: str) -> np.ndarray:
            return post[name].values.reshape(-1, post[name].shape[-1]).mean(0)

        def cs(name: str) -> float:
            return float(post[name].values.mean())

        theta = inv_logit(
            post["theta_lb_age"].values.reshape(-1, post.sizes["age"]).mean(0)
        )
        edi, edy, eda = cs("eta_detect_int"), cm("eta_detect_year"), cm("eta_detect_age")
        edya = (
            post["eta_detect_year_age"]
            .values.reshape(-1, n_year, post.sizes["age"])
            .mean(0)
        )
        edr, ede, edp = cm("eta_detect_race"), cm("eta_detect_edu"), cm("eta_detect_payer")
        eti, ety, eta_age = cs("eta_term_int"), cm("eta_term_year"), cm("eta_term_age")
        etr, ete = cm("eta_term_race"), cm("eta_term_edu")

    a, r, e, p, yc = idx["age"], idx["race"], idx["edu"], idx["payer"], idx["year"]
    w = theta[a] * n  # DS-pregnancy weight per cell (fixed reference composition)

    # Covariate part of each linear predictor (year terms excluded).
    base_d = edi + eda[a] + edr[r] + ede[e] + edp[p]
    base_t = eti + eta_age[a] + etr[r] + ete[e]

    def avg(values: np.ndarray, mask: np.ndarray | None = None) -> float:
        ww = w if mask is None else w[mask]
        vv = values if mask is None else values[mask]
        return float((vv * ww).sum() / ww.sum())

    # As-observed: each cell at its OWN year (full arrays), averaged within year below.
    det_obs = inv_logit(base_d + edy[yc] + edya[yc, a])
    term_obs = inv_logit(base_t + ety[yc])
    reduc_obs = det_obs * term_obs

    rows = []
    for y in range(n_year):
        # Standardised: every cell evaluated at year y, pooled composition held fixed.
        det_s = inv_logit(base_d + edy[y] + edya[y, a])
        term_s = inv_logit(base_t + ety[y])
        m = yc == y
        rows.append({
            "year": y0 + y,
            "screen_std": avg(det_s), "screen_obs": avg(det_obs, m),
            "term_std": avg(term_s), "term_obs": avg(term_obs, m),
            "reduc_std": avg(det_s * term_s),
            "reduc_obs": avg(reduc_obs, m),
        })
    df = pd.DataFrame(rows)
    yr = df["year"].to_numpy()

    fig, ax = plt.subplots(figsize=styles.FIGSIZE_LG)
    series = [
        ("screen", "Screening detection", styles.COLOUR_BLUE),
        ("term", "Termination if detected", styles.COLOUR_GREEN),
        ("reduc", "Reduction (not born alive)", styles.COLOUR_RED),
    ]
    for key, label, col in series:
        ax.plot(yr, df[f"{key}_std"], "-o", ms=3, color=col, label=f"{label} — standardised")
        ax.plot(yr, df[f"{key}_obs"], "--", color=col, alpha=0.55, label=f"{label} — as observed")
    ax.set_xlabel("Year")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)
    ax.set_title(f"Screening & termination by year, composition-standardised (variant {ns.variant})")
    ax.legend(fontsize=6, ncol=2, loc="center left")
    save_fig(fig, OUTPUT_DIR, "year_standardised", data=df)
    plt.close(fig)

    pd.set_option("display.width", 160)
    print(f"fit: {fit}")
    print(df.round(3).to_string(index=False))
    for key, label, _ in series:
        d = (df[f"{key}_std"] - df[f"{key}_obs"]).abs().max()
        print(f"max |standardised - observed| for {label}: {d:.3f}")
    print(f"\nwrote year_standardised to {OUTPUT_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
