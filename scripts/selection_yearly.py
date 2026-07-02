"""DS livebirths and termination by year (2016-2024) from the selection model.

For each converged reporting variant (A, B, C) computes, per calendar year, the
estimated true DS livebirth COUNT (sum of theta*eta*N over the year's cells) and the
termination reduction (1 - eta), plus the recorded DS count (same across variants).
Two-panel figure -> notes/figures/selection_ds_by_year (png/svg/csv).

A and C pin recording (~0.40) and agree near 40k total; B frees recording and lands
near 48k -- so the band between them is the recording-assumption bound, year by year.

Usage:
    python scripts/selection_yearly.py
"""

from __future__ import annotations  # noqa: I001

import dspopulations_us_birth_certificates.env_guard  # noqa: F401

import json  # noqa: E402
import os  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402
from dse_research_utils.environment import setup  # noqa: E402
from dse_research_utils.plot import styles  # noqa: E402

from dspopulations_us_birth_certificates.plot_utils import save_fig  # noqa: E402
from dspopulations_us_birth_certificates.selection import (  # noqa: E402
    inv_logit,
    latest_fit_dir,
)

OUTPUT_DIR = "notes/figures"
VARIANT_COLOUR = {"A": styles.COLOUR_BLUE, "B": styles.COLOUR_ORANGE, "C": styles.COLOUR_GREEN}


def load_year(variant: str) -> dict:
    fit_dir = latest_fit_dir(variant)
    cells = pd.read_parquet(fit_dir / "cells.parquet")
    with open(fit_dir / "config.json") as fh:
        y0 = int(json.load(fh)["year_range"][0])
    n = cells["N_cell"].to_numpy(float)
    r = cells["R_cell"].to_numpy(float)
    age = cells["age_idx"].to_numpy()
    year = cells["year_idx"].to_numpy()
    with xr.open_dataset(fit_dir / "idata.nc", group="posterior") as post:
        p = post["p_ds_lb"].values.reshape(-1, len(cells)).mean(0)
        theta = inv_logit(
            post["theta_lb_age"].values.reshape(-1, post.sizes["age"]).mean(0)
        )[age]
    n_year = int(year.max()) + 1
    rows = {}
    for y in range(n_year):
        m = year == y
        true_ct = float((p[m] * n[m]).sum())
        natural = float((theta[m] * n[m]).sum())
        rows[y0 + y] = {
            "true_ds": true_ct,
            "recorded": float(r[m].sum()),
            "reduction": 1.0 - true_ct / natural,
        }
    return {"variant": variant, "df": pd.DataFrame(rows).T}


def main() -> int:
    setup.init_script()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    data = {v: load_year(v) for v in ("A", "B", "C")}
    years = data["C"]["df"].index.to_numpy()

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(styles.FIGSIZE_LG[0], styles.FIGSIZE_LG[1] * 1.5), sharex=True
    )
    for v in ("A", "B", "C"):
        df = data[v]["df"]
        ax1.plot(years, df["true_ds"], "-o", ms=3, color=VARIANT_COLOUR[v], label=f"True DS, variant {v}")
        ax2.plot(years, df["reduction"], "-o", ms=3, color=VARIANT_COLOUR[v], label=f"Variant {v}")
    ax1.plot(years, data["C"]["df"]["recorded"], "-s", ms=3, color=styles.COLOUR_RED,
             label="Recorded on certificate")
    ax1.set_ylabel("DS livebirths")
    ax1.set_ylim(0, None)
    ax1.set_title("DS livebirths and termination by year, 2016–2024 (selection model A/B/C)")
    ax1.legend(fontsize=7, ncol=2, loc="lower left")
    ax2.set_ylabel("Termination reduction")
    ax2.set_ylim(0, None)
    ax2.set_xlabel("Year")
    ax2.legend(fontsize=7, loc="lower right")
    out = pd.concat({v: data[v]["df"] for v in data}, axis=1)
    out.index.name = "year"
    save_fig(fig, OUTPUT_DIR, "selection_ds_by_year", data=out.reset_index())
    plt.close(fig)

    print("=== true DS livebirths by year (counts) ===")
    tbl = pd.DataFrame({v: data[v]["df"]["true_ds"].round().astype(int) for v in data})
    tbl["recorded"] = data["C"]["df"]["recorded"].astype(int)
    print(tbl.to_string())
    print("\ntotals 2016-2024:", {v: int(data[v]["df"]["true_ds"].sum()) for v in data})
    print(f"wrote selection_ds_by_year to {OUTPUT_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
