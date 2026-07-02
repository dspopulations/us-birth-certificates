"""Triangulate the 2016-2024 true DS livebirth total across estimates.

Compares the structural selection model (variants A/B/C), the GB-calibrated variant D
(recording off, fit to the C-only demographically-blind probability-summed total), the
project's earlier prevalence model, the recorded count, and external surveillance.
Each fit's total is computed as sum(theta*eta*N) over its cells; the prevalence total
from the prevalence_year table; surveillance is an external published figure. Writes
notes/figures/totals_triangulation (png/svg/csv).

Usage:
    python scripts/totals_triangulation.py
"""

from __future__ import annotations  # noqa: I001

import dspopulations_us_birth_certificates.env_guard  # noqa: F401

import os  # noqa: E402

import duckdb  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402
from dse_research_utils.environment import setup  # noqa: E402
from dse_research_utils.plot import styles  # noqa: E402

from dspopulations_us_birth_certificates.plot_utils import save_fig  # noqa: E402
from dspopulations_us_birth_certificates.selection import latest_fit_dir  # noqa: E402

OUTPUT_DIR = "notes/figures"
SURVEILLANCE = 48000  # de Graaf et al., external published estimate (approximate)


def total_true(variant: str) -> float:
    fit = latest_fit_dir(variant)
    cells = pd.read_parquet(fit / "cells.parquet")
    n = cells["N_cell"].to_numpy(float)
    with xr.open_dataset(fit / "idata.nc", group="posterior") as post:
        p = post["p_ds_lb"].values.reshape(-1, len(cells)).mean(0)
    return float((p * n).sum())


def main() -> int:
    setup.init_script()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    con = duckdb.connect("data/us_births.db", read_only=True)
    prev = con.execute(
        """SELECT SUM(pv.p_ds_lb_wt * c.births) FROM
           (SELECT year, COUNT(*) births FROM us_births WHERE year BETWEEN 2016 AND 2024 GROUP BY year) c
           JOIN prevalence_year pv ON pv.year = c.year"""
    ).fetchone()[0]
    con.close()

    blue, orange, green, grey = (
        styles.COLOUR_BLUE, styles.COLOUR_ORANGE, styles.COLOUR_GREEN, styles.TEXT_COLOUR
    )
    tot = {v: total_true(v) for v in ("A", "B", "C", "D")}
    rows = [
        ("Recorded (certificate)", 17776, grey, "s=1.00"),
        ("Variant D — GB-calibrated", tot["D"], orange, "s≈0.69"),
        ("Structural C (pin recording)", tot["C"], blue, "s≈0.40"),
        ("Structural A (pin recording)", tot["A"], blue, "s≈0.40"),
        ("Previous prevalence model", float(prev), green, ""),
        ("Structural B (free recording)", tot["B"], blue, "s≈0.32"),
        ("Surveillance (de Graaf, ~)", SURVEILLANCE, green, ""),
    ]
    rows.sort(key=lambda x: x[1])
    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    cols = [r[2] for r in rows]
    notes = [r[3] for r in rows]

    fig, ax = plt.subplots(figsize=styles.FIGSIZE_LG)
    y = np.arange(len(rows))
    ax.barh(y, vals, color=cols, alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    for yi, (v, nt) in enumerate(zip(vals, notes, strict=True)):
        ax.text(v + 600, yi, f"{v:,.0f}" + (f"  ({nt})" if nt else ""), va="center", fontsize=7)
    ax.axvspan(min(tot["A"], tot["C"]), tot["B"], color=blue, alpha=0.06)
    ax.set_xlabel("Estimated true DS livebirths, 2016–2024")
    ax.set_title("Triangulating the DS livebirth total across methods")
    ax.set_xlim(0, max(vals) * 1.18)
    save_fig(fig, OUTPUT_DIR, "totals_triangulation",
              data=pd.DataFrame({"estimate": labels, "total": vals, "note": notes}))
    plt.close(fig)

    print(pd.DataFrame({"estimate": labels, "total": [round(v) for v in vals]}).to_string(index=False))
    print(f"wrote totals_triangulation to {OUTPUT_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
