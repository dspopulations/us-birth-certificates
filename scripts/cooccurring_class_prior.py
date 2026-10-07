"""AI-assisted documentation revision by Codex (GPT-6).

Sensitivity analysis for co-occurring conditions among true DS cases.

Uses recorded cases with known condition status and a supplied relative recording
rate R = s_present / s_absent. Under accurate condition labels and no DS false
positives, q_true = rec_present / (rec_present + R * rec_absent).

R=1 assumes equal DS recording with and without the condition. The cited validation
studies do not establish that equality. The script varies R from 1 to 3 and also
plots a classifier-cohort proportion. That cohort is not a verified missed-case
population, and the two curves cannot validate each other.

Outputs are notes/figures/cooccurring_class_prior (PNG, SVG and CSV). See
notes/20260622-predictors-bayesian-model.md for assumptions and limitations.

Usage:
    uv run python scripts/cooccurring_class_prior.py
"""

from __future__ import annotations  # noqa: I001

import dspopulations_us_birth_certificates.env_guard  # noqa: F401

import os  # noqa: E402

import duckdb  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from dse_research_utils.environment import setup  # noqa: E402
from dse_research_utils.plot import styles  # noqa: E402

from dspopulations_us_birth_certificates.plot_utils import save_fig  # noqa: E402

OUTPUT_DIR = "notes/figures"
DB = "data/us_births.db"
PRED_MISSING = "ds_pred_missing_14"  # C-only classifier excluding demographic predictors (variant-D source)
SENS_R = (1.5, 2.0)  # scenarios reported alongside the equal-recording assumption R=1
CONDITIONS = [
    ("ca_cchd", "Cyanotic CHD"),
    ("ab_nicu", "NICU admission"),
]


def _counts(con: duckdb.DuckDBPyConnection, col: str) -> dict:
    q = f"""
    SELECT
      SUM(CASE WHEN down_ind=1 AND UPPER({col})='Y' THEN 1 ELSE 0 END) AS rec_present,
      SUM(CASE WHEN down_ind=1 AND UPPER({col})='N' THEN 1 ELSE 0 END) AS rec_absent,
      SUM(CASE WHEN {PRED_MISSING} AND UPPER({col})='Y' THEN 1 ELSE 0 END) AS pm_present,
      SUM(CASE WHEN {PRED_MISSING} AND UPPER({col})='N' THEN 1 ELSE 0 END) AS pm_absent
    FROM us_births WHERE year BETWEEN 2016 AND 2024
    """
    rp, ra, pp, pa = con.execute(q).fetchone()
    return {"rec_present": rp, "rec_absent": ra, "pm_present": pp, "pm_absent": pa}


def _class_prior(rec_present: float, rec_absent: float, r: np.ndarray) -> np.ndarray:
    return rec_present / (rec_present + r * rec_absent)


def main() -> int:
    setup.init_script()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    con = duckdb.connect(DB, read_only=True)

    r_grid = np.linspace(1.0, 3.0, 60)
    fig, axes = plt.subplots(
        1, len(CONDITIONS), figsize=(styles.FIGSIZE_LG[0] * 1.3, styles.FIGSIZE_LG[1])
    )
    rows = []
    for ax, (col, label) in zip(axes, CONDITIONS, strict=True):
        c = _counts(con, col)
        rec_n = c["rec_present"] + c["rec_absent"]
        pm_n = c["pm_present"] + c["pm_absent"]
        recorded = c["rec_present"] / rec_n
        gb_full = (c["rec_present"] + c["pm_present"]) / (rec_n + pm_n)
        cp = _class_prior(c["rec_present"], c["rec_absent"], r_grid)
        cp_sens = {
            r: float(_class_prior(c["rec_present"], c["rec_absent"], np.array([r]))[0])
            for r in SENS_R
        }

        ax.plot(
            r_grid,
            cp * 100,
            "-",
            color=styles.COLOUR_BLUE,
            lw=2,
            label="Conditional co-occurrence estimate",
        )
        ax.axhline(
            gb_full * 100,
            ls="--",
            color=styles.COLOUR_RED,
            label=f"Classifier-cohort share ({gb_full * 100:.0f}%)",
        )
        ax.axvspan(
            1.0, 1.5, color=styles.TEXT_COLOUR, alpha=0.06
        )  # illustrated ratio range near R=1
        ax.plot(
            [1.0],
            [recorded * 100],
            "o",
            color=styles.COLOUR_GREEN,
            ms=8,
            label=f"R=1, equal recording assumed ({recorded * 100:.1f}%)",
        )
        ax.set_title(f"{label}")
        ax.set_xlabel("Recording-rate ratio R = s(with) / s(without)")
        ax.set_ylabel(f"Conditional % of true DS with {label.lower()}")
        ax.set_ylim(0, max(gb_full, recorded) * 130)
        ax.legend(fontsize=6, loc="upper right")
        rows.append(
            {
                "condition": label,
                "recorded_R1_pct": round(recorded * 100, 1),
                "gb_full_pct": round(gb_full * 100, 1),
                "classprior_R1.5_pct": round(cp_sens[1.5] * 100, 1),
                "classprior_R2_pct": round(cp_sens[2.0] * 100, 1),
                "rec_present": c["rec_present"],
                "rec_absent": c["rec_absent"],
                "pm_present": c["pm_present"],
                "pm_absent": c["pm_absent"],
            }
        )
    con.close()

    fig.suptitle(
        "Co-occurring conditions in the full true-DS population: class prior vs GB prediction"
    )
    df = pd.DataFrame(rows)
    save_fig(fig, OUTPUT_DIR, "cooccurring_class_prior", data=df)
    plt.close(fig)

    pd.set_option("display.width", 180)
    print(df.to_string(index=False))
    print("\nR=1 assumes equal DS recording with and without the condition.")
    print("The source studies do not estimate that within-DS ratio directly.")
    print("The curves also assume accurate condition labels and no DS false positives.")
    print(f"wrote cooccurring_class_prior to {OUTPUT_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
