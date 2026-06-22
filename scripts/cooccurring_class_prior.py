"""Stratified class-prior estimate of co-occurring-condition rates in the FULL
(recorded + missed) true-DS population -- without identifying individual missed cases.

For a clinical condition, the rate among all true DS is estimated by stratifying on the
condition and applying a recording rate to each stratum (a class-prior / inverse-
propensity estimate):

    P(cond | true DS) = (rec_present / s_present) / (rec_present/s_present + rec_absent/s_absent)
                      = rec_present / (rec_present + R * rec_absent),     R = s_present / s_absent

where rec_present / rec_absent are recorded-DS counts with / without the condition and
R is the recording-rate ratio (how much more likely a DS birth WITH the condition is to
be recorded than one without). The absolute recording level cancels -- only the ratio R
matters.

  * R = 1  -> constant-s / neutral baseline: the full-population rate equals the recorded
             rate (the unrecorded cases mirror the recorded ones).
  * R > 1  -> severe cases recorded more: the full-population rate falls BELOW the
             recorded rate, because the missed cases concentrate in the milder,
             condition-absent stratum. This is the "severity-s_g" extension.

Contrasted with the GB individual-prediction estimate, which combines the recorded
cohort with the GB-predicted-missing cohort. Because the GB flags the clinically-florid
tail (see the variant-D / over-medicalisation discussion), it INVERTS the correction --
making missed DS look more affected than recorded DS.

Figure -> notes/figures/cooccurring_class_prior (png/svg/csv).

Usage:
    python scripts/cooccurring_class_prior.py
"""

from __future__ import annotations

import os

os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import duckdb  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from dse_research_utils.environment import setup  # noqa: E402
from dse_research_utils.plot import styles  # noqa: E402

from dspopulations_us_birth_certificates.plot_utils import _save_fig  # noqa: E402

OUTPUT_DIR = "notes/figures"
DB = "data/us_births.db"
PRED_MISSING = "ds_pred_missing_14"  # C-only, demographically-blind flag (variant-D source)
R_DEFAULT = 2.0  # illustrative severity recording-rate ratio (pending a validation-based value)
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
    fig, axes = plt.subplots(1, len(CONDITIONS), figsize=(styles.FIGSIZE_LG[0] * 1.3, styles.FIGSIZE_LG[1]))
    rows = []
    for ax, (col, label) in zip(axes, CONDITIONS, strict=True):
        c = _counts(con, col)
        rec_n = c["rec_present"] + c["rec_absent"]
        pm_n = c["pm_present"] + c["pm_absent"]
        recorded = c["rec_present"] / rec_n
        gb_full = (c["rec_present"] + c["pm_present"]) / (rec_n + pm_n)
        cp = _class_prior(c["rec_present"], c["rec_absent"], r_grid)
        cp_default = float(_class_prior(c["rec_present"], c["rec_absent"], np.array([R_DEFAULT]))[0])

        ax.plot(r_grid, cp * 100, "-", color=styles.COLOUR_BLUE, lw=2,
                label="Class-prior estimate (full true-DS population)")
        ax.axhline(gb_full * 100, ls="--", color=styles.COLOUR_RED,
                   label=f"GB individual-prediction estimate ({gb_full * 100:.0f}%)")
        ax.plot([1.0], [recorded * 100], "o", color=styles.COLOUR_GREEN, ms=7,
                label=f"Recorded rate = constant-s, R=1 ({recorded * 100:.1f}%)")
        ax.axvline(R_DEFAULT, ls=":", color=styles.TEXT_COLOUR, alpha=0.6)
        ax.plot([R_DEFAULT], [cp_default * 100], "s", color=styles.COLOUR_BLUE, ms=6)
        ax.annotate(f"R={R_DEFAULT:g}: {cp_default * 100:.1f}%", (R_DEFAULT, cp_default * 100),
                    textcoords="offset points", xytext=(6, 8), fontsize=7, color=styles.COLOUR_BLUE)
        ax.set_title(f"{label}")
        ax.set_xlabel("Recording-rate ratio R = s(with) / s(without)")
        ax.set_ylabel(f"% of true DS with {label.lower()}")
        ax.set_ylim(0, max(gb_full, recorded) * 130)
        ax.legend(fontsize=6, loc="upper right")
        rows.append({
            "condition": label, "recorded_pct": round(recorded * 100, 1),
            "gb_full_pct": round(gb_full * 100, 1),
            f"classprior_R{R_DEFAULT:g}_pct": round(cp_default * 100, 1),
            "rec_present": c["rec_present"], "rec_absent": c["rec_absent"],
            "pm_present": c["pm_present"], "pm_absent": c["pm_absent"],
        })
    con.close()

    fig.suptitle("Co-occurring conditions in the full true-DS population: class prior vs GB prediction")
    df = pd.DataFrame(rows)
    _save_fig(fig, OUTPUT_DIR, "cooccurring_class_prior", data=df)
    plt.close(fig)

    pd.set_option("display.width", 160)
    print(df.to_string(index=False))
    print(f"\nR_DEFAULT={R_DEFAULT} is illustrative; a validation-based s(with)/s(without) ratio should replace it.")
    print(f"wrote cooccurring_class_prior to {OUTPUT_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
