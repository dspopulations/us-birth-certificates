"""DS livebirths by year (1989-2024) from the previous prevalence/reduction model.

Longer-term context for the 2016-2024 selection model. Reads the project's earlier
``prevalence_year`` (estimated true DS livebirth rate) and ``reduction_rate_year``
tables plus per-year birth/recorded counts from ``us_births``, and plots estimated-true
vs recorded DS livebirths by year. That earlier model is prevalence-based and spans
1989-2024; its prevalence is held flat from 2018 (extrapolated), so 2019-2024 estimated
counts move only with the declining birth total. Writes
``notes/figures/previous_model_ds_by_year`` (png/svg/csv).

Usage:
    python scripts/previous_model_yearly.py
"""

from __future__ import annotations  # noqa: I001

import dspopulations_us_birth_certificates.env_guard  # noqa: F401

import os  # noqa: E402

import duckdb  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from dse_research_utils.environment import setup  # noqa: E402
from dse_research_utils.plot import styles  # noqa: E402

from dspopulations_us_birth_certificates.plot_utils import save_fig  # noqa: E402

OUTPUT_DIR = "notes/figures"
DB = "data/us_births.db"
EXTRAP_FROM = 2018  # prevalence held flat from here in the previous model

QUERY = """
WITH t AS (
    SELECT year, COUNT(*) AS births,
           SUM(CASE WHEN down_ind = 1 THEN 1 ELSE 0 END) AS recorded
    FROM us_births WHERE year IS NOT NULL GROUP BY year
)
SELECT t.year, t.births, t.recorded, pv.p_ds_lb_wt,
       pv.p_ds_lb_wt * t.births AS est_ds_livebirths, rr.reduction
FROM t
JOIN prevalence_year pv ON pv.year = t.year
LEFT JOIN reduction_rate_year rr ON rr.year = t.year
ORDER BY t.year
"""


def main() -> int:
    setup.init_script()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    con = duckdb.connect(DB, read_only=True)
    df = con.execute(QUERY).fetchdf()
    con.close()

    yr = df["year"].to_numpy()
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(styles.FIGSIZE_LG[0], styles.FIGSIZE_LG[1] * 1.5), sharex=True
    )
    ax1.plot(yr, df["est_ds_livebirths"], "-o", ms=3, color=styles.COLOUR_BLUE,
             label="Estimated true (prevalence model)")
    ax1.plot(yr, df["recorded"], "-s", ms=3, color=styles.COLOUR_RED, label="Recorded on certificate")
    ax1.set_ylabel("DS livebirths")
    ax1.set_ylim(0, None)
    ax1.set_title("DS livebirths and termination by year, 1989–2024 (previous prevalence model)")
    ax1.legend(fontsize=8, loc="lower right")

    ax2.plot(yr, df["reduction"], "-^", ms=3, color=styles.COLOUR_GREEN,
             label="Termination reduction (share not born alive)")
    ax2.set_ylabel("Termination reduction")
    ax2.set_ylim(0, None)
    ax2.set_xlabel("Year")
    ax2.legend(fontsize=8, loc="lower right")

    for ax in (ax1, ax2):
        ax.axvspan(EXTRAP_FROM + 0.5, yr.max() + 0.5, color=styles.TEXT_COLOUR, alpha=0.08)
    ax1.text(EXTRAP_FROM + 1.2, ax1.get_ylim()[1] * 0.08,
             "extrapolated\n(prevalence flat,\nreduction linear)",
             fontsize=7, color=styles.TEXT_COLOUR)
    save_fig(fig, OUTPUT_DIR, "previous_model_ds_by_year", data=df)
    plt.close(fig)

    peak = df.loc[df["est_ds_livebirths"].idxmax()]
    print(df[["year", "recorded", "est_ds_livebirths", "reduction"]].to_string(index=False))
    print(f"\nestimated-true peak: {peak['est_ds_livebirths']:,.0f} in {int(peak['year'])}")
    sel = df[(df.year >= 2016) & (df.year <= 2024)]
    print(f"2016-2024 estimated-true total: {sel['est_ds_livebirths'].sum():,.0f}")
    print(f"wrote previous_model_ds_by_year to {OUTPUT_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
