# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: dspop-us-birth-certificates
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Predictions
#

# %%
import duckdb
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from dspopulations_us_birth_certificates import repl_utils
from dspopulations_us_birth_certificates.variables import Variables as vars

plt.style.use("../notebook.mplstyle")

START_TIME = datetime.now()
OUTPUT_DIR = f"output/0011-model-predictions-1/{START_TIME:%Y%m%d-%H%M%S}"

SAVE_PLOTS = True

os.makedirs(OUTPUT_DIR, exist_ok=True)

repl_utils.print_environment_info()

print(f"\n--------------------\nOutput directory: {OUTPUT_DIR}\n--------------------\n")

# %%
con = duckdb.connect("../data/us_births.db", read_only=True)

# %% [markdown]
# ## 3.1. Births predicted from 2016 to 2024
#

# %%
df = con.execute(
    f"""
    SELECT
        b.year,
        sum(b.down_ind) as down_ind,
        sum(b.p_ds_lb_pred_01) as down_pred,
        sum(b.p_ds_lb_pred_01)  / sum(b.down_ind) as ratio,
        sum(b.p_ds_lb_nt * (1 - r.reduction)) as ds_lb_est_reduc
    FROM us_births AS b
    LEFT JOIN reduction_rate_year r
        ON b.year = r.year
    WHERE b.year >= 2016
    GROUP BY b.year
    ORDER BY b.year;
    """
).df()
df

# %%
plt.figure(figsize=(7, 4))
# plt.fill_between(range(2003, 2015), 2250, color="#d8f0ff", alpha=0.4)
plt.bar(df["year"] - 0.2, df["down_ind"], width=0.4, label="DS recorded")
plt.bar(df["year"] + 0.2, df["down_pred"], width=0.4, label="DS predicted")
plt.xlim(2015.2, 2024.8)
plt.xticks(range(2016, 2025))
plt.xlabel("Year")
plt.ylabel("Number of Births")
plt.title(
    "Live births of babies with Down syndrome recorded as confirmed or pending vs predicted"
)
plt.legend(bbox_to_anchor=(0.9, 0.94), loc="center right")
if SAVE_PLOTS:
    plt.savefig(
        os.path.join(OUTPUT_DIR, "births_recorded_predicted.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.savefig(
        os.path.join(OUTPUT_DIR, "births_recorded_predicted.svg"), bbox_inches="tight"
    )
plt.show()

# %%
df = con.execute(
    f"""
    SELECT
        b.year,
        SUM(b.down_ind) AS down_ind,
        (SELECT COUNT(*) FROM us_births WHERE year = b.year AND p_ds_lb_pred_01 >= 0.029) AS down_pred,
        SUM(b.p_ds_lb_nt * (1 - r.reduction)) AS ds_lb_est_reduc
    FROM us_births AS b
    LEFT JOIN reduction_rate_year r
        ON b.year = r.year
    WHERE b.year >= 2016
    GROUP BY b.year
    ORDER BY b.year;
    """
).df()
df

# %%
df = con.execute(
    f"""
    WITH year_quota AS (
        SELECT
            year,
            COUNT(*) AS n_recorded,
            CAST(CEIL(COUNT(*) * 2.5) AS BIGINT) AS n_select
        FROM us_births
        WHERE down_ind = 1
        GROUP BY year
    ),
    ranked AS (
        SELECT
            b.*,
            q.n_recorded,
            q.n_select,
            ROW_NUMBER() OVER (
                PARTITION BY b.year
                ORDER BY b.p_ds_lb_pred_01 DESC
            ) AS rn
        FROM us_births AS b
        JOIN year_quota AS q
        ON q.year = b.year
        WHERE b.p_ds_lb_pred_01 IS NOT NULL
    ),
    selected AS (
        SELECT *
        FROM ranked
        WHERE rn <= n_select
        ORDER BY year, rn
    )
    SELECT
        s.year,
        (SELECT COUNT(*) FROM us_births WHERE year = s.year AND down_ind = 1) AS down_ind_total,
        SUM(down_ind) AS down_ind_sel,
        SUM(down_ind) / (SELECT COUNT(*) FROM us_births WHERE year = s.year AND down_ind = 1) AS down_ind_ratio,
        COUNT(*) AS down_pred
    FROM selected AS s
    LEFT JOIN reduction_rate_year r
        ON s.year = r.year
    WHERE s.year >= 2016
    GROUP BY s.year
    ORDER BY s.year;
    """
).df()
df

# %%

# %%
df = con.execute(
    f"""
    WITH year_month_quota AS (
        SELECT
            year,
            dob_mm,
            COUNT(*) AS n_recorded,
            CAST(CEIL(COUNT(*) * 1.5) AS BIGINT) AS n_select
        FROM us_births
        WHERE down_ind = 1 AND year >= 2016
        GROUP BY year, dob_mm
    ),
    ranked AS (
        SELECT
            b.*,
            q.n_recorded,
            q.n_select,
            ROW_NUMBER() OVER (
                PARTITION BY b.year, b.dob_mm
                ORDER BY b.p_ds_lb_pred_01 DESC
            ) AS rn
        FROM us_births AS b
        JOIN year_month_quota AS q
        ON q.year = b.year AND q.dob_mm = b.dob_mm
        WHERE down_ind = 0
    ),
    missing AS (
        SELECT *
        FROM ranked
        WHERE rn <= n_select
        ORDER BY year, dob_mm, rn
    )
    SELECT
        b.mage_c,
        (SELECT COUNT(*) FROM us_births WHERE mage_c = b.mage_c AND down_ind = 1 AND year >= 2016) as ds_births_recorded,
        COUNT(m.rn) as ds_births_missing,
        (SELECT COUNT(*) FROM us_births WHERE mage_c = b.mage_c AND down_ind = 1 AND year >= 2016) + COUNT(m.rn) as ds_births_total,
        SUM(b.p_ds_lb_nt * (1 - r.reduction)) as ds_lb_est_reduc
    FROM us_births AS b
    FULL OUTER JOIN missing AS m
    ON b.id = m.id
    LEFT JOIN reduction_rate_year r
        ON b.year = r.year
    WHERE b.year >= 2016
    GROUP BY b.mage_c,
    ORDER BY b.mage_c;
    """
).df()
df

# %%
plt.figure(figsize=(7, 4))
plt.plot(df["mage_c"], df["ds_births_recorded"], marker="o", label="DS recorded")
plt.plot(
    df["mage_c"],
    df["ds_births_total"],
    marker="o",
    label="DS recorded + predicted missing",
)
plt.plot(
    df["mage_c"],
    df["ds_lb_est_reduc"],
    marker="o",
    label="DS estimated (given age and reduction)",
)
plt.plot(df["mage_c"], df["ds_births_recorded"] * 2.5, marker="o", label="DS recorded x 2.5 (+ 150%)")

plt.xlabel("Maternal Age (years)")
plt.ylabel("Number of Births")
plt.legend()
plt.show()

# %%
df = con.execute(
    f"""
    WITH year_month_quota AS (
        SELECT
            year,
            dob_mm,
            COUNT(*) AS n_recorded,
            CAST(CEIL(COUNT(*) * 1.5) AS BIGINT) AS n_select
        FROM us_births
        WHERE down_ind = 1 AND year >= 2016
        GROUP BY year, dob_mm
    ),
    ranked AS (
        SELECT
            b.*,
            q.n_recorded,
            q.n_select,
            ROW_NUMBER() OVER (
                PARTITION BY b.year, b.dob_mm
                ORDER BY b.p_ds_lb_pred_01 DESC
            ) AS rn
        FROM us_births AS b
        JOIN year_month_quota AS q
        ON q.year = b.year AND q.dob_mm = b.dob_mm
        WHERE down_ind = 0
    ),
    missing AS (
        SELECT *
        FROM ranked
        WHERE rn <= n_select
        ORDER BY year, dob_mm, rn
    )
    SELECT
        b.mracehisp,
        (SELECT COUNT(*) FROM us_births WHERE mracehisp = b.mracehisp AND down_ind = 1 AND year >= 2016) as ds_births_recorded,
        COUNT(m.rn) as ds_births_missing,        
        COUNT(m.rn) / (SELECT COUNT(*) FROM us_births WHERE mracehisp = b.mracehisp AND down_ind = 1 AND year >= 2016) as ds_births_missing_ratio,
        (SELECT COUNT(*) FROM us_births WHERE mracehisp = b.mracehisp AND down_ind = 1 AND year >= 2016) + COUNT(m.rn) as ds_births_total,
        SUM(b.p_ds_lb_nt * (1 - r.reduction)) as ds_lb_est_reduc
    FROM us_births AS b
    FULL OUTER JOIN missing AS m
    ON b.id = m.id
    LEFT JOIN reduction_rate_year r
        ON b.year = r.year
    WHERE b.year >= 2016
    GROUP BY b.mracehisp
    ORDER BY b.mracehisp;
    """
).df()
df

# %%
x_labels = {
    1: "Non-Hispanic White",
    2: "Non-Hispanic Black",
    3: "Non-Hispanic AIAN",
    4: "Non-Hispanic Asian",
    5: "Non-Hispanic NHOPI",
    6: "Non-Hispanic more than one race",
    7: "Hispanic",
    8: "Origin unknown or not stated",
}

plt.figure(figsize=(7, 6))
plt.bar(
    df["mracehisp"] - 0.3, df["ds_births_recorded"], width=0.3, label="Births recorded"
)
plt.bar(
    df["mracehisp"] - 0.3,
    df["ds_births_missing"],
    width=0.3,
    bottom=df["ds_births_recorded"],
    label="Births predicted missing",
)
plt.bar(
    df["mracehisp"] + 0.3, df["ds_lb_est_reduc"], width=0.3, label="DS estimated (given age and reduction)"
)
plt.bar(
    df["mracehisp"], df["ds_births_recorded"] * 2.5, width=0.3, label="Births recorded x 2.5 (+ 150%)"
)

plt.xlabel("mracehisp")
plt.ylabel("Number of Births")
plt.xticks(
    df["mracehisp"],
    [x_labels.get(x, "Unknown") for x in df["mracehisp"]],
    rotation=30,
    ha="right",
)
plt.legend()
plt.show()

# %%
df = con.execute(
    f"""
    WITH year_month_quota AS (
        SELECT
            year,
            dob_mm,
            COUNT(*) AS n_recorded,
            CAST(CEIL(COUNT(*) * 1.5) AS BIGINT) AS n_select
        FROM us_births
        WHERE down_ind = 1 AND year >= 2016
        GROUP BY year, dob_mm
    ),
    ranked AS (
        SELECT
            b.*,
            q.n_recorded,
            q.n_select,
            ROW_NUMBER() OVER (
                PARTITION BY b.year, b.dob_mm
                ORDER BY b.p_ds_lb_pred_01 DESC
            ) AS rn
        FROM us_births AS b
        JOIN year_month_quota AS q
        ON q.year = b.year AND q.dob_mm = b.dob_mm
        WHERE down_ind = 0
    ),
    missing AS (
        SELECT *
        FROM ranked
        WHERE rn <= n_select
        ORDER BY year, dob_mm, rn
    )
    SELECT
        b.meduc,
        (SELECT COUNT(*) FROM us_births WHERE meduc = b.meduc AND down_ind = 1 AND year >= 2016) as ds_births_recorded,
        COUNT(m.rn) as ds_births_missing,        
        COUNT(m.rn) / (SELECT COUNT(*) FROM us_births WHERE meduc = b.meduc AND down_ind = 1 AND year >= 2016) as ds_births_missing_ratio,
        (SELECT COUNT(*) FROM us_births WHERE meduc = b.meduc AND down_ind = 1 AND year >= 2016) + COUNT(m.rn) as ds_births_total,
        SUM(b.p_ds_lb_nt * (1 - r.reduction)) as ds_lb_est_reduc
    FROM us_births AS b
    FULL OUTER JOIN missing AS m
    ON b.id = m.id
    LEFT JOIN reduction_rate_year r
        ON b.year = r.year
    WHERE b.year >= 2016
    GROUP BY b.meduc
    ORDER BY b.meduc;
    """
).df()
df

# %%
df

# %%
x_labels = {
    1: "8th grade or less",
    2: "9th through 12th grade with no diploma",
    3: "High school graduate or GED completed",
    4: "Some college credit, but not a degree",
    5: "Associate degree",
    6: "Bachelor's degree",
    7: "Master's degree",
    8: "Doctorate or professional degree",
    9: "Unknown",
}

plt.figure(figsize=(9, 6))
plt.bar(
    df["meduc"] - 0.3, df["ds_births_recorded"], width=0.3, label="Births recorded"
)
plt.bar(
    df["meduc"] - 0.3,
    df["ds_births_missing"],
    width=0.3,
    bottom=df["ds_births_recorded"],
    label="Births predicted missing",
)
plt.bar(
    df["meduc"] + 0.3, df["ds_lb_est_reduc"], width=0.3, label="DS estimated (given age and reduction)"
)
plt.bar(
    df["meduc"], df["ds_births_recorded"] * 2.5, width=0.3, label="Births recorded x 2.5 (+ 150%)"
)

plt.xlabel("meduc")
plt.ylabel("Number of Births")
plt.xticks(
    df["meduc"],
    [x_labels.get(x, "Unknown") for x in df["meduc"]],
    rotation=30,
    ha="right",
)
plt.legend()
plt.show()

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1000 bins: [0.000, 0.001), [0.001, 0.002), ..., [0.999, 1.000]
bins_df = con.execute(
    """
    SELECT
        CAST(FLOOR(p_ds_lb_pred_01 * 1000) AS INTEGER) AS bin_idx,
        COUNT(*) AS n
    FROM us_births
    WHERE year >= 2016
      AND down_ind = 0
    GROUP BY 1
    ORDER BY 1
"""
).df()

# Ensure all bins 0..1000 exist (bin_idx=1000 would only happen if p==1.0 exactly)
all_bins = pd.DataFrame({"bin_idx": np.arange(0, 1001, dtype=int)})
bins_df = all_bins.merge(bins_df, on="bin_idx", how="left").fillna({"n": 0})
bins_df["n"] = bins_df["n"].astype(np.int64)

# Bin left-edge as the plotted x (e.g., 0.000, 0.001, 0.002, ...)
bins_df["p_left"] = bins_df["bin_idx"] / 1000.0

plt.figure(figsize=(8, 4))
plt.bar(bins_df["p_left"], bins_df["n"], width=0.001, align="edge")
plt.xlabel("Predicted probability (binned to 0.001)")
plt.ylabel("Count")
plt.title("Distribution of p_ds_lb_pred_01 (year >= 2016; non-recorded DS births)")
plt.tight_layout()
plt.show()

# Often useful for highly skewed distributions:
plt.figure(figsize=(8, 4))
plt.bar(bins_df["p_left"], bins_df["n"], width=0.001, align="edge")
plt.yscale("log")
plt.xlabel("Predicted probability (binned to 0.001)")
plt.ylabel("Count (log scale)")
plt.title("Distribution of p_ds_lb_pred_01 (year >= 2016; non-recorded DS births; log-count y-axis)")
plt.tight_layout()
plt.show()

# %%

# %%

# %%

# %%

# %%
con.close()

# %%
