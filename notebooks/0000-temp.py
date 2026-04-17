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
OUTPUT_DIR = f"output/0002-predictors/{START_TIME:%Y%m%d-%H%M%S}"

SAVE_PLOTS = True

os.makedirs(OUTPUT_DIR, exist_ok=True)

repl_utils.print_environment_info()

print(f"\n--------------------\nOutput directory: {OUTPUT_DIR}\n--------------------\n")

# %%
con = duckdb.connect("../data/us_births.db", read_only=True)

# %%
df = con.execute(
    """
    SELECT
        no_abnorm,
        ca_down_c,
        ca_cchd,
        count(*)
    FROM us_births
    WHERE year >= 2016 
    GROUP BY no_abnorm, ca_down_c, ca_cchd
    ORDER BY no_abnorm, ca_down_c, ca_cchd
    """
).df()
df.T

# %%
df = con.execute(
    """
    SELECT
        no_congen,
        down_ind,
        count(*)
    FROM us_births
    WHERE year >= 2016 
    GROUP BY no_congen, down_ind
    ORDER BY no_congen, down_ind
    """
).df()
df.T

# %%
df = con.execute(
    """
    SELECT
        no_congen,
        ca_down_c,
        ca_cchd,
        count(*)
    FROM us_births
    WHERE year >= 2016 
    GROUP BY no_congen, ca_down_c, ca_cchd
    ORDER BY no_congen, ca_down_c, ca_cchd
    """
).df()
df.T

# %%
df = con.execute(
    """
    SELECT
        year,
        AVG(CASE WHEN no_congen IS NULL THEN 1 ELSE 0 END) AS no_congen,
        AVG(CASE WHEN ab_nicu IS NULL THEN 1 ELSE 0 END) AS ab_nicu,
        AVG(CASE WHEN dbwt IS NULL THEN 1 ELSE 0 END) AS dbwt,
        AVG(CASE WHEN ab_aven1 IS NULL THEN 1 ELSE 0 END) AS ab_aven1,
        AVG(CASE WHEN ab_aven6 IS NULL THEN 1 ELSE 0 END) AS ab_aven6,
        AVG(CASE WHEN ab_surf IS NULL THEN 1 ELSE 0 END) AS ab_surf,
        AVG(CASE WHEN ab_anti IS NULL THEN 1 ELSE 0 END) AS ab_anti,
        AVG(CASE WHEN ab_seiz IS NULL THEN 1 ELSE 0 END) AS ab_seiz,
        AVG(CASE WHEN ca_disor IS NULL THEN 1 ELSE 0 END) AS ca_disor,
        AVG(CASE WHEN ca_cchd IS NULL THEN 1 ELSE 0 END) AS ca_cchd,
        AVG(CASE WHEN gestrec10 IS NULL THEN 1 ELSE 0 END) AS gestrec10,
        AVG(CASE WHEN no_abnorm IS NULL THEN 1 ELSE 0 END) AS no_abnorm,
        AVG(CASE WHEN ca_anen IS NULL THEN 1 ELSE 0 END) AS ca_anen,
        AVG(CASE WHEN ca_mnsb IS NULL THEN 1 ELSE 0 END) AS ca_mnsb,
        AVG(CASE WHEN ca_cdh IS NULL THEN 1 ELSE 0 END) AS ca_cdh,
        AVG(CASE WHEN ca_omph IS NULL THEN 1 ELSE 0 END) AS ca_omph,
        AVG(CASE WHEN ca_gast IS NULL THEN 1 ELSE 0 END) AS ca_gast,
        AVG(CASE WHEN ca_limb IS NULL THEN 1 ELSE 0 END) AS ca_limb,
        AVG(CASE WHEN ca_cleft IS NULL THEN 1 ELSE 0 END) AS ca_cleft,
        AVG(CASE WHEN ca_hypo IS NULL THEN 1 ELSE 0 END) AS ca_hypo,
        AVG(CASE WHEN ca_clpal IS NULL THEN 1 ELSE 0 END) AS ca_clpal,
        AVG(CASE WHEN rf_pdiab IS NULL THEN 1 ELSE 0 END) AS rf_pdiab
    FROM us_births
    WHERE year >= 2005
    GROUP BY year
    ORDER BY year;
    """
).df()
df

# %%
plt.figure(figsize=(6, 5))
plt.plot(df["year"], df["no_congen"], label="no_congen missing")
plt.plot(df["year"], df["ab_nicu"], label="ab_nicu missing")
plt.plot(df["year"], df["ca_disor"], label="ca_disor missing")
plt.plot(df["year"], df["ca_cchd"], label="ca_cchd missing")
plt.xticks(df["year"], rotation=45)
plt.ylabel("Proportion missing")
plt.title("Proportion of missing no_congen values")
plt.legend()
plt.show()

# %%

# %% [markdown]
# ## Recorded DS births
#

# %%
df = con.execute(
    f"""
    WITH base AS (
    SELECT
        b.year,
        ca_down_c,
        b.p_ds_lb_nt * (1 - r.reduction) as ds_lb_est
    FROM us_births AS b
    LEFT JOIN reduction_rate_year r
        ON b.year = r.year
    WHERE b.year >= 1989
    )
    SELECT
        year,
        COUNT(*) FILTER (WHERE ca_down_c = 'C') AS confirmed,
        COUNT(*) FILTER (WHERE ca_down_c = 'P') AS pending,
        COUNT(*) FILTER (WHERE ca_down_c = 'N') AS no,
        COUNT(*) FILTER (WHERE ca_down_c = 'U') AS unknown,
        COUNT(*) FILTER (WHERE ca_down_c IS NULL) AS missing,
        COUNT(*) AS total,
        SUM(ds_lb_est) AS ds_lb_est
    FROM base
    GROUP BY year
    ORDER BY year;
    """
).df()
df

# %%
plt.figure(figsize=(7, 4))
# plt.fill_between(range(2003, 2015), 2250, color="#d8f0ff", alpha=0.4)
plt.bar(df["year"], df["pending"], bottom=df["confirmed"], label="DS diagnosis pending")
plt.bar(df["year"], df["confirmed"], label="DS diagnosis confirmed")
plt.plot(
    df["year"],
    (df["ds_lb_est"] * 0.4),
    marker="o",
    color="#33a066",
    label="40% of estimated live births (annual reduction rates)",
)
plt.xlim(1988.2, 2024.8)
plt.xticks(range(1990, 2025, 2), rotation=45)
plt.xlabel("Year")
plt.ylabel("Number of Births")
# plt.title("Live births of babies with Down syndrome recorded as confirmed or pending")
plt.legend(bbox_to_anchor=(0.03, 0.2), loc="center left")
plt.show()

# %%
plt.figure(figsize=(10, 5))
plt.fill_between(range(2003, 2015), 0.36, 0.43, color="#d8f0ff", alpha=0.4)
plt.plot(
    df["year"],
    ((df["pending"] + df["confirmed"]) / df["ds_lb_est"]),
    marker="o",
    color="#33a066",
    label="40% of estimated live births (annual reduction rates)",
)
plt.xlim(1988.2, 2024.8)
plt.xticks(range(1989, 2025), rotation=45)
plt.xlabel("Year")
plt.ylabel("Proportion of births")
plt.title(
    "Proportion of expected live births of babies with Down syndrome recorded as confirmed or pending"
)
plt.legend()
plt.show()

# %%
plt.figure(figsize=(6, 4))
plt.fill_between(range(2004, 2015), 59900, color="#d8f0ff", alpha=0.4)
plt.plot(df["year"], df["unknown"], marker="o", label="Unknown")
plt.xlim(1988.5, 2024.5)
plt.xticks(range(1989, 2025, 2), rotation=45)
plt.xlabel("Year")
plt.ylabel("Number of Births")
plt.title("Live births of babies with Down syndrome recorded as unknown")
plt.legend()
plt.show()

# %%
plt.figure(figsize=(6, 4))
plt.plot(df["year"], df["no"], marker="o", label="Not DS")
plt.plot(df["year"], df["total"], marker="o", label="Total births")
plt.fill_between(range(2003, 2015), 4500000, color="#d8f0ff", alpha=0.4)
plt.xlim(1988.5, 2024.5)
plt.xticks(range(1989, 2025, 2), rotation=45)
plt.xlabel("Year")
plt.ylim(3250000, 4500000)
plt.ylabel("Number of Births (millions)")
plt.title("Live births of babies with Down syndrome recorded as no")
plt.legend()
plt.show()

# %%
plt.figure(figsize=(7, 4))
# plt.fill_between(range(2003, 2015), 2250, color="#d8f0ff", alpha=0.4)
plt.bar(df["year"], df["pending"], bottom=df["confirmed"], label="DS diagnosis pending")
plt.bar(df["year"], df["confirmed"], label="DS diagnosis confirmed")
plt.xlim(1988.2, 2024.8)
plt.xticks(range(1990, 2025, 2), rotation=45)
plt.xlabel("Year")
plt.ylabel("Number of Births")
# plt.title("Live births of babies with Down syndrome recorded as confirmed or pending")
plt.legend(bbox_to_anchor=(0.35, 0.25))
plt.show()

# %%
plt.figure(figsize=(7, 4))
# plt.fill_between(range(2003, 2015), 2250, color="#d8f0ff", alpha=0.4)
plt.bar(df["year"], df["pending"], bottom=df["confirmed"], label="DS diagnosis pending")
plt.bar(df["year"], df["confirmed"], label="DS diagnosis confirmed")
plt.plot(
    df["year"],
    (df["ds_lb_est"] * 0.4),
    marker="o",
    color="#33a066",
    label="40% of estimated live births (annual reduction rates)",
)
plt.xlim(1988.2, 2024.8)
plt.xticks(range(1990, 2025, 2), rotation=45)
plt.xlabel("Year")
plt.ylabel("Number of Births")
# plt.title("Live births of babies with Down syndrome recorded as confirmed or pending")
plt.legend(bbox_to_anchor=(0.03, 0.2), loc="center left")
plt.show()

# %%
con.close()

# %%
