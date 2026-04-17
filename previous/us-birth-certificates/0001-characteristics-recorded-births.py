# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
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
# # 3.1 Characteristics of recorded births

# %%
import duckdb
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

import repl_utils
from variables import Variables as vars

plt.style.use("../../notebook.mplstyle")

START_TIME = datetime.now()
OUTPUT_DIR = f"output/0001-characteristics-recorded-births/{START_TIME:%Y%m%d-%H%M%S}"

SAVE_PLOTS = True

os.makedirs(OUTPUT_DIR, exist_ok=True)

repl_utils.print_environment_info()

print(f"\n--------------------\nOutput directory: {OUTPUT_DIR}\n--------------------\n")

# %%
con = duckdb.connect("../../data/us_births.db", read_only=True)

# %%
df = con.execute(
    f"""
    SELECT
        b.no_congen,
        AVG(b.down_ind) AS down_ind,
    FROM us_births AS b
    WHERE b.year >= 2016
    GROUP BY b.no_congen
    ORDER BY b.no_congen
    """
).df()
df

# %%
df = con.execute(
    f"""
    SELECT
        b.down_ind,
        SUM(b.no_congen) AS sum_no_congen,
    FROM us_births AS b
    WHERE b.year >= 2016
    GROUP BY b.down_ind
    ORDER BY b.down_ind
    """
).df()
df

# %% [markdown]
# ## 3.1. Births recorded from 1989 to 2024

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
plt.figure(figsize=(7,4))
# plt.fill_between(range(2003, 2015), 2250, color="#d8f0ff", alpha=0.4)
plt.bar(df["year"], df["pending"], bottom=df["confirmed"], label="DS diagnosis pending")
plt.bar(df["year"], df["confirmed"], label="DS diagnosis confirmed")
plt.xlim(1988.2, 2024.8)
plt.xticks(range(1990, 2025, 2), rotation=45)
plt.xlabel("Year")
plt.ylabel("Number of Births")
# plt.title("Live births of babies with Down syndrome recorded as confirmed or pending")
plt.legend(bbox_to_anchor=(0.03, 0.2), loc="center left")
if SAVE_PLOTS:
    plt.savefig(os.path.join(OUTPUT_DIR, "births_confirmed_pending.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTPUT_DIR, "births_confirmed_pending.svg"), bbox_inches="tight")
plt.show()

# %%
plt.figure(figsize=(10, 5))
plt.fill_between(range(2003, 2015), 0.36, 0.43, color="#d8f0ff", alpha=0.4)
plt.plot(df["year"], ((df["pending"] + df["confirmed"]) / df["ds_lb_est"]), marker="o", color="#33a066", label="40% of estimated live births (annual reduction rates)")
plt.xlim(1988.2, 2024.8)
plt.xticks(range(1989, 2025), rotation=45)
plt.xlabel("Year")
plt.ylabel("Proportion of births")
plt.title("Proportion of expected live births of babies with Down syndrome recorded as confirmed or pending")
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
plt.figure(figsize=(7,4))
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
plt.figure(figsize=(7,4))
# plt.fill_between(range(2003, 2015), 2250, color="#d8f0ff", alpha=0.4)
plt.bar(df["year"], df["pending"], bottom=df["confirmed"], label="DS diagnosis pending")
plt.bar(df["year"], df["confirmed"], label="DS diagnosis confirmed")
plt.plot(df["year"], (df["ds_lb_est"] * 0.4), marker="o", color="#33a066", label="40% of estimated live births (annual reduction rates)")
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
