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
# # Notes - using race and ethnicity for prevalence estimation
#

# %%
import duckdb
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from graphviz import Digraph
from sympy.physics.units import magnetic_density
from dspopulations_us_birth_certificates.variables import Variables as vars

plt.style.use("../notebook.mplstyle")

os.makedirs("./output", exist_ok=True)

# %%
con = duckdb.connect("../data/us_births.db")  # , read_only=True)

# %%
prev_est_df = pd.read_csv(
    "./us-births-estimated-prevalence-ethnicity-2000-2018.csv"
).convert_dtypes()
prev_est_df

# %%
con.execute("DROP TABLE IF EXISTS us_births_est_prevalence_ethnicity;")
con.execute(
    """
CREATE TABLE us_births_est_prevalence_ethnicity AS
SELECT * FROM prev_est_df
"""
)

# %%
con.execute(
    """
    SELECT 
    * FROM us_births_est_prevalence_ethnicity;
    """
).df()

# %%
year_counts_df = con.execute(
    """
    SELECT 
        b.year,
        sum(b.p_ds_lb_wt) as p_ds_lb_wt,
        sum(e.prevalence / 10000.0) as p_ds_lb_wt_eth,
        sum(b.p_ds_lb_nt) as p_ds_lb_nt,
        sum(down_ind)
    FROM us_births b 
        JOIN us_births_est_prevalence_ethnicity e 
            ON b.year = e.year AND b.mracehisp_c = e.mracehisp_c
    GROUP BY b.year;
    """
).df()
year_counts_df.to_csv(f"./output/us_births_prevalence_estimates_year.csv", index=False)
year_counts_df

# %%
year_counts_df.describe()

# %%
# plot bar chart of p_ds_lb_wt and prev by year
fig, axs = plt.subplots(figsize=(10, 5))

x = np.arange(len(year_counts_df["year"]))  # numeric positions for each year
width = 0.4  # width of each bar

axs.bar(
    x - width / 2,
    year_counts_df["p_ds_lb_wt"],
    width=width,
    label="Surveillance data, given year",
)
axs.bar(
    x + width / 2,
    year_counts_df["p_ds_lb_wt_eth"],
    width=width,
    label="Surveillance data, given race/ethnicity",
)
axs.set_ylabel("Estimated DS live births")
axs.set_title("Estimated DS live births by estimation method and year")
axs.set_xticks(x)
axs.set_xticklabels(year_counts_df["year"])
axs.legend()

# %%
prev_est_year_race_df = con.execute(
    """
    SELECT *
    FROM (
        SELECT 
            b.year,
            b.mracehisp_c,
            SUM(b.p_ds_lb_wt) AS prev_est_year
        FROM us_births b
        JOIN us_births_est_prevalence_ethnicity e
        ON b.year = e.year 
        AND b.mracehisp_c = e.mracehisp_c
        GROUP BY b.year, b.mracehisp_c
    )
    PIVOT (
        SUM(prev_est_year) 
        FOR mracehisp_c IN (
            '1' AS nh_white,
            '2' AS nh_black,
            '3' AS ai_an,
            '4' AS nh_asian_pi,
            '5' AS hispanic
        )
    )
    ORDER BY year;
    """
).df()

prev_est_year_race_df

# %%
prev_est_ethn_race_df = con.execute(
    """
    SELECT *
    FROM (
        SELECT 
            b.year,
            b.mracehisp_c,
            sum(e.prevalence / 10000.0) AS prev_est_race
        FROM us_births b
        JOIN us_births_est_prevalence_ethnicity e
        ON b.year = e.year 
        AND b.mracehisp_c = e.mracehisp_c
        GROUP BY b.year, b.mracehisp_c
    )
    PIVOT (
        SUM(prev_est_race) 
        FOR mracehisp_c IN (
            '1' AS nh_white,
            '2' AS nh_black,
            '3' AS ai_an,
            '4' AS nh_asian_pi,
            '5' AS hispanic
        )
    )
    ORDER BY year;
    """
).df()

prev_est_ethn_race_df

# %%
import matplotlib.pyplot as plt
import numpy as np

fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
races = ["nh_white", "nh_black", "ai_an", "nh_asian_pi", "hispanic"]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

bottom = np.zeros(len(prev_est_year_race_df))
for race, color in zip(races, colors):
    axs[0].bar(
        prev_est_year_race_df["year"],
        prev_est_year_race_df[race],
        bottom=bottom,
        label=race,
        color=color,
    )
    bottom += prev_est_year_race_df[race]

axs[0].set_ylabel("Estimated DS live births")
axs[0].set_title("Estimated given year")
axs[0].set_xticks(prev_est_year_race_df["year"])
axs[0].tick_params(axis="x", rotation=45)

bottom = np.zeros(len(prev_est_ethn_race_df))
for race, color in zip(races, colors):
    axs[1].bar(
        prev_est_ethn_race_df["year"],
        prev_est_ethn_race_df[race],
        bottom=bottom,
        label=race,
        color=color,
    )
    bottom += prev_est_ethn_race_df[race]

axs[1].set_title("Estimated given race/ethnicity")
axs[1].set_xticks(prev_est_ethn_race_df["year"])
axs[1].tick_params(axis="x", rotation=45)

handles, labels = axs[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.05),
    ncol=len(races),
    frameon=False,
)

fig.suptitle("Estimated DS live birth by race/ethnicity and estimation method", y=1.12)
fig.tight_layout(rect=[0, 0, 1, 0.98])
