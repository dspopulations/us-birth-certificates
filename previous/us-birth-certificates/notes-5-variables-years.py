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
# # Notes 5
#

# %%
import duckdb
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from variables import Variables as vars

plt.style.use('../../notebook.mplstyle')

os.makedirs("./output", exist_ok=True)

# %%
# con.close()
con = duckdb.connect("../../data/us_births.db", read_only=True)

# %%
df = con.execute(
    f"""
    SELECT
        b.year,
        COUNT(*)                as birth_count,
        AVG(b.down_ind)::DOUBLE    as ds_recorded,
        AVG(b.p_ds_lb_nt)::DOUBLE       as ds_est_no_term,
        AVG(b.ds_case_weight)::DOUBLE   as case_weighted,
        AVG(b.down_ind)::DOUBLE / AVG(b.p_ds_lb_nt)::DOUBLE as recorded_ratio
    FROM us_births as b
    WHERE b.year >= 2000
    GROUP BY b.year
    ORDER BY b.year
    """
).df()

# %%
df

# %%
plt.figure(figsize=(10, 6))
plt.plot(df['year'], df['recorded_ratio'], label='Recorded')
plt.xlabel("Year")
plt.ylabel("Proportion of DS live births expected")
plt.title("Proportion of DS live births expected in the absence of termination recorded on birth certificates 2000-2024")
plt.plot()

# %%
base_year = 2000
cols = ["ds_recorded", "ds_est_no_term"]

# get the base (year 2000) value for each column
base = (
    df.loc[df["year"].eq(base_year), cols]
      .iloc[0]                 # assumes exactly one row for year 2000
)

# scale: value / base_value * 100
for c in cols:
    df[c + "_idx2000"] = df[c] / base[c] * 100


# %%
plt.figure(figsize=(10, 6))
plt.plot(df['year'], df['ds_est_no_term'], label='Estimated No Termination')
plt.xlabel("Year")
plt.ylabel("Index value (2000=100)")
plt.title("DS live births recorded vs expected in the absence of termination 2000-2024 (2000=100)")
plt.plot()

# %%
plt.figure(figsize=(10, 6))
plt.plot(df['year'], df['ds_recorded_idx2000'], label='Recorded')
plt.plot(df['year'], df['ds_est_no_term_idx2000'], label='Estimated No Termination')
plt.xlabel("Year")
plt.ylabel("Index value (2000=100)")
plt.title("Probability of DS live birth recorded vs expected in the absence of termination indexed (2000=100), 2000-2024")
plt.plot()

# %%
base_year = 2018
cols = ["ds_recorded", "ds_est_no_term"]

# get the base (year 2000) value for each column
base = (
    df.loc[df["year"].eq(base_year), cols]
      .iloc[0]                 # assumes exactly one row for year 2000
)

# scale: value / base_value * 100
for c in cols:
    df[c + "_idx2018"] = df[c] / base[c] * 100


# %%
plt.figure(figsize=(10, 6))
plt.plot(df['year'], df['ds_recorded_idx2018'], label='Recorded')
plt.plot(df['year'], df['ds_est_no_term_idx2018'], label='Estimated No Termination')
plt.xlabel("Year")
plt.ylabel("Index value (2018=100)")
plt.title("Probability of DS live birth recorded vs expected in the absence of termination indexed (2018=100), 2000-2024")
plt.plot()

# %%
base_year = 2016
cols = ["ds_recorded", "ds_est_no_term"]

# get the base (year 2000) value for each column
base = (
    df.loc[df["year"].eq(base_year), cols]
      .iloc[0]                 # assumes exactly one row for year 2000
)

# scale: value / base_value * 100
for c in cols:
    df[c + "_idx2016"] = df[c] / base[c] * 100


# %%
plt.figure(figsize=(10, 6))
plt.plot(df['year'], df['ds_recorded_idx2016'], label='Recorded')
plt.plot(df['year'], df['ds_est_no_term_idx2016'], label='Estimated No Termination')
plt.xlabel("Year")
plt.ylabel("Index value (2016=100)")
plt.title("Probability of DS live birth recorded vs expected in the absence of termination indexed (2016=100), 2000-2024")
plt.plot()

# %%
post_cfdna_df = con.execute(
    f"""
    SELECT
        pay_rec,
        AVG(b.down_ind)::DOUBLE    as ds_recorded,
        AVG(b.p_ds_lb_nt)::DOUBLE       as ds_est_no_term,
        AVG(b.ds_case_weight)::DOUBLE   as case_weighted,
        AVG(b.down_ind)::DOUBLE / AVG(b.p_ds_lb_nt)::DOUBLE as recorded_ratio
    FROM us_births as b
    WHERE b.year >= 2020
    GROUP BY pay_rec
    """
).df()
post_cfdna_df

# %%
pre_cfdna_df = con.execute(
    f"""
    SELECT
        pay_rec,
        b.down_ind::DOUBLE    as ds_recorded,
        AVG(b.p_ds_lb_nt)::DOUBLE       as ds_est_no_term,
        AVG(b.ds_case_weight)::DOUBLE   as case_weighted,
        AVG(b.down_ind)::DOUBLE / AVG(b.p_ds_lb_nt)::DOUBLE as recorded_ratio
    FROM us_births as b
    WHERE pay_rec IS NOT NULL AND b.year >= 2011 AND b.year <= 2015
    GROUP BY pay_rec
    """
).df()
pre_cfdna_df
