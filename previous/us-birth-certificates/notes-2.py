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
# # Notes
#

# %%
import duckdb
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from graphviz import Digraph
from sympy.physics.units import magnetic_density

from variables import Variables as vars

plt.style.use('../../notebook.mplstyle')

os.makedirs("./output", exist_ok=True)

# %%
# con.close()
con = duckdb.connect("../../data/us_births.db")  #, read_only=True)


# %%
mage_eth_c = con.execute(
    """
    SELECT b.year,
           b.mracehisp_c                                      as ethnicity,
           CASE WHEN b.mage_c < 35 THEN '<35' ELSE '>=35' END as mage_group,
           COUNT(*)                                           as birth_count,
           SUM(b.down_ind)::INT as ds_recorded, SUM(b.p_ds_lb_nt) as ds_est_no_term,
           SUM(b.ds_case_weight)                               as case_weighted,
    -- SUM (b.p_ds_lb_nt * (1 - r.reduction)) as ds_est_reduction,
    FROM us_births as b
             JOIN reduction_rate_year r
                  ON b.year = r.year
    GROUP BY b.year, b.mracehisp_c, mage_group
    ORDER BY b.year, b.mracehisp_c, mage_group
    """
).df()
mage_eth_c.to_csv(f"./output/case-weighted-ethnicity-age-group-{datetime.now().strftime("%Y%m%d%H%M")}.csv",
                  index=False)
mage_eth_c

# %%
mage_eth_c = con.execute(
    """
    SELECT b.year,
           b.mracehisp_c        as ethnicity,
           b.mage_c             as age,
           COUNT(*)             as birth_count,
           SUM(b.down_ind)::INT as ds_recorded, SUM(b.p_ds_lb_nt) as ds_est_no_term,
           SUM(b.ds_case_weight) as case_weighted,
    -- SUM (b.p_ds_lb_nt * (1 - r.reduction)) as ds_est_reduction,
    FROM us_births as b
             JOIN reduction_rate_year r
                  ON b.year = r.year
    GROUP BY b.year, b.mracehisp_c, b.mage_c
    ORDER BY b.year, b.mracehisp_c, b.mage_c
    """
).df()
mage_eth_c.to_csv(f"./output/case-weighted-ethnicity-ages-{datetime.now().strftime("%Y%m%d%H%M")}.csv", index=False)
mage_eth_c

# %%
mage_eth_c = con.execute(
    """
    SELECT b.year,
           b.mracehisp_c        as ethnicity,
           CASE
               WHEN b.mage_c < 20 THEN '<20'
               WHEN b.mage_c < 25 THEN '20-24'
               WHEN b.mage_c < 30 THEN '25-29'
               WHEN b.mage_c < 35 THEN '30-34'
               WHEN b.mage_c < 40 THEN '35-39'
               WHEN b.mage_c < 45 THEN '40-44'
               ELSE '>=45'
               END              as mage_group,
           COUNT(*)             as birth_count,
           SUM(b.down_ind)::INT as ds_recorded, SUM(b.p_ds_lb_nt) as ds_est_no_term,
           SUM(b.ds_case_weight) as case_weighted,
    -- SUM (b.p_ds_lb_nt * (1 - r.reduction)) as ds_est_reduction,
    FROM us_births as b
             JOIN reduction_rate_year r
                  ON b.year = r.year
    GROUP BY b.year, b.mracehisp_c, mage_group
    ORDER BY b.year, b.mracehisp_c, mage_group
    """
).df()
mage_eth_c.to_csv(f"./output/case-weighted-ethnicity-age-group-2-{datetime.now().strftime("%Y%m%d%H%M")}.csv",
                  index=False)
mage_eth_c

# %%
con.close()
