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
# # Notes - maternal age
#

# %%
import duckdb
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from graphviz import Digraph
from sympy.physics.units import magnetic_density

from variables import Variables as vars

plt.style.use('../../notebook.mplstyle')

os.makedirs("./output", exist_ok=True)

# %%
con = duckdb.connect("../../data/us_births.db", read_only=True)

# %%
mage_df = con.execute(
    """
    SELECT mage_c,
           COUNT(*)      as lb,
           SUM(down_ind) as count_down_ind,
           AVG(down_ind) as prob_down_ind,
           SUM(p_ds_lb_wt_mage_reduc) as count_ds_lb_wt_mage_reduc,
           AVG(p_ds_lb_wt_mage_reduc) as prob_ds_lb_wt_mage_reduc,
           SUM(ds_case_weight) as count_ds_case_weight,
           AVG(ds_case_weight) as prob_ds_case_weight,
    FROM us_births
    WHERE
        year >= 1989
    GROUP BY mage_c
    ORDER BY mage_c
    """
).df()
mage_df.set_index("mage_c")
mage_df

# %%
plt.bar(mage_df["mage_c"], mage_df["count_ds_case_weight"], label="Estimated given year and ethnicity, case-weighted")
plt.bar(mage_df["mage_c"], mage_df["count_down_ind"], label="Recorded")
plt.title('DS births by maternal age, 1989-2024')
plt.xlabel('Maternal age (years)')
plt.ylabel('Live births count')
plt.legend()

# %%
plt.bar(mage_df["mage_c"], mage_df["count_ds_lb_wt_mage_reduc"], label="Estimated given maternal age less reduction rate for year")
plt.bar(mage_df["mage_c"], mage_df["count_down_ind"], label="Recorded")
plt.title('DS births by maternal age, 1989-2024')
plt.xlabel('Maternal age (years)')
plt.ylabel('Live births count')
plt.legend()

# %%

# %%
con.close()
