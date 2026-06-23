# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: dspop-us-birth-certificates
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Notes - prevalence estimates based on maternal age
#

# %%
import duckdb
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dspopulations_us_birth_certificates.variables import Variables as vars

plt.style.use('../notebook.mplstyle')

os.makedirs("./output", exist_ok=True)

# %%
#con.close()
con = duckdb.connect("../data/us_births.db")

# %%
year_summary_df = con.execute(
    """
    SELECT
        year, COUNT (*) AS all, COUNT(mage_c) as mage_c, COUNT(umagerpt) as umagerpt, COUNT(mager) as mager, COUNT(dmage) as dmage, COUNT(dmagerpt) as dmagerpt, COUNT (mage_impflg) as mage_impflg, COUNT(mage_repflg) as mage_repflg, COUNT(mager41) as mager41, COUNT(mager9) as mager9
    FROM
        us_births
    WHERE
        year >= 1989
    GROUP BY year
    ORDER BY year
    """
).df()
year_summary_df

# %%
con.close()
