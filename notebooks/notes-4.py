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
# # Notes 4
#

# %%
import duckdb
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from dspopulations_us_birth_certificates.variables import Variables as vars

plt.style.use('../notebook.mplstyle')

os.makedirs("./output", exist_ok=True)

# %%
# con.close()
con = duckdb.connect("../data/us_births.db", read_only=True)

# %%
meduc_df = con.execute(
    f"""
    SELECT
        b.year,
        CASE
            WHEN b.mage_c < 35
            THEN '<35' ELSE '>=35'
        END as mage_group,
        CASE
           WHEN b.meduc < 6 THEN 'Less than BA'
           WHEN b.meduc >= 6 AND b.meduc < 9 THEN 'BA or higher'
           ELSE NULL
        END as meduc_group,
        CASE
           WHEN b.feduc < 6 THEN 'Less than BA'
           WHEN b.feduc >= 6 AND b.feduc < 9 THEN 'BA or higher'
           ELSE NULL
        END as feduc_group,
        COUNT(*)                as birth_count,
        SUM(b.down_ind)::INT    as ds_recorded,
        SUM(b.p_ds_lb_nt)       as ds_est_no_term,
        SUM(b.ds_case_weight)   as case_weighted,
    FROM us_births as b
    WHERE b.year >= 2009
    GROUP BY b.year, mage_group, meduc_group, feduc_group
    ORDER BY b.year, mage_group, meduc_group, feduc_group
    """
).df()
meduc_df.to_csv(f"./output/year_meduc_feduc_1-{datetime.now().strftime("%Y%m%d%H%M")}.csv", index=False)
meduc_df

# %%
meduc_df = con.execute(
    f"""
    SELECT
        b.year,
           CASE
               WHEN b.mage_c < 30 THEN '<30'
               WHEN b.mage_c < 35 THEN '30-34'
               WHEN b.mage_c < 40 THEN '35-39'
               ELSE '>=40'
           END              as mage_group,
        CASE
           WHEN b.meduc < 9 THEN b.meduc
           ELSE NULL
        END as meduc,
        CASE
           WHEN b.feduc < 9 THEN b.feduc
           ELSE NULL
        END as feduc,
        COUNT(*)                as birth_count,
        SUM(b.down_ind)::INT    as ds_recorded,
        SUM(b.p_ds_lb_nt)       as ds_est_no_term,
        SUM(b.ds_case_weight)   as case_weighted,
    FROM us_births as b
    WHERE b.year >= 2009
        AND b.bmi >= 13.0 AND b.bmi < 99.0
    GROUP BY b.year, mage_group, meduc, feduc
    ORDER BY b.year, mage_group, meduc, feduc
    """
).df()
meduc_df.to_csv(f"./output/year_meduc_feduc_2-{datetime.now().strftime("%Y%m%d%H%M")}.csv", index=False)
meduc_df

# %%

# %%

# %%
con.close()
