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
# # Notes 3
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
    """
    SELECT b.year,
           CASE WHEN b.mage_c < 35 THEN '<35' ELSE '>=35' END as mage_group,
           b.meduc,
           COUNT(*)                                           as birth_count,
           SUM(b.down_ind)::INT as ds_recorded, SUM(b.p_ds_lb_nt) as ds_est_no_term,
           SUM(b.ds_case_weight)                               as case_weighted,
    FROM us_births as b
    GROUP BY b.year, mage_group, b.meduc
    ORDER BY b.year, mage_group, b.meduc
    """
).df()
meduc_df.to_csv(f"./output/meduc_age_group_by_year-{datetime.now().strftime("%Y%m%d%H%M")}.csv", index=False)
meduc_df

# %%
meduc_df = con.execute(
    """
    SELECT b.year,
           CASE
               WHEN b.mage_c < 20 THEN '<20'
               WHEN b.mage_c < 25 THEN '20-24'
               WHEN b.mage_c < 30 THEN '25-29'
               WHEN b.mage_c < 35 THEN '30-34'
               WHEN b.mage_c < 40 THEN '35-39'
               WHEN b.mage_c < 45 THEN '40-44'
               ELSE '>=45'
               END              as mage_group,
           b.meduc,
           COUNT(*)             as birth_count,
           SUM(b.down_ind)::INT as ds_recorded, SUM(b.p_ds_lb_nt) as ds_est_no_term,
           SUM(b.ds_case_weight) as case_weighted,
    FROM us_births as b
    GROUP BY b.year, mage_group, b.meduc
    ORDER BY b.year, mage_group, b.meduc
    """
).df()
meduc_df.to_csv(f"./output/meduc_age_group_2_by_year-{datetime.now().strftime("%Y%m%d%H%M")}.csv", index=False)
meduc_df

# %%
meduc_df = con.execute(
    """
    SELECT b.year,
           CASE WHEN b.mage_c < 35 THEN '<35' ELSE '>=35' END as mage_group,
           CASE
               WHEN b.meduc < 6 THEN 'Less than BA'
               WHEN b.meduc >= 6 AND b.meduc < 9 THEN 'BA or higher'
               ELSE NULL
               END              as meduc_group,
           CASE
               WHEN b.mracehisp_c = 1 THEN 'NH White'
               WHEN b.mracehisp_c = 2 THEN 'NH Black'
               WHEN b.mracehisp_c = 3 THEN 'NH AI/AN'
               WHEN b.mracehisp_c = 4 THEN 'NH Asian/PI'
               WHEN b.mracehisp_c = 5 THEN 'Hispanic'
               ELSE NULL
               END              as race_ethnicity,
           COUNT(*)             as birth_count,
           SUM(b.down_ind)::INT as ds_recorded, SUM(b.p_ds_lb_nt) as ds_est_no_term,
           SUM(b.ds_case_weight) as case_weighted,
    FROM us_births as b
    WHERE b.year >= 2003
    GROUP BY b.year, mage_group, meduc_group, race_ethnicity
    ORDER BY b.year, mage_group, meduc_group, race_ethnicity
    """
).df()
meduc_df.to_csv(f"./output/meduc_age_group_3_by_year-{datetime.now().strftime("%Y%m%d%H%M")}.csv", index=False)
meduc_df

# %%
meduc_df = con.execute(
    f"""
    SELECT b.year,
           CASE WHEN b.mage_c < 35 THEN '<35' ELSE '>=35' END as mage_group,
           CASE
               WHEN b.meduc < 6 THEN 'Less than BA'
               WHEN b.meduc >= 6 AND b.meduc < 9 THEN 'BA or higher'
               ELSE NULL
           END as meduc_group,
           CASE
                WHEN {vars.BMI} >= 70 THEN NULL
                WHEN {vars.BMI} < 30 THEN 'BMI<30'
                WHEN {vars.BMI} >= 30 THEN 'BMI>=30'
                ELSE NULL
           END as bmi_group,
           COUNT(*)             as birth_count,
           SUM(b.down_ind)::INT as ds_recorded, SUM(b.p_ds_lb_nt) as ds_est_no_term,
           SUM(b.ds_case_weight) as case_weighted,
    FROM us_births as b
    WHERE b.year >= 2009
        AND b.bmi >= 13.0 AND b.bmi < 99.0
    GROUP BY b.year, mage_group, meduc_group, bmi_group
    ORDER BY b.year, mage_group, meduc_group, bmi_group
    """
).df()
meduc_df.to_csv(f"./output/meduc_age_group_4_by_year-{datetime.now().strftime("%Y%m%d%H%M")}.csv", index=False)
meduc_df

# %%
meduc_df = con.execute(
    f"""
    SELECT b.year,
           CASE
               WHEN b.mage_c < 20 THEN '<20'
               WHEN b.mage_c < 25 THEN '20-24'
               WHEN b.mage_c < 30 THEN '25-29'
               WHEN b.mage_c < 35 THEN '30-34'
               WHEN b.mage_c < 40 THEN '35-39'
               WHEN b.mage_c < 45 THEN '40-44'
               ELSE '>=45'
           END              as mage_group,
           CASE
               WHEN b.meduc < 6 THEN 'Less than BA'
               WHEN b.meduc >= 6 AND b.meduc < 9 THEN 'BA or higher'
               ELSE NULL
           END as meduc_group,
           CASE
                WHEN {vars.BMI} >= 70 THEN NULL
                WHEN {vars.BMI} < 30 THEN 'BMI<30'
                WHEN {vars.BMI} >= 30 THEN 'BMI>=30'
                ELSE NULL
           END as bmi_group,
           COUNT(*)             as birth_count,
           SUM(b.down_ind)::INT as ds_recorded, SUM(b.p_ds_lb_nt) as ds_est_no_term,
           SUM(b.ds_case_weight) as case_weighted,
    FROM us_births as b
    WHERE b.year >= 2009
        AND b.bmi >= 13.0 AND b.bmi < 99.0
    GROUP BY b.year, mage_group, meduc_group, bmi_group
    ORDER BY b.year, mage_group, meduc_group, bmi_group
    """
).df()
meduc_df.to_csv(f"./output/meduc_age_group_5_by_year-{datetime.now().strftime("%Y%m%d%H%M")}.csv", index=False)
meduc_df

# %%
bmi_df = con.execute(
    f"""
    SELECT b.bmi as bmi,
           COUNT(*)             as birth_count,
           SUM(b.down_ind)::INT as ds_recorded,
           SUM(b.p_ds_lb_nt) as ds_est_no_term,
           (SUM(b.down_ind) / COUNT(*)) * 10000 as prevalence_recorded
    FROM us_births as b
    WHERE b.bmi >= 13.0 and b.bmi < 99.0
    GROUP BY b.bmi
    ORDER BY bmi
    """
).df()
bmi_df

# %%
plt.scatter(bmi_df['bmi'], bmi_df['prevalence_recorded'], alpha=0.3)
plt.xlabel("Maternal BMI")
plt.ylabel("DS prevalence recorded per 10,000 births")

# %%
# histogram of maternal BMI
plt.hist(bmi_df['bmi'], bins=50, weights=bmi_df['ds_est_no_term'], label="Estimated DS cases (no terminations)", alpha=0.75)
plt.hist(bmi_df['bmi'], bins=50, weights=bmi_df['ds_recorded'], label="DS cases recorded")
plt.xlabel("Maternal BMI")
plt.ylabel("Number of births")
plt.title("Recorded DS cases by maternal BMI, 1989 to 2024")
plt.legend()

# %%
bmi_df = con.execute(
    f"""
    SELECT b.year,
           count(b.bmi),
           count(b.bmi_r),
           count(b.pwgt_r),
           count(b.dwgt_r),
           count(b.wtgain),
           count(b.wtgain_rec),
           count(b.m_ht_in)
    FROM us_births as b
    GROUP BY b.year
    ORDER BY year
    """
).df()
bmi_df.to_csv(f"./output/bmi_height_weight_by_year-{datetime.now().strftime("%Y%m%d%H%M")}.csv", index=False)
bmi_df

# %%

# %%

# %%

# %%
con.close()
