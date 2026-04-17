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
# # Notes - race and ethnicity
#

# %%
import duckdb
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from dspopulations_us_birth_certificates.variables import Variables as vars

plt.style.use("../notebook.mplstyle")

os.makedirs("./output", exist_ok=True)

# %%
con = duckdb.connect("../data/us_births.db", read_only=True)

# %%
race_hisp_totals_df = (
    con.execute(
        """
        SELECT 
        mracehisp_c,
        AVG(down_ind) as prob_down_ind,
        AVG(p_ds_lb_nt) as prob_ds_lb_nt,
        AVG(p_ds_lb_wt_mage_reduc) as prob_ds_lb_wt_mage_reduc,
        AVG(ds_case_weight) as prob_ds_case_weight
        FROM us_births
        GROUP BY mracehisp_c;
        """
    )
    .df()
    .dropna()
)

race_hisp_totals_df[vars.MRACEHISP_C] = race_hisp_totals_df[vars.MRACEHISP_C].map(
    {1: "NH White", 2: "NH Black", 3: "NH AI/AN", 4: "NH Asian/PI", 5: "Hispanic"}
)
race_hisp_totals_df

# %%
plt.title("Probability of DS live birth by race/ethnicity, 1989-2024")
plt.xlabel("Race/ethnicity")
plt.ylabel("Probability of DS live birth")
plt.bar(
    race_hisp_totals_df[vars.MRACEHISP_C],
    race_hisp_totals_df["prob_ds_lb_nt"],
    color="#89d2ff",
    label="Absent elective terminations",
)
plt.bar(
    race_hisp_totals_df[vars.MRACEHISP_C],
    race_hisp_totals_df["prob_ds_lb_wt_mage_reduc"],
    color="#45b0e6",
    label="Predicted by maternal age less reduction rate for year",
)
plt.bar(
    race_hisp_totals_df[vars.MRACEHISP_C],
    race_hisp_totals_df["prob_ds_case_weight"],
    color="#30e08666",
    label="Predicted by ethnicity and year, case weighted",
)
plt.bar(
    race_hisp_totals_df[vars.MRACEHISP_C],
    race_hisp_totals_df["prob_down_ind"],
    color="#1977a6",
    label="Recorded",
)
for i, row in race_hisp_totals_df.iterrows():
    plt.text(
        i - 1,
        row["prob_down_ind"] - 0.00012,
        f"{(row['prob_down_ind'] / row['prob_ds_lb_wt_mage_reduc'] * 100):.1f}% (T)",
        ha="center",
        va="bottom",
        color="white",
    )
for i, row in race_hisp_totals_df.iterrows():
    plt.text(
        i - 1,
        row["prob_down_ind"] - 0.00024,
        f"{(row['prob_down_ind'] / row['prob_ds_lb_nt'] * 100):.1f}% (¬T)",
        ha="center",
        va="bottom",
        color="white",
    )
for i, row in race_hisp_totals_df.iterrows():
    plt.text(
        i - 1,
        row["prob_ds_case_weight"] - 0.00012,
        f"r = {((1 - row['prob_ds_case_weight'] / row['prob_ds_lb_nt']) * 100):.1f}%",
        ha="center",
        va="bottom",
        color="white",
    )
plt.legend()

# %%
con.close()
