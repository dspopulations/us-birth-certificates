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
from graphviz import Digraph
from sympy.physics.units import magnetic_density

from dspopulations_us_birth_certificates.variables import Variables as vars

plt.style.use('../notebook.mplstyle')

os.makedirs("./output", exist_ok=True)

# %%
con = duckdb.connect("../data/us_births.db", read_only=True)

# %% [markdown]
# ### Estimate prevalence given year and ethnicity

# %%
eth_prev_df = con.execute(
    """
    SELECT
        year, mracehisp, down_ind, p_ds_lb_nt, p_ds_lb_wt
    FROM us_births
    WHERE year >= 2003 AND year <= 2023
    """
).df().convert_dtypes()

eth_prev_df

# %%
eth_ds_chance_df = pd.read_csv("./us-births-estimated-prevalence-ethnicity-2000-2023.csv").convert_dtypes()
eth_ds_chance_df.set_index(["year", "ethnicity"], inplace=True)
eth_ds_chance_df = eth_ds_chance_df.unstack("ethnicity")
eth_ds_chance_df

# %%
for i in range(0, 5):
    col = eth_ds_chance_df.columns[i]
    eth_ds_chance_df[col] = eth_ds_chance_df[col].fillna(
        eth_ds_chance_df[col].rolling(window=10, min_periods=1).mean()) / 10000
eth_ds_chance_df

# %%
eth_ds_chance_df.index

# %%
lookup = eth_ds_chance_df.copy()
lookup.columns = eth_ds_chance_df.columns.droplevel(0)
lookup.columns = [col.replace("\n", "_") for col in lookup.columns]
lookup = lookup.reset_index().melt(id_vars="year", var_name="label", value_name="rate").set_index("year")
lookup

# %%
mracehisp_to_label = {
    1: "Non-Hispanic White",
    2: "Non-Hispanic Black",
    3: "American Indian or Alaska Native",
    4: "Non-Hispanic Asian or Pacific Islander",
    5: "Non-Hispanic Asian or Pacific Islander",
    6: "Non-Hispanic_more_than_one",
    7: "Hispanic",
}
eth_prev_df["label"] = eth_prev_df["mracehisp"].map(mracehisp_to_label)

# %%
eth_prev_df["p_ds_lb_nt_eth"] = eth_prev_df.apply(
    lambda row: eth_ds_chance_df.loc[(row["year"], row["mracehisp"]), "p_ds_lb_wt_eth"] if row["mracehisp"] != 5 else 0,
)
eth_prev_df

# %%
plt.figure(figsize=(12, 5))
plt.title("Estimated vs recorded annual DS live births")
plt.xlabel("Year")
plt.ylabel("Estimate/count")
plt.bar(mage_df.index, mage_df[vars.P_DS_LB_NT], label="DS live births absent terminations")
plt.bar(mage_df.index, mage_df[vars.P_DS_LB_WT], label="DS live births")
plt.plot(mage_df.index, mage_df[vars.DOWN_IND], marker='o', color="#ff9060", label='Recorded DS live births')
plt.legend()

# %% [markdown]
#

# %%
age_ds_df = df[[vars.MAGE_C, vars.P_DS_LB_NT, vars.P_DS_LB_WT]].groupby(vars.MAGE_C).describe()
age_ds_df

# %%
recorded_df = df[df[vars.DOWN_IND] == 1]

# %%
age_ds_recorded_df = recorded_df[[vars.MAGE_C, vars.P_DS_LB_NT, vars.P_DS_LB_WT]].groupby(vars.MAGE_C).describe()
age_ds_recorded_df

# %%
plt.figure(figsize=(12, 5))
plt.plot(age_ds_recorded_df.index, age_ds_recorded_df[(vars.P_DS_LB_WT, 'mean')], color="#99ccff",
         label="Chance of DS live birth (recorded)")
plt.plot(age_ds_df.index, age_ds_df[(vars.P_DS_LB_WT, 'mean')], color="#99d066", label="Chance of DS live birth (all)")
plt.legend()

# %%
fig, axs = plt.subplots(1, 2, sharex=True, figsize=(12, 6))

axs[0].set_title("Recorded DS births")
axs[0].bar(age_ds_recorded_df.index, age_ds_recorded_df[(vars.P_DS_LB_NT, 'count')], color="#99ccff")

axs[1].set_title("DS births (all, estimated given maternal age)")
axs[1].bar(age_ds_df.index, age_ds_df[(vars.P_DS_LB_NT, 'count')], color="#99d066")


# %%

# %%
from graphviz import Digraph

dag = Digraph()

dag.attr(fontname="Helvetica")
dag.attr("node", fontname="Helvetica")
dag.attr("edge", fontname="Helvetica")

# set font sizes
dag.attr(size="12,12")
dag.attr("node", fontsize="16", style="filled", fillcolor="#e8f8ff")
dag.attr("edge", fontsize="16")

dag.attr(rankdir="LR", splines="spline")  # Top-to-bottom flow
dag.attr("node", shape="circle", fixedsize="true", width="1.75")

edges = [
    ('Age', 'Case'),
    ('Age', 'Screening'),
    ('Age', 'Termination'),
    ('Age', 'Income'),
    ('Age', 'Recorded'),
    ('Year', 'Age'),
    ('Year', 'Screening'),
    ('Year', 'Termination'),
    ('Year', 'Income'),
    ('Year', 'Recorded'),
    ('Ethnicity', 'Income'),
    ('Ethnicity', 'Screening'),
    ('Ethnicity', 'Termination'),
    ('Ethnicity', 'Recorded'),
    ('Income', 'Screening'),
    ('Case', 'Termination'),
    ('Case', 'DS birth'),
    ('Screening', 'Termination'),
    ('Termination', 'DS birth'),
    ('DS birth', 'Recorded'),
]

for src, dst in edges:
    dag.edge(src, dst)

from IPython import display

display.display_png(dag)

