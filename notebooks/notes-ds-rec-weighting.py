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
# # Notes - weighting recorded DS births

# %%
import duckdb
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

plt.style.use("../notebook.mplstyle")

os.makedirs("./output", exist_ok=True)

# %%
con = duckdb.connect("../data/us_births.db", read_only=True)

# %%
weighted_est_df = con.execute(
    """
    SELECT
        b.year,
        SUM (b.ds_case_weight) as case_weighted,
        SUM (b.p_ds_lb_wt) as year_prev_est,
        SUM (b.p_ds_lb_wt_mage) as year_age_prev_est,
        sum(e.prevalence / 10000.0) as year_ethn_prev_est,
        SUM (b.p_ds_lb_nt) as no_term_age_est,
        SUM (b.p_ds_lb_nt * (1 - r.reduction)) as term_under_est,
        SUM (b.down_ind) as down_ind,
    FROM
        us_births as b
        LEFT JOIN us_births_est_prevalence_ethnicity e
            ON b.year = e.year AND b.mracehisp_c = e.mracehisp_c
        LEFT JOIN reduction_rate_year r
            ON b.year = r.year
    WHERE
        b.year >= 1989
    GROUP BY
        b.year
    ORDER BY
        b.year
    """
).df()
weighted_est_df.to_csv(f"./output/us_births_weighted_ds_estimates_year.csv", index=False)
weighted_est_df

# %%
# plot by year
plt.figure(figsize=(8, 6))
plt.plot(weighted_est_df["year"], weighted_est_df["no_term_age_est"], label="Estimated live births (no terminations)")
plt.plot(weighted_est_df["year"], weighted_est_df["case_weighted"], label="Estimated live births (case-weighted)")
plt.plot(weighted_est_df["year"], weighted_est_df["year_ethn_prev_est"], label="Estimated live births (year/ethnicity obs-weighted)")
plt.plot(weighted_est_df["year"], weighted_est_df["year_age_prev_est"], label="Estimated live births (year/maternal age obs-weighted)")
plt.plot(weighted_est_df["year"], weighted_est_df["year_prev_est"], label="Estimated live births (year obs-weighted)")
plt.plot(weighted_est_df["year"], weighted_est_df["term_under_est"], label="Estimated live births (year/reduction-rate obs-weighted)")
plt.plot(weighted_est_df["year"], weighted_est_df["down_ind"], label="Recorded live births")
plt.title("DS births by year, 1989-2024")
plt.xlabel("Year")
plt.ylabel("DS counts")
plt.legend()

# %%
weighted_est_ages_df = con.execute(
    """
    SELECT
        b.mage_c,
        SUM (b.ds_case_weight) as case_weighted,
        SUM (b.p_ds_lb_wt) as year_prev_est,
        SUM (b.p_ds_lb_wt_mage) as year_age_prev_est,
        SUM (e.prevalence / 10000.0) as year_ethn_prev_est,
        SUM (b.p_ds_lb_nt) as no_term_age_est,
        SUM (b.p_ds_lb_nt * (1 - r.reduction)) as term_under_est,
        SUM (b.down_ind) as down_ind,
    FROM
        us_births as b
        LEFT JOIN us_births_est_prevalence_ethnicity e
            ON b.year = e.year AND b.mracehisp_c = e.mracehisp_c
        JOIN reduction_rate_year r
            ON b.year = r.year
    WHERE
        b.year >= 1989
    GROUP BY
        b.mage_c
    ORDER BY
        b.mage_c
    """
).df()
weighted_est_ages_df

# %%
plt.figure(figsize=(8,6))
plt.bar(weighted_est_ages_df["mage_c"], weighted_est_ages_df["no_term_age_est"], color="#9098ac80", label="Estimated live births (no terminations)")
plt.bar(weighted_est_ages_df["mage_c"], weighted_est_ages_df["year_prev_est"], color="#cf966080", label="Estimated live births (year obs-weighted)")
plt.bar(weighted_est_ages_df["mage_c"], weighted_est_ages_df["term_under_est"], color="#30e0af80", label="Estimated live births (year/reduction-rate obs-weighted)")
plt.bar(weighted_est_ages_df["mage_c"], weighted_est_ages_df["case_weighted"], color="#6666d080", label="Estimated live births (case-weighted)")
plt.bar(weighted_est_ages_df["mage_c"], weighted_est_ages_df["down_ind"], color="#99336680", label="Recorded live births")
plt.title("DS births by maternal age, 1989-2024")
plt.xlabel("Maternal age (years)")
plt.ylabel("DS counts")
plt.legend()

# %%
ethnicity_df = con.execute(
    """
    SELECT
        b.mracehisp_c,
        SUM (b.ds_case_weight) as case_weighted,
        SUM (b.p_ds_lb_wt) as year_prev_est,
        SUM (b.p_ds_lb_wt_mage) as year_age_prev_est,
        SUM (e.prevalence / 10000.0) as year_ethn_prev_est,
        SUM (b.p_ds_lb_nt) as no_term_age_est,
        SUM (b.p_ds_lb_nt * (1 - r.reduction)) as term_under_est,
        SUM (b.down_ind) as down_ind,
    FROM
        us_births as b
        LEFT JOIN us_births_est_prevalence_ethnicity e
            ON b.year = e.year AND b.mracehisp_c = e.mracehisp_c
        JOIN reduction_rate_year r
            ON b.year = r.year
    WHERE
        b.year >= 1989
    GROUP BY
        b.mracehisp_c
    ORDER BY
        b.mracehisp_c
    """
).df().dropna()

race_labels = {
    1: "NH White",
    2: "NH Black",
    3: "AI/AN",
    4: "NH Asian/PI",
    5: "Hispanic"
}

ethnicity_df["mracehisp_c_label"] = ethnicity_df["mracehisp_c"].map(race_labels)

ethnicity_df

# %%
plt.figure(figsize=(10, 6))

x = ethnicity_df["mracehisp_c"]
labels = ethnicity_df["mracehisp_c_label"]

width = 0.15

plt.bar(x - 2*width, ethnicity_df["no_term_age_est"],
        width=width, color="#9098ac80",
        label="Estimated live births (no terminations)")

plt.bar(x - width, ethnicity_df["year_prev_est"],
        width=width, color="#cf966080",
        label="Estimated live births (year obs-weighted)")

plt.bar(x, ethnicity_df["term_under_est"],
        width=width, color="#30e0af80",
        label="Estimated live births (year/reduction-rate obs-weighted)")

plt.bar(x + width, ethnicity_df["case_weighted"],
        width=width, color="#6666d080",
        label="Estimated live births (case-weighted)")

plt.bar(x + 2*width, ethnicity_df["down_ind"],
        width=width, color="#99336680",
        label="Recorded live births")

plt.title("DS births by maternal age, 1989-2024")
plt.xlabel("Ethinicity")
plt.ylabel("DS counts")
plt.xticks(x, labels)
plt.legend()


# %%
ethnicity_2_df = con.execute(
    """
    SELECT
        b.mracehisp_c,
        SUM (b.ds_case_weight) as case_weighted,
        SUM (b.p_ds_lb_wt) as year_prev_est,
        SUM (b.p_ds_lb_wt_mage) as year_age_prev_est,
        SUM (e.prevalence / 10000.0) as year_ethn_prev_est,
        SUM (b.p_ds_lb_nt) as no_term_age_est,
        SUM (b.p_ds_lb_nt * (1 - r.reduction)) as term_under_est,
        SUM (b.down_ind) as down_ind,
    FROM
        us_births as b
        LEFT JOIN us_births_est_prevalence_ethnicity e
            ON b.year = e.year AND b.mracehisp_c = e.mracehisp_c
        JOIN reduction_rate_year r
            ON b.year = r.year
    WHERE
        b.year >= 2000 AND b.year <= 2014
    GROUP BY
        b.mracehisp_c
    ORDER BY
        b.mracehisp_c
    """
).df().dropna()

race_labels = {
    1: "NH White",
    2: "NH Black",
    3: "AI/AN",
    4: "NH Asian/PI",
    5: "Hispanic"
}

ethnicity_2_df["mracehisp_c_label"] = ethnicity_2_df["mracehisp_c"].map(race_labels)

ethnicity_2_df

# %%
plt.figure(figsize=(8, 6))

x = ethnicity_2_df["mracehisp_c"]
labels = ethnicity_2_df["mracehisp_c_label"]

width = 0.14

plt.bar(x - 2*width, ethnicity_2_df["no_term_age_est"],
        width=width, color="#9098ac80",
        label="Estimated live births (no terminations)")

plt.bar(x - width, ethnicity_2_df["year_prev_est"],
        width=width, color="#cf966080",
        label="Estimated live births (year obs-weighted)")

plt.bar(x, ethnicity_2_df["term_under_est"],
        width=width, color="#30e0af80",
        label="Estimated live births (year/reduction-rate obs-weighted)")

plt.bar(x + width, ethnicity_2_df["year_prev_est"],
        width=width, color="#0096c080",
        label="Estimated live births (year/ethnicity obs-weighted)")

plt.bar(x + 2*width, ethnicity_2_df["case_weighted"],
        width=width, color="#6666d080",
        label="Estimated live births (case-weighted)")

plt.bar(x + 3*width, ethnicity_2_df["down_ind"],
        width=width, color="#99336680",
        label="Recorded live births")

plt.title("DS births by ethnic group, 2000-2014")
plt.xlabel("Ethinicity")
plt.ylabel("DS counts")
plt.xticks(x, labels)
plt.legend()


# %%
educ_df = con.execute(
    """
    SELECT
        b.meduc,
        SUM (b.ds_case_weight) as case_weighted,
        SUM (b.p_ds_lb_wt) as year_prev_est,
        SUM (b.p_ds_lb_wt_mage) as year_age_prev_est,
        SUM (e.prevalence / 10000.0) as year_ethn_prev_est,
        SUM (b.p_ds_lb_nt) as no_term_age_est,
        SUM (b.p_ds_lb_nt * (1 - r.reduction)) as term_under_est,
        SUM (b.down_ind) as down_ind,
    FROM
        us_births as b
        LEFT JOIN us_births_est_prevalence_ethnicity e
            ON b.year = e.year AND b.mracehisp_c = e.mracehisp_c
        JOIN reduction_rate_year r
            ON b.year = r.year
    WHERE
        b.year >= 2003
    GROUP BY
        b.meduc
    ORDER BY
        b.meduc
    """
).df().dropna()

meduc_labels = {
    1: "8th grade or less",
    2: "9th through 12th grade",
    3: "High school graduate",
    4: "Some college credit",
    5: "Associate degree",
    6: "Bachelor's degree",
    7: "Master's degree",
    8: "Doctorate or Professional Degree",
    9: "Unknown",
}

educ_df["meduc_label"] = educ_df["meduc"].map(meduc_labels)

educ_df

# %%
plt.figure(figsize=(8, 8))

x = educ_df["meduc"]
labels = educ_df["meduc_label"]

width = 0.14

plt.bar(x - 2*width, educ_df["no_term_age_est"],
        width=width, color="#9098ac80",
        label="Estimated live births (no terminations)")

plt.bar(x - width, educ_df["year_prev_est"],
        width=width, color="#cf966080",
        label="Estimated live births (year obs-weighted)")

plt.bar(x, educ_df["term_under_est"],
        width=width, color="#30e0af80",
        label="Estimated live births (year/reduction-rate obs-weighted)")

plt.bar(x + width, educ_df["year_prev_est"],
        width=width, color="#0096c080",
        label="Estimated live births (year/ethnicity obs-weighted)")

plt.bar(x + 2*width, educ_df["case_weighted"],
        width=width, color="#6666d080",
        label="Estimated live births (case-weighted)")

plt.bar(x + 3*width, educ_df["down_ind"],
        width=width, color="#99336680",
        label="Recorded live births")

plt.title("DS births by mothers education group, 2003-2024")
plt.xlabel("Education")
plt.ylabel("DS counts")
plt.xticks(x, labels, rotation=90)
plt.legend()


# %%
con.close()
