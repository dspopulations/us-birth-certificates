# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
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
# # 1. Variable selection
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
df = con.execute(
    f"""
    SELECT
        b.year,
        COUNT(*) as count_obs,
        COUNT(b.dob_mm) as count_dob_mm,
        COUNT(b.dob_wk) as count_dob_wk,
        COUNT(b.dob_tt) as count_dob_tt,
        COUNT(b.sex) as count_sex,
        COUNT(b.dbwt) as count_dbwt,
        COUNT(b.dwgt_r) as count_dwgt_r,
        COUNT(b.dplural) as count_dplural,
        COUNT(b.setorder_r) as count_setorder_r,
        COUNT(b.bfacil3) as count_bfacil3,
        COUNT(b.precare) as count_precare,
        COUNT(b.gestrec10) as count_gestrec10,
        COUNT(b.apgar5) as count_apgar5,
        COUNT(b.apgar5r) as count_apgar5r,
        COUNT(b.apgar10) as count_apgar10,
        COUNT(b.apgar10r) as count_apgar10r,
        COUNT(b.ab_aven1) as count_ab_aven1,
        COUNT(b.ab_aven6) as count_ab_aven6,
        COUNT(b.ab_nicu) as count_ab_nicu,
        COUNT(b.ab_surf) as count_ab_surf,
        COUNT(b.ab_anti) as count_ab_anti,
        COUNT(b.ab_seiz) as count_ab_seiz,
        COUNT(b.no_abnorm) as count_no_abnorm,
        COUNT(b.ca_anen) as count_ca_anen,
        COUNT(b.ca_mnsb) as count_ca_mnsb,
        COUNT(b.ca_cchd) as count_ca_cchd,
        COUNT(b.ca_cdh) as count_ca_cdh,
        COUNT(b.ca_omph) as count_ca_omph,
        COUNT(b.ca_gast) as count_ca_gast,
        COUNT(b.ca_limb) as count_ca_limb,
        COUNT(b.ca_cleft) as count_ca_cleft,
        COUNT(b.ca_clpal) as count_ca_clpal,
        COUNT(b.ca_disor) as count_ca_disor,
        COUNT(b.ca_hypo) as count_ca_hypo,
        COUNT(b.downs) as count_downs,
        COUNT(b.uca_downs) as count_uca_downs,
        COUNT(b.ca_down) as count_ca_down,
        COUNT(b.ca_downs) as count_ca_downs,
        COUNT(b.ca_down_c) as count_ca_down_c,
        COUNT(b.down_ind) as count_down_ind,
        COUNT(b.no_congen) as count_no_congen,
        COUNT(b.m_ht_in) as count_m_ht_in,
        COUNT(b.bmi) as count_bmi,
        COUNT(b.bmi_r) as count_bmi_r,
        COUNT(b.pwgt_r) as count_pwgt_r,
        COUNT(b.wtgain) as count_wtgain,
        COUNT(b.restatus) as count_restatus,
        COUNT(b.mager) as count_mager,
        COUNT(b.dmage) as count_dmage,
        COUNT(b.mager14) as count_mager14,
        COUNT(b.mager9) as count_mager9,
        COUNT(b.mage36) as count_mage36,
        COUNT(b.mracerec) as count_mracerec,
        COUNT(b.mrace31) as count_mrace31,
        COUNT(b.mrace6) as count_mrace6,
        COUNT(b.mbstate_rec) as count_mbstate_rec,
        COUNT(b.mrace15) as count_mrace15,
        COUNT(b.mhisp_r) as count_mhisp_r,
        COUNT(b.mracehisp) as count_mracehisp,
        COUNT(b.mar) as count_mar,
        COUNT(b.dmar) as count_dmar,
        COUNT(b.mar_p) as count_mar_p,
        COUNT(b.dmeduc) as count_dmeduc,
        COUNT(b.meduc) as count_meduc,
        COUNT(b.umeduc) as count_umeduc,
        COUNT(b.meduc6) as count_meduc6,
        COUNT(b.meduc_rec) as count_meduc_rec,
        COUNT(b.dfage) as count_dfage,
        COUNT(b.dfagerpt) as count_dfagerpt,
        COUNT(b.fage11) as count_fage11,
        COUNT(b.fagerpt) as count_fagerpt,
        COUNT(b.fagecomb) as count_fagecomb,
        COUNT(b.fagerec11) as count_fagerec11,
        COUNT(b.frace) as count_frace,
        COUNT(b.fraceimp) as count_fraceimp,
        COUNT(b.fracerec) as count_fracerec,
        COUNT(b.frace31) as count_frace31,
        COUNT(b.frace6) as count_frace6,
        COUNT(b.frace15) as count_frace15,
        COUNT(b.fhispx) as count_fhispx,
        COUNT(b.fhisp_r) as count_fhisp_r,
        COUNT(b.fracehisp) as count_fracehisp,
        COUNT(b.feduc) as count_feduc,
        COUNT(b.bfed) as count_bfed,
        COUNT(b.previs) as count_previs,
        COUNT(b.previs_rec) as count_previs_rec,
        COUNT(b.priorlive) as count_priorlive,
        COUNT(b.priordead) as count_priordead,
        COUNT(b.priorterm) as count_priorterm,
        COUNT(b.lbo_rec) as count_lbo_rec,
        COUNT(b.tbo_rec) as count_tbo_rec,
        COUNT(b.illb_r11) as count_illb_r11,
        COUNT(b.ilop_r11) as count_ilop_r11,
        COUNT(b.ilp_r11) as count_ilp_r11,
        COUNT(b.pay_rec) as count_pay_rec,
        COUNT(b.wic) as count_wic,
        COUNT(b.rf_pdiab) as count_rf_pdiab,
        COUNT(b.rf_gdiab) as count_rf_gdiab,
        COUNT(b.rf_phype) as count_rf_phype,
        COUNT(b.rf_ghype) as count_rf_ghype,
        COUNT(b.rf_ehype) as count_rf_ehype,
        COUNT(b.rf_ppterm) as count_rf_ppterm,
        COUNT(b.rf_inftr) as count_rf_inftr,
        COUNT(b.rf_fedrg) as count_rf_fedrg,
        COUNT(b.rf_artec) as count_rf_artec,
        COUNT(b.rf_cesar) as count_rf_cesar,
        COUNT(b.rf_cesarn) as count_rf_cesarn,
        COUNT(b.no_risks) as count_no_risks,
        COUNT(b.ld_indl) as count_ld_indl,
        COUNT(b.ld_augm) as count_ld_augm,
        COUNT(b.ld_anes) as count_ld_anes,
        COUNT(b.me_pres) as count_me_pres,
        COUNT(b.rdmeth_rec) as count_rdmeth_rec,
        COUNT(b.dmeth_rec) as count_dmeth_rec,
        COUNT(b.attend) as count_attend,
        COUNT(b.mrace_c) as count_mrace_c,
        COUNT(b.mhisp_c) as count_mhisp_c,
        COUNT(b.mracehisp_c) as count_mracehisp_c,
    FROM us_births as b
    WHERE b.year >= 2005
    GROUP BY b.year
    ORDER BY b.year
    """
).df()
df.to_csv("./output/us_births_variables.csv", index=False)
df

# %%
df = con.execute(
    f"""
    SELECT year, count(dbwt) as count_dbwt
    FROM us_births
    WHERE year >= 2004
    GROUP BY year
    ORDER BY year;
    """
).df()
df

# %%
df = con.execute(
    f"""
    SELECT
        b.mracehisp,
        COUNT(*) as count_obs
    FROM us_births as b
    WHERE b.year >= 2004
    GROUP BY b.mracehisp
    ORDER BY b.mracehisp
    """
).df()
df

# %%
df = con.execute(
    f"""
    SELECT
        b.fracehisp,
        COUNT(*) as count_obs
    FROM us_births as b
    WHERE b.year >= 2004
    GROUP BY b.fracehisp
    ORDER BY b.fracehisp
    """
).df()
df

# %%
df = con.execute(
    f"""
    SELECT
        b.frace15,
        COUNT(*) as count_obs
    FROM us_births as b
    WHERE b.year >= 2004
    GROUP BY b.frace15
    ORDER BY b.frace15
    """
).df()
df

# %%
df = con.execute(
    f"""
    SELECT
        b.mrace15,
        COUNT(*) as count_obs
    FROM us_births as b
    WHERE b.year >= 2004
    GROUP BY b.mrace15
    ORDER BY b.mrace15
    """
).df()
df

# %%
con.close()
