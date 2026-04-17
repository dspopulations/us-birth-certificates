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
import pymc as pm
import pytensor.tensor as pt
import arviz as az
import pandas as pd
import matplotlib.pyplot as plt
from dspopulations_us_birth_certificates.chance import get_ds_lb_nt_probability_array
from graphviz import Digraph
from sympy.physics.units import magnetic_density

from dspopulations_us_birth_certificates.variables import Variables as vars

plt.style.use("../notebook.mplstyle")

os.makedirs("./output", exist_ok=True)

RANDOM_SEED = 1673025012
np.random.seed(RANDOM_SEED)

# %%
con = duckdb.connect("../data/us_births.db", read_only=True)

# %%
MIN_YEAR = 2004
MAX_YEAR = 2024

# %%
counts_df = con.execute(
    f"""
        SELECT
            year as time,
            mage_c as age,
            count(mage_c) as observed,
            SUM(down_ind) as ds_recorded
        FROM
            us_births
        WHERE
            year >= {MIN_YEAR} AND year <= {MAX_YEAR}
        GROUP BY year, mage_c
        ORDER BY year, mage_c
        """
).df()
counts_df

# %%
age = np.asarray(counts_df["age"], dtype=np.int64)
observed = np.asarray(counts_df["observed"], dtype=np.int64)
ds_recorded = np.asarray(counts_df["ds_recorded"], dtype=np.int64)

# %%

with pm.Model() as model:
    age_data = pm.Data("age", age)

    c = pm.Normal("c", mu=37.23, sigma=5.0)
    k = pm.Normal("k", mu=0.2815, sigma=0.1)

    inner = pm.math.sigmoid(k * (age_data - c))

    beta0 = pm.Normal("beta0", mu=-7.33, sigma=2.0)
    beta1 = pm.Normal("beta1", mu=4.211, sigma=2.0)

    eta = beta0 + beta1 * inner
    r   = pm.Deterministic("r", pm.math.sigmoid(eta))

    y_obs = pm.Binomial("y_obs", n=observed, p=r, observed=ds_recorded)

# %%
print(f"Free random variables: {model.free_RVs}")
print(f"Observed random variables: {model.observed_RVs}")

# set font for model graph

graph = pm.model_to_graphviz(model)
graph.graph_attr["fontname"] = "Helvetica"
graph.node_attr["fontname"] = "Helvetica"
graph.edge_attr["fontname"] = "Helvetica"
graph

# %%
with model:
    idata_prior = pm.sample_prior_predictive(
        samples=1000,
        var_names=["c", "k", "beta0", "beta1", "r", "y_obs"],
    )

# %%
r_prior = idata_prior.prior["r"]
r_mean = r_prior.mean()
r_hdi  = az.hdi(r_prior)
r_prior

# %%


for i in range(50):
    plt.plot(age, r_prior["r_dim_0"], alpha=0.1)

# mean and 90% band
plt.plot(age, r_mean, lw=2, label="prior mean r(age)")
plt.fill_between(age, r_hdi["higher"], r_hdi["lower"], alpha=0.3, label="90% prior band")

plt.xlabel("Age")
plt.ylabel("Probability r(age)")
plt.legend()
plt.title("Prior predictive for r(age)")

# %%
for i in range(50):
    plt.plot(age, r_prior[i, :], alpha=0.1)

# mean and 90% band
plt.plot(age, r_mean, lw=2, label="prior mean r(age)")
plt.fill_between(age, r_hdi[:, 0], r_hdi[:, 1], alpha=0.3, label="90% prior band")

plt.xlabel("Age")
plt.ylabel("Probability r(age)")
plt.legend()
plt.title("Prior predictive for r(age)")

# %%
with model:
    idata = pm.sample(
        2000,
        tune=2000,
        target_accept=0.8,
        return_inferencedata=True,
        random_seed=RANDOM_SEED,
    )

# %%
az.summary(idata, var_names=["c", "k", "beta0", "beta1", "r"])

# %%
with model:
    prior_idata = pm.sample_prior_predictive()

# %%
az.plot_ppc(
    prior_idata,
    group="prior",
    num_pp_samples=400,
    figsize=(6, 4),
    random_seed=RANDOM_SEED,
)

# %%
az.plot_trace(idata, var_names=["c", "k", "beta0", "beta1"], figsize=(6, 6))

# %%
az.plot_posterior(
    idata, var_names=["c", "k", "beta0", "beta1"], figsize=(12, 4)
)

# %%
az.plot_energy(idata, figsize=(6, 3))

# %%
with model:
    ppc = pm.sample_posterior_predictive(idata, var_names=["y_obs"])

# %%
az.plot_ppc(ppc, num_pp_samples=400, random_seed=RANDOM_SEED, figsize=(8, 6))

# %%
az.plot_ppc(ppc, num_pp_samples=200, kind="cumulative", random_seed=RANDOM_SEED, figsize=(6, 4))

# %%
# Posterior predictive draws: shape (chain, draw, obs)
y_ppc = ppc.posterior_predictive["y_obs"]

y_mean = y_ppc.mean(dim=("chain", "draw"))

hdi = az.hdi(y_ppc, hdi_prob=0.95)
lower = hdi.sel(hdi="lower")
upper = hdi.sel(hdi="higher")

# Sort indices by age
order = np.argsort(age)
x_sorted = age[order]
y_obs_sorted = ds_recorded[order]
y_mean_sorted = y_mean.values[order]
lower_sorted = lower["y_obs"][order]
upper_sorted = upper["y_obs"][order]


# %%
# m: posterior samples of the mean probability per observation
# dims typically: ("chain", "draw", "age") or ("chain", "draw", "m_dim_0")
m_post = idata.posterior["m"]

# Convert to latent expected *score* (mean of Y given parameters)
score_post = m_post * n_max   # broadcast over chain/draw/obs

# Posterior mean latent score per observation
score_mean = score_post.mean(dim=("chain", "draw"))

# 95% HDI for latent mean score per observation
score_hdi = az.hdi(score_post, hdi_prob=0.95)

m_lower = score_hdi.sel(hdi="lower")
m_upper = score_hdi.sel(hdi="higher")

score_mean_sorted  = score_mean.values[order]
m_lower_sorted = m_lower["m"][order]
m_upper_sorted = m_upper["m"][order]

