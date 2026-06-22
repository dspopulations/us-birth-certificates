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
# # Predictors 11 - Predictors of recorded DS live births
#
# Runs experiment `exp_0011` via the shared training/evaluation pipeline.
# See `experiments/exp_0011.py` for the full configuration.

# %%
from experiment_runner import run_experiment
from experiments.exp_0011 import config

# %%
results = run_experiment(config)

# %% [markdown]
# ## Results summary

# %%
for r in results:
    print(f"
Model {r.variant.index}: {r.variant.label}")
    print(f"  Features: {len(r.features)}")
    print(f"  Best iteration: {r.best_iteration}")
    for metric, value in r.metrics.items():
        print(f"  {metric}: {value:.6f}")
