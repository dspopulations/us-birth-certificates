"""Orchestrator: load data, train models, evaluate, save outputs.

Can be called from notebooks or from the command line::

    python experiment_runner.py --experiment exp_0011
"""

from __future__ import annotations

import json
import os
from datetime import datetime

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import data_utils
import repl_utils
import training
from evaluation import ModelResult, evaluate_model
from experiment_config import ExperimentConfig


def create_output_dir(config: ExperimentConfig) -> str:
    """Create a timestamped output directory for the experiment."""
    os.makedirs("./output", exist_ok=True)
    start_time = datetime.now()
    output_dir = f"output/{config.name}/{start_time:%Y%m%d-%H%M%S}"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def _setup_environment(config: ExperimentConfig):
    """One-time setup: style, seed, print env info."""
    pd.options.mode.copy_on_write = True
    plt.style.use("../../notebook.mplstyle")
    np.random.seed(config.random_seed)
    repl_utils.print_environment_info()


def run_experiment(config: ExperimentConfig) -> list[ModelResult]:
    """Run a full experiment: load → train → evaluate for all model variants.

    Parameters
    ----------
    config : ExperimentConfig
        Full experiment configuration.

    Returns
    -------
    list[ModelResult]
        One result per model variant.
    """
    _setup_environment(config)
    output_dir = create_output_dir(config)
    print(f"\nOutput directory: {output_dir}\n{'=' * 60}")

    # Load data
    df = data_utils.load_predictors_data(
        from_year=config.start_year,
        to_year=config.end_year,
        include_unknown=config.include_unknown,
    )
    print(f"Loaded {len(df):,} records ({config.start_year}-{config.end_year})")

    # Prepare initial features (no drops yet)
    X, y, features, categorical = training.prepare_features(df, config)

    # Split once — all variants share the same train/valid split
    X_train, X_valid, y_train, y_valid, train_data, valid_data = (
        training.create_datasets(X, y, categorical, config)
    )
    training.print_balance(y_train, y_valid)

    # Hyperparameter search (once for the experiment)
    best_params = training.optimize_hyperparameters(train_data, valid_data, config)

    # Train and evaluate each model variant
    results: list[ModelResult] = []

    for variant in config.model_variants:
        print(f"\n{'=' * 60}")
        print(f"Model {variant.index}: {variant.label}")
        print(f"{'=' * 60}")

        # Apply feature drops for this variant
        if variant.drop_features:
            drop_cols = [c for c in variant.drop_features if c in X_train.columns]
            X_train = X_train.drop(columns=drop_cols)
            X_valid = X_valid.drop(columns=drop_cols)
            features = X_train.columns.to_list()
            categorical = [c for c in categorical if c not in drop_cols]

            # Rebuild datasets with updated features
            train_data = lgb.Dataset(
                X_train,
                label=y_train,
                categorical_feature=categorical,
                free_raw_data=False,
            )
            valid_data = lgb.Dataset(
                X_valid,
                label=y_valid,
                categorical_feature=categorical,
                reference=train_data,
                free_raw_data=False,
            )

        # Train
        model = training.train_model(
            train_data,
            valid_data,
            best_params,
            config,
            output_dir=output_dir,
            model_idx=variant.index,
        )

        # Evaluate
        result = evaluate_model(
            model,
            X_train,
            X_valid,
            y_valid,
            features,
            categorical,
            variant,
            config,
            output_dir,
        )
        results.append(result)

    # Save manifest
    save_manifest(config, results, output_dir)

    print(f"\nExperiment '{config.name}' complete.")
    print(f"Results saved to: {output_dir}")

    return results


def save_manifest(
    config: ExperimentConfig,
    results: list[ModelResult],
    output_dir: str,
):
    """Save experiment manifest (config + per-variant metrics) as JSON."""
    manifest = {
        "experiment": config.name,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "start_year": config.start_year,
            "end_year": config.end_year,
            "training_split": config.training_split,
            "num_boost_round": config.num_boost_round,
            "early_stopping_rounds": config.early_stopping_rounds,
            "optimize_trials": config.optimize_trials,
            "random_seed": config.random_seed,
            "num_features": len(config.all_features),
        },
        "models": [
            {
                "index": r.variant.index,
                "label": r.variant.label,
                "num_features": len(r.features),
                "best_iteration": r.best_iteration,
                "metrics": r.metrics,
            }
            for r in results
        ],
    }

    with open(f"{output_dir}/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)


def main():
    import argparse
    import importlib

    parser = argparse.ArgumentParser(description="Run a predictor experiment.")
    parser.add_argument(
        "--experiment",
        "-e",
        required=True,
        help="Experiment module name in experiments/ (e.g. exp_0011)",
    )
    args = parser.parse_args()

    mod = importlib.import_module(f"experiments.{args.experiment}")
    run_experiment(mod.config)


if __name__ == "__main__":
    main()
