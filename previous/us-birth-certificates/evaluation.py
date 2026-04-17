"""Evaluation pipeline: metrics, feature importance, SHAP analysis, correlation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from sklearn.inspection import permutation_importance

import ml_utils
import plot_utils
import stats_utils
from experiment_config import ExperimentConfig, ModelVariantConfig


@dataclass
class ModelResult:
    """Structured results for one model variant."""

    experiment_name: str
    variant: ModelVariantConfig
    features: list[str]
    metrics: dict = field(default_factory=dict)
    best_iteration: int = 0


def compute_metrics(
    model,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    config: ExperimentConfig,
    output_dir: str,
    model_idx: int,
) -> tuple[dict, pd.DataFrame]:
    """Compute and save validation metrics (AUC, AP, log loss, precision@K).

    Returns (metrics_dict, metrics_dataframe).
    """
    p_valid = model.predict(X_valid, num_iteration=model.best_iteration)

    metrics_df, fpr, tpr, thresholds, tp, fp, n_pos = ml_utils.get_metrics(
        y_valid, p_valid, K=10000, thr=0.01
    )

    metrics_df.to_csv(
        f"{output_dir}/model_{model_idx}_validation_metrics.csv", index=False
    )

    plot_utils.plot_roc_curve(
        fpr,
        tpr,
        model_idx,
        save=config.save_plots,
        output_dir=output_dir,
        file_name=f"model_{model_idx}_roc_curve",
    )
    plot_utils.plot_precision_recall_curve(
        fpr,
        tpr,
        model_idx,
        save=config.save_plots,
        output_dir=output_dir,
        file_name=f"model_{model_idx}_precision_recall_curve",
    )

    metrics_dict = dict(zip(metrics_df["metric"], metrics_df["value"]))
    return metrics_dict, metrics_df


def compute_feature_importance(
    model,
    features: list[str],
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    categorical: list[str],
    config: ExperimentConfig,
    output_dir: str,
    model_idx: int,
):
    """Compute gain-based and permutation importance; save CSVs and plots.

    Returns (X_eval, y_eval, perm_result) for downstream SHAP use.
    """
    # Gain importance
    importance_gain = model.feature_importance(importance_type="gain")
    df_gain = pd.DataFrame(
        {"feature": features, "importance_gain": importance_gain}
    ).sort_values("importance_gain", ascending=False)
    df_gain.to_csv(
        f"{output_dir}/model_{model_idx}_feature_importance_gain.csv", index=False
    )

    # Explanation set
    X_eval, y_eval = ml_utils.build_explain_set(
        model, X_valid, y_valid, categorical, seed=config.random_seed
    )
    model_wrapped = ml_utils.LGBMEstimator(model)

    # Permutation importance
    result = permutation_importance(
        model_wrapped,
        X_eval,
        y_eval,
        scoring=ml_utils.ap_scorer,
        n_repeats=20,
        n_jobs=8,
        random_state=config.random_seed,
    )

    perm_df = pd.DataFrame(
        {
            "feature": X_eval.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    perm_df.to_csv(
        f"{output_dir}/model_{model_idx}_permutation_importance.csv", index=False
    )

    plot_utils.plot_permutation_importances(
        result,
        X_eval,
        model_idx,
        save=config.save_plots,
        output_dir=output_dir,
        file_name=f"model_{model_idx}_permutation_importances",
    )

    return X_eval, y_eval, result


def compute_correlation_analysis(
    X_eval: pd.DataFrame,
    model_idx: int,
    config: ExperimentConfig,
    output_dir: str,
):
    """Distance correlation + hierarchical clustering + heatmap."""
    distance, corr = stats_utils.distance_corr_dissimilarity(X_eval)
    condensed = squareform(distance, checks=True)
    dist_linkage = hierarchy.linkage(condensed, method="average")
    dendro_labels = X_eval.columns.to_list()

    dendro = plot_utils.plot_dendrogram(
        dist_linkage,
        dendro_labels,
        model_idx,
        save=config.save_plots,
        output_dir=output_dir,
        file_name=f"model_{model_idx}_dendrogram",
    )

    plot_utils.plot_correlation_heatmap(
        corr,
        dendro,
        label_threshold=0.3,
        model_idx=model_idx,
        save=config.save_plots,
        output_dir=output_dir,
        file_name=f"model_{model_idx}_correlation_heatmap",
    )


def compute_shap_analysis(
    model,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    variant: ModelVariantConfig,
    config: ExperimentConfig,
    output_dir: str,
):
    """SHAP TreeExplainer + importance CSV + bar/beeswarm/scatter plots."""
    import shap

    model_idx = variant.index
    explainer = shap.TreeExplainer(
        model, feature_perturbation="tree_path_dependent", model_output="raw"
    )
    explanation = explainer(X_eval)
    shap_values = explanation.values

    # Importance CSV
    shap_importance = pd.DataFrame(
        {
            "feature": X_eval.columns,
            "mean_abs_shap": np.mean(np.abs(shap_values), axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    shap_importance.to_csv(
        f"{output_dir}/model_{model_idx}_shap_importance.csv", index=False
    )

    # Bar plot
    plot_utils.plot_shap_bar(
        explanation,
        model_idx,
        max_display=variant.shap_bar_max_display,
        save=config.save_plots,
        output_dir=output_dir,
    )

    # Beeswarm plot
    plot_utils.plot_shap_beeswarm(
        explanation,
        model_idx,
        max_display=variant.shap_beeswarm_max_display,
        plot_size=variant.shap_beeswarm_plot_size,
        save=config.save_plots,
        output_dir=output_dir,
    )

    # Scatter plots for specified feature pairs
    for feature_x, feature_color in variant.shap_scatter_pairs:
        if feature_x in X_eval.columns and feature_color in X_eval.columns:
            plot_utils.plot_shap_scatter(
                explanation,
                feature_x,
                feature_color,
                model_idx,
                save=config.save_plots,
                output_dir=output_dir,
            )


def evaluate_model(
    model,
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    features: list[str],
    categorical: list[str],
    variant: ModelVariantConfig,
    config: ExperimentConfig,
    output_dir: str,
) -> ModelResult:
    """Run the full evaluation pipeline for one model variant.

    Calls metrics, feature importance, correlation analysis, and SHAP in sequence.
    """
    model_idx = variant.index
    print(f"\n{'=' * 60}")
    print(f"Evaluating Model {model_idx}: {variant.label}")
    print(f"{'=' * 60}")

    # Metrics
    metrics_dict, _ = compute_metrics(
        model, X_valid, y_valid, config, output_dir, model_idx
    )

    # Feature importance + explanation set
    X_eval, y_eval, _ = compute_feature_importance(
        model, features, X_valid, y_valid, categorical, config, output_dir, model_idx
    )

    # Correlation analysis
    compute_correlation_analysis(X_eval, model_idx, config, output_dir)

    # SHAP
    compute_shap_analysis(model, X_eval, y_eval, variant, config, output_dir)

    return ModelResult(
        experiment_name=config.name,
        variant=variant,
        features=features,
        metrics=metrics_dict,
        best_iteration=model.best_iteration,
    )
