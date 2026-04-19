"""Cross-experiment comparison: load manifests, compare metrics, plot results."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure


def find_latest_manifest(experiment_name: str, output_root: str = "output") -> str:
    """Find the most recent manifest.json for a given experiment name.

    Searches ``output_root/<experiment_name>/*/manifest.json`` and returns
    the path with the latest timestamp directory.
    """
    exp_dir = Path(output_root) / experiment_name
    if not exp_dir.exists():
        raise FileNotFoundError(f"No output directory found for '{experiment_name}'")

    manifests = sorted(exp_dir.glob("*/manifest.json"))
    if not manifests:
        raise FileNotFoundError(f"No manifest.json found in '{exp_dir}'")

    return str(manifests[-1])


def load_manifest(path: str) -> dict:
    """Load a single experiment manifest."""
    with open(path) as f:
        return json.load(f)


def load_experiment_results(
    experiment_names: list[str] | None = None,
    manifest_paths: list[str] | None = None,
    output_root: str = "output",
) -> list[dict]:
    """Load manifests from multiple experiments.

    Provide either ``experiment_names`` (auto-discovers latest run) or
    explicit ``manifest_paths``.
    """
    if manifest_paths:
        return [load_manifest(p) for p in manifest_paths]

    if experiment_names:
        paths = [find_latest_manifest(n, output_root) for n in experiment_names]
        return [load_manifest(p) for p in paths]

    raise ValueError("Provide either experiment_names or manifest_paths")


def compare_metrics(manifests: list[dict]) -> pd.DataFrame:
    """Build a comparison DataFrame across experiments and model variants.

    Returns a DataFrame with columns:
    experiment, model_index, model_label, num_features, best_iteration,
    and one column per metric.
    """
    rows = []
    for m in manifests:
        for model in m["models"]:
            row = {
                "experiment": m["experiment"],
                "model_index": model["index"],
                "model_label": model["label"],
                "num_features": model["num_features"],
                "best_iteration": model["best_iteration"],
            }
            row.update(model["metrics"])
            rows.append(row)

    df = pd.DataFrame(rows)
    return df


def compare_final_models(manifests: list[dict]) -> pd.DataFrame:
    """Compare only the final (last) model variant from each experiment."""
    rows = []
    for m in manifests:
        model = m["models"][-1]
        row = {
            "experiment": m["experiment"],
            "model_label": model["label"],
            "num_features": model["num_features"],
            "best_iteration": model["best_iteration"],
            "training_split": m["config"]["training_split"],
            "num_boost_round": m["config"]["num_boost_round"],
        }
        row.update(model["metrics"])
        rows.append(row)

    return pd.DataFrame(rows)


def plot_metric_comparison(
    df: pd.DataFrame,
    metric: str = "Validation AP",
    title: str | None = None,
    save: bool = False,
    output_dir: str = ".",
    file_name: str = "experiment_comparison",
) -> Figure:
    """Bar chart comparing a metric across experiments and model variants."""
    if metric not in df.columns:
        available = [
            c
            for c in df.columns
            if c
            not in {
                "experiment",
                "model_index",
                "model_label",
                "num_features",
                "best_iteration",
            }
        ]
        raise ValueError(f"Metric '{metric}' not found. Available: {available}")

    pivot = df.pivot(index="experiment", columns="model_index", values=metric)
    ax = pivot.plot.bar(figsize=(10, 5), rot=0)
    ax.set_ylabel(metric)
    ax.set_title(title or f"{metric} by Experiment and Model Variant")
    ax.legend(title="Model", labels=[f"Model {i}" for i in pivot.columns])

    fig = ax.figure
    if save:
        fig.savefig(f"{output_dir}/{file_name}.png", dpi=300, bbox_inches="tight")
        fig.savefig(f"{output_dir}/{file_name}.svg", bbox_inches="tight")
    return fig


def compare_feature_importance(
    experiment_names: list[str],
    model_index: int = -1,
    importance_type: str = "shap",
    output_root: str = "output",
    top_n: int = 20,
) -> pd.DataFrame:
    """Side-by-side feature importance from multiple experiments.

    Parameters
    ----------
    experiment_names : list of str
        Experiment names to compare.
    model_index : int
        Which model variant to compare (-1 for last/final).
    importance_type : str
        "shap" for SHAP importance, "permutation" for permutation importance.
    output_root : str
        Root output directory.
    top_n : int
        Number of top features to include.

    Returns
    -------
    pd.DataFrame
        Wide-format DataFrame with features as rows and experiments as columns.
    """
    file_suffix = (
        "shap_importance" if importance_type == "shap" else "permutation_importance"
    )
    value_col = "mean_abs_shap" if importance_type == "shap" else "importance_mean"

    dfs = {}
    for name in experiment_names:
        manifest_path = find_latest_manifest(name, output_root)
        manifest = load_manifest(manifest_path)
        run_dir = os.path.dirname(manifest_path)

        # Resolve model index
        models = manifest["models"]
        idx = models[model_index]["index"] if model_index < 0 else model_index

        csv_path = f"{run_dir}/model_{idx}_{file_suffix}.csv"
        if os.path.exists(csv_path):
            imp = pd.read_csv(csv_path)
            dfs[name] = imp.set_index("feature")[value_col]

    if not dfs:
        return pd.DataFrame()

    combined = pd.DataFrame(dfs)

    # Rank by mean importance across experiments
    combined["mean_rank"] = combined.rank(ascending=False).mean(axis=1)
    combined = combined.sort_values("mean_rank").head(top_n)
    combined = combined.drop(columns="mean_rank")

    return combined
