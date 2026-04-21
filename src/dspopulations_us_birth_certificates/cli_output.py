"""Pretty CLI output helpers built on ``rich``.

Centralises the formatting used by ``scripts/fit_model.py``,
``scripts/tune_model.py``, the pipeline steps, and the Optuna harness so
that every command-line invocation of the LightGBM pipeline produces a
consistent, scannable run log.

All helpers write to a shared ``Console`` — callers can either import
``console`` directly or use the convenience functions below. No function
here raises; formatting helpers degrade gracefully when optional inputs
are ``None`` so they can be called unconditionally at the relevant step
in the pipeline.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rich.box import SIMPLE_HEAVY
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

console = Console(highlight=False, soft_wrap=False)


# ---------------------------------------------------------------------------
# Section scaffolding
# ---------------------------------------------------------------------------


def banner(title: str, subtitle: str | None = None) -> None:
    """Print a large, framed banner — one per command invocation."""
    body = Text(title, style="bold white on blue", justify="center")
    if subtitle:
        body.append("\n")
        body.append(subtitle, style="white")
    console.print()
    console.print(Panel(body, border_style="blue", padding=(1, 2)))


def section(title: str) -> None:
    """Print a section rule. Use once per logical pipeline step."""
    console.print()
    console.print(Rule(f"[bold cyan]{title}[/bold cyan]", style="cyan"))


def subsection(title: str) -> None:
    """Print a minor section marker (e.g. sub-step within a pipeline stage)."""
    console.print(f"\n[bold yellow]{title}[/bold yellow]")


def success(message: str) -> None:
    console.print(f"[bold green][OK][/bold green] {message}")


def warning(message: str) -> None:
    console.print(f"[bold yellow][!][/bold yellow] {message}")


def info(message: str) -> None:
    console.print(f"[dim]*[/dim] {message}")


# ---------------------------------------------------------------------------
# Table helpers
# ---------------------------------------------------------------------------


def _fmt(value: Any) -> str:
    """Shared renderer for mixed config/metric values."""
    if value is None:
        return "[dim]-[/dim]"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if value == 0:
            return "0"
        if abs(value) >= 1000 or abs(value) < 1e-3:
            return f"{value:.4g}"
        return f"{value:.6f}".rstrip("0").rstrip(".")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        if not value:
            return "[dim](none)[/dim]"
        if all(isinstance(v, (str, int, float, bool)) for v in value):
            return ", ".join(str(v) for v in value)
    return str(value)


def kv_table(
    title: str,
    items: Iterable[tuple[str, Any]],
    *,
    key_header: str = "Setting",
    value_header: str = "Value",
    key_style: str = "cyan",
    value_style: str = "white",
) -> Table:
    """Build a two-column key/value table suitable for config summaries."""
    table = Table(
        title=title,
        title_style="bold",
        box=SIMPLE_HEAVY,
        show_header=True,
        header_style="bold magenta",
        expand=False,
    )
    table.add_column(key_header, style=key_style, no_wrap=True)
    table.add_column(value_header, style=value_style, overflow="fold")
    for k, v in items:
        table.add_row(str(k), _fmt(v))
    return table


def print_kv(title: str, items: Iterable[tuple[str, Any]]) -> None:
    console.print(kv_table(title, items))


# ---------------------------------------------------------------------------
# Config / run setup
# ---------------------------------------------------------------------------


def print_run_header(
    *,
    command: str,
    profile: str | None,
    output_dir: Path,
    model_id: str | None = None,
) -> None:
    """Top-of-run banner + short parameter summary."""
    title = f"{command}"
    subtitle = f"profile = {profile or 'default'}"
    if model_id:
        subtitle += f" • model = {model_id}"
    subtitle += f" • started {datetime.now(UTC).isoformat(timespec='seconds')}"
    banner(title, subtitle)
    info(f"Output directory: [blue]{output_dir}[/blue]")


def print_fit_config(config: Any) -> None:
    """Render a ``FitConfig``-ish dataclass as a table (best-effort, by dict())."""
    data = vars(config) if hasattr(config, "__dict__") else dict(config)
    print_kv("Fit configuration", data.items())


def print_model_config(model_config: Any) -> None:
    """Summarise a ``ModelConfig`` instance in a single table + feature lists."""
    rows = [
        ("model_id", getattr(model_config, "model_id", None)),
        ("variant_of", getattr(model_config, "variant_of", None)),
        ("target_var", getattr(model_config, "target_var", None)),
        ("year_range", getattr(model_config, "year_range", None)),
        ("include_unknown", getattr(model_config, "include_unknown", None)),
        (
            "numeric features",
            f"{len(model_config.numeric_features)}",
        ),
        (
            "categorical features",
            f"{len(model_config.categorical_features)}",
        ),
    ]
    console.print(kv_table("Model configuration", rows))

    if model_config.numeric_features:
        console.print(
            Panel(
                ", ".join(model_config.numeric_features),
                title="Numeric features",
                title_align="left",
                border_style="green",
                padding=(0, 1),
            )
        )
    if model_config.categorical_features:
        console.print(
            Panel(
                ", ".join(model_config.categorical_features),
                title="Categorical features",
                title_align="left",
                border_style="green",
                padding=(0, 1),
            )
        )


def print_run_config(run_config: Any) -> None:
    """Render a ``RunConfig`` as a single-row table."""
    rows = [
        ("preset", getattr(run_config, "name", None)),
        ("n_trials", getattr(run_config, "n_trials", None)),
        ("num_boost_round", getattr(run_config, "num_boost_round", None)),
        (
            "early_stopping_rounds",
            getattr(run_config, "early_stopping_rounds", None),
        ),
        ("cv_splits", getattr(run_config, "cv_splits", None)),
        ("shap_mode", getattr(run_config, "shap_mode", None)),
        ("shap_subsample_size", getattr(run_config, "shap_subsample_size", None)),
        ("random_seed", getattr(run_config, "random_seed", None)),
    ]
    console.print(kv_table("Run configuration", rows))


def print_params(title: str, params: Mapping[str, Any], *, style: str = "cyan") -> None:
    """Pretty-print a parameter dict (e.g. best Optuna params)."""
    if not params:
        console.print(f"[dim]{title}: (none)[/dim]")
        return
    items = sorted(params.items())
    console.print(kv_table(title, items, key_style=style))


# ---------------------------------------------------------------------------
# Data / split summaries
# ---------------------------------------------------------------------------


def print_data_summary(
    df_rows: int,
    target_var: str,
    positives: int,
    *,
    year_range: tuple[int, int] | None = None,
    include_unknown: bool | None = None,
) -> None:
    base_rate = positives / df_rows if df_rows else 0.0
    rows: list[tuple[str, Any]] = [
        ("rows", df_rows),
        ("target", target_var),
        ("positives", positives),
        ("base rate", f"{base_rate:.6f}  ({base_rate * 100:.3f}%)"),
    ]
    if year_range is not None:
        rows.append(("year range", f"{year_range[0]}-{year_range[1]}"))
    if include_unknown is not None:
        rows.append(("include unknown", include_unknown))
    console.print(kv_table("Predictors frame", rows))


def print_split_summary(
    X_train,
    X_valid,
    y_train,
    y_valid,
) -> None:
    """Single table showing train/valid row counts, positives, and rates."""

    def _rate(y) -> float:
        n = len(y)
        return float(sum(y)) / n if n else 0.0

    table = Table(
        title="Train / valid split",
        title_style="bold",
        box=SIMPLE_HEAVY,
        header_style="bold magenta",
    )
    table.add_column("Split", style="cyan", no_wrap=True)
    table.add_column("Rows", justify="right")
    table.add_column("Positives", justify="right")
    table.add_column("Rate", justify="right")
    for label, X, y in (("train", X_train, y_train), ("valid", X_valid, y_valid)):
        n = len(X)
        pos = int(sum(y))
        rate = _rate(y)
        table.add_row(label, f"{n:,}", f"{pos:,}", f"{rate * 100:.3f}%")
    console.print(table)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def print_metrics_table(metrics: Mapping[str, Any]) -> None:
    """Render the ``compute_metrics`` output as a two-column table."""
    # Order metrics from most informative downward; stragglers follow.
    preferred = (
        "average_precision",
        "roc_auc",
        "log_loss",
        "brier_score",
        "mean_predicted_prob",
        "best_iteration",
        "n_valid",
        "n_positive_valid",
    )
    items: list[tuple[str, Any]] = []
    for key in preferred:
        if key in metrics:
            items.append((key, metrics[key]))
    for key, value in metrics.items():
        if key not in preferred:
            items.append((key, value))
    console.print(kv_table("Validation metrics", items, key_header="Metric"))


# ---------------------------------------------------------------------------
# Feature importances
# ---------------------------------------------------------------------------


def _importance_table(
    title: str, df, feature_col: str, value_col: str, *, n: int = 15
) -> Table:
    if df is None or len(df) == 0:
        raise ValueError(f"{title}: importance frame is empty.")
    ordered = df.sort_values(value_col, ascending=False).head(n)
    table = Table(
        title=f"{title} (top {len(ordered)} of {len(df)})",
        title_style="bold",
        box=SIMPLE_HEAVY,
        header_style="bold magenta",
    )
    table.add_column("Rank", justify="right", style="dim")
    table.add_column("Feature", style="cyan")
    table.add_column(value_col, justify="right")
    for rank, (_idx, row) in enumerate(ordered.iterrows(), start=1):
        value = row[value_col]
        table.add_row(str(rank), str(row[feature_col]), f"{value:,.4g}")
    return table


def print_gain_importance(df, *, n: int = 15) -> None:
    if df is None or len(df) == 0:
        warning("No gain-importance frame available.")
        return
    console.print(
        _importance_table("Gain importance", df, "feature", "importance_gain", n=n)
    )


def print_permutation_importance(df, *, n: int = 15) -> None:
    if df is None or len(df) == 0:
        warning("No permutation-importance frame available.")
        return
    console.print(
        _importance_table(
            "Permutation importance (mean drop in AP)",
            df,
            "feature",
            "importance_mean",
            n=n,
        )
    )


def print_shap_importance(df, *, n: int = 15) -> None:
    if df is None or len(df) == 0:
        warning("No SHAP-importance frame available.")
        return
    console.print(
        _importance_table(
            "SHAP importance (mean |value|)", df, "feature", "mean_abs_shap", n=n
        )
    )


# ---------------------------------------------------------------------------
# Optuna
# ---------------------------------------------------------------------------


def print_optuna_search_space(space: Mapping[str, Any]) -> None:
    """Best-effort summary of a hyperparameter search space dict."""
    rows = [(k, v) for k, v in sorted(space.items())]
    console.print(kv_table("LightGBM search space", rows, key_header="Param"))


def print_optuna_summary(study: Any, *, n_top: int = 10) -> None:
    """Summarise a completed Optuna study: best value, best params, top trials."""
    try:
        best_value = float(study.best_value)
        best_params = dict(study.best_params)
        trials = list(study.trials)
    except Exception as exc:  # noqa: BLE001 - diagnostics should never abort a run
        warning(f"Could not summarise study: {exc}")
        return

    from optuna.trial import TrialState

    completed = [t for t in trials if t.state == TrialState.COMPLETE]
    pruned = [t for t in trials if t.state == TrialState.PRUNED]
    failed = [t for t in trials if t.state == TrialState.FAIL]

    overview_rows = [
        ("best AP", f"{best_value:.6f}"),
        ("trials (total)", len(trials)),
        ("trials (completed)", len(completed)),
        ("trials (pruned)", len(pruned)),
        ("trials (failed)", len(failed)),
    ]
    console.print(kv_table("Optuna study", overview_rows, key_header="Summary"))

    print_params("Best params", best_params)

    if completed:
        table = Table(
            title=f"Top {min(n_top, len(completed))} trials by AP",
            title_style="bold",
            box=SIMPLE_HEAVY,
            header_style="bold magenta",
        )
        table.add_column("Trial", justify="right", style="dim")
        table.add_column("AP", justify="right")
        table.add_column("learning_rate", justify="right")
        table.add_column("num_leaves", justify="right")
        table.add_column("min_data", justify="right")
        table.add_column("feat_frac", justify="right")
        table.add_column("bag_frac", justify="right")
        ranked = sorted(completed, key=lambda t: (t.value is None, -(t.value or 0.0)))[
            :n_top
        ]
        for t in ranked:
            p = t.params
            table.add_row(
                str(t.number),
                f"{t.value:.6f}" if t.value is not None else "-",
                _fmt(p.get("learning_rate")),
                _fmt(p.get("num_leaves")),
                _fmt(p.get("min_data_in_leaf")),
                _fmt(p.get("feature_fraction")),
                _fmt(p.get("bagging_fraction")),
            )
        console.print(table)


# ---------------------------------------------------------------------------
# Artefact listing
# ---------------------------------------------------------------------------


def print_artefact_summary(output_dir: Path) -> None:
    """Walk ``output_dir`` and group artefacts by type for a tidy final report."""
    if not output_dir.exists():
        warning(f"Output directory {output_dir} does not exist.")
        return

    entries = sorted(p for p in output_dir.rglob("*") if p.is_file())
    if not entries:
        warning(f"No artefacts under {output_dir}.")
        return

    by_suffix: dict[str, list[Path]] = {}
    for path in entries:
        suffix = path.suffix.lower() or "(no-ext)"
        by_suffix.setdefault(suffix, []).append(path)

    table = Table(
        title=f"Artefacts in {output_dir}",
        title_style="bold",
        box=SIMPLE_HEAVY,
        header_style="bold magenta",
    )
    table.add_column("Type", style="cyan", no_wrap=True)
    table.add_column("Count", justify="right")
    table.add_column("Files", overflow="fold")
    for suffix in sorted(by_suffix):
        paths = by_suffix[suffix]
        rels = [str(p.relative_to(output_dir)) for p in paths]
        table.add_row(suffix, str(len(paths)), ", ".join(rels))
    console.print(table)
