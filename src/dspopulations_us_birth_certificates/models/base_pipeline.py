"""Abstract pipeline that orchestrates one model run.

``EstimatorPipeline`` defines the sequence of steps — each a method that
reads and writes ``self.context`` (a ``ModelFitContext``). Subclasses
(``LGBMClassifierPipeline``, future sklearn wrappers) override
``configure_estimator``, ``train_fold``, and ``train_final``.

Every step is individually callable so a notebook or a test can drive the
pipeline at any granularity. ``fit()`` is the full-run convenience.
"""

from __future__ import annotations

import json
import logging
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from dspopulations_us_birth_certificates import (
    cli_output,
    data_utils,
    ml_utils,
    plot_utils,
    stats_utils,
)
from dspopulations_us_birth_certificates.explain import calibration, shap_analysis
from dspopulations_us_birth_certificates.models.common import (
    ModelConfig,
    ModelFitContext,
    RunConfig,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class EstimatorPipeline(ABC):
    """Orchestrates one model run end to end."""

    def __init__(
        self,
        config: ModelConfig,
        run_config: RunConfig,
        output_dir: Path | str,
    ) -> None:
        self.context = self._build_context(config, run_config, Path(output_dir))

    def _build_context(
        self,
        config: ModelConfig,
        run_config: RunConfig,
        output_dir: Path,
    ) -> ModelFitContext:
        """Build an empty ``ModelFitContext`` rooted at the run's output dir."""
        output_dir.mkdir(parents=True, exist_ok=True)
        return ModelFitContext(
            config=config, run_config=run_config, output_dir=output_dir
        )

    # ---- data ----------------------------------------------------------------

    def load_data(
        self,
        *,
        db_path: str | None = None,
    ) -> pd.DataFrame:
        """Load the harmonised predictors frame for the configured year range."""
        cli_output.section("Load data")
        cfg = self.context.config
        cli_output.info(
            f"Year range [bold]{cfg.year_range[0]}-{cfg.year_range[1]}[/bold], "
            f"include_unknown=[bold]{cfg.include_unknown}[/bold]"
        )
        if db_path is not None:
            cli_output.info(f"DuckDB: [blue]{db_path}[/blue]")
        kwargs: dict[str, Any] = dict(
            from_year=cfg.year_range[0],
            to_year=cfg.year_range[1],
            include_unknown=cfg.include_unknown,
        )
        if db_path is not None:
            kwargs["db_path"] = db_path
        df = data_utils.load_predictors_data(**kwargs)
        target = cfg.target_var
        positives = int(df[target].fillna(0).astype(int).sum())
        cli_output.print_data_summary(
            df_rows=len(df),
            target_var=target,
            positives=positives,
            year_range=cfg.year_range,
            include_unknown=cfg.include_unknown,
        )
        return df

    def prepare_features(self, df: pd.DataFrame) -> None:
        """Populate ``X_train/y_train/X_valid/y_valid`` on the context."""
        cli_output.section("Prepare features")
        cfg = self.context.config
        numeric = list(cfg.numeric_features)
        categorical = list(cfg.categorical_features)
        features = categorical + numeric

        X = df[features].copy()
        y = df[cfg.target_var].replace({pd.NA: 0, np.nan: 0}).astype(np.int32)
        X[categorical] = X[categorical].astype("category")

        training_split = cfg.train_config.get("training_split", 0.8)
        random_seed = self.context.run_config.random_seed

        cli_output.info(
            f"Features: [bold]{len(features)}[/bold] "
            f"([cyan]{len(categorical)} categorical[/cyan], "
            f"[cyan]{len(numeric)} numeric[/cyan])"
        )
        cli_output.info(
            f"Stratified split {training_split:.0%}/{1 - training_split:.0%} "
            f"(seed={random_seed})"
        )

        X_train, X_valid, y_train, y_valid = train_test_split(
            X,
            y,
            test_size=1 - training_split,
            stratify=y,
            random_state=random_seed,
        )

        self.context.X_train = X_train
        self.context.y_train = y_train
        self.context.X_valid = X_valid
        self.context.y_valid = y_valid

        cli_output.print_split_summary(X_train, X_valid, y_train, y_valid)

    # ---- training ------------------------------------------------------------

    @abstractmethod
    def configure_estimator(self) -> Any:
        """Return a fresh, unfitted estimator configured for this variant."""

    @abstractmethod
    def train_fold(self, fold_idx: int) -> Any:
        """Train one fold's estimator and append it to ``context.fold_models``."""

    @abstractmethod
    def train_final(self) -> Any:
        """Fit the final estimator on all training data for artefact export."""

    def cross_validate(self) -> None:
        """Run k-fold CV using ``run_config.cv_splits``.

        Not implemented in step 4; the single train/valid split from
        ``prepare_features`` is used instead. Tracked for a follow-up.
        """
        raise NotImplementedError(
            "cross_validate will land in a follow-up PR; "
            "step 4 uses the single stratified split from prepare_features."
        )

    # ---- evaluation ----------------------------------------------------------

    def compute_metrics(self) -> None:
        """Compute AP, log-loss, Brier, ROC-AUC, P@K, tail calibration."""
        cli_output.section("Compute metrics")
        ctx = self.context
        if ctx.final_model is None:
            raise RuntimeError("train_final() must run before compute_metrics().")

        p_valid = self._predict_valid()
        ctx.p_valid = p_valid

        y_valid = np.asarray(ctx.y_valid)
        metrics = {
            "average_precision": float(average_precision_score(y_valid, p_valid)),
            "roc_auc": float(roc_auc_score(y_valid, p_valid)),
            "log_loss": float(log_loss(y_valid, p_valid, labels=[0, 1])),
            "brier_score": float(brier_score_loss(y_valid, p_valid)),
            "mean_predicted_prob": float(np.asarray(p_valid).mean()),
            "best_iteration": (
                int(ctx.best_iteration) if ctx.best_iteration is not None else None
            ),
            "n_valid": int(len(y_valid)),
            "n_positive_valid": int(y_valid.sum()),
        }
        ctx.metrics = metrics
        cli_output.print_metrics_table(metrics)

    def permutation_importance_analysis(self) -> None:
        """Populate ``context.permutation_importance``."""
        cli_output.section("Permutation importance")
        ctx = self.context
        if ctx.final_model is None:
            raise RuntimeError(
                "train_final() must run before permutation_importance_analysis()."
            )

        X_eval, y_eval = ml_utils.build_explain_set(
            ctx.final_model,
            ctx.X_valid,
            ctx.y_valid,
            list(ctx.config.categorical_features),
        )
        model_wrapped = ml_utils.LGBMEstimator(ctx.final_model)
        # Derive n_jobs from the model's train_config when set; otherwise let
        # sklearn pick a default. This avoids hard-coded CPU oversubscription
        # in CI/containers and honours the CLI --num-threads flag.
        n_jobs: int | None = None
        train_config = getattr(ctx.config, "train_config", None)
        if isinstance(train_config, dict):
            configured = train_config.get("num_threads")
            try:
                configured_int = int(configured) if configured is not None else None
            except TypeError, ValueError:
                configured_int = None
            if configured_int is not None and configured_int > 0:
                n_jobs = configured_int

        cli_output.info(
            f"Permuting {X_eval.shape[1]} features × 20 repeats "
            f"on {len(X_eval):,} rows "
            f"(n_jobs={n_jobs or 'auto'})"
        )

        result = permutation_importance(
            model_wrapped,
            X_eval,
            y_eval,
            scoring=ml_utils.ap_scorer,
            n_repeats=20,
            n_jobs=n_jobs,
            random_state=ctx.run_config.random_seed,
        )
        ctx.permutation_importance = {
            "result": result,
            "X_eval": X_eval,
            "y_eval": y_eval,
        }
        perm_df = pd.DataFrame(
            {
                "feature": X_eval.columns,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        ).sort_values("importance_mean", ascending=False)
        cli_output.print_permutation_importance(perm_df)

    def shap_analysis(self) -> None:
        """Populate ``context.shap_explanation`` subject to ``run_config.shap_mode``."""
        cli_output.section("SHAP analysis")
        ctx = self.context
        if ctx.final_model is None:
            raise RuntimeError("train_final() must run before shap_analysis().")
        if ctx.run_config.shap_mode == "skip":
            cli_output.info("SHAP skipped per run_config.shap_mode='skip'.")
            logger.info("SHAP skipped per run_config.shap_mode='skip'.")
            return

        if ctx.permutation_importance is None:
            # Share the explain set with permutation importance when available.
            X_eval, _ = ml_utils.build_explain_set(
                ctx.final_model,
                ctx.X_valid,
                ctx.y_valid,
                list(ctx.config.categorical_features),
            )
        else:
            X_eval = ctx.permutation_importance["X_eval"]

        if ctx.run_config.shap_mode == "subsample":
            size = ctx.run_config.shap_subsample_size
            if size is not None and size < len(X_eval):
                rng = np.random.default_rng(ctx.run_config.random_seed)
                sampled = rng.choice(len(X_eval), size=size, replace=False)
                X_eval = X_eval.iloc[sampled]

        cli_output.info(
            f"Mode=[bold]{ctx.run_config.shap_mode}[/bold], "
            f"evaluating SHAP on [bold]{len(X_eval):,}[/bold] rows × "
            f"[bold]{X_eval.shape[1]}[/bold] features"
        )

        ctx.shap_explanation = shap_analysis.compute_explanation(
            ctx.final_model, X_eval
        )
        shap_df = shap_analysis.shap_importance(ctx.shap_explanation, X_eval.columns)
        cli_output.print_shap_importance(shap_df)

    # ---- outputs -------------------------------------------------------------

    def save_artefacts(self, *, save_plots: bool = True) -> None:
        """Write ``model.txt``, predictions, metrics, importances, plots.

        ``save_plots=False`` skips the plot generation hook so callers that
        just want the tabular artefacts (CLI ``--no-plots``, faster CI runs)
        don't pay the matplotlib cost or trigger GUI rendering.
        """
        cli_output.section("Save artefacts")
        ctx = self.context
        out = ctx.output_dir
        cli_output.info(f"Writing to [blue]{out}[/blue] (plots={save_plots})")

        # Config + metrics
        (out / "config.json").write_text(json.dumps(ctx.config.to_dict(), indent=2))
        (out / "metrics.json").write_text(json.dumps(ctx.metrics, indent=2))

        # Model
        if ctx.final_model is not None and ctx.best_iteration is not None:
            self._save_final_model(out / "model.txt")

        # Validation predictions
        if ctx.p_valid is not None:
            pd.DataFrame(
                {
                    "y_true": np.asarray(ctx.y_valid),
                    "p_pred": np.asarray(ctx.p_valid),
                }
            ).to_parquet(out / "predictions_valid.parquet", index=False)

        # Built-in gain importance
        imp = self._gain_importance()
        if imp is not None:
            imp.to_csv(out / "feature_importance_gain.csv", index=False)
            cli_output.print_gain_importance(imp)

        # Calibration tables
        if ctx.p_valid is not None:
            pr = calibration.precision_recall_at_k(
                np.asarray(ctx.y_valid), np.asarray(ctx.p_valid)
            )
            pr.to_csv(out / "precision_recall_at_k.csv", index=False)
            tail = calibration.tail_calibration_table(
                np.asarray(ctx.y_valid), np.asarray(ctx.p_valid)
            )
            tail.to_csv(out / "calibration_tail.csv", index=False)

        # Permutation importance
        if ctx.permutation_importance is not None:
            result = ctx.permutation_importance["result"]
            X_eval = ctx.permutation_importance["X_eval"]
            perm_df = pd.DataFrame(
                {
                    "feature": X_eval.columns,
                    "importance_mean": result.importances_mean,
                    "importance_std": result.importances_std,
                }
            ).sort_values("importance_mean", ascending=False)
            perm_df.to_csv(out / "permutation_importance.csv", index=False)

        # SHAP
        if ctx.shap_explanation is not None:
            X_eval = (
                ctx.permutation_importance["X_eval"]
                if ctx.permutation_importance is not None
                else None
            )
            feature_names = (
                X_eval.columns
                if X_eval is not None
                else list(ctx.config.categorical_features)
                + list(ctx.config.numeric_features)
            )
            shap_df = shap_analysis.shap_importance(ctx.shap_explanation, feature_names)
            shap_df.to_csv(out / "shap_importance.csv", index=False)

        # Plots
        if save_plots:
            plots_dir = out / "plots"
            plots_dir.mkdir(exist_ok=True)
            self._save_plots(plots_dir)

        cli_output.success(f"Artefacts written to {out}")
        cli_output.print_artefact_summary(out)

    def write_manifest(self) -> None:
        """Delegate to ``manifest.write_manifest`` for this run."""
        cli_output.section("Write manifest")
        from dspopulations_us_birth_certificates import manifest

        path = manifest.write_manifest(self.context, self.context.output_dir)
        cli_output.success(f"Manifest: {path}")

    def report(
        self,
        render: bool = False,
        template_path: Path | None = None,
    ) -> None:
        """Copy the Quarto template into the run dir; render it if requested.

        Non-rendering is cheap and always safe — the ``index.qmd`` ends up
        alongside the run's artefacts so the report can be rendered later.
        ``render=True`` invokes the ``quarto`` CLI; misses the CLI are
        logged rather than raised so the pipeline run still completes.
        """
        cli_output.section("Report")
        from dspopulations_us_birth_certificates import reporting

        tpl = template_path or reporting.DEFAULT_TEMPLATE
        try:
            qmd = reporting.copy_template(self.context, tpl)
        except FileNotFoundError as exc:
            cli_output.warning(f"Skipping report: {exc}")
            logger.warning("Skipping report: %s", exc)
            return

        cli_output.success(f"Copied template to {qmd}")

        if render:
            try:
                reporting.render_quarto_report(qmd)
                cli_output.success("Quarto render complete")
            except (FileNotFoundError, subprocess.CalledProcessError) as exc:
                # Expected operational failures (missing CLI, non-zero exit)
                # are logged but don't fail the run. Any other exception is
                # programmer error and should surface.
                cli_output.warning(f"Quarto render failed: {exc}")
                logger.warning("Quarto render failed: %s", exc)
        else:
            cli_output.info(f"To render: [blue]quarto render {qmd}[/blue]")

    # ---- convenience ---------------------------------------------------------

    def fit(
        self,
        render: bool = False,
        db_path: str | None = None,
        *,
        run_permutation: bool | None = None,
        save_plots: bool = True,
    ) -> ModelFitContext:
        """Run every step in order and return the populated context.

        ``run_permutation`` defaults to ``True`` for ``test``/``reporting``
        presets and ``False`` for ``dev`` (since permutation importance is
        slow and the dev preset exists for the fast inner loop). Pass an
        explicit value to override.
        """
        if run_permutation is None:
            run_permutation = self.context.run_config.name != "dev"

        df = self.load_data(db_path=db_path)
        self.prepare_features(df)
        self.train_final()
        self.compute_metrics()
        if run_permutation:
            self.permutation_importance_analysis()
        self.shap_analysis()
        self.save_artefacts(save_plots=save_plots)
        self.write_manifest()
        self.report(render=render)
        return self.context

    # ---- helpers subclasses can override -------------------------------------

    def _predict_valid(self) -> np.ndarray:
        """Default prediction path — overridden when the estimator needs it."""
        raise NotImplementedError

    def _save_final_model(self, path: Path) -> None:
        """Serialise the final model — overridden by concrete pipelines."""
        raise NotImplementedError

    def _gain_importance(self) -> pd.DataFrame | None:
        """Return a (feature, importance_gain) frame if the estimator supports it."""
        return None

    def _save_plots(self, plots_dir: Path) -> None:
        """Overridable hook for standard ROC/PR/dendrogram/heatmap/SHAP plots.

        Plot helpers return ``Figure`` objects; this method saves them and
        then closes each one so CLI runs don't accumulate figures in the
        matplotlib registry.
        """
        import matplotlib.pyplot as plt

        ctx = self.context
        if ctx.p_valid is None:
            return

        from sklearn.metrics import precision_recall_curve, roc_curve

        fpr, tpr, _ = roc_curve(np.asarray(ctx.y_valid), np.asarray(ctx.p_valid))
        prec, rec, _ = precision_recall_curve(
            np.asarray(ctx.y_valid), np.asarray(ctx.p_valid)
        )

        figs: list = []

        figs.append(
            plot_utils.plot_roc_curve(
                fpr,
                tpr,
                0,
                save=True,
                output_dir=str(plots_dir),
                file_name="roc_curve",
            )
        )
        figs.append(
            plot_utils.plot_precision_recall_curve(
                rec,
                prec,
                0,
                save=True,
                output_dir=str(plots_dir),
                file_name="precision_recall_curve",
            )
        )

        if ctx.permutation_importance is not None:
            figs.append(
                plot_utils.plot_permutation_importances(
                    ctx.permutation_importance["result"],
                    ctx.permutation_importance["X_eval"],
                    0,
                    save=True,
                    output_dir=str(plots_dir),
                    file_name="permutation_importances",
                )
            )

            X_eval = ctx.permutation_importance["X_eval"]
            distance, corr = stats_utils.distance_corr_dissimilarity(X_eval)
            condensed = squareform(distance, checks=True)
            linkage = hierarchy.linkage(condensed, method="average")
            dendro_fig, dendro = plot_utils.plot_dendrogram(
                linkage,
                X_eval.columns.to_list(),
                0,
                save=True,
                output_dir=str(plots_dir),
                file_name="dendrogram",
            )
            figs.append(dendro_fig)
            figs.append(
                plot_utils.plot_correlation_heatmap(
                    corr,
                    dendro,
                    label_threshold=0.3,
                    model_idx=0,
                    save=True,
                    output_dir=str(plots_dir),
                    file_name="correlation_heatmap",
                )
            )

        if ctx.shap_explanation is not None:
            figs.append(
                shap_analysis.plot_bar(
                    ctx.shap_explanation,
                    save=True,
                    output_dir=str(plots_dir),
                    file_stem="shap_bar",
                )
            )
            figs.append(
                shap_analysis.plot_beeswarm(
                    ctx.shap_explanation,
                    save=True,
                    output_dir=str(plots_dir),
                    file_stem="shap_beeswarm",
                )
            )
            for spec in ctx.config.shap_scatter_specs:
                figs.append(
                    shap_analysis.plot_scatter(
                        ctx.shap_explanation,
                        x_feature=spec.x_feature,
                        colour_feature=spec.colour_by_feature,
                        save=True,
                        output_dir=str(plots_dir),
                        file_stem=(
                            f"shap_{spec.x_feature}"
                            + (
                                f"_vs_{spec.colour_by_feature}"
                                if spec.colour_by_feature
                                else ""
                            )
                        ),
                    )
                )

        for fig in figs:
            plt.close(fig)
