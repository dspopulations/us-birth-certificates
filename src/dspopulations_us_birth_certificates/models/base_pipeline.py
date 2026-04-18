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
        cfg = self.context.config
        kwargs: dict[str, Any] = dict(
            from_year=cfg.year_range[0],
            to_year=cfg.year_range[1],
            include_unknown=cfg.include_unknown,
        )
        if db_path is not None:
            kwargs["db_path"] = db_path
        return data_utils.load_predictors_data(**kwargs)

    def prepare_features(self, df: pd.DataFrame) -> None:
        """Populate ``X_train/y_train/X_valid/y_valid`` on the context."""
        cfg = self.context.config
        numeric = list(cfg.numeric_features)
        categorical = list(cfg.categorical_features)
        features = categorical + numeric

        X = df[features].copy()
        y = df[cfg.target_var].replace({pd.NA: 0, np.nan: 0}).astype(np.int32)
        X[categorical] = X[categorical].astype("category")

        training_split = cfg.train_config.get("training_split", 0.8)
        random_seed = self.context.run_config.random_seed

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

    def permutation_importance_analysis(self) -> None:
        """Populate ``context.permutation_importance``."""
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
        result = permutation_importance(
            model_wrapped,
            X_eval,
            y_eval,
            scoring=ml_utils.ap_scorer,
            n_repeats=20,
            n_jobs=8,
            random_state=ctx.run_config.random_seed,
        )
        ctx.permutation_importance = {
            "result": result,
            "X_eval": X_eval,
            "y_eval": y_eval,
        }

    def shap_analysis(self) -> None:
        """Populate ``context.shap_explanation`` subject to ``run_config.shap_mode``."""
        ctx = self.context
        if ctx.run_config.shap_mode == "skip":
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

        ctx.shap_explanation = shap_analysis.compute_explanation(
            ctx.final_model, X_eval
        )

    # ---- outputs -------------------------------------------------------------

    def save_artefacts(self) -> None:
        """Write ``model.txt``, predictions, metrics, importances, plots."""
        ctx = self.context
        out = ctx.output_dir

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
                else list(ctx.config.numeric_features)
                + list(ctx.config.categorical_features)
            )
            shap_df = shap_analysis.shap_importance(
                ctx.shap_explanation, feature_names
            )
            shap_df.to_csv(out / "shap_importance.csv", index=False)

        # Plots
        plots_dir = out / "plots"
        plots_dir.mkdir(exist_ok=True)
        self._save_plots(plots_dir)

    def write_manifest(self) -> None:
        """Delegate to ``manifest.write_manifest`` for this run."""
        from dspopulations_us_birth_certificates import manifest

        manifest.write_manifest(self.context, self.context.output_dir)

    def report(self, render: bool = False) -> None:
        """Copy the Quarto template into the run dir — populated in step 7."""
        logger.info(
            "Quarto reporting is a no-op until refactor step 7 lands."
        )

    # ---- convenience ---------------------------------------------------------

    def fit(self, render: bool = False, db_path: str | None = None) -> ModelFitContext:
        """Run every step in order and return the populated context."""
        df = self.load_data(db_path=db_path)
        self.prepare_features(df)
        self.train_final()
        self.compute_metrics()
        self.permutation_importance_analysis()
        self.shap_analysis()
        self.save_artefacts()
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
        """Overridable hook for standard ROC/PR/dendrogram/heatmap/SHAP plots."""
        ctx = self.context
        if ctx.p_valid is None:
            return

        # ROC + PR curves
        from sklearn.metrics import precision_recall_curve, roc_curve

        fpr, tpr, _ = roc_curve(np.asarray(ctx.y_valid), np.asarray(ctx.p_valid))
        prec, rec, _ = precision_recall_curve(
            np.asarray(ctx.y_valid), np.asarray(ctx.p_valid)
        )

        plot_utils.plot_roc_curve(
            fpr,
            tpr,
            0,
            save=True,
            output_dir=str(plots_dir),
            file_name="roc_curve",
        )
        plot_utils.plot_precision_recall_curve(
            rec,
            prec,
            0,
            save=True,
            output_dir=str(plots_dir),
            file_name="precision_recall_curve",
        )

        # Permutation importance plot
        if ctx.permutation_importance is not None:
            plot_utils.plot_permutation_importances(
                ctx.permutation_importance["result"],
                ctx.permutation_importance["X_eval"],
                0,
                save=True,
                output_dir=str(plots_dir),
                file_name="permutation_importances",
            )

            # Dendrogram + correlation heatmap
            X_eval = ctx.permutation_importance["X_eval"]
            distance, corr = stats_utils.distance_corr_dissimilarity(X_eval)
            condensed = squareform(distance, checks=True)
            linkage = hierarchy.linkage(condensed, method="average")
            dendro = plot_utils.plot_dendrogram(
                linkage,
                X_eval.columns.to_list(),
                0,
                save=True,
                output_dir=str(plots_dir),
                file_name="dendrogram",
            )
            plot_utils.plot_correlation_heatmap(
                corr,
                dendro,
                label_threshold=0.3,
                model_idx=0,
                save=True,
                output_dir=str(plots_dir),
                file_name="correlation_heatmap",
            )

        # SHAP plots
        if ctx.shap_explanation is not None:
            shap_analysis.plot_bar(
                ctx.shap_explanation,
                save=True,
                output_dir=str(plots_dir),
                file_stem="shap_bar",
                show=False,
            )
            shap_analysis.plot_beeswarm(
                ctx.shap_explanation,
                save=True,
                output_dir=str(plots_dir),
                file_stem="shap_beeswarm",
                show=False,
            )
            for spec in ctx.config.shap_scatter_specs:
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
                    show=False,
                )
