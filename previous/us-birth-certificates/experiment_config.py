"""Dataclasses defining experiment parameters for predictor model experiments."""

from dataclasses import dataclass, field


@dataclass
class ModelVariantConfig:
    """Configuration for a single model variant within an experiment.

    Each experiment typically trains 2-3 model variants that progressively
    remove low-importance features.
    """

    index: int
    label: str
    drop_features: list[str] = field(default_factory=list)
    shap_scatter_pairs: list[tuple[str, str]] = field(default_factory=list)
    """Feature pairs for SHAP dependence scatter plots, e.g. [("year", "mage_c")]."""
    shap_bar_max_display: int = 45
    shap_beeswarm_max_display: int = 45
    shap_beeswarm_plot_size: tuple[int, int] = (8, 10)


@dataclass
class ExperimentConfig:
    """Full configuration for a predictor experiment.

    Captures everything that varies between predictor notebooks:
    year range, split ratios, boost parameters, feature sets,
    hyperparameter search settings, and model variant definitions.
    """

    name: str
    start_year: int = 2016
    end_year: int = 2024
    include_unknown: bool = True
    training_split: float = 0.75
    num_boost_round: int = 50_000
    early_stopping_rounds: int = 200
    select_hyperparameters: bool = True
    optimize_trials: int = 200
    random_seed: int = 47
    save_plots: bool = True
    numeric_features: list[str] = field(default_factory=list)
    categorical_features: list[str] = field(default_factory=list)
    base_params: dict = field(default_factory=dict)
    prior_best_params: dict = field(default_factory=dict)
    model_variants: list[ModelVariantConfig] = field(default_factory=list)

    @property
    def all_features(self) -> list[str]:
        return self.categorical_features + self.numeric_features


DEFAULT_BASE_PARAMS: dict = {
    "objective": "binary",
    "metric": ["average_precision", "binary_logloss"],
    "boosting_type": "gbdt",
    "max_bin": 255,
    "scale_pos_weight": 1,
    "force_col_wise": True,
}

DEFAULT_PRIOR_BEST_PARAMS: dict = {
    "learning_rate": 0.009461164726049449,
    "num_leaves": 180,
    "min_data_in_leaf": 756,
    "min_gain_to_split": 0.9285634625013361,
    "feature_fraction": 0.9239582799934513,
    "bagging_fraction": 0.9185684081749333,
    "bagging_freq": 2,
    "lambda_l1": 0.0005836073944757167,
    "lambda_l2": 0.6142323696066677,
}
