"""One specification for DSP construction, persistence and reporting."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from dspopulations_us_birth_certificates.selection.anomaly_panel import (
    DEFAULT_PANEL_CONDITION_TREND_SIGMA,
    DEFAULT_PANEL_FACTOR_SIGMA,
    DEFAULT_PANEL_IDIOSYNCRATIC_SIGMA,
    DEFAULT_PANEL_LOADING_SIGMA,
    DEFAULT_PANEL_PREVALENCE_TREND_SIGMA,
    AnomalyPanel,
)
from dspopulations_us_birth_certificates.selection.core_models import (
    CoreModelDefinition,
    validate_core_model_definition,
)
from dspopulations_us_birth_certificates.selection.core_reduction import (
    DEFAULT_ANCHOR_LEVEL_SIGMA,
    DEFAULT_ANCHOR_OBS_SIGMA,
    DEFAULT_ANCHOR_OBS_SIGMA_FIXED,
    DEFAULT_ANCHOR_TREND_SIGMA,
    CoreReductionModelConfig,
    CoreReductionPriors,
    SurveillanceAnchor,
    build_core_reduction_model,
)
from dspopulations_us_birth_certificates.selection.core_validation import integer_array


@dataclass(frozen=True)
class AnchorSettings:
    level_sigma: float = DEFAULT_ANCHOR_LEVEL_SIGMA
    trend_sigma: float = DEFAULT_ANCHOR_TREND_SIGMA
    obs_sigma: float = DEFAULT_ANCHOR_OBS_SIGMA
    obs_sigma_fixed: float | None = DEFAULT_ANCHOR_OBS_SIGMA_FIXED
    forecast_flat: bool = False
    overlap_share: float = 1.0
    level_prior_prevalence: float = 0.0013


@dataclass(frozen=True)
class PanelSettings:
    prevalence_trend_sigma: float = DEFAULT_PANEL_PREVALENCE_TREND_SIGMA
    loading_sigma: float = DEFAULT_PANEL_LOADING_SIGMA
    loading_fixed: float | None = None
    factor_sigma: float = DEFAULT_PANEL_FACTOR_SIGMA
    condition_trend_sigma: float = DEFAULT_PANEL_CONDITION_TREND_SIGMA
    idiosyncratic_sigma: float = DEFAULT_PANEL_IDIOSYNCRATIC_SIGMA


@dataclass(frozen=True)
class CoreFitSpecification:
    definition: CoreModelDefinition
    priors: CoreReductionPriors
    year_range: tuple[int, int]
    recorded_definition: str = "confirmed_or_pending"
    anchor: SurveillanceAnchor | None = None
    panel: AnomalyPanel | None = None
    anchor_settings: AnchorSettings = field(default_factory=AnchorSettings)
    panel_settings: PanelSettings = field(default_factory=PanelSettings)

    def __post_init__(self):
        validate_core_model_definition(self.definition)
        years = integer_array(self.year_range, "year range")
        if years.shape != (2,):
            raise ValueError("year range must contain its start and end years")
        if self.year_range[1] < self.year_range[0]:
            raise ValueError("year range must be ordered")
        if (self.definition.reduction_model == "anchor") != (self.anchor is not None):
            raise ValueError("anchor input must match the model definition")
        if (self.definition.recording_panel == "anomaly") != (self.panel is not None):
            raise ValueError("panel input must match the model definition")

    def to_config(self) -> CoreReductionModelConfig:
        return CoreReductionModelConfig.from_priors(
            year_range=self.year_range,
            priors_obj=self.priors,
            model_definition=self.definition,
            recorded_definition=self.recorded_definition,
            anchor=self.anchor,
            anchor_hyperpriors=asdict(self.anchor_settings),
            panel=self.panel,
            panel_hyperpriors={
                "panel_" + k: v for k, v in asdict(self.panel_settings).items()
            },
        )

    def build(self, cells):
        recorded_range = cells.attrs.get("year_range")
        if recorded_range is not None and tuple(recorded_range) != self.year_range:
            raise ValueError("cell calendar years do not match the specification")
        if self.anchor is not None and self.anchor.mid_years:
            if not np.array_equal(
                np.asarray(self.anchor.mid_years) - self.year_range[0],
                self.anchor.mid_year_idx,
            ):
                raise ValueError("anchor calendar years do not match the specification")
        if self.panel is not None:
            if not np.array_equal(
                np.asarray(self.panel.years) - self.year_range[0], self.panel.year_idx
            ):
                raise ValueError("panel calendar years do not match the specification")
        exact = "maternal_age" in cells
        if exact != (self.definition.age_model == "single_year"):
            raise ValueError("cell ages do not match the model definition")
        model = build_core_reduction_model(
            cells,
            self.priors,
            n_year=self.year_range[1] - self.year_range[0] + 1,
            recording_model=self.definition.recording_model,
            reduction_model=self.definition.reduction_model,
            recording_drift=self.definition.recording_drift,
            recording_panel=self.definition.recording_panel,
            anchor=self.anchor,
            panel=self.panel,
            **{"anchor_" + k: v for k, v in asdict(self.anchor_settings).items()},
            **{"panel_" + k: v for k, v in asdict(self.panel_settings).items()},
        )
        model.dsp_specification = self.to_config().to_dict()
        return model
