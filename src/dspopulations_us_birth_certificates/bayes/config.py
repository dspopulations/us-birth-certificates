"""Run / model configuration for the Bayesian regression pipeline.

Mirrors the shape of ``models/common.py`` so contributors moving between
the LightGBM and Bayesian pipelines can rely on the same mental model:

- ``BayesRunConfig`` — speed / fidelity preset chosen at CLI time (``dev``
  for fast inner-loop smoke runs, ``reporting`` for publication-quality
  posteriors). Presets are authoritative — individual flags override.
- ``BayesModelConfig`` — serialisable snapshot of a ``BayesModelDefinition``
  at fit time, written to ``config.json`` next to the InferenceData.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

BayesRunConfigName = Literal["dev", "reporting"]
OutcomeName = Literal["recorded", "recorded_plus_predicted"]


_PRESETS: dict[str, dict[str, Any]] = {
    "dev": {
        "draws": 500,
        "tune": 500,
        "chains": 2,
        "target_accept": 0.8,
        "prior_predictive_samples": 500,
        "posterior_predictive": True,
        "nuts_sampler": "nutpie",
    },
    "reporting": {
        "draws": 2000,
        "tune": 2000,
        "chains": 4,
        "target_accept": 0.9,
        "prior_predictive_samples": 1000,
        "posterior_predictive": True,
        "nuts_sampler": "nutpie",
    },
}


@dataclass(frozen=True)
class BayesRunConfig:
    """Speed / fidelity preset for a Bayesian fit."""

    name: BayesRunConfigName
    draws: int
    tune: int
    chains: int
    target_accept: float
    prior_predictive_samples: int
    posterior_predictive: bool
    nuts_sampler: str
    random_seed: int = 47

    @classmethod
    def from_name(
        cls, name: BayesRunConfigName, *, random_seed: int = 47
    ) -> BayesRunConfig:
        if name not in _PRESETS:
            raise ValueError(
                f"Unknown BayesRunConfig preset {name!r}. "
                f"Valid names: {sorted(_PRESETS)}"
            )
        return cls(name=name, random_seed=random_seed, **_PRESETS[name])

    @classmethod
    def preset_names(cls) -> tuple[str, ...]:
        return tuple(_PRESETS)


@dataclass
class BayesModelConfig:
    """Serialisable snapshot of a ``BayesModelDefinition`` at fit time."""

    model_id: str
    dims: tuple[str, ...]
    year_range: tuple[int, int]
    outcome: OutcomeName
    outcome_params: dict[str, Any] = field(default_factory=dict)
    priors: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "dims": list(self.dims),
            "year_range": list(self.year_range),
            "outcome": self.outcome,
            "outcome_params": dict(self.outcome_params),
            "priors": dict(self.priors),
            "notes": self.notes,
        }


@dataclass
class BayesFitContext:
    """Mutable state threaded through the Bayesian fit steps.

    Populated progressively by the CLI: cells → model → idata → artefacts.
    """

    config: BayesModelConfig
    run_config: BayesRunConfig
    output_dir: Any = None  # pathlib.Path, kept loose to avoid import cost
    cells: Any = None
    model: Any = None
    idata: Any = None
    summary: Any = None
    metrics: dict[str, Any] = field(default_factory=dict)


def asdict_bayes_run_config(cfg: BayesRunConfig) -> dict[str, Any]:
    """JSON-serialisable dict for a run config."""
    return asdict(cfg)
