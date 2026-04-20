"""Run / model configuration for the selection-model fit pipeline.

Parallels ``dspopulations_us_birth_certificates.bayes.config`` and reuses
its :class:`BayesRunConfig` dataclass — the shape (draws / tune / chains
/ target_accept / sampler / seed) is identical. The presets diverge:

- ``dev`` — 400 draws, 2 chains, target_accept 0.9. Runs in tens of
  seconds on nutpie for iteration.
- ``reporting`` — 1500 draws, 4 chains, target_accept 0.95. Higher than
  the bayes ``reporting`` preset (0.9) because the selection model has
  the known η/s identification challenge and needs tighter stepping.

:class:`SelectionModelConfig` is the serialisable snapshot of a fit,
written to ``config.json`` next to the InferenceData so runs are
reproducible from artefacts alone.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np

from dspopulations_us_birth_certificates.bayes.config import BayesRunConfig
from dspopulations_us_birth_certificates.selection.priors import (
    VARIANTS,
    ModelPriors,
)

SelectionRunConfigName = Literal["dev", "reporting"]
Variant = Literal["A", "B", "C", "D"]
Spec = Literal["theta_only", "theta_s", "single_eta", "full"]

MODEL_ID = "selection"


_SELECTION_PRESETS: dict[str, dict[str, Any]] = {
    "dev": {
        "draws": 400,
        "tune": 400,
        "chains": 2,
        "target_accept": 0.9,
        "prior_predictive_samples": 400,
        "posterior_predictive": True,
        "nuts_sampler": "nutpie",
    },
    "reporting": {
        "draws": 1500,
        "tune": 1500,
        "chains": 4,
        "target_accept": 0.95,
        "prior_predictive_samples": 1000,
        "posterior_predictive": True,
        "nuts_sampler": "nutpie",
    },
}


def selection_run_config(
    name: SelectionRunConfigName, *, random_seed: int = 47
) -> BayesRunConfig:
    """Return a ``BayesRunConfig`` instance with selection-tuned presets."""
    if name not in _SELECTION_PRESETS:
        raise ValueError(
            f"Unknown selection run profile {name!r}. "
            f"Valid names: {sorted(_SELECTION_PRESETS)}"
        )
    return BayesRunConfig(name=name, random_seed=random_seed, **_SELECTION_PRESETS[name])


def preset_names() -> tuple[str, ...]:
    return tuple(_SELECTION_PRESETS)


def _priors_to_dict(priors: ModelPriors) -> dict[str, Any]:
    """Serialise a ``ModelPriors`` to a JSON-safe dict.

    Numpy arrays become plain lists; scalar floats pass through.
    """
    out: dict[str, Any] = {}
    for key, value in asdict(priors).items():
        if isinstance(value, np.ndarray):
            out[key] = value.tolist()
        else:
            out[key] = value
    return out


@dataclass
class SelectionModelConfig:
    """Serialisable snapshot of a selection-model fit.

    ``model_id`` is always ``"selection"`` so
    :func:`bayes.io.copy_docs_template` resolves to
    ``docs/models/selection/index.qmd`` regardless of variant. ``variant``
    and ``spec`` are separate fields the Quarto template branches on.
    """

    variant: Variant
    spec: Spec
    year_range: tuple[int, int]
    post_dobbs_year: int
    priors: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    model_id: str = MODEL_ID

    @classmethod
    def from_priors(
        cls,
        *,
        variant: Variant,
        spec: Spec,
        year_range: tuple[int, int],
        post_dobbs_year: int,
        priors_obj: ModelPriors | None = None,
        notes: str = "",
    ) -> SelectionModelConfig:
        """Build a config from a concrete priors instance (or variant letter)."""
        if priors_obj is None:
            priors_obj = VARIANTS[variant]()
        return cls(
            variant=variant,
            spec=spec,
            year_range=year_range,
            post_dobbs_year=post_dobbs_year,
            priors=_priors_to_dict(priors_obj),
            notes=notes,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "variant": self.variant,
            "spec": self.spec,
            "year_range": list(self.year_range),
            "post_dobbs_year": self.post_dobbs_year,
            "priors": dict(self.priors),
            "notes": self.notes,
        }


__all__ = [
    "MODEL_ID",
    "SelectionModelConfig",
    "SelectionRunConfigName",
    "Spec",
    "Variant",
    "preset_names",
    "selection_run_config",
]
