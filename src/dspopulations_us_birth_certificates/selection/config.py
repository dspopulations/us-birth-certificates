"""Run / model configuration for the selection-model fit pipeline.

- :class:`RunConfig` — sampler preset (draws / tune / chains /
  target_accept / sampler / seed). Build via :func:`selection_run_config`.
- :class:`FitContext` — mutable state threaded through fit steps
  (cells → model → idata → artefacts).
- :class:`SelectionModelConfig` — serialisable snapshot of a fit, written
  to ``config.json`` next to the InferenceData so runs are reproducible
  from artefacts alone.

Run profiles
------------
- ``dev`` — 1000 tune + 1000 draws × 2 chains, target_accept 0.9.
  Enough posterior support to land ESS above 400 on the named RVs on
  the full spec; a few minutes on nutpie for theta_only, ~30 min for
  full.
- ``reporting`` — 1500 draws, 4 chains, target_accept 0.95. The
  selection model has a known η/s identification challenge and needs
  tighter stepping than the usual 0.9.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np

from dspopulations_us_birth_certificates.selection.model import Spec
from dspopulations_us_birth_certificates.selection.priors import (
    VARIANTS,
    ModelPriors,
)

SelectionRunConfigName = Literal["dev", "reporting"]
Variant = Literal["A", "B", "C", "D"]

MODEL_ID = "selection"


_SELECTION_PRESETS: dict[str, dict[str, Any]] = {
    "dev": {
        "draws": 1000,
        "tune": 1000,
        "chains": 2,
        "target_accept": 0.9,
        "prior_predictive_samples": 500,
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


@dataclass(frozen=True)
class RunConfig:
    """Speed / fidelity preset for a selection-model fit."""

    name: SelectionRunConfigName
    draws: int
    tune: int
    chains: int
    target_accept: float
    prior_predictive_samples: int
    posterior_predictive: bool
    nuts_sampler: str
    random_seed: int = 47


@dataclass
class FitContext:
    """Mutable state threaded through the selection-model fit steps.

    Populated progressively by the CLI: cells → model → idata → artefacts.
    """

    config: Any  # SelectionModelConfig (duck-typed on ``.to_dict()``)
    run_config: RunConfig
    output_dir: Any = None  # pathlib.Path, kept loose to avoid import cost
    cells: Any = None
    model: Any = None
    idata: Any = None
    summary: Any = None
    metrics: dict[str, Any] = field(default_factory=dict)


def selection_run_config(
    name: SelectionRunConfigName, *, random_seed: int = 47
) -> RunConfig:
    """Return a :class:`RunConfig` with selection-tuned presets."""
    if name not in _SELECTION_PRESETS:
        raise ValueError(
            f"Unknown selection run profile {name!r}. "
            f"Valid names: {sorted(_SELECTION_PRESETS)}"
        )
    return RunConfig(name=name, random_seed=random_seed, **_SELECTION_PRESETS[name])


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
    :func:`selection.io.copy_docs_template` resolves to
    ``docs/models/selection/index.qmd`` regardless of variant. ``variant``
    and ``spec`` are separate fields the Quarto template branches on.
    """

    variant: Variant
    spec: Spec
    year_range: tuple[int, int]
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
            priors=_priors_to_dict(priors_obj),
            notes=notes,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "variant": self.variant,
            "spec": self.spec,
            "year_range": list(self.year_range),
            "priors": dict(self.priors),
            "notes": self.notes,
        }


__all__ = [
    "MODEL_ID",
    "FitContext",
    "RunConfig",
    "SelectionModelConfig",
    "SelectionRunConfigName",
    "Spec",
    "Variant",
    "preset_names",
    "selection_run_config",
]
