"""Model registry and pipeline scaffolding.

`ModelDefinition` subclasses auto-register into `MODELS` on import (see
`base_model.py`). Import the concrete model modules here so their
subclasses are discoverable when a CLI script does
`from dspopulations_us_birth_certificates.models import MODELS`.
"""

from __future__ import annotations

# Import concrete model modules so their __init_subclass__ hook populates
# MODELS. Add new variants here as they land.
from dspopulations_us_birth_certificates.models import (  # noqa: F401
    usbc10,
    usbc10_cn,
    usbc11,
    usbc11_cn,
)
from dspopulations_us_birth_certificates.models.base_model import (
    MODELS,
    ModelDefinition,
)
from dspopulations_us_birth_certificates.models.common import (
    ModelConfig,
    ModelFitContext,
    RunConfig,
    SelectionStep,
    ShapScatterSpec,
)

__all__ = [
    "MODELS",
    "ModelConfig",
    "ModelDefinition",
    "ModelFitContext",
    "RunConfig",
    "SelectionStep",
    "ShapScatterSpec",
]
