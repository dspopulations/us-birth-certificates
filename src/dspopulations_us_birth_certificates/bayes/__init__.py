"""Bayesian regression pipeline for Down-syndrome live-birth analysis.

Mirrors the shape of ``dspopulations_us_birth_certificates.models`` so
contributors moving between the LightGBM and Bayesian pipelines can rely
on the same mental model:

- ``BayesRunConfig`` — speed / fidelity preset at CLI time.
- ``BayesModelDefinition`` — declarative model class; subclasses auto-
  register into ``MODELS``.
- ``load_cells`` — aggregate DuckDB births into year×dim cells for a
  given outcome construction (``recorded`` vs ``recorded_plus_predicted``).
- ``sample`` — run prior / posterior / posterior-predictive sampling.
- ``save_artefacts`` — persist the fit's InferenceData, cells, and configs.

Designed to be driven from ``scripts/fit_bayes_model.py`` (reproducible
CLI) and from notebooks (exploratory fits, plots, diagnostics).
"""

from __future__ import annotations

from dspopulations_us_birth_certificates.bayes import models  # noqa: F401
from dspopulations_us_birth_certificates.bayes.config import (
    BayesFitContext,
    BayesModelConfig,
    BayesRunConfig,
)
from dspopulations_us_birth_certificates.bayes.data import load_cells
from dspopulations_us_birth_certificates.bayes.hsgp import (
    HSGPComponent,
    make_hsgp_component,
)
from dspopulations_us_birth_certificates.bayes.io import (
    copy_docs_template,
    render_quarto,
    save_artefacts,
    save_summary,
)
from dspopulations_us_birth_certificates.bayes.outcomes import (
    OutcomeSpec,
    outcome_spec_from_name,
    recorded_plus_predicted_spec,
    recorded_spec,
)
from dspopulations_us_birth_certificates.bayes.registry import (
    MODELS,
    BayesModelDefinition,
)
from dspopulations_us_birth_certificates.bayes.sampling import sample

__all__ = [
    "MODELS",
    "BayesFitContext",
    "BayesModelConfig",
    "BayesModelDefinition",
    "BayesRunConfig",
    "HSGPComponent",
    "OutcomeSpec",
    "copy_docs_template",
    "load_cells",
    "make_hsgp_component",
    "outcome_spec_from_name",
    "recorded_plus_predicted_spec",
    "recorded_spec",
    "render_quarto",
    "sample",
    "save_artefacts",
    "save_summary",
]
