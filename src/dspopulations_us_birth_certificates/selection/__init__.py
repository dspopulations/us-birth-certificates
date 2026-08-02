"""Three-stage Bayesian selection model for DS livebirth ascertainment.

Decomposes observed recording into baseline livebirth rate × screening-
and-termination pass-through × birth-certificate sensitivity. See
``notes/20260622-predictors-bayesian-model.md`` for the design rationale.

Public surface
--------------
- :func:`build_model` — PyMC model builder with four staged specs.
- :class:`ModelPriors` and the ``variant_*`` factories — prior sets that
  encode Morris/Natoli/Kuppermann/Boulet evidence at four sensitivity
  settings.
- :func:`prepare_cells` — DuckDB aggregator producing the cell frame.
- :func:`simulate_cells` / :class:`TrueParams` — forward simulator for
  parameter-recovery validation.
"""

from __future__ import annotations

from dspopulations_us_birth_certificates.selection import diagnostics
from dspopulations_us_birth_certificates.selection.config import (
    MODEL_ID,
    FitContext,
    RunConfig,
    SelectionModelConfig,
    preset_names,
    selection_run_config,
)
from dspopulations_us_birth_certificates.selection.core_reduction import (
    CORE_REDUCTION_MODEL_ID,
    CoreReductionModelConfig,
    CoreReductionPriors,
    build_core_reduction_model,
    core_year_summary,
    prepare_core_age_year_cells,
)
from dspopulations_us_birth_certificates.selection.core_reporting import (
    render_core_all,
)
from dspopulations_us_birth_certificates.selection.data import (
    DEFAULT_DB_PATH,
    DEFAULT_YEAR_RANGE,
    prepare_cells,
    summarise_cells,
)
from dspopulations_us_birth_certificates.selection.io import (
    copy_docs_template,
    latest_fit_dir,
    render_quarto,
    render_report,
    save_artefacts,
    save_summary,
)
from dspopulations_us_birth_certificates.selection.model import (
    SPECS,
    build_model,
    extract_true_counts,
    posterior_subgroup_rate,
)
from dspopulations_us_birth_certificates.selection.priors import (
    AGE_LEVELS,
    EDU_LEVELS,
    N_AGE,
    N_EDU,
    N_PAYER,
    N_RACE,
    PAYER_LEVELS,
    RACE_LEVELS,
    VARIANTS,
    ModelPriors,
    inv_logit,
    logit,
    variant_A_tight_s,
    variant_B_tight_eta_term,
    variant_C_default,
    variant_D_recording_off,
)
from dspopulations_us_birth_certificates.selection.sampling import sample
from dspopulations_us_birth_certificates.selection.simulate import (
    TrueParams,
    simulate_cells,
)

__all__ = [
    "AGE_LEVELS",
    "CORE_REDUCTION_MODEL_ID",
    "DEFAULT_DB_PATH",
    "DEFAULT_YEAR_RANGE",
    "EDU_LEVELS",
    "MODEL_ID",
    "FitContext",
    "ModelPriors",
    "N_AGE",
    "N_EDU",
    "N_PAYER",
    "N_RACE",
    "PAYER_LEVELS",
    "RACE_LEVELS",
    "RunConfig",
    "SPECS",
    "SelectionModelConfig",
    "CoreReductionModelConfig",
    "CoreReductionPriors",
    "TrueParams",
    "VARIANTS",
    "build_model",
    "build_core_reduction_model",
    "copy_docs_template",
    "core_year_summary",
    "diagnostics",
    "extract_true_counts",
    "inv_logit",
    "latest_fit_dir",
    "logit",
    "posterior_subgroup_rate",
    "prepare_cells",
    "prepare_core_age_year_cells",
    "preset_names",
    "render_quarto",
    "render_core_all",
    "render_report",
    "sample",
    "save_artefacts",
    "save_summary",
    "selection_run_config",
    "simulate_cells",
    "summarise_cells",
    "variant_A_tight_s",
    "variant_B_tight_eta_term",
    "variant_C_default",
    "variant_D_recording_off",
]
