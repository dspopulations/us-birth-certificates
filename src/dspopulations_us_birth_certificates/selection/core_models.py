"""Declarative registry for the core reduction-recording model family."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

RecordingModel = Literal["constant", "year", "revision"]
ReductionModel = Literal["year", "year_age", "anchor"]
AgeModel = Literal["band", "single_year"]

CORE_REDUCTION_FAMILY_ID = "selection_core_reduction"


@dataclass(frozen=True)
class CoreModelDefinition:
    """Complete definition for one core accounting model."""

    model_id: str
    slug: str
    title: str
    description: str
    recording_model: RecordingModel
    introduced: str
    reduction_model: ReductionModel = "year"
    age_model: AgeModel = "band"
    template_id: str = CORE_REDUCTION_FAMILY_ID
    comparison_parent: str | None = None


DSP001 = CoreModelDefinition(
    model_id="DSP001",
    slug="constant_s",
    title="Core reduction-recording model with constant s",
    description=(
        "Baseline age-by-year accounting model with surveillance-informed "
        "combined reduction and one global certificate recording sensitivity."
    ),
    recording_model="constant",
    introduced="2026-08-02",
)

DSP002 = CoreModelDefinition(
    model_id="DSP002",
    slug="s_year",
    title="Core reduction-recording model with s_year",
    description=(
        "First extension of DSP001 with partially pooled year-specific "
        "certificate recording sensitivity."
    ),
    recording_model="year",
    introduced="2026-08-02",
    comparison_parent="DSP001",
)

DSP003 = CoreModelDefinition(
    model_id="DSP003",
    slug="rho_year_age",
    title="Core reduction-recording model with age-specific reduction",
    description=(
        "Extension of DSP001 with a smooth maternal-age reduction effect, "
        "calibrated to preserve each year's surveillance-informed national margin."
    ),
    recording_model="constant",
    reduction_model="year_age",
    age_model="single_year",
    introduced="2026-08-02",
    comparison_parent="DSP001",
)

DSP004 = CoreModelDefinition(
    model_id="DSP004",
    slug="constant_s_exact_age",
    title="Core reduction-recording model with constant s and exact age",
    description=(
        "Exact-age Morris-curve ablation of DSP001 with one national reduction "
        "parameter per year and one global certificate recording sensitivity."
    ),
    recording_model="constant",
    reduction_model="year",
    age_model="single_year",
    introduced="2026-08-02",
    comparison_parent="DSP001",
)

DSP005 = CoreModelDefinition(
    model_id="DSP005",
    slug="s_year_exact_age",
    title="Core reduction-recording model with s_year and exact age",
    description=(
        "Exact-age Morris-curve extension of DSP004 with partially pooled "
        "year-specific certificate recording sensitivity."
    ),
    recording_model="year",
    reduction_model="year",
    age_model="single_year",
    introduced="2026-08-02",
    comparison_parent="DSP004",
)

DSP006 = CoreModelDefinition(
    model_id="DSP006",
    slug="s_revision_exact_age",
    title="Core reduction-recording model with revision-split s and exact age",
    description=(
        "Exact-age extension of DSP004 giving separate certificate recording "
        "sensitivity to revised and unrevised birth certificates. Requires a "
        "year range spanning the 2004-2015 revision phase-in to be identified; "
        "from 2016 every record is revised and the offset reverts to its prior."
    ),
    recording_model="revision",
    reduction_model="year",
    age_model="single_year",
    introduced="2026-08-03",
    comparison_parent="DSP004",
)

DSP007 = CoreModelDefinition(
    model_id="DSP007",
    slug="anchor_constant_s_exact_age",
    title="Surveillance-anchored core model with constant s and exact age",
    description=(
        "Replaces the reduction-rate prior with a latent annual Down-syndrome "
        "live-birth prevalence, observed through the surveillance programmes' "
        "overlapping five-year window means and following a local linear trend. "
        "The reduction becomes a consequence of the anchored prevalence rather "
        "than an imported prior, so the level is identified by data. Direct "
        "counterpart of DSP004."
    ),
    recording_model="constant",
    reduction_model="anchor",
    age_model="single_year",
    introduced="2026-08-04",
    comparison_parent="DSP004",
)

DSP008 = CoreModelDefinition(
    model_id="DSP008",
    slug="anchor_s_revision_exact_age",
    title="Surveillance-anchored core model with revision-split s and exact age",
    description=(
        "Combines the surveillance-anchored level of DSP007 with the "
        "revised/unrevised recording split of DSP006. Requires a year range "
        "spanning both the 2004-2015 revision phase-in and at least one "
        "surveillance window."
    ),
    recording_model="revision",
    reduction_model="anchor",
    age_model="single_year",
    introduced="2026-08-04",
    comparison_parent="DSP007",
)

CORE_MODEL_REGISTRY: dict[str, CoreModelDefinition] = {
    "dsp001": DSP001,
    "dsp002": DSP002,
    "dsp003": DSP003,
    "dsp004": DSP004,
    "dsp005": DSP005,
    "dsp006": DSP006,
    "dsp007": DSP007,
    "dsp008": DSP008,
}


def validate_core_model_definition(definition: CoreModelDefinition) -> None:
    """Fail early when a declarative core-model specification is invalid."""
    if not re.fullmatch(r"DSP\d{3}", definition.model_id):
        raise ValueError(
            f"{definition.model_id!r} must have the form DSP001, DSP002, ..."
        )
    if not re.fullmatch(r"[A-Za-z0-9]+(?:[A-Za-z0-9_-]*[A-Za-z0-9])?", definition.slug):
        raise ValueError(f"{definition.model_id}.slug must be path safe.")
    if definition.recording_model not in {"constant", "year", "revision"}:
        raise ValueError(
            f"{definition.model_id}.recording_model must be 'constant', 'year' "
            "or 'revision'."
        )
    if definition.reduction_model not in {"year", "year_age", "anchor"}:
        raise ValueError(
            f"{definition.model_id}.reduction_model must be 'year', 'year_age' "
            "or 'anchor'."
        )
    if definition.age_model not in {"band", "single_year"}:
        raise ValueError(
            f"{definition.model_id}.age_model must be 'band' or 'single_year'."
        )
    if (
        definition.reduction_model == "year_age"
        and definition.age_model != "single_year"
    ):
        raise ValueError(
            f"{definition.model_id} must use age_model='single_year' when "
            "reduction_model='year_age'."
        )
    if definition.comparison_parent is not None and not re.fullmatch(
        r"DSP\d{3}", definition.comparison_parent
    ):
        raise ValueError(
            f"{definition.model_id}.comparison_parent must have the form DSP001."
        )


def validate_core_model_registry() -> None:
    """Validate every registered core model and registry key."""
    seen_slugs: set[str] = set()
    seen_ids: set[str] = set()
    for key, definition in CORE_MODEL_REGISTRY.items():
        validate_core_model_definition(definition)
        if key != definition.model_id.lower():
            raise ValueError(
                f"Registry key {key!r} does not match {definition.model_id!r}."
            )
        if definition.model_id in seen_ids:
            raise ValueError(f"Duplicate core model_id: {definition.model_id}.")
        if definition.slug in seen_slugs:
            raise ValueError(f"Duplicate core model slug: {definition.slug}.")
        seen_ids.add(definition.model_id)
        seen_slugs.add(definition.slug)

    registered_ids = {
        definition.model_id for definition in CORE_MODEL_REGISTRY.values()
    }
    for definition in CORE_MODEL_REGISTRY.values():
        parent = definition.comparison_parent
        if parent is not None and parent not in registered_ids:
            raise ValueError(
                f"{definition.model_id}.comparison_parent={parent!r} is not registered."
            )


def get_core_model_definition(model: str) -> CoreModelDefinition:
    """Return a core model definition, accepting upper/lower-case IDs."""
    key = model.lower()
    try:
        return CORE_MODEL_REGISTRY[key]
    except KeyError as exc:
        valid = ", ".join(
            definition.model_id for definition in CORE_MODEL_REGISTRY.values()
        )
        raise ValueError(
            f"Unknown core model {model!r}. Valid models: {valid}"
        ) from exc


def core_model_names() -> tuple[str, ...]:
    """Return registered core-model IDs in registry order."""
    return tuple(definition.model_id for definition in CORE_MODEL_REGISTRY.values())


validate_core_model_registry()


__all__ = [
    "CORE_MODEL_REGISTRY",
    "CORE_REDUCTION_FAMILY_ID",
    "DSP001",
    "DSP002",
    "DSP003",
    "DSP004",
    "DSP005",
    "DSP006",
    "DSP007",
    "DSP008",
    "AgeModel",
    "CoreModelDefinition",
    "RecordingModel",
    "ReductionModel",
    "core_model_names",
    "get_core_model_definition",
    "validate_core_model_definition",
    "validate_core_model_registry",
]
