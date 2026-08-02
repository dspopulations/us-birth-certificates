"""Core age-reduction-recording Bayesian model.

This is the deliberately simple baseline for the selection-model family:

    maternal-age expected DS livebirths
    × combined survival after prenatal selection
    × certificate recording sensitivity
    = recorded DS births.

The model estimates the combined reduction before livebirth (``rho_year``)
instead of decomposing it into prenatal detection and termination. That
decomposition can be layered on later after the central accounting model is
stable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import duckdb
import numpy as np
import pandas as pd

from dspopulations_us_birth_certificates.chance import (
    get_ds_lb_nt_probability_array,
)
from dspopulations_us_birth_certificates.intervals import posterior_mean_eti
from dspopulations_us_birth_certificates.selection.core_models import (
    CORE_REDUCTION_FAMILY_ID,
    DSP001,
    AgeModel,
    CoreModelDefinition,
    RecordingModel,
    ReductionModel,
)
from dspopulations_us_birth_certificates.selection.data import (
    DEFAULT_COLUMNS,
    DEFAULT_YEAR_RANGE,
)
from dspopulations_us_birth_certificates.selection.priors import (
    FALSE_POSITIVE_RATE,
    MORRIS_THETA_LB_PER_1000,
    N_AGE,
    logit,
)

CORE_REDUCTION_MODEL_ID = CORE_REDUCTION_FAMILY_ID
DEFAULT_REDUCTION_CSV = Path("data/us-births-reduction-rates-1989-2024.csv")
DEFAULT_EXTRAPOLATED_REDUCTION_START = 2020
DEFAULT_REDUCTION_AGE_STEP_SIGMA = 0.10
RecordedDefinition = Literal["confirmed_or_pending", "confirmed_only"]


@dataclass(frozen=True)
class CoreReductionPriors:
    """Prior bundle for the core reduction-recording model."""

    theta_lb_age: np.ndarray = field(
        default_factory=lambda: MORRIS_THETA_LB_PER_1000 / 1000.0
    )
    reduction_mean: np.ndarray = field(default_factory=lambda: np.empty(0))
    reduction_logit: np.ndarray = field(default_factory=lambda: np.empty(0))
    reduction_sigma: np.ndarray = field(default_factory=lambda: np.empty(0))
    recording_s_logit: float = field(default_factory=lambda: float(logit(0.5)))
    recording_s_sigma: float = 1.0
    recording_s_year_sigma: float = 0.35
    reduction_age_step_sigma: float = DEFAULT_REDUCTION_AGE_STEP_SIGMA
    false_positive_rate: float = FALSE_POSITIVE_RATE
    reduction_source: str = str(DEFAULT_REDUCTION_CSV)
    extrapolated_reduction_start: int = DEFAULT_EXTRAPOLATED_REDUCTION_START

    @classmethod
    def from_reduction_csv(
        cls,
        *,
        year_range: tuple[int, int] = DEFAULT_YEAR_RANGE,
        path: Path | str = DEFAULT_REDUCTION_CSV,
        observed_logit_sigma: float = 0.20,
        extrapolated_logit_sigma: float = 0.45,
        extrapolated_start: int = DEFAULT_EXTRAPOLATED_REDUCTION_START,
        recording_s_mean: float = 0.5,
        recording_s_sigma: float = 1.0,
        recording_s_year_sigma: float = 0.35,
        reduction_age_step_sigma: float = DEFAULT_REDUCTION_AGE_STEP_SIGMA,
        false_positive_rate: float = FALSE_POSITIVE_RATE,
    ) -> CoreReductionPriors:
        """Build priors from the tracked year-level reduction-rate CSV.

        ``observed_logit_sigma`` applies before ``extrapolated_start``;
        ``extrapolated_logit_sigma`` applies from that year onward because the
        current CSV linearly extrapolates the lagged surveillance series.
        """
        if not 0.0 < recording_s_mean < 1.0:
            raise ValueError("recording_s_mean must lie in (0, 1)")
        if observed_logit_sigma <= 0.0 or extrapolated_logit_sigma <= 0.0:
            raise ValueError("reduction logit sigmas must be positive")
        if recording_s_sigma <= 0.0:
            raise ValueError("recording_s_sigma must be positive")
        if recording_s_year_sigma < 0.0:
            raise ValueError("recording_s_year_sigma must be non-negative")
        if reduction_age_step_sigma <= 0.0:
            raise ValueError("reduction_age_step_sigma must be positive")
        if not 0.0 <= false_positive_rate < 1.0:
            raise ValueError("false_positive_rate must lie in [0, 1)")

        path = Path(path)
        table = pd.read_csv(path, encoding="utf-8-sig")
        required = {"year", "reduction"}
        missing = required - set(table.columns)
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

        years = np.arange(year_range[0], year_range[1] + 1)
        indexed = table.set_index("year")
        missing_years = [int(y) for y in years if y not in indexed.index]
        if missing_years:
            raise ValueError(
                f"{path} has no reduction prior for years: {missing_years}"
            )

        reduction = indexed.loc[years, "reduction"].to_numpy(dtype=float)
        if not np.all((0.0 < reduction) & (reduction < 1.0)):
            raise ValueError("reduction priors must lie strictly between 0 and 1")

        sigma = np.where(
            years >= extrapolated_start,
            extrapolated_logit_sigma,
            observed_logit_sigma,
        ).astype(float)

        return cls(
            reduction_mean=reduction,
            reduction_logit=logit(reduction),
            reduction_sigma=sigma,
            recording_s_logit=float(logit(recording_s_mean)),
            recording_s_sigma=recording_s_sigma,
            recording_s_year_sigma=recording_s_year_sigma,
            reduction_age_step_sigma=reduction_age_step_sigma,
            false_positive_rate=false_positive_rate,
            reduction_source=str(path),
            extrapolated_reduction_start=extrapolated_start,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation."""
        out: dict[str, Any] = {}
        for key, value in asdict(self).items():
            if isinstance(value, np.ndarray):
                out[key] = value.tolist()
            else:
                out[key] = value
        return out


@dataclass(frozen=True)
class CoreReductionModelConfig:
    """Serialisable snapshot for a core reduction fit."""

    year_range: tuple[int, int]
    priors: dict[str, Any]
    notes: str = ""
    model_id: str = DSP001.model_id
    model_slug: str = DSP001.slug
    family_id: str = CORE_REDUCTION_FAMILY_ID
    recording_model: RecordingModel = DSP001.recording_model
    reduction_model: ReductionModel = DSP001.reduction_model
    age_model: AgeModel = DSP001.age_model
    recorded_definition: RecordedDefinition = "confirmed_or_pending"
    theta_model: str = "seven_band_fixed"
    age_endpoint_convention: dict[str, str] | None = None
    template_id: str = DSP001.template_id
    comparison_parent: str | None = DSP001.comparison_parent

    @classmethod
    def from_priors(
        cls,
        *,
        year_range: tuple[int, int],
        priors_obj: CoreReductionPriors,
        model_definition: CoreModelDefinition = DSP001,
        recorded_definition: RecordedDefinition = "confirmed_or_pending",
        notes: str = "",
    ) -> CoreReductionModelConfig:
        if recorded_definition not in {"confirmed_or_pending", "confirmed_only"}:
            raise ValueError(
                "recorded_definition must be 'confirmed_or_pending' or 'confirmed_only'"
            )
        priors = priors_obj.to_dict()
        exact_age = model_definition.age_model == "single_year"
        priors["theta_lb_age_used"] = not exact_age
        return cls(
            year_range=year_range,
            priors=priors,
            notes=notes,
            model_id=model_definition.model_id,
            model_slug=model_definition.slug,
            family_id=CORE_REDUCTION_FAMILY_ID,
            recording_model=model_definition.recording_model,
            reduction_model=model_definition.reduction_model,
            age_model=model_definition.age_model,
            recorded_definition=recorded_definition,
            theta_model=(
                "morris_double_logistic_by_age_code"
                if exact_age
                else "seven_band_fixed"
            ),
            age_endpoint_convention=(
                {
                    "12": "10-12; Morris evaluated at age 12",
                    "50": "50+; Morris evaluated at age 50",
                }
                if exact_age
                else None
            ),
            template_id=model_definition.template_id,
            comparison_parent=model_definition.comparison_parent,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_slug": self.model_slug,
            "family_id": self.family_id,
            "recording_model": self.recording_model,
            "reduction_model": self.reduction_model,
            "age_model": self.age_model,
            "recorded_definition": self.recorded_definition,
            "theta_model": self.theta_model,
            "age_endpoint_convention": (
                dict(self.age_endpoint_convention)
                if self.age_endpoint_convention is not None
                else None
            ),
            "template_id": self.template_id,
            "comparison_parent": self.comparison_parent,
            "year_range": list(self.year_range),
            "priors": dict(self.priors),
            "notes": self.notes,
        }


def _age_case(column: str) -> str:
    """Return the shared seven-band maternal-age SQL CASE expression."""
    return (
        f"CASE "
        f"WHEN {column} < 20 THEN 0 "
        f"WHEN {column} < 25 THEN 1 "
        f"WHEN {column} < 30 THEN 2 "
        f"WHEN {column} < 35 THEN 3 "
        f"WHEN {column} < 40 THEN 4 "
        f"WHEN {column} < 45 THEN 5 "
        f"ELSE 6 END"
    )


def prepare_core_age_year_cells(
    con: duckdb.DuckDBPyConnection,
    *,
    year_range: tuple[int, int] = DEFAULT_YEAR_RANGE,
    table: str = "us_births",
    columns: dict[str, str] | None = None,
    age_model: AgeModel = "band",
    recorded_definition: RecordedDefinition = "confirmed_or_pending",
) -> pd.DataFrame:
    """Aggregate births into age-by-year cells for the core model.

    Unlike ``prepare_cells`` for the full selection model, this does not
    require complete clinical flags. The core model only uses year, maternal
    age, and the recorded DS indicator.
    """
    if age_model not in {"band", "single_year"}:
        raise ValueError("age_model must be 'band' or 'single_year'")
    if recorded_definition not in {"confirmed_or_pending", "confirmed_only"}:
        raise ValueError(
            "recorded_definition must be 'confirmed_or_pending' or 'confirmed_only'"
        )

    cols = {"ca_down_c": "ca_down_c", **DEFAULT_COLUMNS, **(columns or {})}
    from_year, to_year = year_range
    if age_model == "band":
        age_select = f"{_age_case(cols['mage_c'])} AS age_idx"
        age_columns = "age_idx"
    else:
        # ``mage_c`` is era-dependent: recent NCHS data already use capped
        # endpoint codes, while earlier years can contain literal ages 10-54.
        # Normalise first so the declared 10-12 and 50+ cells remain true for
        # every accepted year range.
        age_select = (
            "CASE "
            f"WHEN CAST({cols['mage_c']} AS INTEGER) <= 12 THEN 12 "
            f"WHEN CAST({cols['mage_c']} AS INTEGER) >= 50 THEN 50 "
            f"ELSE CAST({cols['mage_c']} AS INTEGER) END AS maternal_age"
        )
        age_columns = "maternal_age"
    if recorded_definition == "confirmed_or_pending":
        recorded_expr = f"CAST({cols['down_ind']} AS INTEGER)"
    else:
        recorded_expr = (
            f"CASE WHEN UPPER(CAST({cols['ca_down_c']} AS VARCHAR)) = 'C' "
            "THEN 1 ELSE 0 END"
        )
    sql = f"""
        WITH coded AS (
            SELECT
                CAST({cols["year"]} AS INTEGER) - {from_year} AS year_idx,
                {age_select},
                {recorded_expr} AS recorded_ind
            FROM {table}
            WHERE {cols["year"]} BETWEEN {from_year} AND {to_year}
              AND {cols["mage_c"]} IS NOT NULL
              AND {cols["down_ind"]} IS NOT NULL
        )
        SELECT
            year_idx,
            {age_columns},
            COUNT(*) AS N_cell,
            SUM(recorded_ind) AS R_cell
        FROM coded
        GROUP BY year_idx, {age_columns}
        ORDER BY year_idx, {age_columns}
    """
    cells = con.execute(sql).df()
    cells["year_idx"] = cells["year_idx"].astype("int32")
    if age_model == "band":
        cells["age_idx"] = cells["age_idx"].astype("int32")
        age_values = list(range(N_AGE))
    else:
        cells["maternal_age"] = cells["maternal_age"].astype("int16")
        age_values = sorted(cells["maternal_age"].unique().tolist())
        age_lookup = {age: idx for idx, age in enumerate(age_values)}
        cells.insert(
            1,
            "age_idx",
            cells["maternal_age"].map(age_lookup).astype("int32"),
        )
        cells.insert(
            3,
            "maternal_age_label",
            cells["maternal_age"].map(
                lambda age: "10-12" if age == 12 else "50+" if age == 50 else str(age)
            ),
        )
    cells["N_cell"] = cells["N_cell"].astype("int64")
    cells["R_cell"] = cells["R_cell"].astype("int64")
    n_year = to_year - from_year + 1
    n_age = len(age_values)
    cells.attrs.update(
        {
            "n_year": n_year,
            "n_age": n_age,
            "age_model": age_model,
            "age_values": age_values,
            "age_labels": [
                "10-12" if age == 12 else "50+" if age == 50 else str(age)
                for age in age_values
            ],
            "recorded_definition": recorded_definition,
            "year_range": year_range,
            "N_total": int(cells["N_cell"].sum()),
            "R_total": int(cells["R_cell"].sum()),
        }
    )
    _check_core_cells(cells, n_year=n_year, n_age=n_age)
    return cells


def _check_core_cells(cells: pd.DataFrame, *, n_year: int, n_age: int) -> None:
    """Validate age-year cell bounds before model construction."""
    for col, limit in {"year_idx": n_year, "age_idx": n_age}.items():
        if not cells[col].between(0, limit - 1).all():
            bad = cells.loc[~cells[col].between(0, limit - 1), col].unique()
            raise ValueError(
                f"{col} values out of range [0, {limit - 1}]: "
                f"found {sorted(bad.tolist())!r}"
            )
    if (cells["R_cell"] > cells["N_cell"]).any():
        raise ValueError("R_cell > N_cell in at least one core cell")


def build_core_reduction_model(
    cells: pd.DataFrame,
    priors: CoreReductionPriors,
    *,
    n_year: int | None = None,
    recording_model: RecordingModel = "constant",
    reduction_model: ReductionModel = "year",
) -> Any:
    """Build the PyMC core reduction-recording model."""
    import pymc as pm
    import pytensor.tensor as pt

    if recording_model not in {"constant", "year"}:
        raise ValueError("recording_model must be 'constant' or 'year'")
    if reduction_model not in {"year", "year_age"}:
        raise ValueError("reduction_model must be 'year' or 'year_age'")
    if recording_model == "year" and priors.recording_s_year_sigma <= 0.0:
        raise ValueError(
            "recording_s_year_sigma must be positive when recording_model='year'"
        )
    if not 0.0 <= priors.false_positive_rate < 1.0:
        raise ValueError("false_positive_rate must lie in [0, 1)")
    if cells.empty:
        raise ValueError("core model requires at least one age-year cell")
    if n_year is None:
        n_year = int(cells.attrs.get("n_year", cells["year_idx"].max() + 1))
    if len(priors.reduction_logit) != n_year:
        raise ValueError(
            f"reduction prior has length {len(priors.reduction_logit)}, "
            f"but n_year={n_year}"
        )

    age_idx = cells["age_idx"].to_numpy()
    year_idx = cells["year_idx"].to_numpy()
    n_cell = cells["N_cell"].to_numpy(dtype=float)
    r_cell = cells["R_cell"].to_numpy(dtype=int)
    exact_age = "maternal_age" in cells
    if reduction_model == "year_age" and not exact_age:
        raise ValueError(
            "reduction_model='year_age' requires exact-age cells with a "
            "maternal_age column"
        )
    if exact_age:
        age_table = (
            cells[["age_idx", "maternal_age"]].drop_duplicates().sort_values("age_idx")
        )
        if not np.array_equal(
            age_table["age_idx"].to_numpy(), np.arange(len(age_table))
        ):
            raise ValueError("exact-age cells must use contiguous age_idx values")
        age_values = age_table["maternal_age"].to_numpy(dtype=int)
        theta = np.asarray(get_ds_lb_nt_probability_array(age_values), dtype=float)
    else:
        theta = np.asarray(priors.theta_lb_age, dtype=float)
        if len(theta) != N_AGE:
            raise ValueError(f"theta_lb_age must have length {N_AGE}")
        age_values = np.arange(N_AGE)
    n_age = len(theta)
    if reduction_model == "year_age" and n_age < 2:
        raise ValueError(
            "reduction_model='year_age' requires at least two maternal ages"
        )
    if age_idx.min() < 0 or age_idx.max() >= n_age:
        raise ValueError("age_idx values do not match the model age coordinate")

    coords = {
        "year": np.arange(n_year),
        "age": age_values,
        "cell": np.arange(len(cells)),
    }
    if reduction_model == "year_age":
        coords["age_step"] = age_values[1:]
        age_step_scale = np.sqrt(np.diff(age_values).astype(float))

    year_age_n = np.zeros((n_year, n_age), dtype=float)
    np.add.at(year_age_n, (year_idx, age_idx), n_cell)
    natural_count_year_age = year_age_n * theta[np.newaxis, :]
    natural_count_year = natural_count_year_age.sum(axis=1)
    if np.any(natural_count_year <= 0.0):
        raise ValueError("each modelled year must have positive age-expected DS births")
    natural_ds_weight_year_age = (
        natural_count_year_age / natural_count_year[:, np.newaxis]
    )
    year_by_cell_n = np.vstack(
        [np.where(year_idx == y, n_cell, 0.0) for y in range(n_year)]
    )

    with pm.Model(coords=coords) as model:
        pm.Data("theta_lb_age", theta, dims="age")
        pm.Data("natural_count_year", natural_count_year, dims="year")
        natural_weight = pm.Data(
            "natural_ds_weight_year_age",
            natural_ds_weight_year_age,
            dims=("year", "age"),
        )

        rho_logit = pm.Normal(
            "rho_logit_year",
            mu=priors.reduction_logit,
            sigma=priors.reduction_sigma,
            dims="year",
        )
        if reduction_model == "year":
            rho_year = pm.Deterministic(
                "rho_year", pm.math.invlogit(rho_logit), dims="year"
            )
            eta_year = pm.Deterministic("eta_year", 1.0 - rho_year, dims="year")
            rho_year_age = None
        else:
            rho_year_anchor = pm.Deterministic(
                "rho_year_anchor", pm.math.invlogit(rho_logit), dims="year"
            )
            rho_age_step = pm.Normal(
                "rho_age_step",
                mu=0.0,
                sigma=priors.reduction_age_step_sigma * age_step_scale,
                dims="age_step",
            )
            rho_age_offset_uncentred = pt.concatenate(
                [pt.zeros((1,), dtype=rho_age_step.dtype), pt.cumsum(rho_age_step)]
            )
            rho_age_offset = pm.Deterministic(
                "rho_age_offset",
                rho_age_offset_uncentred - rho_age_offset_uncentred.mean(),
                dims="age",
            )

            # Calibrate a separate intercept for each year so the smooth age
            # curve preserves the sampled surveillance-informed national
            # reduction margin. The weights are expected DS births absent
            # prenatal reduction: N(year, age) * Morris(age).
            rho_year_intercept = rho_logit
            for _ in range(12):
                rho_candidate = pm.math.invlogit(
                    rho_year_intercept[:, None] + rho_age_offset[None, :]
                )
                margin_residual = (natural_weight * rho_candidate).sum(
                    axis=1
                ) - rho_year_anchor
                margin_derivative = (
                    natural_weight * rho_candidate * (1.0 - rho_candidate)
                ).sum(axis=1)
                rho_year_intercept = rho_year_intercept - margin_residual / pt.maximum(
                    margin_derivative, 1e-12
                )
            rho_year_intercept = pm.Deterministic(
                "rho_year_intercept", rho_year_intercept, dims="year"
            )
            rho_year_age = pm.Deterministic(
                "rho_year_age",
                pm.math.invlogit(rho_year_intercept[:, None] + rho_age_offset[None, :]),
                dims=("year", "age"),
            )
            rho_year = pm.Deterministic(
                "rho_year", (natural_weight * rho_year_age).sum(axis=1), dims="year"
            )
            pm.Deterministic(
                "rho_year_margin_error",
                rho_year - rho_year_anchor,
                dims="year",
            )
            eta_year = pm.Deterministic("eta_year", 1.0 - rho_year, dims="year")
            pm.Deterministic("eta_year_age", 1.0 - rho_year_age, dims=("year", "age"))

        s_logit = pm.Normal(
            "recording_s_logit",
            mu=priors.recording_s_logit,
            sigma=priors.recording_s_sigma,
        )
        recording_s = pm.Deterministic("recording_s", pm.math.invlogit(s_logit))
        if recording_model == "constant":
            recording_s_year = pm.Deterministic(
                "recording_s_year",
                pt.ones((n_year,)) * recording_s,
                dims="year",
            )
        else:
            s_year_offset_raw = pm.Normal(
                "recording_s_year_offset_raw",
                mu=0.0,
                sigma=priors.recording_s_year_sigma,
                dims="year",
            )
            s_year_offset = pm.Deterministic(
                "recording_s_year_offset",
                s_year_offset_raw - s_year_offset_raw.mean(),
                dims="year",
            )
            recording_s_year = pm.Deterministic(
                "recording_s_year",
                pm.math.invlogit(s_logit + s_year_offset),
                dims="year",
            )

        if rho_year_age is None:
            p_ds_lb_value = theta[age_idx] * eta_year[year_idx]
        else:
            p_ds_lb_value = theta[age_idx] * (1.0 - rho_year_age[year_idx, age_idx])
        p_ds_lb = pm.Deterministic("p_ds_lb", p_ds_lb_value, dims="cell")
        p_recorded = pm.Deterministic(
            "p_recorded",
            p_ds_lb * recording_s_year[year_idx]
            + (1.0 - p_ds_lb) * priors.false_positive_rate,
            dims="cell",
        )

        pm.Deterministic(
            "true_count_year",
            pt.dot(year_by_cell_n, p_ds_lb),
            dims="year",
        )
        pm.Deterministic(
            "recorded_count_year_mu",
            pt.dot(year_by_cell_n, p_recorded),
            dims="year",
        )
        pm.Deterministic("true_count_total", pt.dot(n_cell, p_ds_lb))

        pm.Binomial(
            "R_obs",
            n=n_cell.astype("int64"),
            p=p_recorded,
            observed=r_cell,
            dims="cell",
        )

    return model


def core_year_summary(idata: Any, cells: pd.DataFrame) -> pd.DataFrame:
    """Posterior year summary for the core reduction model."""
    year_range = cells.attrs.get("year_range")
    n_year = int(cells.attrs["n_year"])
    labels = (
        list(range(int(year_range[0]), int(year_range[0]) + n_year))
        if year_range is not None
        else list(range(n_year))
    )
    observed = cells.groupby("year_idx", observed=True)["R_cell"].sum()
    births = cells.groupby("year_idx", observed=True)["N_cell"].sum()

    rows = []
    for y, label in enumerate(labels):
        row = {
            "year": label,
            "births": int(births.get(y, 0)),
            "recorded_ds": int(observed.get(y, 0)),
        }
        for var in (
            "rho_year",
            "eta_year",
            "recording_s_year",
            "true_count_year",
            "recorded_count_year_mu",
        ):
            stats = posterior_mean_eti(idata.posterior[var].sel(year=y).values)
            row[f"{var}_mean"] = stats["mean"]
            row[f"{var}_lo"] = stats["lo"]
            row[f"{var}_hi"] = stats["hi"]
        rows.append(row)
    return pd.DataFrame(rows)


__all__ = [
    "CORE_REDUCTION_MODEL_ID",
    "DEFAULT_REDUCTION_AGE_STEP_SIGMA",
    "DEFAULT_EXTRAPOLATED_REDUCTION_START",
    "DEFAULT_REDUCTION_CSV",
    "CoreReductionModelConfig",
    "CoreReductionPriors",
    "RecordedDefinition",
    "build_core_reduction_model",
    "core_year_summary",
    "prepare_core_age_year_cells",
]
