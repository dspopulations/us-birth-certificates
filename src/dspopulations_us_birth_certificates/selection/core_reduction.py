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
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from dspopulations_us_birth_certificates.intervals import posterior_mean_eti
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

CORE_REDUCTION_MODEL_ID = "selection_core_reduction"
DEFAULT_REDUCTION_CSV = Path("data/us-births-reduction-rates-1989-2024.csv")
DEFAULT_EXTRAPOLATED_REDUCTION_START = 2020


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
    model_id: str = CORE_REDUCTION_MODEL_ID

    @classmethod
    def from_priors(
        cls,
        *,
        year_range: tuple[int, int],
        priors_obj: CoreReductionPriors,
        notes: str = "",
    ) -> CoreReductionModelConfig:
        return cls(
            year_range=year_range,
            priors=priors_obj.to_dict(),
            notes=notes,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
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
) -> pd.DataFrame:
    """Aggregate births into age-by-year cells for the core model.

    Unlike ``prepare_cells`` for the full selection model, this does not
    require complete clinical flags. The core model only uses year, maternal
    age, and the recorded DS indicator.
    """
    cols = {**DEFAULT_COLUMNS, **(columns or {})}
    from_year, to_year = year_range
    age_case = _age_case(cols["mage_c"])
    sql = f"""
        WITH coded AS (
            SELECT
                CAST({cols["year"]} AS INTEGER) - {from_year} AS year_idx,
                {age_case} AS age_idx,
                CAST({cols["down_ind"]} AS INTEGER) AS down_ind
            FROM {table}
            WHERE {cols["year"]} BETWEEN {from_year} AND {to_year}
              AND {cols["mage_c"]} IS NOT NULL
              AND {cols["down_ind"]} IS NOT NULL
        )
        SELECT
            year_idx,
            age_idx,
            COUNT(*) AS N_cell,
            SUM(down_ind) AS R_cell
        FROM coded
        GROUP BY year_idx, age_idx
        ORDER BY year_idx, age_idx
    """
    cells = con.execute(sql).df()
    cells["year_idx"] = cells["year_idx"].astype("int32")
    cells["age_idx"] = cells["age_idx"].astype("int32")
    cells["N_cell"] = cells["N_cell"].astype("int64")
    cells["R_cell"] = cells["R_cell"].astype("int64")
    n_year = to_year - from_year + 1
    cells.attrs.update(
        {
            "n_year": n_year,
            "year_range": year_range,
            "N_total": int(cells["N_cell"].sum()),
            "R_total": int(cells["R_cell"].sum()),
        }
    )
    _check_core_cells(cells, n_year=n_year)
    return cells


def _check_core_cells(cells: pd.DataFrame, *, n_year: int) -> None:
    """Validate age-year cell bounds before model construction."""
    for col, limit in {"year_idx": n_year, "age_idx": N_AGE}.items():
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
) -> Any:
    """Build the PyMC core reduction-recording model."""
    import pymc as pm
    import pytensor.tensor as pt

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
    theta = np.asarray(priors.theta_lb_age, dtype=float)
    if len(theta) != N_AGE:
        raise ValueError(f"theta_lb_age must have length {N_AGE}")

    coords = {
        "year": np.arange(n_year),
        "age": np.arange(N_AGE),
        "cell": np.arange(len(cells)),
    }

    natural_count_year = np.zeros(n_year, dtype=float)
    for y in range(n_year):
        mask = year_idx == y
        natural_count_year[y] = float((n_cell[mask] * theta[age_idx[mask]]).sum())
    year_by_cell_n = np.vstack(
        [np.where(year_idx == y, n_cell, 0.0) for y in range(n_year)]
    )

    with pm.Model(coords=coords) as model:
        pm.Data("theta_lb_age", theta, dims="age")
        pm.Data("natural_count_year", natural_count_year, dims="year")

        rho_logit = pm.Normal(
            "rho_logit_year",
            mu=priors.reduction_logit,
            sigma=priors.reduction_sigma,
            dims="year",
        )
        rho_year = pm.Deterministic(
            "rho_year", pm.math.invlogit(rho_logit), dims="year"
        )
        eta_year = pm.Deterministic("eta_year", 1.0 - rho_year, dims="year")

        s_logit = pm.Normal(
            "recording_s_logit",
            mu=priors.recording_s_logit,
            sigma=priors.recording_s_sigma,
        )
        recording_s = pm.Deterministic("recording_s", pm.math.invlogit(s_logit))

        p_ds_lb = pm.Deterministic(
            "p_ds_lb", theta[age_idx] * eta_year[year_idx], dims="cell"
        )
        p_recorded = pm.Deterministic(
            "p_recorded",
            p_ds_lb * recording_s + (1.0 - p_ds_lb) * priors.false_positive_rate,
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
    "DEFAULT_EXTRAPOLATED_REDUCTION_START",
    "DEFAULT_REDUCTION_CSV",
    "CoreReductionModelConfig",
    "CoreReductionPriors",
    "build_core_reduction_model",
    "core_year_summary",
    "prepare_core_age_year_cells",
]
