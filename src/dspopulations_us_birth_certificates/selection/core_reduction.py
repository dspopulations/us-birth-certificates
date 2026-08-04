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
    RecordingDrift,
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
DEFAULT_REDUCTION_ERROR_CORRELATION = 0.0
DEFAULT_REDUCTION_CALIBRATION_SHIFT_LOGIT = 0.0
RecordedDefinition = Literal["confirmed_or_pending", "confirmed_only"]

DEFAULT_ANCHOR_CSV = Path("output/degraaf_surveillance/expected_births_anchor.csv")
DEFAULT_ANCHOR_WINDOW_HALF_WIDTH = 2
# Year-to-year SD of log surveillance prevalence measured at 1.5-1.8% in the
# model-family review, so a half-normal at 0.02 is weakly informative around it
# rather than an invention.
DEFAULT_ANCHOR_LEVEL_SIGMA = 0.02
DEFAULT_ANCHOR_TREND_SIGMA = 0.01
# The workbook attaches no uncertainty to the surveillance prevalences at all.
# This is the prior scale for a *free* observation SD, so the fit reports how
# well the windows are actually reconciled instead of asserting it.
DEFAULT_ANCHOR_OBS_SIGMA = 0.05

# Per-year logit-scale SD of the random walk on recording sensitivity over the
# years no surveillance window covers. Calibrated so the cumulative prior width
# across a four-year unanchored tail spans this repository's own bracketing
# allocation of the post-2018 flag-rate decline: the de Graaf-derived recording
# anchor in notes/figures/recording_rates_anchor.csv has s for NH White falling
# 17% over 2016-2024, which is about 0.12 logit units over four years, or one
# cumulative SD at this value. It is an assumption, not evidence. The split of
# the post-window decline between prevalence and recording is prior-determined,
# so a drifted fit must be reported alongside both corners: 0.0 here (all
# prevalence, identical to DSP008) and ``anchor_forecast_flat=True`` (all
# recording).
DEFAULT_RECORDING_S_DRIFT_SIGMA = 0.06


@dataclass(frozen=True)
class SurveillanceAnchor:
    """Observed surveillance prevalence for overlapping centred windows.

    ``mid_year_idx`` is zero-based against the model's first year.  Each window
    constrains the mean prevalence over ``2 * half_width + 1`` consecutive years,
    which is what makes the overlap between windows harmless: overlapping
    windows share latent years rather than double-counting evidence.

    ``log_prevalence`` is the log of the surveillance prevalence per live birth
    (not per 10,000), so it composes directly with the model's probabilities.
    """

    mid_year_idx: np.ndarray
    log_prevalence: np.ndarray
    half_width: int = DEFAULT_ANCHOR_WINDOW_HALF_WIDTH
    source: str = ""
    mid_years: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if len(self.mid_year_idx) != len(self.log_prevalence):
            raise ValueError("anchor index and prevalence arrays must align")
        if len(self.mid_year_idx) == 0:
            raise ValueError("surveillance anchor requires at least one window")
        if self.half_width < 0:
            raise ValueError("anchor half_width must be non-negative")
        if not np.all(np.isfinite(self.log_prevalence)):
            raise ValueError("anchor log prevalences must be finite")

    @classmethod
    def from_csv(
        cls,
        *,
        year_range: tuple[int, int],
        path: Path | str = DEFAULT_ANCHOR_CSV,
        half_width: int = DEFAULT_ANCHOR_WINDOW_HALF_WIDTH,
        prevalence_column: str = "prevalence_per10k",
    ) -> SurveillanceAnchor:
        """Load windows whose mid-year falls inside ``year_range``.

        The latent prevalence series is padded by ``half_width`` either side of
        the modelled years, so a window centred on the first or last modelled
        year is still usable.  Windows centred outside the range are dropped
        rather than partially applied.
        """
        path = Path(path)
        table = pd.read_csv(path)
        required = {"mid_year", prevalence_column}
        missing = required - set(table.columns)
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

        from_year, to_year = year_range
        inside = table[
            (table["mid_year"] >= from_year) & (table["mid_year"] <= to_year)
        ].sort_values("mid_year")
        if inside.empty:
            raise ValueError(
                f"{path} has no surveillance window centred within "
                f"{from_year}-{to_year}"
            )
        prevalence = inside[prevalence_column].to_numpy(dtype=float) / 1e4
        if not np.all(prevalence > 0.0):
            raise ValueError("anchor prevalences must be positive")
        return cls(
            mid_year_idx=inside["mid_year"].to_numpy(dtype=int) - from_year,
            log_prevalence=np.log(prevalence),
            half_width=half_width,
            source=str(path),
            mid_years=tuple(int(y) for y in inside["mid_year"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "half_width": self.half_width,
            "mid_years": list(self.mid_years),
            "n_windows": len(self.mid_year_idx),
            "log_prevalence": self.log_prevalence.tolist(),
            "prevalence_per10k": (np.exp(self.log_prevalence) * 1e4).tolist(),
            # Overlapping windows are not independent observations. The birth-year
            # span they jointly cover, divided by the window width, is the honest
            # count: 17 five-year windows over 2000-2018 span 23 birth-years and
            # so carry about 4.6 independent observations, not 17.
            "effective_independent_windows": (
                int(self.mid_year_idx.max() - self.mid_year_idx.min())
                + 1
                + 2 * self.half_width
            )
            / (2 * self.half_width + 1),
        }


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
    recording_s_drift_sigma: float = DEFAULT_RECORDING_S_DRIFT_SIGMA
    reduction_age_step_sigma: float = DEFAULT_REDUCTION_AGE_STEP_SIGMA
    false_positive_rate: float = FALSE_POSITIVE_RATE
    reduction_source: str = str(DEFAULT_REDUCTION_CSV)
    extrapolated_reduction_start: int = DEFAULT_EXTRAPOLATED_REDUCTION_START
    reduction_error_correlation: float = DEFAULT_REDUCTION_ERROR_CORRELATION
    reduction_calibration_shift_logit: float = DEFAULT_REDUCTION_CALIBRATION_SHIFT_LOGIT

    @classmethod
    def from_reduction_csv(
        cls,
        *,
        year_range: tuple[int, int] = DEFAULT_YEAR_RANGE,
        path: Path | str = DEFAULT_REDUCTION_CSV,
        observed_logit_sigma: float = 0.20,
        extrapolated_logit_sigma: float = 0.45,
        reduction_error_correlation: float = DEFAULT_REDUCTION_ERROR_CORRELATION,
        reduction_calibration_shift_logit: float = (
            DEFAULT_REDUCTION_CALIBRATION_SHIFT_LOGIT
        ),
        extrapolated_start: int = DEFAULT_EXTRAPOLATED_REDUCTION_START,
        recording_s_mean: float = 0.5,
        recording_s_sigma: float = 1.0,
        recording_s_year_sigma: float = 0.35,
        recording_s_drift_sigma: float = DEFAULT_RECORDING_S_DRIFT_SIGMA,
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
        if (
            not np.isfinite(reduction_error_correlation)
            or not 0.0 <= reduction_error_correlation < 1.0
        ):
            raise ValueError("reduction_error_correlation must lie in [0, 1)")
        if not np.isfinite(reduction_calibration_shift_logit):
            raise ValueError("reduction_calibration_shift_logit must be finite")
        if recording_s_sigma <= 0.0:
            raise ValueError("recording_s_sigma must be positive")
        if recording_s_year_sigma < 0.0:
            raise ValueError("recording_s_year_sigma must be non-negative")
        if not np.isfinite(recording_s_drift_sigma) or recording_s_drift_sigma < 0.0:
            raise ValueError("recording_s_drift_sigma must be finite and non-negative")
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
            reduction_error_correlation=reduction_error_correlation,
            reduction_calibration_shift_logit=reduction_calibration_shift_logit,
            recording_s_logit=float(logit(recording_s_mean)),
            recording_s_sigma=recording_s_sigma,
            recording_s_year_sigma=recording_s_year_sigma,
            recording_s_drift_sigma=recording_s_drift_sigma,
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


def _reduction_error_covariance(
    sigma: np.ndarray,
    correlation: float,
) -> np.ndarray:
    """Return the logit-error covariance with equicorrelated standard errors."""
    sigma = np.asarray(sigma, dtype=float)
    if sigma.ndim != 1 or not len(sigma):
        raise ValueError("reduction sigma must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(sigma)) or np.any(sigma <= 0.0):
        raise ValueError("reduction sigma values must be finite and positive")
    if not np.isfinite(correlation) or not 0.0 <= correlation < 1.0:
        raise ValueError("reduction error correlation must lie in [0, 1)")
    standardised = (1.0 - correlation) * np.eye(len(sigma)) + correlation
    return np.outer(sigma, sigma) * standardised


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
    recording_drift: RecordingDrift = DSP001.recording_drift
    recorded_definition: RecordedDefinition = "confirmed_or_pending"
    theta_model: str = "seven_band_fixed"
    age_endpoint_convention: dict[str, str] | None = None
    template_id: str = DSP001.template_id
    comparison_parent: str | None = DSP001.comparison_parent
    surveillance_anchor: dict[str, Any] | None = None

    @classmethod
    def from_priors(
        cls,
        *,
        year_range: tuple[int, int],
        priors_obj: CoreReductionPriors,
        model_definition: CoreModelDefinition = DSP001,
        recorded_definition: RecordedDefinition = "confirmed_or_pending",
        notes: str = "",
        anchor: SurveillanceAnchor | None = None,
        anchor_hyperpriors: dict[str, Any] | None = None,
    ) -> CoreReductionModelConfig:
        if recorded_definition not in {"confirmed_or_pending", "confirmed_only"}:
            raise ValueError(
                "recorded_definition must be 'confirmed_or_pending' or 'confirmed_only'"
            )
        priors = priors_obj.to_dict()
        exact_age = model_definition.age_model == "single_year"
        priors["theta_lb_age_used"] = not exact_age
        anchored = model_definition.reduction_model == "anchor"
        if anchored and anchor is None:
            raise ValueError(
                f"{model_definition.model_id} is anchored but no "
                "SurveillanceAnchor was supplied to the config"
            )
        # An anchored fit does not consume the reduction prior. Say so in the
        # serialised config, so a reader of the report cannot mistake the
        # retained comparison series for the prior the model actually used.
        priors["reduction_prior_enters_likelihood"] = not anchored
        # A drifted fit divides the post-window flag decline between prevalence
        # and recording by prior width alone, so record whether the drift is live
        # rather than leaving a reader to infer it from the sigma.
        drifted = model_definition.recording_drift == "post_anchor"
        priors["recording_drift_enters_likelihood"] = (
            drifted and priors_obj.recording_s_drift_sigma > 0.0
        )
        anchor_record: dict[str, Any] | None = None
        if anchor is not None:
            anchor_record = anchor.to_dict()
            anchor_record["hyperpriors"] = dict(anchor_hyperpriors or {})
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
            recording_drift=model_definition.recording_drift,
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
            surveillance_anchor=anchor_record,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_slug": self.model_slug,
            "family_id": self.family_id,
            "recording_model": self.recording_model,
            "reduction_model": self.reduction_model,
            "age_model": self.age_model,
            "recording_drift": self.recording_drift,
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
            "surveillance_anchor": (
                dict(self.surveillance_anchor)
                if self.surveillance_anchor is not None
                else None
            ),
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
    split_revision: bool = False,
) -> pd.DataFrame:
    """Aggregate births into age-by-year cells for the core model.

    Unlike ``prepare_cells`` for the full selection model, this does not
    require complete clinical flags. The core model only uses year, maternal
    age, and the recorded DS indicator.

    ``split_revision`` additionally splits every cell by whether the record
    carries the 2003-revision congenital-anomaly checkbox (``ca_down`` or
    ``ca_downs`` populated) rather than the unrevised ``uca_downs`` field only.
    Revised certificates record materially more Down syndrome than unrevised
    ones in the same year, so a window spanning the 2004-2015 phase-in can
    identify that measurement shift directly instead of absorbing it into a
    time trend. The split is redundant from 2016, when coverage reaches 100%.
    """
    if age_model not in {"band", "single_year"}:
        raise ValueError("age_model must be 'band' or 'single_year'")
    if recorded_definition not in {"confirmed_or_pending", "confirmed_only"}:
        raise ValueError(
            "recorded_definition must be 'confirmed_or_pending' or 'confirmed_only'"
        )

    cols = {
        "ca_down_c": "ca_down_c",
        "ca_down": "ca_down",
        "ca_downs": "ca_downs",
        **DEFAULT_COLUMNS,
        **(columns or {}),
    }
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
    if split_revision:
        revision_select = (
            f"CASE WHEN {cols['ca_down']} IS NOT NULL "
            f"OR {cols['ca_downs']} IS NOT NULL THEN 1 ELSE 0 END AS revised,"
        )
        group_columns = f"{age_columns}, revised"
    else:
        revision_select = ""
        group_columns = age_columns
    sql = f"""
        WITH coded AS (
            SELECT
                CAST({cols["year"]} AS INTEGER) - {from_year} AS year_idx,
                {age_select},
                {revision_select}
                {recorded_expr} AS recorded_ind
            FROM {table}
            WHERE {cols["year"]} BETWEEN {from_year} AND {to_year}
              AND {cols["mage_c"]} IS NOT NULL
              AND {cols["down_ind"]} IS NOT NULL
        )
        SELECT
            year_idx,
            {group_columns},
            COUNT(*) AS N_cell,
            SUM(recorded_ind) AS R_cell
        FROM coded
        GROUP BY year_idx, {group_columns}
        ORDER BY year_idx, {group_columns}
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
    if split_revision:
        cells["revised"] = cells["revised"].astype("int8")
    n_year = to_year - from_year + 1
    n_age = len(age_values)
    cells.attrs.update(
        {
            "n_year": n_year,
            "n_age": n_age,
            "age_model": age_model,
            "age_values": age_values,
            "split_revision": split_revision,
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
    recording_drift: RecordingDrift = "none",
    anchor: SurveillanceAnchor | None = None,
    anchor_level_sigma: float = DEFAULT_ANCHOR_LEVEL_SIGMA,
    anchor_trend_sigma: float = DEFAULT_ANCHOR_TREND_SIGMA,
    anchor_obs_sigma: float = DEFAULT_ANCHOR_OBS_SIGMA,
    anchor_obs_sigma_fixed: float | None = None,
    anchor_forecast_flat: bool = False,
) -> Any:
    """Build the PyMC core reduction-recording model.

    With ``reduction_model='anchor'`` the year level is not sampled from the
    reduction-rate CSV at all.  Instead a latent annual log Down-syndrome
    live-birth prevalence follows a local linear trend, is observed through the
    surveillance programmes' overlapping five-year window means, and sets
    ``eta_year`` by the accounting identity

        eta_year = prevalence_year / Morris_expected_prevalence_year

    so the reduction becomes a *consequence* of an anchored prevalence rather
    than an imported prior.  Years beyond the last observed window are forecast
    by the same state equation, with intervals that widen as they should.

    ``recording_drift='post_anchor'`` adds a random walk on ``logit s`` over
    exactly those unanchored years.  Holding ``s`` constant there — what every
    other anchored model does — means a falling recorded rate can only be read as
    falling prevalence, so the allocation is decided by a modelling default
    rather than by evidence.  The drift makes it an explicit prior instead.  It
    does **not** identify the split: with no window to constrain the level, the
    division between prevalence and recording is set by
    ``priors.recording_s_drift_sigma`` against the anchor's own level and trend
    variances.  Report the two corners with it —
    ``recording_s_drift_sigma=0.0`` puts the whole decline on prevalence and
    reproduces the undrifted fit exactly, and ``anchor_forecast_flat=True``
    holds latent prevalence at its last anchored value and puts the whole
    decline on recording.
    """
    import pymc as pm
    import pytensor.tensor as pt

    if recording_model not in {"constant", "year", "revision"}:
        raise ValueError("recording_model must be 'constant', 'year' or 'revision'")
    if recording_model == "revision" and "revised" not in cells:
        raise ValueError(
            "recording_model='revision' requires cells prepared with "
            "split_revision=True"
        )
    if reduction_model not in {"year", "year_age", "anchor"}:
        raise ValueError("reduction_model must be 'year', 'year_age' or 'anchor'")
    if reduction_model == "anchor":
        if anchor is None:
            raise ValueError("reduction_model='anchor' requires a SurveillanceAnchor")
        for name, value in (
            ("anchor_level_sigma", anchor_level_sigma),
            ("anchor_trend_sigma", anchor_trend_sigma),
            ("anchor_obs_sigma", anchor_obs_sigma),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if anchor_obs_sigma_fixed is not None and (
            not np.isfinite(anchor_obs_sigma_fixed) or anchor_obs_sigma_fixed <= 0.0
        ):
            raise ValueError("anchor_obs_sigma_fixed must be finite and positive")
    elif anchor is not None:
        raise ValueError(
            "a SurveillanceAnchor was supplied but reduction_model is "
            f"{reduction_model!r}; pass reduction_model='anchor' to use it"
        )
    if recording_drift not in {"none", "post_anchor"}:
        raise ValueError("recording_drift must be 'none' or 'post_anchor'")
    if recording_drift == "post_anchor":
        if reduction_model != "anchor":
            raise ValueError(
                "recording_drift='post_anchor' requires reduction_model='anchor'; "
                "without an anchor there is no last covered year to drift from"
            )
        if recording_model == "year":
            raise ValueError(
                "recording_drift='post_anchor' cannot be combined with "
                "recording_model='year'; centred year offsets and a post-anchor "
                "random walk parameterise the same year-varying sensitivity"
            )
        if (
            not np.isfinite(priors.recording_s_drift_sigma)
            or priors.recording_s_drift_sigma < 0.0
        ):
            raise ValueError("recording_s_drift_sigma must be finite and non-negative")
    if anchor_forecast_flat and reduction_model != "anchor":
        raise ValueError("anchor_forecast_flat requires reduction_model='anchor'")
    if recording_model == "year" and priors.recording_s_year_sigma <= 0.0:
        raise ValueError(
            "recording_s_year_sigma must be positive when recording_model='year'"
        )
    if not 0.0 <= priors.false_positive_rate < 1.0:
        raise ValueError("false_positive_rate must lie in [0, 1)")
    if (
        not np.isfinite(priors.reduction_error_correlation)
        or not 0.0 <= priors.reduction_error_correlation < 1.0
    ):
        raise ValueError("reduction_error_correlation must lie in [0, 1)")
    if not np.isfinite(priors.reduction_calibration_shift_logit):
        raise ValueError("reduction_calibration_shift_logit must be finite")
    if cells.empty:
        raise ValueError("core model requires at least one age-year cell")
    if n_year is None:
        n_year = int(cells.attrs.get("n_year", cells["year_idx"].max() + 1))
    # An anchored fit does not consume the reduction prior, but the fit script
    # still loads it so the report can contrast the superseded prior with the
    # anchored posterior. Only require alignment, not presence.
    require_reduction_prior = reduction_model != "anchor"
    if require_reduction_prior or len(priors.reduction_logit):
        if len(priors.reduction_logit) != n_year:
            raise ValueError(
                f"reduction prior has length {len(priors.reduction_logit)}, "
                f"but n_year={n_year}"
            )
    if require_reduction_prior or len(priors.reduction_sigma):
        if len(priors.reduction_sigma) != n_year:
            raise ValueError(
                f"reduction prior sigma has length {len(priors.reduction_sigma)}, "
                f"but n_year={n_year}"
            )
        if not np.all(np.isfinite(priors.reduction_sigma)) or np.any(
            priors.reduction_sigma <= 0.0
        ):
            raise ValueError("reduction_sigma values must be finite and positive")
    if anchor is not None:
        if anchor.mid_year_idx.min() < 0 or anchor.mid_year_idx.max() >= n_year:
            raise ValueError(
                "anchor window mid-years must fall inside the modelled year range"
            )

    # The last modelled year any window reaches. A window centred on model-year
    # index m constrains the mean over m - half_width ... m + half_width, so the
    # anchored span ends half a window past the final mid-year. Years after that
    # carry no surveillance observation at all.
    last_anchored_year_idx = (
        int(anchor.mid_year_idx.max()) + anchor.half_width
        if anchor is not None
        else n_year - 1
    )
    n_drift_year = 0
    if recording_drift == "post_anchor":
        n_drift_year = n_year - 1 - min(last_anchored_year_idx, n_year - 1)
        if n_drift_year <= 0:
            raise ValueError(
                "recording_drift='post_anchor' needs at least one modelled year "
                "beyond the last surveillance window, but the windows reach the "
                f"end of the {n_year}-year range; use an undrifted model instead"
            )

    age_idx = cells["age_idx"].to_numpy()
    year_idx = cells["year_idx"].to_numpy()
    n_cell = cells["N_cell"].to_numpy(dtype=float)
    r_cell = cells["R_cell"].to_numpy(dtype=int)
    revised_cell = (
        cells["revised"].to_numpy(dtype=float)
        if recording_model == "revision"
        else None
    )
    if revised_cell is not None and not np.all(np.isin(revised_cell, (0.0, 1.0))):
        raise ValueError("revised must be 0/1 for recording_model='revision'")
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
    if reduction_model == "anchor":
        assert anchor is not None
        # The latent prevalence series is padded either side of the modelled
        # years so a window centred on the first or last modelled year still has
        # all five of its years available.
        n_latent = n_year + 2 * anchor.half_width
        coords["latent_year"] = np.arange(
            -anchor.half_width, n_year + anchor.half_width
        )
        coords["anchor_window"] = np.asarray(anchor.mid_years, dtype=int)

    year_age_n = np.zeros((n_year, n_age), dtype=float)
    np.add.at(year_age_n, (year_idx, age_idx), n_cell)
    natural_count_year_age = year_age_n * theta[np.newaxis, :]
    natural_count_year = natural_count_year_age.sum(axis=1)
    if np.any(natural_count_year <= 0.0):
        raise ValueError("each modelled year must have positive age-expected DS births")
    births_year = year_age_n.sum(axis=1)
    # Morris-expected prevalence absent any prenatal reduction. Dividing an
    # anchored prevalence by this gives eta directly.
    natural_prevalence_year = natural_count_year / births_year
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

        if reduction_model == "anchor":
            assert anchor is not None
            pm.Data("natural_prevalence_year", natural_prevalence_year, dims="year")
            half = anchor.half_width
            width = 2 * half + 1

            # Local linear trend on log prevalence, non-centred. The level
            # innovation absorbs year-to-year variation; the slope innovation lets
            # the trend itself drift, which is what makes the forecast interval
            # widen rather than extrapolate one fixed gradient forever.
            # Centred on the first observed window. 0.25 on the log scale is
            # +-28% at one SD: weakly informative about the level, while keeping
            # theta * eta a valid probability even in prior-predictive draws.
            level_start = pm.Normal(
                "anchor_log_level_start",
                mu=float(anchor.log_prevalence[0]),
                sigma=0.25,
            )
            slope_start = pm.Normal(
                "anchor_log_slope_start", mu=0.0, sigma=anchor_trend_sigma * 5.0
            )
            sigma_level = pm.HalfNormal("anchor_level_sigma", sigma=anchor_level_sigma)
            sigma_trend = pm.HalfNormal("anchor_trend_sigma", sigma=anchor_trend_sigma)
            level_innovation = pm.Normal(
                "anchor_level_innovation_raw", mu=0.0, sigma=1.0, shape=n_latent - 1
            )
            trend_innovation = pm.Normal(
                "anchor_trend_innovation_raw", mu=0.0, sigma=1.0, shape=n_latent - 1
            )
            slope = slope_start + pt.concatenate(
                [pt.zeros((1,)), pt.cumsum(trend_innovation * sigma_trend)]
            )
            latent_increment = slope[:-1] + level_innovation * sigma_level
            if anchor_forecast_flat:
                # The corner opposite a drifting s: hold latent prevalence at its
                # last anchored value so the whole post-window decline in the
                # recorded rate has to be absorbed by recording. Increment j
                # feeds latent year j + 1, so zeroing every increment from the
                # last anchored latent index onward leaves the path flat from
                # that year. Latent index i is model year i - half_width, hence
                # the 2 * half offset.
                last_anchored_latent_idx = int(anchor.mid_year_idx.max()) + 2 * half
                latent_increment = latent_increment * pt.as_tensor_variable(
                    (np.arange(n_latent - 1) < last_anchored_latent_idx).astype(float)
                )
            log_prevalence_latent = pm.Deterministic(
                "anchor_log_prevalence_latent",
                level_start
                + pt.concatenate([pt.zeros((1,)), pt.cumsum(latent_increment)]),
                dims="latent_year",
            )
            prevalence_latent = pt.exp(log_prevalence_latent)

            # Each window constrains the mean of its five latent years. Averaging
            # inside the observation equation is what makes overlapping windows
            # share latent years instead of double-counting evidence.
            window_starts = anchor.mid_year_idx  # latent index of window start
            window_means = pt.stack(
                [
                    prevalence_latent[start : start + width].mean()
                    for start in window_starts
                ]
            )
            pm.Deterministic(
                "anchor_window_prevalence", window_means, dims="anchor_window"
            )
            # Estimating the observation SD measures only whether the windows are
            # mutually consistent with a smooth path -- it cannot measure whether
            # the surveillance programmes' prevalences are *accurate*, because the
            # workbook supplies no uncertainty. Fixing it larger is therefore the
            # honest sensitivity axis, not a wider prior (which the data would
            # simply overrule).
            if anchor_obs_sigma_fixed is None:
                sigma_obs: Any = pm.HalfNormal(
                    "anchor_obs_sigma", sigma=anchor_obs_sigma
                )
            else:
                sigma_obs = anchor_obs_sigma_fixed
                pm.Deterministic(
                    "anchor_obs_sigma", pt.as_tensor_variable(anchor_obs_sigma_fixed)
                )
            pm.Normal(
                "anchor_obs",
                mu=pt.log(window_means),
                sigma=sigma_obs,
                observed=anchor.log_prevalence,
                dims="anchor_window",
            )

            prevalence_year = pm.Deterministic(
                "prevalence_year",
                prevalence_latent[half : half + n_year],
                dims="year",
            )
            # The accounting identity: reduction is whatever reconciles an
            # anchored prevalence with the Morris no-reduction expectation.
            eta_year = pm.Deterministic(
                "eta_year",
                prevalence_year / natural_prevalence_year,
                dims="year",
            )
            pm.Deterministic("rho_year", 1.0 - eta_year, dims="year")
            rho_year_age = None
            rho_logit = None
        else:
            reduction_mu = (
                priors.reduction_logit + priors.reduction_calibration_shift_logit
            )
            correlation = priors.reduction_error_correlation
            if correlation == 0.0:
                rho_logit = pm.Normal(
                    "rho_logit_year",
                    mu=reduction_mu,
                    sigma=priors.reduction_sigma,
                    dims="year",
                )
            else:
                covariance = _reduction_error_covariance(
                    priors.reduction_sigma,
                    correlation,
                )
                cholesky = np.linalg.cholesky(covariance)
                raw_error = pm.Normal(
                    "rho_logit_year_raw",
                    mu=0.0,
                    sigma=1.0,
                    dims="year",
                )
                rho_logit = pm.Deterministic(
                    "rho_logit_year",
                    reduction_mu + pt.dot(cholesky, raw_error),
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
                    rho_year_intercept = (
                        rho_year_intercept
                        - margin_residual / pt.maximum(margin_derivative, 1e-12)
                    )
                rho_year_intercept = pm.Deterministic(
                    "rho_year_intercept", rho_year_intercept, dims="year"
                )
                rho_year_age = pm.Deterministic(
                    "rho_year_age",
                    pm.math.invlogit(
                        rho_year_intercept[:, None] + rho_age_offset[None, :]
                    ),
                    dims=("year", "age"),
                )
                rho_year = pm.Deterministic(
                    "rho_year",
                    (natural_weight * rho_year_age).sum(axis=1),
                    dims="year",
                )
                pm.Deterministic(
                    "rho_year_margin_error",
                    rho_year - rho_year_anchor,
                    dims="year",
                )
                eta_year = pm.Deterministic("eta_year", 1.0 - rho_year, dims="year")
                pm.Deterministic(
                    "eta_year_age", 1.0 - rho_year_age, dims=("year", "age")
                )

        s_logit = pm.Normal(
            "recording_s_logit",
            mu=priors.recording_s_logit,
            sigma=priors.recording_s_sigma,
        )
        recording_s = pm.Deterministic("recording_s", pm.math.invlogit(s_logit))

        # ``recording_s`` stays the anchored-era level in every model, drifted or
        # not, so it remains directly comparable across the family. The drift is
        # carried as a separate per-year logit offset that is exactly zero while
        # the anchor still speaks.
        s_drift_logit: Any = None
        if recording_drift == "post_anchor" and priors.recording_s_drift_sigma > 0.0:
            drift_innovation = pm.Normal(
                "recording_s_drift_innovation_raw",
                mu=0.0,
                sigma=1.0,
                shape=n_drift_year,
            )
            s_drift_logit = pm.Deterministic(
                "recording_s_drift_logit",
                pt.concatenate(
                    [
                        pt.zeros((n_year - n_drift_year,)),
                        pt.cumsum(drift_innovation * priors.recording_s_drift_sigma),
                    ]
                ),
                dims="year",
            )
        s_logit_year = s_logit if s_drift_logit is None else s_logit + s_drift_logit
        # Keep the undrifted graph byte-identical to what it was before the drift
        # existed, so a zero drift sigma reproduces the parent model exactly.
        recording_s_year_value = (
            pt.ones((n_year,)) * recording_s
            if s_drift_logit is None
            else pm.math.invlogit(s_logit_year)
        )

        if recording_model == "revision":
            # ``recording_s`` is the revised-certificate sensitivity, so it stays
            # directly comparable with fits confined to 2016 onward where every
            # record is revised. The unrevised sensitivity is a logit offset from
            # it, identified by years in which both certificate versions are in
            # use. Sum-to-zero centring would be wrong here: the two levels are
            # distinguishable measurement regimes, not exchangeable groups.
            s_unrevised_offset = pm.Normal(
                "recording_s_unrevised_offset",
                mu=0.0,
                sigma=priors.recording_s_sigma,
            )
            pm.Deterministic(
                "recording_s_unrevised",
                pm.math.invlogit(s_logit + s_unrevised_offset),
            )
            recording_s_year = pm.Deterministic(
                "recording_s_year",
                recording_s_year_value,
                dims="year",
            )
        elif recording_model == "constant":
            recording_s_year = pm.Deterministic(
                "recording_s_year",
                recording_s_year_value,
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

        if s_drift_logit is not None:
            # The headline the drift exists to report: how far the final modelled
            # year's revised sensitivity has moved from its anchored-era level.
            # A value below 1 means recording has taken part of the recorded-rate
            # decline that an undrifted fit books entirely as falling prevalence.
            pm.Deterministic(
                "recording_s_drift_ratio", recording_s_year[-1] / recording_s
            )

        if rho_year_age is None:
            p_ds_lb_value = theta[age_idx] * eta_year[year_idx]
            if reduction_model == "anchor":
                # eta is a ratio of prevalences, not a bounded probability, so an
                # extreme draw could in principle push theta * eta above 1 and
                # make the Binomial invalid. The guard never binds at posterior
                # scale (eta is around 0.6) but keeps prior-predictive and early
                # tuning draws well defined. Note rho < 0 is deliberately NOT
                # excluded: it would mean the anchored prevalence exceeds the
                # Morris no-reduction expectation, which is a real diagnostic.
                p_ds_lb_value = pt.clip(p_ds_lb_value, 1e-12, 1.0 - 1e-9)
        else:
            p_ds_lb_value = theta[age_idx] * (1.0 - rho_year_age[year_idx, age_idx])
        p_ds_lb = pm.Deterministic("p_ds_lb", p_ds_lb_value, dims="cell")
        if recording_model == "revision":
            # The drift shifts both certificate versions together: it models the
            # certificate's recording behaviour over time, not a change in the gap
            # between the two versions. Unrevised records only exist before 2016,
            # which is always inside the anchored span, so in practice the drift
            # term is zero wherever ``revised_cell`` is 0.
            s_cell = pm.math.invlogit(
                (s_logit if s_drift_logit is None else s_logit_year[year_idx])
                + s_unrevised_offset * (1.0 - revised_cell)
            )
        else:
            s_cell = recording_s_year[year_idx]
        p_recorded = pm.Deterministic(
            "p_recorded",
            p_ds_lb * s_cell + (1.0 - p_ds_lb) * priors.false_positive_rate,
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
    "DEFAULT_RECORDING_S_DRIFT_SIGMA",
    "DEFAULT_REDUCTION_AGE_STEP_SIGMA",
    "DEFAULT_EXTRAPOLATED_REDUCTION_START",
    "DEFAULT_REDUCTION_CSV",
    "DEFAULT_REDUCTION_CALIBRATION_SHIFT_LOGIT",
    "DEFAULT_REDUCTION_ERROR_CORRELATION",
    "CoreReductionModelConfig",
    "CoreReductionPriors",
    "RecordedDefinition",
    "build_core_reduction_model",
    "core_year_summary",
    "prepare_core_age_year_cells",
]
