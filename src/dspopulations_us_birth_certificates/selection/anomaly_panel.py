"""Control-condition panel from the shared congenital-anomaly certificate item.

The 2003 birth-certificate revision records Down syndrome as one checkbox in a
single congenital-anomaly item.  Several other checkboxes on that same item
describe conditions with no prenatal detection-and-termination channel worth
speaking of, so their recorded rate is close to a direct reading of the item's
recording sensitivity rather than a mixture of prevalence and recording.

That is an *exclusion restriction*, and it is the only route identified so far
that could divide the post-2020 decline in the recorded Down syndrome rate rather
than parameterise the division as ``DSP009`` does.  It works in exactly the years
the surveillance anchor does not reach.

Two things limit it, and both are represented explicitly here rather than assumed
away:

* The controls' own birth prevalence must be stable, or its trend must be known.
  ``true_trend_log_per_year`` in the curation table carries a known trend as a
  fixed offset; the *common* component of an unknown one is carried in the model
  as a prior, because no internal comparison can detect it.
* The controls must actually agree with one another.  They do not.
  :func:`panel_heterogeneity` measures the disagreement at load time so a fit
  cannot quietly present a shared factor the panel itself refutes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from dspopulations_us_birth_certificates.selection.core_validation import (
    integer_array,
    probability_array,
)

DEFAULT_ANOMALY_CONDITIONS_CSV = Path("data/us-births-anomaly-panel-conditions.csv")

# Every record carries the revised congenital-anomaly item only from 2016; before
# that the revised subset is a changing, non-random set of states, so a panel
# reaching back further would read state composition as recording behaviour.
DEFAULT_PANEL_FROM_YEAR = 2016

# Prior SD on the common true-prevalence trend across the control conditions, log
# per year. This is the one assumption the panel cannot check internally: a trend
# shared by hypospadias, both clefts and limb reduction would be read as recording
# and there is no comparison inside the panel that would notice.
#
# 0.004 is a deliberate judgement, not a measurement. It puts one prior SD at
# about 3.3% over the eight-year panel span, against a common recorded decline of
# roughly 10%. So it admits that up to about a third of that decline might be real
# prevalence, while still asserting the exclusion restriction does most of the
# work. Raising it moves DSP010 towards DSP009; setting it to zero asserts the
# restriction exactly. Report across it.
DEFAULT_PANEL_PREVALENCE_TREND_SIGMA = 0.004

# Prior SD on the Down syndrome loading, centred on 1. A loading of 1 is the
# shared log-odds restriction: DS sensitivity odds move with the panel factor. The
# anchored panel years inform it, so this is a starting point rather than the
# answer, but it must be wide enough not to force the restriction -- the note's
# CCHD counter-example shows a single item-wide factor is refuted for at least one
# checkbox, and the controls disagree among themselves too.
DEFAULT_PANEL_LOADING_SIGMA = 0.5

# Half-normal prior scale for the SD of the per-condition trend deviations. This
# is a hyperprior on how much the controls are allowed to disagree, and the panel
# measures that disagreement directly: :func:`panel_heterogeneity` returns a
# between-condition SD near 0.012 log per year on the current control set, so 0.02
# is generous around what the data show rather than a constraint on it.
#
# The deviations are deliberately not centred to sum to zero. Centring would
# assert the controls' trends average exactly to the item-wide recording factor,
# which given the observed heterogeneity is the fixed-effect fallacy; see the
# rationale in ``build_core_reduction_model``.
DEFAULT_PANEL_CONDITION_TREND_SIGMA = 0.02
# Half-normal prior scale for the year-by-condition idiosyncratic term. The
# controls' disagreement is mostly trend-like rather than year-to-year noise, so
# this usually settles well below its prior.
DEFAULT_PANEL_IDIOSYNCRATIC_SIGMA = 0.05
# Prior scale for the year-to-year innovation in the common log-rate change. The
# controls move by a few percent a year, so 0.03 is generous around that.
DEFAULT_PANEL_FACTOR_SIGMA = 0.03


@dataclass(frozen=True)
class AnomalyPanelConditions:
    """The curated congenital-anomaly checkbox table."""

    table: pd.DataFrame
    source: str = ""

    REQUIRED_COLUMNS = (
        "condition",
        "label",
        "role",
        "prenatal_reduction",
        "true_trend_log_per_year",
        "reason",
    )

    def __post_init__(self) -> None:
        missing = set(self.REQUIRED_COLUMNS) - set(self.table.columns)
        if missing:
            raise ValueError(
                f"anomaly condition table is missing columns: {sorted(missing)}"
            )
        bad_roles = set(self.table["role"]) - {"control", "excluded"}
        if bad_roles:
            raise ValueError(f"role must be 'control' or 'excluded'; got {bad_roles}")
        if self.table["condition"].duplicated().any():
            duplicated = self.table.loc[
                self.table["condition"].duplicated(), "condition"
            ].tolist()
            raise ValueError(f"duplicate conditions in the table: {duplicated}")
        if not np.all(np.isfinite(self.table["true_trend_log_per_year"])):
            raise ValueError("true_trend_log_per_year values must be finite")
        if self.controls().empty:
            raise ValueError("the anomaly condition table declares no controls")

    @classmethod
    def from_csv(
        cls, path: Path | str = DEFAULT_ANOMALY_CONDITIONS_CSV
    ) -> AnomalyPanelConditions:
        path = Path(path)
        return cls(table=pd.read_csv(path), source=str(path))

    def controls(self) -> pd.DataFrame:
        return self.table[self.table["role"] == "control"].reset_index(drop=True)

    def excluded(self) -> pd.DataFrame:
        return self.table[self.table["role"] == "excluded"].reset_index(drop=True)

    def select(self, conditions: list[str] | tuple[str, ...] | None) -> pd.DataFrame:
        """Return the rows to use as controls, honouring an explicit override.

        An explicit list may name any curated condition, including an excluded
        one, so a sensitivity fit can add a condition back deliberately.  Naming
        a condition absent from the table is an error, not a silent drop.
        """
        if conditions is None:
            return self.controls()
        wanted = list(conditions)
        known = set(self.table["condition"])
        unknown = [c for c in wanted if c not in known]
        if unknown:
            raise ValueError(
                f"unknown anomaly conditions {unknown}; curated conditions are "
                f"{sorted(known)}"
            )
        if len(set(wanted)) != len(wanted):
            raise ValueError(f"duplicate conditions requested: {wanted}")
        if len(wanted) < 2:
            raise ValueError(
                "the panel needs at least two control conditions; with one there "
                "is no disagreement to measure and the shared factor would be "
                "that condition's own trend"
            )
        indexed = self.table.set_index("condition")
        return indexed.loc[wanted].reset_index()


@dataclass(frozen=True)
class AnomalyPanel:
    """Control-condition flag counts by year, with a composition offset.

    ``year_idx`` is zero-based against the *model's* first year, so the panel
    plugs into the same year coordinate as the certificate cells.  It normally
    covers only the tail of that range: the panel starts when every record
    carries the revised anomaly item.

    ``expected_share`` is the rate each condition would show in each year if its
    age-specific rates were fixed at the panel-pooled profile and only the
    maternal-age composition moved.  Entering it as a fixed offset removes
    composition shift from the shared factor.  It matters little for these
    conditions -- at most 1.1 percentage points over the panel span -- but it is
    cheap, and it matters a great deal for conditions with a steep age gradient,
    which is exactly how gastroschisis was caught.
    """

    condition: tuple[str, ...]
    year_idx: np.ndarray
    years: tuple[int, ...]
    flags: np.ndarray
    births: np.ndarray
    expected_share: np.ndarray
    true_trend_log_per_year: np.ndarray
    reference_year_idx: int
    source: str = ""
    labels: tuple[str, ...] = ()
    conditions_source: str = ""
    excluded: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if np.asarray(self.flags).ndim != 2:
            raise ValueError("flags must be a year-by-condition matrix")
        integer_array(self.flags, "panel flags")
        integer_array(self.births, "panel births", minimum=1)
        indices = integer_array(self.year_idx, "panel year_idx")
        years = integer_array(self.years, "panel years")
        if len(years) < 2:
            raise ValueError("the panel needs at least two years")
        if len(set(self.condition)) != len(self.condition):
            raise ValueError("panel condition names must be distinct")
        if len(years) != len(indices) or not np.all(
            years - indices == years[0] - indices[0]
        ):
            raise ValueError("panel calendar years must align with model indices")
        if not np.all(np.isfinite(self.true_trend_log_per_year)):
            raise ValueError("panel true trends must be finite")
        probability_array(self.expected_share, "expected_share")
        n_year, n_condition = self.flags.shape
        if n_condition != len(self.condition):
            raise ValueError("flags columns must align with the condition names")
        if len(self.year_idx) != n_year or len(self.years) != n_year:
            raise ValueError("panel year arrays must align with the flag rows")
        if self.births.shape != (n_year,):
            raise ValueError("births must carry one denominator per panel year")
        if self.expected_share.shape != self.flags.shape:
            raise ValueError("expected_share must align with flags")
        if len(self.true_trend_log_per_year) != n_condition:
            raise ValueError("true_trend_log_per_year must align with the conditions")
        if n_condition < 2:
            raise ValueError("the panel needs at least two control conditions")
        if n_year < 2:
            raise ValueError("the panel needs at least two years")
        if np.any(self.flags < 0) or np.any(self.flags > self.births[:, np.newaxis]):
            raise ValueError("panel flags must lie between zero and the denominator")
        if np.any(self.expected_share <= 0.0):
            raise ValueError("expected_share values must be positive")
        if not 0 <= self.reference_year_idx < n_year:
            raise ValueError("reference_year_idx must index a panel year")
        if not np.array_equal(self.year_idx, np.sort(self.year_idx)):
            raise ValueError("panel years must be sorted")
        if len(set(self.year_idx.tolist())) != n_year:
            raise ValueError("panel years must be distinct")

    @property
    def n_year(self) -> int:
        return int(self.flags.shape[0])

    @property
    def n_condition(self) -> int:
        return int(self.flags.shape[1])

    @property
    def years_since_reference(self) -> np.ndarray:
        """Signed year offset from the reference year, for the trend terms."""
        return (self.year_idx - self.year_idx[self.reference_year_idx]).astype(float)

    def observed_rate(self) -> np.ndarray:
        return self.flags / self.births[:, np.newaxis]

    def anchored_overlap_years(self, last_anchored_year_idx: int) -> tuple[int, ...]:
        """Panel years a surveillance window still reaches.

        These are the years in which both channels speak about the same
        recording sensitivity, so they are what makes the Down syndrome loading
        estimable rather than purely prior-driven.
        """
        return tuple(
            int(year)
            for year, idx in zip(self.years, self.year_idx, strict=True)
            if idx <= last_anchored_year_idx
        )

    def to_dict(self) -> dict[str, Any]:
        heterogeneity = panel_heterogeneity(self)
        return {
            "source": self.source,
            "conditions_source": self.conditions_source,
            "conditions": list(self.condition),
            "labels": list(self.labels),
            "years": list(self.years),
            "year_idx": self.year_idx.tolist(),
            "flags": self.flags.tolist(),
            "births": self.births.tolist(),
            "expected_share": self.expected_share.tolist(),
            "reference_year": int(self.years[self.reference_year_idx]),
            "n_condition": self.n_condition,
            "n_year": self.n_year,
            "flags_per_year": [int(v) for v in self.flags.sum(axis=1)],
            "true_trend_log_per_year": self.true_trend_log_per_year.tolist(),
            "excluded_conditions": [
                {"condition": name, "reason": reason} for name, reason in self.excluded
            ],
            # The panel's own internal consistency. A shared recording factor is
            # only as credible as the controls' agreement about it, so this
            # travels with every fit rather than living in a one-off analysis.
            "heterogeneity": heterogeneity,
        }


def panel_heterogeneity(
    panel: AnomalyPanel,
    *,
    span_years: int = 3,
) -> dict[str, Any]:
    """Measure whether the controls agree about the common recording change.

    Compares each condition's composition-adjusted log-rate change between the
    first and last ``span_years`` of the panel, then reports a standard
    random-effects heterogeneity summary over the conditions.  A high ``i_squared``
    means the "shared factor" the model is about to estimate is not something the
    controls agree on, and the fit's interval should be read accordingly.

    Poisson counting error is the only within-condition uncertainty used here.
    Omitting other within-condition error can inflate Q and I-squared.
    These are descriptive checks under the counting-error assumption, not a
    lower bound on heterogeneity or a test of the panel's causal assumptions.
    """
    n_year = panel.n_year
    span = int(min(span_years, n_year // 2))
    if span < 1:
        return {"available": False, "reason": "panel too short to compare spans"}

    early = slice(0, span)
    late = slice(n_year - span, n_year)
    rate = panel.observed_rate()
    adjusted = rate / panel.expected_share

    early_flags = panel.flags[early].sum(axis=0)
    late_flags = panel.flags[late].sum(axis=0)
    if np.any(early_flags < 1) or np.any(late_flags < 1):
        return {"available": False, "reason": "a condition has no flags in a span"}

    # Weight each span mean by its births so the ratio matches the pooled rate.
    early_rate = (adjusted[early] * panel.births[early, np.newaxis]).sum(
        axis=0
    ) / panel.births[early].sum()
    late_rate = (adjusted[late] * panel.births[late, np.newaxis]).sum(
        axis=0
    ) / panel.births[late].sum()
    log_change = np.log(late_rate / early_rate)
    variance = 1.0 / early_flags + 1.0 / late_flags

    weight = 1.0 / variance
    fixed_mean = float((log_change * weight).sum() / weight.sum())
    q_statistic = float((weight * (log_change - fixed_mean) ** 2).sum())
    degrees = panel.n_condition - 1
    i_squared = (
        max(0.0, (q_statistic - degrees) / q_statistic) if q_statistic > 0 else 0.0
    )
    # DerSimonian-Laird between-condition variance.
    weight_scale = weight.sum() - (weight**2).sum() / weight.sum()
    tau_squared = (
        max(0.0, (q_statistic - degrees) / weight_scale) if weight_scale > 0 else 0.0
    )
    random_weight = 1.0 / (variance + tau_squared)
    random_mean = float((log_change * random_weight).sum() / random_weight.sum())
    random_se = float(np.sqrt(1.0 / random_weight.sum()))

    return {
        "available": True,
        "span_years": span,
        "early_years": [int(y) for y in panel.years[early]],
        "late_years": [int(y) for y in panel.years[late]],
        "condition_log_change": dict(
            zip(panel.condition, [float(v) for v in log_change], strict=True)
        ),
        "condition_se": dict(
            zip(
                panel.condition,
                [float(v) for v in np.sqrt(variance)],
                strict=True,
            )
        ),
        "fixed_effect_log_change": fixed_mean,
        "fixed_effect_se": float(np.sqrt(1.0 / weight.sum())),
        "q_statistic": q_statistic,
        "degrees_of_freedom": degrees,
        "i_squared": float(i_squared),
        "tau": float(np.sqrt(tau_squared)),
        # The honest headline: a random-effects mean, whose SE widens with the
        # controls' disagreement instead of ignoring it.
        "random_effect_log_change": random_mean,
        "random_effect_se": random_se,
    }


def prepare_anomaly_panel(
    con: duckdb.DuckDBPyConnection,
    *,
    year_range: tuple[int, int],
    panel_from_year: int = DEFAULT_PANEL_FROM_YEAR,
    conditions: AnomalyPanelConditions | None = None,
    condition_names: list[str] | tuple[str, ...] | None = None,
    table: str = "us_births",
    year_column: str = "dob_yy",
    age_column: str = "mage_c",
    recorded_column: str = "down_ind",
    reference_year: int | None = None,
) -> AnomalyPanel:
    """Aggregate control-condition flags into a year-by-condition panel.

    The row filter matches :func:`prepare_core_age_year_cells` exactly, so the
    panel's denominators equal the certificate cells' per-year births and the two
    likelihood channels describe the same population.  ``build_core_reduction_model``
    checks that.

    Hypospadias is male-only, so its denominator is nominally wrong.  The male
    share of US births is stable to within 0.08% across 2016-2024, against
    double-digit changes in the recorded rates, so the mis-scaling is absorbed by
    the condition's own level and contributes no trend.
    """
    from_year, to_year = year_range
    if panel_from_year < from_year or panel_from_year > to_year:
        raise ValueError(
            f"panel_from_year={panel_from_year} must fall inside the modelled "
            f"range {from_year}-{to_year}"
        )
    if panel_from_year < DEFAULT_PANEL_FROM_YEAR:
        raise ValueError(
            f"panel_from_year={panel_from_year} precedes {DEFAULT_PANEL_FROM_YEAR}, "
            "when revised-certificate coverage reaches 100%. Before then the "
            "revised subset is a changing set of states and the panel would read "
            "state composition as recording behaviour."
        )
    if to_year - panel_from_year + 1 < 2:
        raise ValueError("the panel needs at least two years")

    conditions = conditions or AnomalyPanelConditions.from_csv()
    selected = conditions.select(condition_names)
    names = tuple(str(c) for c in selected["condition"])

    flag_selects = ",\n            ".join(
        f"SUM(CASE WHEN UPPER(CAST({name} AS VARCHAR)) = 'Y' THEN 1 ELSE 0 END) "
        f"AS {name}"
        for name in names
    )
    sql = f"""
        WITH coded AS (
            SELECT
                CAST({year_column} AS INTEGER) AS year,
                CASE
                    WHEN CAST({age_column} AS INTEGER) <= 12 THEN 12
                    WHEN CAST({age_column} AS INTEGER) >= 50 THEN 50
                    ELSE CAST({age_column} AS INTEGER)
                END AS maternal_age,
                {", ".join(names)}
            FROM {table}
            WHERE {year_column} BETWEEN {panel_from_year} AND {to_year}
              AND {age_column} IS NOT NULL
              AND {recorded_column} IS NOT NULL
        )
        SELECT
            year,
            maternal_age,
            COUNT(*) AS births,
            {flag_selects}
        FROM coded
        GROUP BY year, maternal_age
        ORDER BY year, maternal_age
    """
    by_age = con.execute(sql).df()
    if by_age.empty:
        raise ValueError(
            f"no births found for the anomaly panel over {panel_from_year}-{to_year}"
        )

    years = tuple(int(y) for y in sorted(by_age["year"].unique()))
    expected_years = tuple(range(panel_from_year, to_year + 1))
    if years != expected_years:
        missing = sorted(set(expected_years) - set(years))
        raise ValueError(f"the anomaly panel has no births for years {missing}")

    flags = (
        by_age.groupby("year")[list(names)].sum().reindex(years).to_numpy(dtype=float)
    )
    births = by_age.groupby("year")["births"].sum().reindex(years).to_numpy(dtype=float)
    empty = [
        name for name, total in zip(names, flags.sum(axis=0), strict=True) if total <= 0
    ]
    if empty:
        raise ValueError(
            f"conditions {empty} carry no flags over {panel_from_year}-{to_year}; "
            "they are not populated in this extract and cannot serve as controls"
        )

    # Composition offset: hold each condition's age-specific rate at the
    # panel-pooled profile and let only the maternal-age distribution move.
    pooled = by_age.groupby("maternal_age")[["births", *names]].sum()
    pooled_rate = pooled[list(names)].to_numpy(dtype=float) / pooled[
        ["births"]
    ].to_numpy(dtype=float)
    age_share = by_age.pivot_table(
        index="year", columns="maternal_age", values="births", aggfunc="sum"
    ).reindex(years)
    age_share = age_share.reindex(columns=pooled.index, fill_value=0.0).fillna(0.0)
    age_share = age_share.to_numpy(dtype=float)
    age_share = age_share / age_share.sum(axis=1, keepdims=True)
    expected_share = age_share @ pooled_rate

    reference_year = years[0] if reference_year is None else int(reference_year)
    if reference_year not in years:
        raise ValueError(
            f"reference_year={reference_year} is not a panel year; panel covers "
            f"{years[0]}-{years[-1]}"
        )

    trend_lookup = dict(
        zip(
            selected["condition"],
            selected["true_trend_log_per_year"].to_numpy(dtype=float),
            strict=True,
        )
    )
    excluded = tuple(
        (str(row.condition), str(row.reason))
        for row in conditions.excluded().itertuples()
        if str(row.condition) not in names
    )
    return AnomalyPanel(
        condition=names,
        year_idx=np.asarray([y - from_year for y in years], dtype=int),
        years=years,
        flags=flags,
        births=births,
        expected_share=expected_share,
        true_trend_log_per_year=np.asarray(
            [trend_lookup[name] for name in names], dtype=float
        ),
        reference_year_idx=years.index(reference_year),
        source=f"{table} {panel_from_year}-{to_year}",
        labels=tuple(str(label) for label in selected["label"]),
        conditions_source=conditions.source,
        excluded=excluded,
    )


__all__ = [
    "DEFAULT_ANOMALY_CONDITIONS_CSV",
    "DEFAULT_PANEL_CONDITION_TREND_SIGMA",
    "DEFAULT_PANEL_FACTOR_SIGMA",
    "DEFAULT_PANEL_FROM_YEAR",
    "DEFAULT_PANEL_IDIOSYNCRATIC_SIGMA",
    "DEFAULT_PANEL_LOADING_SIGMA",
    "DEFAULT_PANEL_PREVALENCE_TREND_SIGMA",
    "AnomalyPanel",
    "AnomalyPanelConditions",
    "panel_heterogeneity",
    "prepare_anomaly_panel",
]
