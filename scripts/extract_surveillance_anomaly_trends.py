"""Extract true-prevalence trends for the DSP010 anomaly-panel control conditions.

The DSP010 panel reads the *common* movement of congenital-anomaly checkboxes that
share the Down syndrome certificate item as the item's recording sensitivity.  That
reading is only valid to the extent the control conditions' own birth prevalence held
still: any real prevalence trend shared by the controls is perfectly confounded with a
recording trend and no comparison inside the panel can see it.  Until now every
``true_trend_log_per_year`` in ``data/us-births-anomaly-panel-conditions.csv`` was
``0.0`` -- "believed stable, not verified".

This script replaces that assumption with an external measurement.

The source is the **Texas Birth Defects Registry** annual report, Table 2A, which
gives case counts and prevalence per 10,000 live births for each monitored defect by
single delivery year.  TBDR qualifies on the one criterion that matters here: it does
not ascertain cases from the birth-certificate anomaly item, so its series cannot
re-import the recording decline the panel is trying to measure.  From the report's own
methods:

    "The Texas Birth Defects Registry uses active surveillance.  This means it does
    not require reporting by hospitals or medical professionals.  Instead, trained
    program staff members regularly visit medical facilities where they have the
    authority to review logs, hospital discharge lists, and other records."

    "Regardless of the source of demographic information for this report, all
    diagnostic information was abstracted from medical records."

Birth and fetal-death certificates enter only to supply demographics for cases already
found, never to find or diagnose them.  The registry covered the whole state for every
year in the series, so there is no programme-roster composition artefact of the kind
that contaminates pooled multi-state trends.

Trends are fitted as quasi-Poisson log-linear regressions of counts on delivery year
with a log live-births offset.  The dispersion scaling matters: annual counts vary far
more than Poisson because ascertainment itself moves, and an unscaled standard error
would claim precision the series does not have.

**Validation gates the pin.**  A prevalence trend is only usable if the series is
smooth: a level shift inside the window means the fitted slope is an averaged
discontinuity rather than a trend, and pinning it would inject a registry artefact into
the model as though it were biology.  Every condition is therefore scanned for a
breakpoint, and any condition carrying a significant level shift is **refused a pin**
and left at zero with the reason recorded.  Discrepancies are reported, never silently
corrected; ``--strict`` turns a refusal into a non-zero exit status.

Outputs (written to ``--output-root``, with the two model-facing files also written to
``data/`` unless ``--no-install`` is passed):

``us-births-anomaly-surveillance-trends.csv``
    One row per condition per fitting window: fitted slope in log per year, both the
    Poisson and dispersion-scaled standard errors, the dispersion, the breakpoint scan
    result, and whether the condition earned a pin.
``us-births-anomaly-panel-conditions-pinned.csv``
    The curation table with ``true_trend_log_per_year`` filled in from the primary
    window.  Feed it to the model with ``--panel-conditions-csv``.
``surveillance_series.csv``
    The parsed per-year counts, denominators and rates, so the fit can be checked
    without re-reading the workbook.
``national_cross_check.csv``
    The NBDPN pooled-cohort national estimates against the same spans computed from
    Texas, as an external check that one state tracks the national series.
"""

from __future__ import annotations

import argparse
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dse_research_utils.environment.setup import init_script

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"

DEFAULT_WORKBOOK = Path("data/1999-2022-tbdr-2-prevbyyear.xlsx")
WORKBOOK_URL = (
    "https://www.dshs.texas.gov/sites/default/files/birthdefects/annualreport/"
    "1999-2022-tbdr-2-prevbyyear.xlsx"
)
DEFAULT_CONDITIONS_CSV = Path("data/us-births-anomaly-panel-conditions.csv")
DEFAULT_OUTPUT_ROOT = Path("output/surveillance_anomaly_trends")
DEFAULT_DATA_ROOT = Path("data")

# The primary window is deliberately longer than the panel's own 2016-2024 span.  A
# birth-prevalence trend is a slow quantity, and the short window is demonstrably
# unreliable here: over 2016-2022 the limb-reduction slope comes out *positive* and
# disagrees in sign with the national pooled series, while over 2010-2022 it agrees to
# within 0.001 log per year.  Seven annual points on a few hundred cases cannot
# separate a trend from noise; thirteen can.
DEFAULT_PRIMARY_WINDOW = (2010, 2022)
# Reported alongside as a sensitivity, because it is the span the panel actually covers
# and a reader is entitled to see what pinning to it would do.
DEFAULT_SENSITIVITY_WINDOW = (2016, 2022)

# A level shift this large relative to its standard error means the series is not a
# trend.  Three sigma is deliberately permissive -- it refuses only unmistakable
# discontinuities, not ordinary annual wobble.
DEFAULT_BREAK_Z = 3.0
# The model-free companion to the breakpoint scan: if a straight line fits this much
# worse than Poisson, the series is not a trend plus noise and its slope means nothing,
# whatever the reason.  Set well above the 1.2-1.8 the well-behaved controls show.
DEFAULT_MAX_DISPERSION = 3.0
# Each side of a candidate breakpoint needs enough points to define a slope.
MIN_SEGMENT_YEARS = 3

# TBDR label -> our condition code.  ``ca_limb`` sums two TBDR categories because the
# certificate carries a single limb-reduction checkbox where TBDR splits upper from
# lower.  A child with both is counted in both TBDR rows, so the summed *level* is
# slightly overstated; the summed *trend* is unaffected as long as that overlap share
# is stable, which is the same assumption the certificate's own single box makes.
CONDITION_LABELS: dict[str, tuple[str, ...]] = {
    "ca_hypo": ("Hypospadias (among males)",),
    "ca_clpal": ("Cleft palate alone (without cleft lip)",),
    "ca_cleft": ("Cleft lip with or without cleft palate",),
    "ca_limb": (
        "Reduction defects of the upper limbs",
        "Reduction defects of the lower limbs",
    ),
    "ca_gast": ("Gastroschisis",),
}
# Hypospadias is reported per 10,000 *male* live births, so it needs its own
# denominator or its trend picks up the sex ratio.
MALE_DENOMINATOR = frozenset({"ca_hypo"})
# Carried through the fit purely as an independent reference series.  Down syndrome is
# the model's own outcome, so it is never pinned -- but TBDR counts it from medical
# records, which makes it worth writing down next to the controls.
REFERENCE_LABELS: dict[str, tuple[str, ...]] = {
    "reference_ds": ("Down syndrome (trisomy 21)",),
}

# NBDPN pooled national estimates, per 10,000 live births, from the 2016-2020 national
# report and its predecessors.  Used ONLY as a sign-and-magnitude cross-check that a
# single state tracks the national series -- never as a pinned value, because pooled
# multi-year cohorts cannot give an annual slope and the contributing-programme roster
# changes between cohorts.  Hypospadias is absent from the national tables, which is
# precisely why Texas is load-bearing for the one control that matters most.
NATIONAL_POOLED: dict[str, dict[str, float]] = {
    "ca_clpal": {
        "1999-2001": 6.39,
        "2004-2006": 6.35,
        "2010-2014": 5.93,
        "2016-2020": 6.32,
    },
    "ca_cleft": {
        "1999-2001": 10.47,
        "2004-2006": 10.63,
        "2010-2014": 10.00,
        "2016-2020": 9.69,
    },
    "ca_limb": {"2010-2014": 5.15, "2016-2020": 4.84},
    "ca_gast": {
        "1999-2001": 3.73,
        "2004-2006": 4.49,
        "2010-2014": 5.12,
        "2016-2020": 4.10,
    },
}
NATIONAL_PERIOD_MIDPOINT = {
    "1999-2001": 2000.0,
    "2004-2006": 2005.0,
    "2010-2014": 2012.0,
    "2016-2020": 2018.0,
}
NATIONAL_SOURCE = (
    "Stallings et al. (2024) National population-based estimates for major birth "
    "defects, 2016-2020, Birth Defects Research 116(1):e2301, and the 1999-2001, "
    "2004-2006 and 2010-2014 predecessor estimates it tabulates"
)
TBDR_SOURCE = (
    "Texas Birth Defects Registry Annual Report, Table 2A, deliveries 1999-2022 "
    "(Texas DSHS, September 2025); active surveillance, diagnoses abstracted from "
    "medical records"
)

# US male share of live births sits just above 0.51; anything outside this band means
# the hypospadias denominator was not recovered correctly.
MALE_SHARE_BAND = (0.505, 0.517)

# The recovered denominators are checked against an entirely separate source already in
# the repository. A wrong denominator would rescale every slope silently, and the report
# tabulates its own denominators only in an appendix this script does not parse.
DEFAULT_WONDER_CSV = Path("data/us-births-wonder-state-year-2016-2024.csv")
DEFAULT_WONDER_STATE = "Texas"
MAX_DENOMINATOR_DRIFT = 0.01


def _column_index(ref: str) -> int:
    """Translate a spreadsheet cell reference's column letters to a 0-based index."""
    letters = re.match(r"[A-Z]+", ref)
    if letters is None:
        raise ValueError(f"unparseable cell reference: {ref!r}")
    index = 0
    for char in letters.group(0):
        index = index * 26 + (ord(char) - 64)
    return index - 1


@dataclass
class Findings:
    """Everything the run wants to tell the reader rather than fix behind their back."""

    refused_pins: list[dict[str, Any]] = field(default_factory=list)
    denominator_notes: list[str] = field(default_factory=list)
    sign_disagreements: list[dict[str, Any]] = field(default_factory=list)
    missing_years: list[dict[str, Any]] = field(default_factory=list)
    denominator_drift: list[dict[str, Any]] = field(default_factory=list)

    @property
    def has_discrepancies(self) -> bool:
        return bool(
            self.refused_pins
            or self.sign_disagreements
            or self.missing_years
            or self.denominator_drift
        )


def read_single_year_sheet(path: Path) -> dict[str, dict[int, tuple[int, float]]]:
    """Parse Table 2A into ``{defect label: {year: (cases, rate per 10,000)}}``.

    Read with the standard library only, so the canonical conda environment needs no
    Excel dependency.  Long defect blocks repeat their label when the printed table
    breaks across a page, so the label is carried forward and repeated blocks merge.
    """
    with zipfile.ZipFile(path) as archive:
        shared = [
            "".join(node.text or "" for node in item.iter(f"{NS}t"))
            for item in ET.fromstring(archive.read("xl/sharedStrings.xml")).iter(
                f"{NS}si"
            )
        ]
        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))

    rows: dict[int, dict[int, str]] = {}
    for row in sheet.iter(f"{NS}row"):
        cells: dict[int, str] = {}
        for cell in row.iter(f"{NS}c"):
            value = cell.find(f"{NS}v")
            if value is None or value.text is None:
                continue
            text = (
                shared[int(value.text)] if cell.attrib.get("t") == "s" else value.text
            )
            cells[_column_index(cell.attrib["r"])] = text.strip()
        if cells:
            rows[int(row.attrib["r"])] = cells

    series: dict[str, dict[int, tuple[int, float]]] = {}
    label: str | None = None
    for index in sorted(rows):
        cells = rows[index]
        # A label with no year beside it is a body-system heading, not a defect.
        if 0 in cells and 1 not in cells:
            label = None
            continue
        if 0 in cells and 1 in cells:
            label = " ".join(cells[0].split())
        if label is None or not {1, 2, 3} <= cells.keys():
            continue
        try:
            year = int(cells[1])
        except ValueError:
            continue
        series.setdefault(label, {})[year] = (int(cells[2]), float(cells[3]))
    return series


def derive_denominators(
    series: dict[str, dict[int, tuple[int, float]]], findings: Findings
) -> tuple[dict[int, float], dict[int, float]]:
    """Recover live-birth and male-live-birth denominators from counts and rates.

    The report tabulates its denominators in an appendix we do not parse, but every
    defect block implies one via ``cases / rate * 10_000``.  Taking the median across
    all whole-population blocks beats any single block, whose implied value is limited
    by the rate being printed to two decimals.
    """
    all_years = sorted({year for block in series.values() for year in block})
    live_births: dict[int, float] = {}
    male_births: dict[int, float] = {}
    for year in all_years:
        implied = [
            cases / rate * 1e4
            for label, block in series.items()
            if "males" not in label and year in block
            for cases, rate in [block[year]]
            if rate > 0
        ]
        if implied:
            live_births[year] = float(np.median(implied))
        male_block = series["Hypospadias (among males)"]
        if year in male_block:
            cases, rate = male_block[year]
            if rate > 0:
                male_births[year] = cases / rate * 1e4

    for year in sorted(set(live_births) & set(male_births)):
        share = male_births[year] / live_births[year]
        if not MALE_SHARE_BAND[0] <= share <= MALE_SHARE_BAND[1]:
            findings.denominator_notes.append(
                f"{year}: implied male share {share:.4f} outside "
                f"{MALE_SHARE_BAND[0]}-{MALE_SHARE_BAND[1]}"
            )
    return live_births, male_births


def check_denominators_against_wonder(
    live_births: dict[int, float],
    wonder_csv: Path,
    state: str,
    findings: Findings,
) -> pd.DataFrame:
    """Compare the recovered denominators against CDC WONDER resident live births.

    The two numbers come from unrelated places -- one backed out of printed rates in a
    state registry report, the other a national vital-statistics query already tracked in
    this repository -- so agreement is real evidence that the denominators are right.
    """
    if not wonder_csv.exists():
        findings.denominator_notes.append(
            f"denominator cross-check skipped: {wonder_csv} not present"
        )
        return pd.DataFrame()
    wonder = pd.read_csv(wonder_csv)
    official = wonder[wonder["state"] == state].set_index("year")["births"]
    records: list[dict[str, Any]] = []
    for year in sorted(set(live_births) & set(official.index)):
        implied = float(live_births[year])
        reference = float(official.loc[year])
        drift = implied / reference - 1
        if abs(drift) > MAX_DENOMINATOR_DRIFT:
            findings.denominator_drift.append(
                {"year": year, "implied": implied, "wonder": reference, "drift": drift}
            )
        records.append(
            {
                "year": year,
                "wonder_births": reference,
                "implied_births": implied,
                "relative_drift": drift,
                "state": state,
                "source": f"CDC WONDER via {wonder_csv}",
            }
        )
    return pd.DataFrame.from_records(records)


def quasi_poisson_trend(
    years: np.ndarray,
    counts: np.ndarray,
    offsets: np.ndarray,
    extra: np.ndarray | None = None,
) -> dict[str, Any]:
    """Fit ``log E[counts] = a + b*(year - year0) + [c*step] + log(offset)`` by IRLS.

    Returns the slope, its Poisson standard error, the dispersion-scaled (quasi-
    Poisson) standard error, and the same for any extra column supplied.  The scaled
    error is the one to quote: annual ascertainment wobble makes these counts
    over-dispersed, and the unscaled error would overstate what the series pins down.
    """
    counts = np.asarray(counts, dtype=float)
    centred = np.asarray(years, dtype=float) - float(np.min(years))
    columns = [np.ones_like(centred), centred]
    if extra is not None:
        columns.append(np.asarray(extra, dtype=float))
    design = np.column_stack(columns)
    log_offset = np.log(np.asarray(offsets, dtype=float))

    beta = np.zeros(design.shape[1])
    beta[0] = np.log(counts.sum() / np.exp(log_offset).sum())
    for _ in range(100):
        eta = design @ beta + log_offset
        mu = np.exp(eta)
        working = eta - log_offset + (counts - mu) / mu
        step = np.linalg.solve(
            design.T @ (mu[:, None] * design), design.T @ (mu * working)
        )
        if np.max(np.abs(step - beta)) < 1e-12:
            beta = step
            break
        beta = step

    mu = np.exp(design @ beta + log_offset)
    covariance = np.linalg.inv(design.T @ (mu[:, None] * design))
    dof = len(counts) - design.shape[1]
    pearson = float(np.sum((counts - mu) ** 2 / mu))
    # Never scale *down*: an under-dispersed short series is luck, not extra precision.
    dispersion = max(pearson / dof, 1.0) if dof > 0 else float("nan")
    poisson_se = float(np.sqrt(covariance[1, 1]))
    result = {
        "slope_log_per_year": float(beta[1]),
        "se_poisson": poisson_se,
        "se_quasi": poisson_se * float(np.sqrt(dispersion)),
        "dispersion": float(pearson / dof) if dof > 0 else float("nan"),
        "n_year": int(len(counts)),
        "cases": int(counts.sum()),
    }
    if extra is not None:
        extra_se = float(np.sqrt(covariance[2, 2])) * float(np.sqrt(dispersion))
        result["extra_coefficient"] = float(beta[2])
        result["extra_se"] = extra_se
        result["extra_z"] = float(beta[2]) / extra_se if extra_se > 0 else float("nan")
    return result


def scan_for_break(
    years: np.ndarray, counts: np.ndarray, offsets: np.ndarray
) -> dict[str, Any]:
    """Find the most significant level shift inside the window.

    A trend and a step look alike to a straight line but mean opposite things: a step
    is an ascertainment or coding change, and pinning it as biology would inject a
    registry artefact into the model.  Each interior year is tried as a breakpoint and
    the largest standardised shift is returned.
    """
    best: dict[str, Any] = {
        "break_year": None,
        "break_log_shift": float("nan"),
        "break_z": 0.0,
    }
    candidates = [
        year
        for year in years
        if (year - years.min()) >= MIN_SEGMENT_YEARS
        and (years.max() - year) >= MIN_SEGMENT_YEARS - 1
    ]
    for candidate in candidates:
        indicator = (years >= candidate).astype(float)
        try:
            fit = quasi_poisson_trend(years, counts, offsets, extra=indicator)
        except np.linalg.LinAlgError:
            continue
        if abs(fit["extra_z"]) > abs(best["break_z"]):
            best = {
                "break_year": int(candidate),
                "break_log_shift": fit["extra_coefficient"],
                "break_z": fit["extra_z"],
            }
    return best


def build_series_frame(
    series: dict[str, dict[int, tuple[int, float]]],
    live_births: dict[int, float],
    male_births: dict[int, float],
    findings: Findings,
) -> pd.DataFrame:
    """Collapse the TBDR blocks onto our condition codes, one row per condition-year."""
    records: list[dict[str, Any]] = []
    for condition, labels in {**CONDITION_LABELS, **REFERENCE_LABELS}.items():
        missing = [label for label in labels if label not in series]
        if missing:
            raise ValueError(f"{condition}: workbook has no block for {missing}")
        years = sorted(set.intersection(*(set(series[label]) for label in labels)))
        denominators = male_births if condition in MALE_DENOMINATOR else live_births
        for year in years:
            if year not in denominators:
                findings.missing_years.append({"condition": condition, "year": year})
                continue
            cases = sum(series[label][year][0] for label in labels)
            records.append(
                {
                    "condition": condition,
                    "year": year,
                    "cases": cases,
                    "denominator": denominators[year],
                    "rate_per_10k": cases / denominators[year] * 1e4,
                    "tbdr_labels": " + ".join(labels),
                }
            )
    return pd.DataFrame.from_records(records)


def fit_windows(
    frame: pd.DataFrame,
    windows: dict[str, tuple[int, int]],
    break_z: float,
    max_dispersion: float,
    findings: Findings,
) -> pd.DataFrame:
    """Fit every condition over every window and decide which trends may be pinned.

    Two independent gates, because each alone rests on a parametric choice the other
    does not.  The breakpoint scan asks "is there a level shift?", but it measures the
    shift against a fitted trend, so a sceptic can always dispute the trend and shrink
    the shift.  The dispersion gate asks the model-free question "does a straight line
    describe this series at all?" -- and a line that fits five times worse than Poisson
    has no slope worth pinning whatever the reason.
    """
    records: list[dict[str, Any]] = []
    for window_name, (start, end) in windows.items():
        for condition in [*CONDITION_LABELS, *REFERENCE_LABELS]:
            block = frame[
                (frame["condition"] == condition)
                & (frame["year"] >= start)
                & (frame["year"] <= end)
            ].sort_values("year")
            if len(block) < MIN_SEGMENT_YEARS + 1:
                continue
            years = block["year"].to_numpy()
            counts = block["cases"].to_numpy()
            offsets = block["denominator"].to_numpy()
            fit = quasi_poisson_trend(years, counts, offsets)
            fit.update(scan_for_break(years, counts, offsets))
            failed = []
            if abs(fit["break_z"]) > break_z:
                failed.append(
                    f"level shift of {100 * (np.exp(fit['break_log_shift']) - 1):+.1f}% "
                    f"at {fit['break_year']} (z={fit['break_z']:+.1f}), so the fitted "
                    f"slope averages a discontinuity rather than measuring a trend"
                )
            if fit["dispersion"] > max_dispersion:
                failed.append(
                    f"dispersion {fit['dispersion']:.2f} against a limit of "
                    f"{max_dispersion:.2f}, so a straight line does not describe the "
                    f"series and its slope is not a prevalence trend"
                )
            pinnable = condition in CONDITION_LABELS and not failed
            reason = ""
            if condition not in CONDITION_LABELS:
                reason = "reference series only, never pinned"
            elif failed:
                reason = "; ".join(failed)
                if window_name == "primary":
                    findings.refused_pins.append(
                        {
                            "condition": condition,
                            "window": f"{start}-{end}",
                            "reason": reason,
                        }
                    )
            records.append(
                {
                    "condition": condition,
                    "window": f"{start}-{end}",
                    "window_role": window_name,
                    **fit,
                    "pinnable": pinnable,
                    "pin_note": reason,
                    "source": TBDR_SOURCE,
                }
            )
    return pd.DataFrame.from_records(records)


def cross_check_national(frame: pd.DataFrame, findings: Findings) -> pd.DataFrame:
    """Compare Texas against NBDPN pooled cohorts over the pooled cohorts' own spans.

    The two most recent cohorts are used, not the widest available pair.  Gastroschisis
    is the reason: it rose for a decade and then fell, so a 1999-to-2020 comparison
    reports a rise for a condition that has been declining throughout the panel's span.
    Adjacent recent cohorts also sit closest to the years being pinned.
    """
    records: list[dict[str, Any]] = []
    for condition, pooled in NATIONAL_POOLED.items():
        periods = sorted(pooled, key=lambda key: NATIONAL_PERIOD_MIDPOINT[key])
        early, late = periods[-2], periods[-1]
        span = NATIONAL_PERIOD_MIDPOINT[late] - NATIONAL_PERIOD_MIDPOINT[early]
        national_slope = float(np.log(pooled[late] / pooled[early]) / span)
        texas: dict[str, float] = {}
        for period in (early, late):
            first, last = (int(part) for part in period.split("-"))
            block = frame[
                (frame["condition"] == condition)
                & (frame["year"] >= first)
                & (frame["year"] <= last)
            ]
            texas[period] = float(block["rate_per_10k"].mean())
        texas_slope = float(np.log(texas[late] / texas[early]) / span)
        agree = national_slope * texas_slope > 0
        if not agree:
            findings.sign_disagreements.append(
                {
                    "condition": condition,
                    "national_slope": national_slope,
                    "texas_slope": texas_slope,
                }
            )
        records.append(
            {
                "condition": condition,
                "early_period": early,
                "late_period": late,
                "span_years": span,
                "national_early": pooled[early],
                "national_late": pooled[late],
                "national_slope_log_per_year": national_slope,
                "texas_early": texas[early],
                "texas_late": texas[late],
                "texas_slope_log_per_year": texas_slope,
                "signs_agree": agree,
                "national_source": NATIONAL_SOURCE,
            }
        )
    return pd.DataFrame.from_records(records)


def build_pinned_conditions(
    conditions_csv: Path, trends: pd.DataFrame, window_role: str
) -> pd.DataFrame:
    """Write the fitted slopes into a copy of the curation table.

    Conditions that failed the breakpoint scan keep ``0.0`` and gain a reason, so the
    zero is a recorded decision rather than the silent default it replaces.
    """
    table = pd.read_csv(conditions_csv)
    chosen = trends[trends["window_role"] == window_role].set_index("condition")
    pinned: list[float] = []
    reasons: list[str] = []
    for _, row in table.iterrows():
        condition = row["condition"]
        reason = str(row["reason"])
        if condition not in chosen.index:
            pinned.append(0.0)
            reasons.append(
                f"{reason} Prevalence trend not pinned: no active-surveillance series "
                f"extracted for this condition."
            )
            continue
        fit = chosen.loc[condition]
        window = fit["window"]
        if bool(fit["pinnable"]):
            pinned.append(round(float(fit["slope_log_per_year"]), 6))
            reasons.append(
                f"{reason} Prevalence trend pinned to "
                f"{float(fit['slope_log_per_year']):+.5f} log per year "
                f"(SE {float(fit['se_quasi']):.5f}) from {TBDR_SOURCE}, {window}."
            )
        else:
            pinned.append(0.0)
            reasons.append(
                f"{reason} Prevalence trend left at zero: {fit['pin_note']}."
            )
    table["true_trend_log_per_year"] = pinned
    table["reason"] = reasons
    table["source"] = [
        f"curation; trend from TBDR {chosen.loc[row['condition'], 'window']}"
        if row["condition"] in chosen.index
        else "curation"
        for _, row in table.iterrows()
    ]
    return table


def report(
    trends: pd.DataFrame,
    cross_check: pd.DataFrame,
    findings: Findings,
    window_role: str,
) -> None:
    """Print the numbers a reader needs to judge the pin, not just its result."""
    primary = trends[trends["window_role"] == window_role]
    print("\n== fitted prevalence trends (log per year) ==")
    print(
        f"{'condition':12} {'window':10} {'slope':>10} {'SE':>9} {'disp':>6} "
        f"{'break':>18} {'pin':>5}"
    )
    for _, row in primary.sort_values("condition").iterrows():
        break_note = (
            f"{100 * (np.exp(row['break_log_shift']) - 1):+.1f}%@{int(row['break_year'])} "
            f"z={row['break_z']:+.1f}"
            if pd.notna(row["break_year"])
            else "none"
        )
        print(
            f"{row['condition']:12} {row['window']:10} {row['slope_log_per_year']:>+10.5f} "
            f"{row['se_quasi']:>9.5f} {row['dispersion']:>6.2f} {break_note:>18} "
            f"{'yes' if row['pinnable'] else 'NO':>5}"
        )

    controls = primary[
        primary["condition"].isin(CONDITION_LABELS)
        & (primary["condition"] != "ca_gast")
    ]
    effective = np.where(
        controls["pinnable"].to_numpy(), controls["slope_log_per_year"].to_numpy(), 0.0
    )
    print(
        f"\nshared prevalence trend implied by the pinned controls: "
        f"{effective.mean():+.5f} log per year "
        f"(spread {effective.min():+.5f} to {effective.max():+.5f}); "
        f"the DSP010 prior on this quantity was 0 +- 0.00400"
    )

    print("\n== national cross-check (pooled cohorts, sign and magnitude only) ==")
    for _, row in cross_check.iterrows():
        print(
            f"{row['condition']:12} national {row['national_slope_log_per_year']:+.5f} "
            f"vs Texas {row['texas_slope_log_per_year']:+.5f} over "
            f"{row['span_years']:.0f} years -> "
            f"{'agree' if row['signs_agree'] else 'SIGN DISAGREEMENT'}"
        )

    if findings.refused_pins:
        print("\n== pins refused ==")
        for item in findings.refused_pins:
            print(f"  {item['condition']} ({item['window']}): {item['reason']}")
    if findings.denominator_drift:
        print("\n== denominators disagree with CDC WONDER ==")
        for item in findings.denominator_drift:
            print(
                f"  {item['year']}: implied {item['implied']:,.0f} vs WONDER "
                f"{item['wonder']:,.0f} ({100 * item['drift']:+.2f}%)"
            )
    if findings.denominator_notes:
        print("\n== denominator warnings ==")
        for note in findings.denominator_notes:
            print(f"  {note}")
    if findings.missing_years:
        print(
            f"\n== {len(findings.missing_years)} condition-years dropped for a missing denominator =="
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK)
    parser.add_argument("--conditions-csv", type=Path, default=DEFAULT_CONDITIONS_CSV)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--wonder-csv", type=Path, default=DEFAULT_WONDER_CSV)
    parser.add_argument("--wonder-state", default=DEFAULT_WONDER_STATE)
    parser.add_argument(
        "--primary-window",
        type=int,
        nargs=2,
        metavar=("FIRST", "LAST"),
        default=DEFAULT_PRIMARY_WINDOW,
        help="years used for the pinned trend (default %(default)s)",
    )
    parser.add_argument(
        "--sensitivity-window",
        type=int,
        nargs=2,
        metavar=("FIRST", "LAST"),
        default=DEFAULT_SENSITIVITY_WINDOW,
        help="second window reported for comparison (default %(default)s)",
    )
    parser.add_argument(
        "--break-z",
        type=float,
        default=DEFAULT_BREAK_Z,
        help="refuse to pin a condition whose level shift exceeds this z (default %(default)s)",
    )
    parser.add_argument(
        "--max-dispersion",
        type=float,
        default=DEFAULT_MAX_DISPERSION,
        help="refuse to pin a condition whose straight-line fit is this over-dispersed (default %(default)s)",
    )
    parser.add_argument(
        "--no-install",
        action="store_true",
        help="write only to --output-root, leaving the tracked data/ copies alone",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero if any pin was refused or the national check disagrees",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    init_script()

    if not args.workbook.exists():
        print(
            f"workbook not found: {args.workbook}\ndownload it with:\n"
            f'  curl -sSL -o "{args.workbook}" "{WORKBOOK_URL}"',
            file=sys.stderr,
        )
        return 2

    findings = Findings()
    series = read_single_year_sheet(args.workbook)
    print(f"parsed {len(series)} defect blocks from {args.workbook}")

    live_births, male_births = derive_denominators(series, findings)
    span = f"{min(live_births)}-{max(live_births)}"
    print(
        f"recovered live-birth denominators for {len(live_births)} years ({span}); "
        f"male share {np.mean([male_births[y] / live_births[y] for y in male_births]):.4f}"
    )

    denominator_check = check_denominators_against_wonder(
        live_births, args.wonder_csv, args.wonder_state, findings
    )
    if not denominator_check.empty:
        worst = denominator_check["relative_drift"].abs().max()
        print(
            f"denominators cross-checked against CDC WONDER {args.wonder_state} for "
            f"{len(denominator_check)} years; worst drift {100 * worst:.2f}%"
        )

    frame = build_series_frame(series, live_births, male_births, findings)
    windows = {
        "primary": tuple(args.primary_window),
        "sensitivity": tuple(args.sensitivity_window),
    }
    trends = fit_windows(frame, windows, args.break_z, args.max_dispersion, findings)
    cross_check = cross_check_national(frame, findings)
    pinned = build_pinned_conditions(args.conditions_csv, trends, "primary")
    report(trends, cross_check, findings, "primary")

    args.output_root.mkdir(parents=True, exist_ok=True)
    outputs = {
        "us-births-anomaly-surveillance-trends.csv": trends,
        "us-births-anomaly-panel-conditions-pinned.csv": pinned,
        "surveillance_series.csv": frame,
        "national_cross_check.csv": cross_check,
        "denominator_cross_check.csv": denominator_check,
    }
    for name, table in outputs.items():
        table.to_csv(args.output_root / name, index=False)
    print(f"\nwrote {len(outputs)} files to {args.output_root}")

    if not args.no_install:
        args.data_root.mkdir(parents=True, exist_ok=True)
        for name in (
            "us-births-anomaly-surveillance-trends.csv",
            "us-births-anomaly-panel-conditions-pinned.csv",
        ):
            outputs[name].to_csv(args.data_root / name, index=False)
        print(f"installed the two model-facing files into {args.data_root}")

    if args.strict and findings.has_discrepancies:
        print(
            "\nstrict mode: a pin was refused or the national check disagreed; "
            "see us-births-anomaly-surveillance-trends.csv",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
