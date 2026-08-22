"""Extract the de Graaf surveillance-prevalence workbook and validate it.

The workbook (``ALT3``) carries, for five maternal race/Hispanic-origin groups
and overlapping five-year windows, the Down-syndrome live-birth prevalence
observed in US birth-defect **surveillance programmes**.  That surveillance
series is the only genuinely external quantity in the file: everything else --
birth-certificate counts, the five-year running prevalences, and the resulting
"percentage reported" -- is arithmetic on data we already hold.

This script therefore extracts the surveillance series and *recomputes* the rest
from ``data/us_births.db`` rather than trusting the workbook's own denominators,
which disagree with ours in specific years -- the multi-race code introduced in
2014 is included in some years and excluded in others, and Pacific Islander
births are grouped with American Indian/Alaska Native in four years.  It reads
the ``.xlsx`` with the standard library only, so the canonical environment does
not need an Excel dependency.

The primary outputs are:

``surveillance_prevalence_race_window.csv``
    The external input: one row per (window mid-year, race) with the
    surveillance prevalence.  Only observed windows appear, so the absent
    mid-years are themselves the record of what is missing.
``reporting_fraction_window.csv``
    Recomputed on our own birth-certificate counts: five-year recorded flags,
    births, expected cases and the implied reporting fraction, per race and
    pooled.
``expected_births_anchor.csv``
    The pooled surveillance-based expected Down-syndrome live births per window
    -- the quantity intended to anchor the DSP model's level.

Four further files record provenance and validation: ``workbook_panel.csv``
(the workbook's own per-year figures, with each reporting fraction marked
observed, pasted or fitted), ``workbook_trendlines.csv``,
``workbook_multirace_treatment.csv`` and ``workbook_count_mismatches.csv``.

Validation is reported, never silently corrected.  ``--strict`` turns the
workbook-versus-database discrepancies into a non-zero exit status.
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

import duckdb
import numpy as np
import pandas as pd
from dse_research_utils.environment.setup import init_script

DEFAULT_WORKBOOK = Path(
    "data/cor verwissel jaren overzicht prevalencties races usa birth cert "
    "vanaf 2000 ALT3.xlsx"
)
DEFAULT_DB_PATH = Path("data/us_births.db")
DEFAULT_OUTPUT_ROOT = Path("output/degraaf_surveillance")

FIRST_YEAR = 2000
LAST_YEAR = 2024
WINDOW_HALF_WIDTH = 2

# Workbook row order within each year block, and the mracehisp_c code each
# label maps onto. "tot weighted" is de Graaf's birth-weighted pooled row.
RACE_ORDER = ("nhw", "nhb", "his", "as/pi", "ai/an", "tot weighted")
RACE_CODES: dict[str, int | None] = {
    "nhw": 1,
    "nhb": 2,
    "his": 5,
    "as/pi": 4,
    "ai/an": 3,
    "tot weighted": None,
}
SINGLE_RACES = tuple(r for r in RACE_ORDER if RACE_CODES[r] is not None)

# Column letters in the main per-year block.
COL_YEAR = "A"
COL_RACE = "B"
COL_RECORDED = "C"
COL_BIRTHS = "D"
COL_YEAR_INDEX = "F"
COL_REPORTING_FILLED = "G"
COL_SURVEILLANCE = "L"
COL_BC_PREV_5YR = "Q"
COL_SURVEILLANCE_5YR = "R"
COL_REPORTING_OBSERVED = "U"

# Side block holding the surveillance matrix: mid-years in row 4, one row per
# race, columns AA(27) onward.
SIDE_MIDYEAR_ROW = 4
SIDE_WINDOW_LABEL_ROW = 3
SIDE_RACE_ROWS = {"nhw": 5, "nhb": 6, "his": 7, "as/pi": 8, "ai/an": 9}
SIDE_FIRST_COL = 27
SIDE_LAST_COL = 45

# de Graaf's hardcoded linear-trend coefficients (slope, intercept) in column G,
# read off the chart trendlines and pasted in as literals. Reproduced here so
# the extraction can prove it recovers them from the observed points.
TRENDLINE_COEFFICIENTS: dict[str, tuple[float, float]] = {
    "nhw": (0.001692095492241, 0.421962461509317),
    "nhb": (0.005150633225139, 0.220699386696071),
    "his": (0.001980572041, 0.316338542655),
    "as/pi": (0.001836717736925, 0.320411014238408),
    "ai/an": (0.009189841181544, 0.335633908102391),
    "tot weighted": (0.001714237881126, 0.366315772420995),
}

_SPREADSHEET_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
_CELL_REF = re.compile(r"([A-Z]+)(\d+)")


def _column_index(letters: str) -> int:
    index = 0
    for char in letters:
        index = index * 26 + (ord(char) - 64)
    return index


@dataclass
class Cell:
    value: str | None
    formula: str | None

    def number(self) -> float | None:
        """Numeric value, or None for blanks and Excel error literals."""
        if self.value is None:
            return None
        try:
            return float(self.value)
        except ValueError:
            return None  # e.g. '#DIV/0!'


def read_worksheet(path: Path) -> dict[tuple[int, int], Cell]:
    """Read the first worksheet into a {(row, column index): Cell} mapping."""
    with zipfile.ZipFile(path) as archive:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall("m:si", _SPREADSHEET_NS):
                shared.append(
                    "".join(
                        node.text or ""
                        for node in item.iter(f"{{{_SPREADSHEET_NS['m']}}}t")
                    )
                )
        root = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))

    cells: dict[tuple[int, int], Cell] = {}
    for node in root.iter(f"{{{_SPREADSHEET_NS['m']}}}c"):
        ref = node.get("r")
        if ref is None:
            continue
        match = _CELL_REF.match(ref)
        if match is None:
            continue
        column, row = _column_index(match.group(1)), int(match.group(2))
        cell_type = node.get("t")
        value_node = node.find("m:v", _SPREADSHEET_NS)
        formula_node = node.find("m:f", _SPREADSHEET_NS)
        inline_node = node.find("m:is", _SPREADSHEET_NS)

        value: str | None = None
        if cell_type == "s" and value_node is not None:
            value = shared[int(value_node.text or "0")]
        elif cell_type == "inlineStr" and inline_node is not None:
            value = "".join(
                n.text or "" for n in inline_node.iter(f"{{{_SPREADSHEET_NS['m']}}}t")
            )
        elif value_node is not None:
            value = value_node.text
        cells[(row, column)] = Cell(
            value=value,
            formula=formula_node.text if formula_node is not None else None,
        )
    return cells


def chart_trendline_sources(path: Path) -> list[dict[str, Any]]:
    """Describe each chart's series and trendline, to document provenance.

    Excel stores the trendline *specification* but not its fitted coefficients,
    so the numbers in column G can only have arrived by being read off a chart
    and typed in. This records which series each line was fitted to.
    """
    charts: list[dict[str, Any]] = []
    with zipfile.ZipFile(path) as archive:
        names = sorted(
            n for n in archive.namelist() if re.fullmatch(r"xl/charts/chart\d+\.xml", n)
        )
        for name in names:
            xml = archive.read(name).decode("utf-8")
            charts.append(
                {
                    "chart": name.rsplit("/", 1)[-1],
                    "series_refs": re.findall(r"<c:f>([^<]*)</c:f>", xml),
                    "trendline_types": re.findall(r'trendlineType val="(\w+)"', xml),
                    "displays_equation": 'dispEq val="1"' in xml,
                    "forecast_periods": re.findall(
                        r"<c:(forward|backward)[^>]*val=\"([^\"]*)\"", xml
                    ),
                }
            )
    return charts


@dataclass
class Findings:
    """Validation results, reported rather than silently patched."""

    arithmetic: list[tuple[str, float]] = field(default_factory=list)
    trendlines: list[dict[str, Any]] = field(default_factory=list)
    count_mismatches: list[dict[str, Any]] = field(default_factory=list)
    multirace_treatment: list[dict[str, Any]] = field(default_factory=list)

    @property
    def has_discrepancies(self) -> bool:
        return bool(self.count_mismatches) or any(
            not np.isfinite(v) or v > 1e-9 for _, v in self.arithmetic
        )


def extract_panel(cells: dict[tuple[int, int], Cell]) -> pd.DataFrame:
    """Pull the main per-(year, race) block, recording each cell's provenance."""

    def cell(row: int, letter: str) -> Cell:
        return cells.get((row, _column_index(letter)), Cell(None, None))

    records: list[dict[str, Any]] = []
    for offset in range(LAST_YEAR - FIRST_YEAR + 1):
        year = FIRST_YEAR + offset
        for position, race in enumerate(RACE_ORDER):
            row = 2 + len(RACE_ORDER) * offset + position
            found_year = cell(row, COL_YEAR).number()
            found_race = cell(row, COL_RACE).value
            if found_year is None or int(found_year) != year or found_race != race:
                raise ValueError(
                    f"workbook layout changed: row {row} holds "
                    f"({found_year!r}, {found_race!r}), expected ({year}, {race!r})"
                )

            formula = (cell(row, COL_REPORTING_FILLED).formula or "").strip()
            if formula.startswith(COL_REPORTING_OBSERVED):
                source = "observed"
            elif formula:
                source = "fitted"
            elif cell(row, COL_REPORTING_FILLED).value is not None:
                source = "pasted"
            else:
                source = None

            records.append(
                {
                    "row": row,
                    "year": year,
                    "race": race,
                    "mracehisp_c": RACE_CODES[race],
                    "wb_recorded": cell(row, COL_RECORDED).number(),
                    "wb_births": cell(row, COL_BIRTHS).number(),
                    "year_index": cell(row, COL_YEAR_INDEX).number(),
                    "wb_reporting_filled": cell(row, COL_REPORTING_FILLED).number(),
                    "wb_reporting_source": source,
                    "wb_reporting_observed": cell(row, COL_REPORTING_OBSERVED).number(),
                    "wb_surveillance_5yr": cell(row, COL_SURVEILLANCE_5YR).number(),
                    "wb_surveillance_annual_col": cell(row, COL_SURVEILLANCE).number(),
                    "wb_bc_prev_5yr": cell(row, COL_BC_PREV_5YR).number(),
                }
            )
    return pd.DataFrame.from_records(records)


def extract_surveillance(cells: dict[tuple[int, int], Cell]) -> pd.DataFrame:
    """Pull the surveillance matrix from the side block: the external input."""
    records: list[dict[str, Any]] = []
    for column in range(SIDE_FIRST_COL, SIDE_LAST_COL + 1):
        midyear_cell = cells.get((SIDE_MIDYEAR_ROW, column), Cell(None, None))
        midyear = midyear_cell.number()
        if midyear is None:
            continue
        label = cells.get((SIDE_WINDOW_LABEL_ROW, column), Cell(None, None)).value
        for race, row in SIDE_RACE_ROWS.items():
            prevalence = cells.get((row, column), Cell(None, None)).number()
            if prevalence is None:
                continue
            records.append(
                {
                    "mid_year": int(midyear),
                    "window": label,
                    "race": race,
                    "mracehisp_c": RACE_CODES[race],
                    "surveillance_prev_per10k": prevalence,
                }
            )
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        raise ValueError("no surveillance values found in the side block")
    return frame.sort_values(["mid_year", "race"], ignore_index=True)


def check_arithmetic(panel: pd.DataFrame, findings: Findings) -> None:
    """Confirm the workbook's internal identities hold, so it is reconstructible."""

    def max_relative_error(left: Any, right: Any) -> float:
        left = np.asarray(left, dtype=float)
        right = np.asarray(right, dtype=float)
        usable = np.isfinite(left) & np.isfinite(right) & (right != 0)
        if not usable.any():
            return float("nan")
        return float(
            np.max(np.abs(left[usable] - right[usable]) / np.abs(right[usable]))
        )

    single = panel[panel.race.isin(SINGLE_RACES)]
    observed = single.dropna(
        subset=["wb_reporting_observed", "wb_bc_prev_5yr", "wb_surveillance_5yr"]
    )
    findings.arithmetic.append(
        (
            "U == Q / R (reporting fraction is 5yr BC prevalence / surveillance)",
            max_relative_error(
                observed.wb_reporting_observed,
                observed.wb_bc_prev_5yr / observed.wb_surveillance_5yr,
            ),
        )
    )
    matched = observed.dropna(subset=["wb_reporting_filled"])
    findings.arithmetic.append(
        (
            "G == U on cells where surveillance was observed",
            max_relative_error(
                matched.wb_reporting_filled, matched.wb_reporting_observed
            ),
        )
    )
    fitted = single[single.wb_reporting_source == "fitted"].dropna(
        subset=["wb_reporting_filled"]
    )
    predicted = [
        TRENDLINE_COEFFICIENTS[race][0] * index + TRENDLINE_COEFFICIENTS[race][1]
        for race, index in zip(fitted.race, fitted.year_index, strict=True)
    ]
    findings.arithmetic.append(
        (
            "G == fitted line on cells where surveillance was missing",
            max_relative_error(fitted.wb_reporting_filled, predicted),
        )
    )


def check_trendlines(panel: pd.DataFrame, findings: Findings) -> None:
    """Refit each trendline from the observed points and compare to the literals."""
    for race in RACE_ORDER:
        rows = panel[panel.race == race]
        observed = rows.dropna(subset=["wb_reporting_observed"])
        if race != "tot weighted":
            observed = observed.dropna(subset=["wb_surveillance_annual_col"])
        if len(observed) < 3:
            continue
        x = observed.year_index.to_numpy(dtype=float)
        y = observed.wb_reporting_observed.to_numpy(dtype=float)
        design = np.vstack([x, np.ones_like(x)]).T
        (slope, intercept), *_ = np.linalg.lstsq(design, y, rcond=None)
        residual = y - (slope * x + intercept)
        total = float(np.sum((y - y.mean()) ** 2))
        expected_slope, expected_intercept = TRENDLINE_COEFFICIENTS[race]
        findings.trendlines.append(
            {
                "race": race,
                "n_observed": len(x),
                "slope_refit": slope,
                "slope_workbook": expected_slope,
                "slope_rel_error": abs(slope - expected_slope) / abs(expected_slope),
                "intercept_refit": intercept,
                "intercept_workbook": expected_intercept,
                "intercept_rel_error": abs(intercept - expected_intercept)
                / abs(expected_intercept),
                "r_squared": 1.0 - float(np.sum(residual**2)) / total
                if total
                else float("nan"),
            }
        )


def load_birth_counts(db_path: Path) -> pd.DataFrame:
    """Recorded flags and births per year and race group, from our own data.

    Two years either side of the requested span are needed for the centred
    five-year windows. Births whose ``mracehisp_c`` is neither 1-5 -- the
    multi-race code introduced in 2014, and unknown origin -- are returned under
    group 0 so the pooled anchor can account for them explicitly.
    """
    query = f"""
        select year,
               case when mracehisp_c in (1, 2, 3, 4, 5) then mracehisp_c else 0 end
                   as race_group,
               count(*) as births,
               sum(case when down_ind = 1 then 1 else 0 end) as recorded,
               sum(case when mracehisp_c = 6 then 1 else 0 end) as multirace,
               sum(case when mracehisp_c is null then 1 else 0 end) as origin_unknown
        from us_births
        where year between {FIRST_YEAR - WINDOW_HALF_WIDTH}
                      and {LAST_YEAR + WINDOW_HALF_WIDTH}
        group by 1, 2
        order by 1, 2
    """
    with duckdb.connect(str(db_path), read_only=True) as connection:
        return connection.sql(query).df()


def check_counts(panel: pd.DataFrame, counts: pd.DataFrame, findings: Findings) -> None:
    """Compare the workbook's birth-certificate counts with ours, year by year.

    Differences are not corrected. They are classified, because the pattern is
    informative: whether a year's five race rows include or exclude the
    multi-race code, and whether any pair of groups has been transposed.
    """
    by_key = {
        (int(r.year), int(r.race_group)): (int(r.births), int(r.recorded))
        for r in counts.itertuples()
    }
    multirace = {
        int(r.year): int(r.multirace) for r in counts.itertuples() if r.multirace
    }

    for year in range(FIRST_YEAR, LAST_YEAR + 1):
        rows = panel[(panel.year == year) & panel.race.isin(SINGLE_RACES)]
        workbook_births = int(rows.wb_births.sum())
        ours_single = sum(
            by_key.get((year, RACE_CODES[r]), (0, 0))[0] for r in SINGLE_RACES
        )
        multi = multirace.get(year, 0)
        if workbook_births == ours_single:
            treatment = "excludes multi-race"
        elif workbook_births == ours_single + multi:
            treatment = "includes multi-race (allocated across groups)"
        else:
            treatment = f"neither (differs by {workbook_births - ours_single:+d})"
        findings.multirace_treatment.append(
            {
                "year": year,
                "workbook_births": workbook_births,
                "our_births_single_race": ours_single,
                "multi_race_births": multi,
                "treatment": treatment,
            }
        )

        for row in rows.itertuples():
            ours = by_key.get((year, int(row.mracehisp_c)))
            if ours is None:
                continue
            births_delta = int(row.wb_births) - ours[0]
            recorded_delta = int(row.wb_recorded) - ours[1]
            if births_delta or recorded_delta:
                findings.count_mismatches.append(
                    {
                        "year": year,
                        "race": row.race,
                        "workbook_births": int(row.wb_births),
                        "our_births": ours[0],
                        "births_delta": births_delta,
                        "workbook_recorded": int(row.wb_recorded),
                        "our_recorded": ours[1],
                        "recorded_delta": recorded_delta,
                    }
                )


def build_reporting_fractions(
    surveillance: pd.DataFrame, counts: pd.DataFrame
) -> pd.DataFrame:
    """Recompute five-year reporting fractions on our own counts."""
    by_key = {
        (int(r.year), int(r.race_group)): (int(r.births), int(r.recorded))
        for r in counts.itertuples()
    }

    def window_totals(mid_year: int, code: int) -> tuple[int, int] | None:
        parts = [
            by_key.get((mid_year + offset, code))
            for offset in range(-WINDOW_HALF_WIDTH, WINDOW_HALF_WIDTH + 1)
        ]
        if any(part is None for part in parts):
            return None
        return (
            sum(part[0] for part in parts),  # type: ignore[index]
            sum(part[1] for part in parts),  # type: ignore[index]
        )

    records: list[dict[str, Any]] = []
    for mid_year, group in surveillance.groupby("mid_year", sort=True):
        prevalence = dict(zip(group.race, group.surveillance_prev_per10k, strict=True))
        window = group.window.iloc[0]
        expected_total = births_total = recorded_total = 0.0
        complete = True
        for race in SINGLE_RACES:
            code = RACE_CODES[race]
            assert code is not None
            totals = window_totals(int(mid_year), code)
            rate = prevalence.get(race)
            if totals is None or rate is None:
                complete = False
                continue
            births, recorded = totals
            expected = rate * births / 1e4
            records.append(
                {
                    "mid_year": int(mid_year),
                    "window": window,
                    "race": race,
                    "mracehisp_c": code,
                    "surveillance_prev_per10k": rate,
                    "births_5yr": births,
                    "recorded_5yr": recorded,
                    "expected_5yr": expected,
                    "reporting_fraction": recorded / expected if expected else np.nan,
                }
            )
            expected_total += expected
            births_total += births
            recorded_total += recorded
        if complete and births_total:
            records.append(
                {
                    "mid_year": int(mid_year),
                    "window": window,
                    "race": "pooled (5 groups)",
                    "mracehisp_c": None,
                    "surveillance_prev_per10k": expected_total / births_total * 1e4,
                    "births_5yr": int(births_total),
                    "recorded_5yr": int(recorded_total),
                    "expected_5yr": expected_total,
                    "reporting_fraction": recorded_total / expected_total,
                }
            )
    return pd.DataFrame.from_records(records)


def build_anchor(reporting: pd.DataFrame, counts: pd.DataFrame) -> pd.DataFrame:
    """Pooled expected Down-syndrome live births per window, all races.

    The five named groups do not exhaust the birth cohort: multi-race (from
    2014) and unknown-origin births have no surveillance prevalence of their
    own.  They are carried at the composition-weighted mean of the five
    observed groups, which is the only assumption the data support, and the
    residual share is reported so the choice stays visible.
    """
    by_key = {
        (int(r.year), int(r.race_group)): (int(r.births), int(r.recorded))
        for r in counts.itertuples()
    }

    def residual_totals(mid_year: int) -> tuple[int, int] | None:
        parts = [
            by_key.get((mid_year + offset, 0))
            for offset in range(-WINDOW_HALF_WIDTH, WINDOW_HALF_WIDTH + 1)
        ]
        if any(part is None for part in parts):
            return None
        return (
            sum(part[0] for part in parts),  # type: ignore[index]
            sum(part[1] for part in parts),  # type: ignore[index]
        )

    pooled = reporting[reporting.race == "pooled (5 groups)"]
    records: list[dict[str, Any]] = []
    for row in pooled.itertuples():
        residual = residual_totals(int(row.mid_year)) or (0, 0)
        mean_prevalence = row.expected_5yr / row.births_5yr * 1e4
        births_all = row.births_5yr + residual[0]
        recorded_all = row.recorded_5yr + residual[1]
        expected_all = row.expected_5yr + mean_prevalence * residual[0] / 1e4
        records.append(
            {
                "mid_year": row.mid_year,
                "window": row.window,
                "births_5yr_named": row.births_5yr,
                "births_5yr_all": births_all,
                "residual_share": residual[0] / births_all if births_all else np.nan,
                "prevalence_per10k": mean_prevalence,
                "expected_ds_5yr": expected_all,
                "expected_ds_per_year": expected_all / (2 * WINDOW_HALF_WIDTH + 1),
                "recorded_ds_5yr": recorded_all,
                "recorded_ds_per_year": recorded_all / (2 * WINDOW_HALF_WIDTH + 1),
                "reporting_fraction": recorded_all / expected_all
                if expected_all
                else np.nan,
            }
        )
    return pd.DataFrame.from_records(records)


def summarise_trend(anchor: pd.DataFrame) -> dict[str, float]:
    """Log-linear trend in pooled surveillance prevalence, with a caveat.

    The windows overlap by four of five years, so the seventeen rows are not
    seventeen independent observations; the effective count is the span in
    birth-years divided by the window width.
    """
    mid = anchor.mid_year.to_numpy(dtype=float)
    prevalence = anchor.prevalence_per10k.to_numpy(dtype=float)
    centred = mid - FIRST_YEAR
    design = np.vstack([centred, np.ones_like(centred)]).T
    (slope, intercept), *_ = np.linalg.lstsq(design, np.log(prevalence), rcond=None)
    residual = np.log(prevalence) - (slope * centred + intercept)
    degrees = max(len(mid) - 2, 1)
    variance = float(np.sum(residual**2)) / degrees
    spread = float(np.sum((centred - centred.mean()) ** 2))
    span = int(mid.max() - mid.min()) + 2 * WINDOW_HALF_WIDTH + 1
    return {
        "log_slope_per_year": float(slope),
        "log_slope_se": float(np.sqrt(variance / spread)) if spread else float("nan"),
        "residual_sd": float(residual.std(ddof=2)),
        "n_windows": float(len(mid)),
        "birth_years_spanned": float(span),
        "effective_independent_windows": span / (2 * WINDOW_HALF_WIDTH + 1),
    }


def report(findings: Findings, anchor: pd.DataFrame, trend: dict[str, float]) -> None:
    print("\n--- workbook internal arithmetic (max relative error) ---")
    for label, error in findings.arithmetic:
        verdict = "OK" if np.isfinite(error) and error <= 1e-9 else "CHECK"
        print(f"  [{verdict}] {label}: {error:.3e}")

    print("\n--- trendline coefficients: refit vs the literals in column G ---")
    print(
        f"  {'race':<14}{'n':>3}{'slope refit':>15}{'rel err':>10}"
        f"{'intercept refit':>18}{'rel err':>10}{'R2':>7}"
    )
    for row in findings.trendlines:
        print(
            f"  {row['race']:<14}{row['n_observed']:>3}{row['slope_refit']:>15.9f}"
            f"{row['slope_rel_error']:>10.1e}{row['intercept_refit']:>18.9f}"
            f"{row['intercept_rel_error']:>10.1e}{row['r_squared']:>7.3f}"
        )

    print("\n--- multi-race treatment by year (workbook vs our counts) ---")
    for row in findings.multirace_treatment:
        if row["treatment"].startswith("excludes") and not row["multi_race_births"]:
            continue  # pre-2014: nothing to include, nothing to report
        print(
            f"  {row['year']}  workbook {row['workbook_births']:>9,}  "
            f"ours {row['our_births_single_race']:>9,}  "
            f"multi-race {row['multi_race_births']:>7,}  {row['treatment']}"
        )

    if findings.count_mismatches:
        frame = pd.DataFrame(findings.count_mismatches)
        print(
            f"\n--- count mismatches: {len(frame)} race-year cells "
            f"in {frame.year.nunique()} years ---"
        )
        for year, group in frame.groupby("year"):
            parts = ", ".join(
                f"{r.race} {r.births_delta:+,} births / {r.recorded_delta:+,} flags"
                for r in group.itertuples()
            )
            print(f"  {year}: {parts}")

    print("\n--- pooled surveillance anchor ---")
    print(
        f"  {'mid':<5}{'window':<12}{'births 5yr':>13}{'prev/10k':>10}"
        f"{'expected/yr':>13}{'recorded/yr':>13}{'reported':>10}{'residual':>10}"
    )
    for row in anchor.itertuples():
        print(
            f"  {row.mid_year:<5}{str(row.window):<12}{row.births_5yr_all:>13,}"
            f"{row.prevalence_per10k:>10.4f}{row.expected_ds_per_year:>13,.1f}"
            f"{row.recorded_ds_per_year:>13,.1f}{row.reporting_fraction:>10.4f}"
            f"{row.residual_share:>9.1%}"
        )

    print("\n--- prevalence trend ---")
    print(
        f"  log-linear slope {100 * trend['log_slope_per_year']:.3f}%/yr "
        f"(SE {100 * trend['log_slope_se']:.3f}pp), residual SD "
        f"{100 * trend['residual_sd']:.2f}%"
    )
    print(
        f"  {trend['n_windows']:.0f} overlapping windows span "
        f"{trend['birth_years_spanned']:.0f} birth-years, so roughly "
        f"{trend['effective_independent_windows']:.1f} independent observations"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero when the workbook disagrees with the database",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    init_script()

    if not args.workbook.exists():
        print(f"workbook not found: {args.workbook}", file=sys.stderr)
        return 2
    if not args.db_path.exists():
        print(f"database not found: {args.db_path}", file=sys.stderr)
        return 2

    cells = read_worksheet(args.workbook)
    panel = extract_panel(cells)
    surveillance = extract_surveillance(cells)
    counts = load_birth_counts(args.db_path)

    findings = Findings()
    check_arithmetic(panel, findings)
    check_trendlines(panel, findings)
    check_counts(panel, counts, findings)

    # Only observed windows exist in the side block, so the extracted file needs
    # no "observed" flag; the absent mid-years are the record of what is missing.
    observed_years = sorted(surveillance.mid_year.unique())
    gaps = sorted(
        set(range(min(observed_years), max(observed_years) + 1)) - set(observed_years)
    )
    print(
        f"surveillance windows observed: {len(observed_years)} mid-years "
        f"{min(observed_years)}-{max(observed_years)}, gaps at {gaps or 'none'}"
    )
    charts = chart_trendline_sources(args.workbook)
    print(
        f"charts carrying a displayed linear trendline: "
        f"{sum(1 for c in charts if c['trendline_types'] and c['displays_equation'])}"
        f" of {len(charts)}; forecast periods declared: "
        f"{sorted({p for c in charts for p in c['forecast_periods']}) or 'none'}"
    )

    reporting = build_reporting_fractions(surveillance, counts)
    anchor = build_anchor(reporting, counts)
    trend = summarise_trend(anchor)
    report(findings, anchor, trend)

    args.output_root.mkdir(parents=True, exist_ok=True)
    outputs = {
        "surveillance_prevalence_race_window.csv": surveillance,
        "reporting_fraction_window.csv": reporting,
        "expected_births_anchor.csv": anchor,
        "workbook_count_mismatches.csv": pd.DataFrame(findings.count_mismatches),
        "workbook_trendlines.csv": pd.DataFrame(findings.trendlines),
        "workbook_panel.csv": panel,
        "workbook_multirace_treatment.csv": pd.DataFrame(findings.multirace_treatment),
    }
    for name, frame in outputs.items():
        frame.to_csv(args.output_root / name, index=False)
    print(f"\nwrote {len(outputs)} files to {args.output_root}")

    if args.strict and findings.has_discrepancies:
        print(
            "\nstrict mode: workbook disagrees with the database; "
            "see workbook_count_mismatches.csv",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
