"""Tests for the anomaly-panel surveillance trend extraction.

The numerics here decide what gets pinned into the model as *known* true prevalence,
carrying no uncertainty once it lands in ``true_trend_log_per_year``.  A silently wrong
slope, or a step change mistaken for a trend, would inject a registry artefact into
DSP010 as though it were biology, so the fit, the dispersion scaling and the breakpoint
refusal all get exercised directly.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "extract_surveillance_anomaly_trends.py"
CONDITIONS_CSV = REPO_ROOT / "data" / "us-births-anomaly-panel-conditions.csv"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "extract_surveillance_anomaly_trends_cli", SCRIPT_PATH
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["extract_surveillance_anomaly_trends_cli"] = module
    spec.loader.exec_module(module)
    return module


EXTRACT = _load_module()


def _poisson_series(
    slope: float,
    years: np.ndarray,
    births: float = 400_000.0,
    base_rate: float = 5e-4,
    seed: int = 0,
):
    """Counts drawn from a log-linear Poisson process with a known slope."""
    rng = np.random.default_rng(seed)
    centred = years - years.min()
    mu = births * base_rate * np.exp(slope * centred)
    return rng.poisson(mu).astype(float), np.full(len(years), births)


def test_trend_recovers_a_known_slope():
    years = np.arange(2010, 2023)
    for slope in (-0.03, 0.0, 0.02):
        counts, offsets = _poisson_series(slope, years, seed=abs(int(slope * 1000)))
        fit = EXTRACT.quasi_poisson_trend(years, counts, offsets)
        assert fit["slope_log_per_year"] == pytest.approx(
            slope, abs=3 * fit["se_quasi"]
        )
        assert fit["n_year"] == len(years)
        assert fit["cases"] == int(counts.sum())


def test_a_flat_series_gives_a_slope_indistinguishable_from_zero():
    years = np.arange(2010, 2023)
    counts, offsets = _poisson_series(0.0, years, seed=7)
    fit = EXTRACT.quasi_poisson_trend(years, counts, offsets)
    assert abs(fit["slope_log_per_year"]) < 2 * fit["se_quasi"]


def test_dispersion_scaling_never_shrinks_the_standard_error():
    """An under-dispersed short series is luck, not extra precision."""
    years = np.arange(2016, 2023)
    # Rates held exactly on the model line, so Pearson chi-square lands below its
    # degrees of freedom and the raw dispersion is under one.
    offsets = np.full(len(years), 400_000.0)
    counts = np.round(offsets * 5e-4).astype(float)
    fit = EXTRACT.quasi_poisson_trend(years, counts, offsets)
    assert fit["dispersion"] < 1.0
    assert fit["se_quasi"] == pytest.approx(fit["se_poisson"])


def test_overdispersion_widens_the_standard_error():
    years = np.arange(2010, 2023)
    counts, offsets = _poisson_series(0.0, years, seed=3)
    # Inject year-to-year ascertainment wobble on top of Poisson noise.
    wobble = np.array([1.25 if index % 2 else 0.75 for index in range(len(years))])
    fit = EXTRACT.quasi_poisson_trend(years, np.round(counts * wobble), offsets)
    assert fit["dispersion"] > 1.0
    assert fit["se_quasi"] > fit["se_poisson"]


def test_break_scan_finds_an_injected_step():
    """Sized on the real refusal: hypospadias, ~1,400 cases a year, a 16% step at 2019."""
    years = np.arange(2010, 2023)
    counts, offsets = _poisson_series(0.0, years, base_rate=3.5e-3, seed=11)
    counts = counts * np.where(years >= 2019, 0.85, 1.0)
    found = EXTRACT.scan_for_break(years, np.round(counts), offsets)
    assert found["break_year"] == 2019
    assert found["break_log_shift"] < 0.0
    assert abs(found["break_z"]) > EXTRACT.DEFAULT_BREAK_Z


def test_break_scan_leaves_a_clean_trend_alone():
    years = np.arange(2010, 2023)
    counts, offsets = _poisson_series(-0.02, years, seed=5)
    found = EXTRACT.scan_for_break(years, counts, offsets)
    assert abs(found["break_z"]) <= EXTRACT.DEFAULT_BREAK_Z


def test_denominators_come_from_the_median_across_whole_population_blocks():
    """A single block's implied denominator is limited by the printed rate's 2 dp."""
    births = 400_000.0
    series = {
        f"Defect {index}": {2016: (int(round(births * rate / 1e4)), rate)}
        for index, rate in enumerate((5.0, 10.0, 4.0, 6.5, 12.0))
    }
    # A rate this high implies only 140,000 male births against 400,000 live births,
    # a share of 0.35 that no real birth cohort produces. It must be reported rather
    # than quietly accepted, because a wrong denominator silently rescales the trend.
    series["Hypospadias (among males)"] = {2016: (1_400, 100.0)}
    findings = EXTRACT.Findings()
    live, male = EXTRACT.derive_denominators(series, findings)
    assert live[2016] == pytest.approx(births, rel=1e-3)
    assert male[2016] == pytest.approx(140_000.0, rel=1e-9)
    assert findings.denominator_notes
    assert "0.35" in findings.denominator_notes[0]


def test_page_broken_blocks_merge_into_one_series(tmp_path: Path):
    """Long defect blocks repeat their label when the printed table breaks a page."""
    workbook = _write_workbook(
        tmp_path / "table.xlsx",
        [
            ("Central Nervous System", None, None, None),
            ("Gastroschisis", 2016, 188, 4.74),
            (None, 2017, 176, 4.61),
            ("Gastroschisis", 2018, 182, 4.83),
            (None, 2019, 172, 4.55),
        ],
    )
    series = EXTRACT.read_single_year_sheet(workbook)
    assert set(series) == {"Gastroschisis"}
    assert sorted(series["Gastroschisis"]) == [2016, 2017, 2018, 2019]
    assert series["Gastroschisis"][2018] == (182, 4.83)


def _write_workbook(
    path: Path, rows: list[tuple[str | None, int | None, int | None, float | None]]
) -> Path:
    """Write a minimal xlsx with the shape of TBDR Table 2A."""
    import zipfile

    strings = [row[0] for row in rows if row[0] is not None]
    shared_index = {value: index for index, value in enumerate(dict.fromkeys(strings))}
    sheet_rows = []
    for number, (label, year, cases, rate) in enumerate(rows, start=1):
        cells = []
        if label is not None:
            cells.append(f'<c r="A{number}" t="s"><v>{shared_index[label]}</v></c>')
        if year is not None:
            cells.append(f'<c r="B{number}"><v>{year}</v></c>')
        if cases is not None:
            cells.append(f'<c r="C{number}"><v>{cases}</v></c>')
        if rate is not None:
            cells.append(f'<c r="D{number}"><v>{rate}</v></c>')
        sheet_rows.append(f'<row r="{number}">{"".join(cells)}</row>')
    namespace = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "xl/sharedStrings.xml",
            f'<sst xmlns="{namespace}">'
            + "".join(f"<si><t>{value}</t></si>" for value in shared_index)
            + "</sst>",
        )
        archive.writestr(
            "xl/worksheets/sheet1.xml",
            f'<worksheet xmlns="{namespace}"><sheetData>{"".join(sheet_rows)}</sheetData></worksheet>',
        )
    return path


def _trend_frame(pinnable: bool) -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "condition": condition,
                "window": "2010-2022",
                "window_role": "primary",
                "slope_log_per_year": -0.005,
                "se_quasi": 0.004,
                "pinnable": pinnable,
                "pin_note": "" if pinnable else "level shift at 2019 (z=-6.0)",
            }
            for condition in ("ca_hypo", "ca_clpal", "ca_cleft", "ca_limb", "ca_gast")
        ]
    )


def test_pinned_table_carries_the_slope_and_its_provenance():
    table = EXTRACT.build_pinned_conditions(
        CONDITIONS_CSV, _trend_frame(True), "primary"
    )
    controls = table[table["condition"].isin(("ca_clpal", "ca_cleft", "ca_limb"))]
    assert controls["true_trend_log_per_year"].to_numpy() == pytest.approx(-0.005)
    assert controls["reason"].str.contains("Prevalence trend pinned").all()
    assert controls["source"].str.contains("TBDR 2010-2022").all()


def test_a_refused_pin_stays_zero_and_records_why():
    """The zero must become a recorded decision, not the silent default it replaces."""
    table = EXTRACT.build_pinned_conditions(
        CONDITIONS_CSV, _trend_frame(False), "primary"
    )
    fitted = table[
        table["condition"].isin(("ca_hypo", "ca_clpal", "ca_cleft", "ca_limb"))
    ]
    assert (fitted["true_trend_log_per_year"] == 0.0).all()
    assert fitted["reason"].str.contains("left at zero").all()
    assert fitted["reason"].str.contains("level shift").all()


def test_conditions_absent_from_the_fit_keep_their_curated_reason():
    table = EXTRACT.build_pinned_conditions(
        CONDITIONS_CSV, _trend_frame(True), "primary"
    )
    unfitted = table[table["condition"] == "ca_cchd"].iloc[0]
    assert unfitted["true_trend_log_per_year"] == 0.0
    assert "pulse-oximetry" in unfitted["reason"]
    assert "not pinned" in unfitted["reason"]


def test_the_shipped_pinned_table_is_loadable_and_agrees_with_the_trend_table():
    """Guard the two tracked artefacts against drifting apart."""
    from dspopulations_us_birth_certificates.selection.anomaly_panel import (
        AnomalyPanelConditions,
    )

    pinned_path = REPO_ROOT / "data" / "us-births-anomaly-panel-conditions-pinned.csv"
    trends_path = REPO_ROOT / "data" / "us-births-anomaly-surveillance-trends.csv"
    if not (pinned_path.exists() and trends_path.exists()):
        pytest.skip("extraction outputs not present")

    conditions = AnomalyPanelConditions.from_csv(pinned_path)
    controls = conditions.controls()["condition"].tolist()
    assert controls == ["ca_hypo", "ca_clpal", "ca_cleft", "ca_limb"]

    trends = pd.read_csv(trends_path)
    primary = trends[trends["window_role"] == "primary"].set_index("condition")
    table = conditions.table.set_index("condition")
    for condition, row in primary.iterrows():
        if condition not in table.index:
            continue
        expected = row["slope_log_per_year"] if row["pinnable"] else 0.0
        assert table.loc[condition, "true_trend_log_per_year"] == pytest.approx(
            expected, abs=1e-6
        ), condition
