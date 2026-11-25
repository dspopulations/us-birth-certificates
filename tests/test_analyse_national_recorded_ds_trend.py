# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Checks for the national recorded Down syndrome trend script.

AI-assisted by Claude Code (Opus 5.5).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "analyse_national_recorded_ds_trend.py"


def _load():
    spec = importlib.util.spec_from_file_location("national_trend", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["national_trend"] = module
    spec.loader.exec_module(module)
    return module


trend = _load()


def _cells(
    annual_factor=0.97,
    seed=3,
    ages=range(15, 48),
    births=20_000,
    years=trend.YEARS,
    revised_share=lambda year: 1.0,
    revised_ratio=1.0,
    unknown_share=0.002,
):
    """Synthetic year by age by certificate counts with a known trend.

    ``revised_share`` gives the share of births on the revised certificate in
    each year. Revised births have ``revised_ratio`` times the 1989-item rate.
    """
    rng = np.random.default_rng(seed)
    centre = (min(years) + max(years)) / 2
    rows = []
    for year in years:
        share = revised_share(year)
        for revised, fraction in ((1, share), (0, 1 - share)):
            n = int(round(births * fraction))
            if n == 0:
                continue
            for age in ages:
                rate = 2e-4 * np.exp(0.08 * (age - 30))
                rate *= annual_factor ** (year - centre)
                rate *= revised_ratio if revised else 1.0
                unknown = rng.poisson(n * unknown_share)
                cases = rng.poisson(n * rate)
                confirmed = rng.binomial(cases, 0.45) if revised else 0
                rows.append(
                    dict(
                        year=year,
                        age=age,
                        revised=revised,
                        births=n,
                        confirmed=confirmed,
                        pending=cases - confirmed if revised else 0,
                        unrevised_yes=0 if revised else cases,
                        no=n - cases - unknown,
                        unknown=unknown,
                    )
                )
    return pd.DataFrame(rows)


def _transition_share(year):
    """Revised-item share: none before 2004, rising to all from 2016."""
    return float(np.clip((year - 2003) / 13, 0, 1))


def _full_cells(ages=range(18, 22), births=20_000):
    return _cells(
        years=trend.ALL_YEARS,
        ages=ages,
        births=births,
        revised_share=_transition_share,
    )


def test_trend_recovers_a_known_annual_change():
    cells = trend.add_derived(_cells(annual_factor=0.97, births=400_000))
    strata = trend.strata_counts(cells)
    result, ratios = trend.age_adjusted_trend(strata.recorded, strata)
    assert result["lower_95"] < -3.0 < result["upper_95"]
    assert result["yearly_vs_linear_df"] == len(trend.YEARS) - 2
    assert set(ratios) == set(trend.YEARS[1:])


def test_standardised_rate_equals_crude_rate_in_the_standard_year():
    cells = trend.add_derived(_cells())
    strata = trend.strata_counts(cells)
    weights = trend.standard_weights(strata)
    rates = trend.standardised_rates(strata, weights)
    national = trend.national_summary(cells).set_index("year")
    first = rates.loc[(rates.year == 2018) & (rates.status == "recorded")].iloc[0]
    assert first.per_10000 == pytest.approx(national.loc[2018, "recorded_per_10000"])
    assert first.lower_95 < first.per_10000 < first.upper_95


def test_unknown_multiplier_zero_reproduces_the_main_trend():
    cells = trend.add_derived(_cells())
    strata = trend.strata_counts(cells)
    main, _ = trend.age_adjusted_trend(strata.recorded, strata)
    sensitivity = trend.unknown_sensitivity(cells, multipliers=(0.0, 10.0))
    assert sensitivity.annual_pct.iloc[0] == pytest.approx(main["annual_pct"])
    assert sensitivity.added_2018.iloc[0] == 0
    assert sensitivity.added_2018.iloc[1] > 0


def test_certificate_term_removes_a_spurious_trend_from_changing_coverage():
    years = np.arange(2004, 2016)
    cells = trend.add_derived(
        _cells(
            annual_factor=1.0,
            births=400_000,
            years=years,
            revised_share=lambda year: (year - 2003) / 13,
            revised_ratio=1.3,
        )
    )
    strata = trend.strata_counts(cells, ("known", "recorded"), keys=("revised",))
    naive_strata = trend.strata_counts(cells, ("known", "recorded"))
    naive, _ = trend.age_adjusted_trend(
        naive_strata.recorded, naive_strata, exposure="known"
    )
    adjusted, _ = trend.age_adjusted_trend(
        strata.recorded, strata, exposure="known", extra=("revised",)
    )
    assert naive["lower_95"] > 0
    assert adjusted["lower_95"] < 0 < adjusted["upper_95"]
    assert adjusted["revised_lower_95"] < 1.3 < adjusted["revised_upper_95"]


def test_certificate_gap_recovers_a_constant_ratio_in_every_year():
    cells = trend.add_derived(
        _cells(
            annual_factor=1.0,
            births=400_000,
            years=np.arange(trend.TRANSITION[0], trend.TRANSITION[1] + 1),
            revised_share=lambda year: 0.5,
            revised_ratio=1.3,
        )
    )
    yearly, test = trend.certificate_gap(cells)
    assert list(yearly.year) == list(
        range(trend.TRANSITION[0], trend.TRANSITION[1] + 1)
    )
    assert test["df"] == len(yearly) - 1
    assert test["constant_rate_ratio"] == pytest.approx(1.3, rel=0.05)
    assert yearly.revised_rate_ratio.median() == pytest.approx(1.3, rel=0.05)
    assert test["p"] > 0.001


def test_long_run_rates_ignore_births_with_unknown_status():
    base = _cells(unknown_share=0.0)
    padded = base.copy()
    padded["unknown"] = padded.births
    padded["births"] = 2 * padded.births
    rates = []
    for frame in (base, padded):
        cells = trend.add_derived(frame)
        weights = trend.standard_weights(trend.strata_counts(cells))
        rates.append(trend.long_standardised_rates(cells, weights).per_10000)
    np.testing.assert_allclose(rates[0], rates[1])


def test_collapsing_certificates_keeps_every_count():
    cells = trend.add_derived(
        _cells(years=np.arange(2010, 2014), revised_share=lambda year: 0.5)
    )
    collapsed = trend.collapse_certificates(cells)
    for column in [*trend.COUNTS, "recorded", "known"]:
        assert collapsed[column].sum() == cells[column].sum()
    assert not collapsed.duplicated(["year", "age"]).any()


def test_validation_accepts_a_complete_long_run_table():
    trend.validate_cells(_full_cells())


def test_validation_rejects_births_that_do_not_match_statuses():
    cells = _full_cells()
    cells.loc[0, "births"] += 1
    with pytest.raises(ValueError, match="sum of all statuses"):
        trend.validate_cells(cells)


def test_validation_rejects_pending_on_the_1989_item():
    cells = _full_cells()
    row = cells.index[cells.revised == 0][0]
    cells.loc[row, ["pending", "no"]] += [1, -1]
    with pytest.raises(ValueError, match="1989 certificate item"):
        trend.validate_cells(cells)


def test_validation_rejects_1989_item_statuses_after_2015():
    cells = _full_cells()
    row = cells.index[cells.year == 2016][0]
    extra = cells.loc[[row]].assign(revised=0, confirmed=0, pending=0)
    extra["no"] = extra.births - extra.unknown
    with pytest.raises(ValueError, match="1989-item statuses from 2016"):
        trend.validate_cells(pd.concat([cells, extra], ignore_index=True))


def test_validation_rejects_the_revised_item_before_2004():
    cells = _full_cells()
    rows = (cells.year == 2003) & (cells.revised == 0)
    cells.loc[rows, "confirmed"] = cells.loc[rows, "unrevised_yes"]
    cells.loc[rows, ["unrevised_yes", "revised"]] = [0, 1]
    with pytest.raises(ValueError, match="Revised Down syndrome item before 2004"):
        trend.validate_cells(cells)


def test_band_test_ignores_bands_absent_from_the_data():
    cells = trend.add_derived(_cells(ages=range(25, 40), births=200_000))
    result = trend.band_slope_test(cells)
    assert result["df"] == 2


def test_main_writes_every_output_from_saved_counts(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    counts = tmp_path / "counts.csv"
    _full_cells(ages=range(15, 48), births=40_000).to_csv(counts, index=False)
    out = tmp_path / "run"
    assert trend.main(["--counts-csv", str(counts), "--output-dir", str(out)]) == 0
    for name in [
        "year_age_certificate_counts.csv",
        "national_summary.csv",
        "age_adjusted_trends.csv",
        "long_run_summary.csv",
        "long_run_standardised_rates_95ci.csv",
        "long_run_period_trends.csv",
        "long_run_period_band_tests.csv",
        "certificate_gap_by_year.csv",
        "certificate_gap_test.csv",
        "manifest.json",
        "national_recorded_trend.png",
        "age_band_recorded_trend.png",
        "long_run_recorded_trend.png",
        "certificate_transition.png",
    ]:
        assert (out / name).exists(), name
    periods = pd.read_csv(out / "long_run_period_trends.csv")
    assert len(periods) == len(trend.ERAS) * (len(trend.BANDS) + 1) + 1
    assert periods.certificate_term.sum() == 1
    keys = ["year", "age", "revised"]
    saved = pd.read_csv(out / "year_age_certificate_counts.csv")
    original = pd.read_csv(counts)[saved.columns].sort_values(keys)
    pd.testing.assert_frame_equal(saved, original.reset_index(drop=True))


def test_age_bands_cover_every_valid_age_once():
    cells = trend.add_derived(_cells(ages=range(10, 55)))
    assert not cells.band.isna().any()
    first = cells.loc[cells.year == 2018].drop_duplicates("age").set_index("age").band
    assert first.loc[24] == "Under 25"
    assert first.loc[25] == "25–29"
    assert first.loc[54] == "40 and over"
