# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""National trends in recorded Down syndrome births, 1989-2025.

AI-assisted by Claude Code (Opus 5.5).
Run from the repository root. Reads the database without changing it, or uses
saved year by single-age by certificate-type aggregate counts.

The recent analysis (2018-2025) counts confirmed or pending records among all
births. The long-run analysis (1989-2025) uses one item that is comparable
across certificate revisions: recorded, not recorded or unknown. Its
denominator is births with a known status, because up to 18% of births lack a
usable status before 1996. All outputs concern certificate recording, not
true prevalence, recording completeness or prenatal reduction.

``revised`` marks births with the revised Down syndrome item. In 2003 the file
gives only the 1989-style item, even for states already on the revised
certificate, so those births have ``revised`` 0.
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import matplotlib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from dse_research_utils.environment.setup import init_script
from dse_research_utils.metadata.provenance import git_snapshot, package_versions
from dse_research_utils.plot.io import save_styled_figure
from dse_research_utils.plot.styles import (
    DPI_FILE,
    FIGSIZE_XL,
    MUTED_TEXT_COLOUR,
    TEXT_COLOUR,
)
from scipy.stats import chi2

from dspopulations_us_birth_certificates import chance, recorded_trends
from dspopulations_us_birth_certificates.file_io import (
    write_atomically,
    write_text_atomically,
)
from dspopulations_us_birth_certificates.ingest.layouts import sha256
from dspopulations_us_birth_certificates.plot_colours import (
    NATURAL_COLOUR,
    RECORDED_COLOUR,
    ordered_palette,
)
from dspopulations_us_birth_certificates.recorded_trends import gamma_interval

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ALL_YEARS = np.arange(1989, 2026)
YEARS = np.arange(2018, 2026)
STANDARD_YEAR = 2018
STRATA = np.arange(19, 46)
COUNTS = ["births", "confirmed", "pending", "unrevised_yes", "no", "unknown"]
STATUSES = {
    "confirmed": ["confirmed"],
    "pending": ["pending"],
    "recorded": ["confirmed", "pending"],
}
BANDS = {
    "Under 25": (0, 24),
    "25–29": (25, 29),
    "30–34": (30, 34),
    "35–39": (35, 39),
    "40 and over": (40, 99),
}
ERAS = {
    "1989–1995": (1989, 1995),
    "1996–2003": (1996, 2003),
    "2004–2015": (2004, 2015),
    "2016–2025": (2016, 2025),
}
"""Periods fixed by certificate design and reporting coverage, not by the trend.

1989-1995: 1989 certificate item; 5-18% of births lack a usable status.
1996-2003: 1989 certificate item; at least 97% of births have a status.
2004-2015: states move to the 2003 revision; both items are present.
2016-2025: revised certificate in every state.
"""
TRANSITION = (2004, 2015)
REVISED_ONLY_FROM = ERAS["2016–2025"][0]
"""Years with both certificate items. The all-ages trend here is also fitted
with a constant certificate-type term, and that constant gap is tested."""
UNKNOWN_MULTIPLIERS = (1.0, 5.0, 10.0)
NBDPN_CSV = Path("data/us-births-nbdpn-trisomy21-2018-2022.csv")
WONDER_CSV = Path("data/us-births-wonder-national-year-2016-2024.csv")
QUERY = """WITH s AS (
    SELECT year, mage_c AS age, mager, down_ind, p_ds_lb_nt,
           COALESCE(ca_down, ca_downs, '') AS rv,
           COALESCE(uca_downs, downs, 0) AS od
    FROM us_births WHERE year BETWEEN 1989 AND 2025
)
SELECT year, age, CAST(rv <> '' AS INTEGER) AS revised,
       COUNT(*) AS births,
       COUNT(*) FILTER (WHERE rv = 'C') AS confirmed,
       COUNT(*) FILTER (WHERE rv = 'P') AS pending,
       COUNT(*) FILTER (WHERE rv = '' AND od = 1) AS unrevised_yes,
       COUNT(*) FILTER (WHERE rv = 'N' OR (rv = '' AND od = 2)) AS no,
       COUNT(*) FILTER (WHERE rv = 'U' OR (rv = '' AND od NOT IN (1, 2))) AS unknown,
       COUNT(*) FILTER (WHERE
           (rv IN ('C', 'P') OR (rv = '' AND od = 1)) <> COALESCE(down_ind = 1, FALSE)
           OR (rv = 'N' OR (rv = '' AND od = 2)) <> COALESCE(down_ind = 0, FALSE)
       ) AS status_mismatches,
       COUNT(*) FILTER (WHERE mager IS NOT NULL AND mager <> age) AS age_mismatches,
       SUM(p_ds_lb_nt) AS morris_expected_db
FROM s
GROUP BY year, age, revised
ORDER BY year, age, revised
"""


def save_csv(frame, out, name):
    write_atomically(out / name, lambda path: frame.to_csv(path, index=False))


def save_figure(fig, out, stem):
    save_styled_figure(
        out, stem, fig=fig, dpi=DPI_FILE, bbox_inches="tight", svg=True, close=True
    )


def pct(value):
    return 100 * (np.exp(value) - 1)


def validate_cells(cells):
    """Require one row per year, age and certificate type, with consistent counts.

    Births must equal the sum of all statuses. Confirmed and pending exist only
    on the revised item; the yes code exists only on the 1989 item. The revised
    item may appear only from 2004, and known 1989-item statuses only before
    2016, so each period holds the certificate items its label states.
    """
    keys = ["year", "age", "revised"]
    if cells.duplicated(keys).any():
        raise ValueError("Duplicate year-age-certificate cell")
    if set(cells.year) != set(ALL_YEARS):
        raise ValueError("Expected every year from 1989 to 2025")
    if cells.age.isna().any() or not cells.age.between(10, 54).all():
        raise ValueError("Maternal age must be recorded and lie between 10 and 54")
    if not cells.revised.isin([0, 1]).all():
        raise ValueError("Certificate item must be 0 (1989 item) or 1 (revised)")
    values = cells[COUNTS].to_numpy()
    if (
        not np.isfinite(values).all()
        or (values < 0).any()
        or (values != np.floor(values)).any()
    ):
        raise ValueError("Counts must be finite non-negative integers")
    if not (cells.births > 0).all() or not np.array_equal(
        cells.births, cells[COUNTS[1:]].sum(axis=1)
    ):
        raise ValueError("Births must be positive and equal the sum of all statuses")
    unrevised = cells.revised == 0
    if (cells.loc[unrevised, ["confirmed", "pending"]].to_numpy() != 0).any():
        raise ValueError("Confirmed or pending counts on the 1989 certificate item")
    if (cells.loc[~unrevised, "unrevised_yes"] != 0).any():
        raise ValueError("1989-item yes counts on revised certificates")
    if (cells.loc[~unrevised, "year"] < TRANSITION[0]).any():
        raise ValueError(f"Revised Down syndrome item before {TRANSITION[0]}")
    late = unrevised & (cells.year >= REVISED_ONLY_FROM)
    if (cells.loc[late, ["unrevised_yes", "no"]].to_numpy() != 0).any():
        raise ValueError(f"1989-item statuses from {REVISED_ONLY_FROM}")


def add_derived(cells):
    """Add recorded and known counts, Morris expected counts and age groups."""
    cells = cells.copy()
    cells["recorded"] = cells.confirmed + cells.pending + cells.unrevised_yes
    cells["known"] = cells.births - cells.unknown
    probability = chance.get_ds_lb_nt_probability_array(cells.age.to_numpy(dtype=float))
    cells["morris_expected"] = cells.births * probability
    cells["morris_expected_known"] = cells.known * probability
    cells["stratum"] = cells.age.clip(STRATA[0], STRATA[-1])
    cells["band"] = pd.cut(
        cells.age,
        [-1, *(high for _, high in BANDS.values())],
        labels=list(BANDS),
    )
    return cells


def collapse_certificates(cells):
    """Sum over certificate type to one row per year and age."""
    summed = cells.groupby(["year", "age"], as_index=False)[COUNTS].sum()
    return add_derived(summed)


def strata_counts(
    cells, columns=("births", "confirmed", "pending", "recorded"), keys=()
):
    """Sum cells to year by 27-stratum counts (under 20, 20 to 44, 45+)."""
    by = ["year", "stratum", *keys]
    return (
        cells.groupby(by, as_index=False, observed=True)[list(columns)]
        .sum()
        .sort_values(by)
        .reset_index(drop=True)
    )


def national_summary(cells):
    rows = []
    for year, part in cells.groupby("year"):
        births = part.births.sum()
        recorded = part.recorded.sum()
        expected = part.morris_expected.sum()
        rows.append(
            dict(
                year=int(year),
                births=int(births),
                confirmed=int(part.confirmed.sum()),
                pending=int(part.pending.sum()),
                recorded=int(recorded),
                unknown=int(part.unknown.sum()),
                confirmed_per_10000=1e4 * part.confirmed.sum() / births,
                pending_per_10000=1e4 * part.pending.sum() / births,
                recorded_per_10000=1e4 * recorded / births,
                unknown_per_10000=1e4 * part.unknown.sum() / births,
                pending_share=part.pending.sum() / recorded,
                mean_maternal_age=float(np.average(part.age, weights=part.births)),
                pct_births_35_plus=100 * part.births[part.age >= 35].sum() / births,
                pct_births_40_plus=100 * part.births[part.age >= 40].sum() / births,
                morris_expected_per_10000=1e4 * expected / births,
                recorded_over_morris_expected=recorded / expected,
            )
        )
    return pd.DataFrame(rows)


def long_summary(cells):
    """Yearly long-run summary on the known-status denominator."""
    rows = []
    for year, part in cells.groupby("year"):
        births = part.births.sum()
        known = part.known.sum()
        recorded = part.recorded.sum()
        expected = part.morris_expected_known.sum()
        rows.append(
            dict(
                year=int(year),
                births=int(births),
                known_births=int(known),
                unknown_share=1 - known / births,
                revised_item_share_of_known=part.known[part.revised == 1].sum() / known,
                recorded=int(recorded),
                recorded_per_10000_known=1e4 * recorded / known,
                mean_maternal_age=float(np.average(part.age, weights=part.births)),
                pct_births_35_plus=100 * part.births[part.age >= 35].sum() / births,
                pct_births_40_plus=100 * part.births[part.age >= 40].sum() / births,
                morris_expected_per_10000_known=1e4 * expected / known,
                recorded_over_morris_expected=recorded / expected,
            )
        )
    return pd.DataFrame(rows)


def standard_weights(strata):
    """Share of 2018 births in each of the 27 maternal-age strata."""
    births = strata.loc[strata.year == STANDARD_YEAR].set_index("stratum").births
    births = births.reindex(STRATA)
    if births.isna().any():
        raise ValueError("Missing 2018 age stratum")
    return births / births.sum()


def direct_rate(part, weights, count, exposure):
    """One direct standardised rate with a pointwise Fay-Feuer interval.

    Returns NaN values when any standard stratum has no exposure.
    """
    part = part.set_index("stratum").reindex(weights.index)
    if part[exposure].isna().any() or (part[exposure] <= 0).any():
        return np.nan, np.nan, np.nan
    return gamma_interval(
        part[count].to_numpy(), part[exposure].to_numpy(), weights.to_numpy()
    )


def standardised_rates(strata, weights):
    """Direct age-standardised rates per status on all births."""
    rows = []
    for year, part in strata.groupby("year"):
        for status in STATUSES:
            rate, lower, upper = direct_rate(part, weights, status, "births")
            rows.append(
                dict(
                    year=int(year),
                    status=status,
                    per_10000=rate,
                    lower_95=lower,
                    upper_95=upper,
                )
            )
    return pd.DataFrame(rows)


def long_standardised_rates(cells, weights):
    """Recorded rate per known-status birth, standardised to 2018 ages.

    "All" covers both certificate types. During the transition the two types
    are also standardised separately.
    """
    rows = []
    combined = strata_counts(cells, ("known", "recorded"))
    by_type = strata_counts(cells, ("known", "recorded"), keys=("revised",))
    for year, part in combined.groupby("year"):
        rate, lower, upper = direct_rate(part, weights, "recorded", "known")
        rows.append(
            dict(
                year=int(year),
                certificate="all",
                per_10000=rate,
                lower_95=lower,
                upper_95=upper,
            )
        )
        if TRANSITION[0] <= year <= TRANSITION[1]:
            for revised, label in ((1, "revised"), (0, "1989 item")):
                type_part = by_type.loc[
                    (by_type.year == year) & (by_type.revised == revised)
                ]
                rate, lower, upper = direct_rate(
                    type_part, weights, "recorded", "known"
                )
                rows.append(
                    dict(
                        year=int(year),
                        certificate=label,
                        per_10000=rate,
                        lower_95=lower,
                        upper_95=upper,
                    )
                )
    return pd.DataFrame(rows)


def _design(strata, year_terms, extra=()):
    """Stratum indicators, extra terms, and a centred year or year indicators."""
    stratum = pd.get_dummies(strata.stratum, prefix="stratum", dtype=float)
    parts = [stratum, strata[list(extra)].astype(float)]
    if year_terms == "linear":
        centre = (strata.year.min() + strata.year.max()) / 2
        parts.append(pd.DataFrame({"year_centred": strata.year - centre}))
    else:
        parts.append(
            pd.get_dummies(strata.year, prefix="year", dtype=float).iloc[:, 1:]
        )
    return pd.concat(parts, axis=1)


def _poisson(counts, strata, year_terms, exposure, extra):
    return sm.GLM(
        np.asarray(counts, dtype=float),
        _design(strata, year_terms, extra),
        family=sm.families.Poisson(),
        offset=np.log(strata[exposure].to_numpy(dtype=float)),
    ).fit()


def age_adjusted_trend(counts, strata, exposure="births", extra=()):
    """Log-linear year trend with separate maternal-age-stratum intercepts.

    Returns the annual percentage change with Wald and dispersion-scaled 95%
    intervals, and a likelihood-ratio check of a straight line against one
    level for each year. ``extra`` names further columns of ``strata`` to
    include as covariates in both models; their rate ratios are returned.
    """
    linear = _poisson(counts, strata, "linear", exposure, extra)
    yearly = _poisson(counts, strata, "yearly", exposure, extra)
    slope = linear.params["year_centred"]
    se = linear.bse["year_centred"]
    dispersion = linear.pearson_chi2 / linear.df_resid
    scaled = se * np.sqrt(max(dispersion, 1.0))
    lr = linear.deviance - yearly.deviance
    df = int(round(linear.df_resid - yearly.df_resid))
    trend = dict(
        first_year=int(strata.year.min()),
        last_year=int(strata.year.max()),
        cases=float(np.sum(counts)),
        annual_pct=pct(slope),
        lower_95=pct(slope - 1.96 * se),
        upper_95=pct(slope + 1.96 * se),
        pearson_dispersion=dispersion,
        lower_95_scaled=pct(slope - 1.96 * scaled),
        upper_95_scaled=pct(slope + 1.96 * scaled),
        change_first_to_last_pct=pct(slope * (strata.year.max() - strata.year.min())),
        yearly_vs_linear_lr=lr,
        yearly_vs_linear_df=df,
        yearly_vs_linear_p=float(chi2.sf(lr, df)),
    )
    for name in extra:
        value, error = linear.params[name], linear.bse[name]
        trend[f"{name}_rate_ratio"] = float(np.exp(value))
        trend[f"{name}_lower_95"] = float(np.exp(value - 1.96 * error))
        trend[f"{name}_upper_95"] = float(np.exp(value + 1.96 * error))
    year_ratios = {
        int(name.split("_")[1]): float(np.exp(value))
        for name, value in yearly.params.items()
        if name.startswith("year_")
    }
    return trend, year_ratios


def band_slope_test(cells, exposure="births"):
    """Approximate LR test that recorded trends differ between age bands."""
    strata = strata_counts(cells, (exposure, "recorded"), keys=("band",))
    base = _design(strata, "linear")
    bands = pd.get_dummies(strata.band.cat.remove_unused_categories(), dtype=float)
    interacted = bands.mul(base.year_centred, axis=0).iloc[:, 1:]
    offset = np.log(strata[exposure].to_numpy(dtype=float))
    common = sm.GLM(
        strata.recorded, base, family=sm.families.Poisson(), offset=offset
    ).fit()
    separate = sm.GLM(
        strata.recorded,
        pd.concat([base, interacted.add_prefix("slope_")], axis=1),
        family=sm.families.Poisson(),
        offset=offset,
    ).fit()
    lr = common.deviance - separate.deviance
    df = interacted.shape[1]
    return dict(lr=lr, df=df, p=float(chi2.sf(lr, df)))


def unknown_sensitivity(cells, multipliers=UNKNOWN_MULTIPLIERS):
    """Add unknown-status births at a multiple of the known-status recorded rate.

    Each year and stratum gains unknown * multiplier * recorded / known cases.
    Multiplier zero reproduces the main recorded-count trend.
    """
    strata = strata_counts(cells, ("births", "recorded", "unknown"))
    known = strata.births - strata.unknown
    rows = []
    for multiplier in multipliers:
        added = multiplier * strata.unknown * strata.recorded / known
        trend, _ = age_adjusted_trend(strata.recorded + added, strata)
        rows.append(
            dict(
                multiplier=multiplier,
                added_2018=float(added[strata.year == YEARS[0]].sum()),
                added_2025=float(added[strata.year == YEARS[-1]].sum()),
                annual_pct=trend["annual_pct"],
                lower_95=trend["lower_95"],
                upper_95=trend["upper_95"],
            )
        )
    return pd.DataFrame(rows)


def era_trends(cells):
    """Age-adjusted trends per period on the known-status denominator.

    Each period is fitted for all ages and for each maternal-age band. The
    transition period is also fitted, for all ages, with a constant revised
    certificate term. ``certificate_gap`` tests that constant-gap assumption.
    """
    rows = []
    for era, (first, last) in ERAS.items():
        part = cells.loc[cells.year.between(first, last)]
        for scope in ["All ages", *BANDS]:
            subset = part if scope == "All ages" else part.loc[part.band == scope]
            strata = strata_counts(subset, ("known", "recorded"))
            trend, _ = age_adjusted_trend(strata.recorded, strata, exposure="known")
            rows.append(dict(era=era, scope=scope, certificate_term=False, **trend))
    part = cells.loc[cells.year.between(*TRANSITION)]
    strata = strata_counts(part, ("known", "recorded"), keys=("revised",))
    trend, _ = age_adjusted_trend(
        strata.recorded, strata, exposure="known", extra=("revised",)
    )
    era = f"{TRANSITION[0]}–{TRANSITION[1]}"
    rows.append(dict(era=era, scope="All ages", certificate_term=True, **trend))
    return pd.DataFrame(rows)


def certificate_gap(cells):
    """Revised versus 1989-item recorded rate ratio in each transition year.

    Both models have year and stratum terms. The first has one certificate
    ratio; the second lets it differ by year. The likelihood-ratio test asks
    whether a constant gap is adequate. The two groups also differ by state.
    """
    part = cells.loc[cells.year.between(*TRANSITION)]
    strata = strata_counts(part, ("known", "recorded"), keys=("revised",))
    offset = np.log(strata.known.to_numpy(dtype=float))
    base = _design(strata, "yearly", ("revised",))
    years = pd.get_dummies(strata.year, prefix="gap", dtype=float)
    varying = pd.concat(
        [base.drop(columns="revised"), years.mul(strata.revised, axis=0)], axis=1
    )
    constant = sm.GLM(
        strata.recorded, base, family=sm.families.Poisson(), offset=offset
    ).fit()
    yearly = sm.GLM(
        strata.recorded, varying, family=sm.families.Poisson(), offset=offset
    ).fit()
    lr = constant.deviance - yearly.deviance
    df = int(round(constant.df_resid - yearly.df_resid))
    rows = [
        dict(
            year=int(name.split("_")[1]),
            revised_rate_ratio=float(np.exp(yearly.params[name])),
            lower_95=float(np.exp(yearly.params[name] - 1.96 * yearly.bse[name])),
            upper_95=float(np.exp(yearly.params[name] + 1.96 * yearly.bse[name])),
        )
        for name in years.columns
    ]
    test = dict(
        constant_rate_ratio=float(np.exp(constant.params["revised"])),
        lr=lr,
        df=df,
        p=float(chi2.sf(lr, df)),
    )
    return pd.DataFrame(rows), test


def era_band_tests(cells):
    """Band-trend test per period on the known-status denominator."""
    rows = []
    for era, (first, last) in ERAS.items():
        part = cells.loc[cells.year.between(first, last)]
        rows.append(dict(era=era, **band_slope_test(part, exposure="known")))
    return pd.DataFrame(rows)


def long_band_rates(cells):
    """Yearly recorded rate and recorded / expected ratio by maternal-age band."""
    frame = cells.groupby(["band", "year"], as_index=False, observed=True)[
        ["known", "recorded", "morris_expected_known"]
    ].sum()
    frame["per_10000_known"] = 1e4 * frame.recorded / frame.known
    frame["recorded_over_morris_expected"] = (
        frame.recorded / frame.morris_expected_known
    )
    return frame


def external_comparison(national):
    """Line up WONDER resident counts and NBDPN annual all-outcome prevalence."""
    wonder = pd.read_csv(WONDER_CSV)[["year", "births", "ds_cp"]].rename(
        columns={
            "births": "wonder_resident_births",
            "ds_cp": "wonder_resident_recorded",
        }
    )
    nbdpn = pd.read_csv(NBDPN_CSV)
    nbdpn = nbdpn.loc[
        nbdpn.estimate_type == "annual", ["period_start", "prevalence_per_10000"]
    ]
    nbdpn = nbdpn.rename(
        columns={"period_start": "year", "prevalence_per_10000": "nbdpn_per_10000"}
    )
    table = national[["year", "births", "recorded", "recorded_per_10000"]].merge(
        wonder, on="year", how="left"
    )
    table = table.merge(nbdpn, on="year", how="left")
    table["certificate_over_nbdpn"] = table.recorded_per_10000 / table.nbdpn_per_10000
    return table


def _errorbar(ax, frame, **kwargs):
    ax.errorbar(
        frame.index,
        frame.per_10000,
        yerr=np.stack(
            [frame.per_10000 - frame.lower_95, frame.upper_95 - frame.per_10000]
        ),
        **kwargs,
    )


def make_recent_figures(out, national, standardised, band_rates, band_trends):
    recorded = standardised.loc[standardised.status == "recorded"].set_index("year")
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_XL, layout="none")
    ax = axes[0]
    ax.plot(
        national.year,
        national.recorded_per_10000,
        linestyle="--",
        marker="o",
        markerfacecolor="white",
        color=RECORDED_COLOUR,
        label="Crude",
    )
    _errorbar(
        ax,
        recorded,
        marker="o",
        color=RECORDED_COLOUR,
        capsize=3,
        label="Standardised to 2018 maternal ages",
    )
    ax.set(
        title="Recorded Down syndrome per 10,000 births",
        xlabel="Year",
        ylim=(0, None),
    )
    ax.legend(loc="lower left", frameon=False)

    ax = axes[1]
    index = national.set_index("year")
    series = [
        ("Births", index.births, MUTED_TEXT_COLOUR, ":"),
        (
            "Expected from maternal age\nwithout terminations",
            index.morris_expected_per_10000,
            NATURAL_COLOUR,
            "-",
        ),
        ("Recorded rate,\nage-standardised", recorded.per_10000, RECORDED_COLOUR, "-"),
    ]
    for label, values, colour, style in series:
        scaled = 100 * values / values.iloc[0]
        ax.plot(scaled.index, scaled, linestyle=style, marker="o", color=colour)
        ax.annotate(
            label,
            (scaled.index[-1], scaled.iloc[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=colour if colour != MUTED_TEXT_COLOUR else TEXT_COLOUR,
        )
    ax.axhline(100, color=MUTED_TEXT_COLOUR, linewidth=0.8)
    ax.set(
        title="Index, 2018 = 100",
        xlabel="Year",
        xlim=(2017.6, 2027.4),
        xticks=YEARS[::2],
    )
    fig.text(
        0.07,
        -0.04,
        "All US occurrence births. Recorded = confirmed or pending certificate status. "
        "Bars are pointwise 95% Fay-Feuer intervals.\n"
        "The expected rate applies the Morris maternal-age curve. It describes age "
        "composition, not an estimate of true prevalence.",
        fontsize=9,
        color=MUTED_TEXT_COLOUR,
    )
    fig.subplots_adjust(left=0.07, right=0.98, top=0.9, bottom=0.12, wspace=0.22)
    save_figure(fig, out, "national_recorded_trend")

    rows = band_trends.reset_index(drop=True)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=FIGSIZE_XL,
        layout="none",
        gridspec_kw=dict(width_ratios=[3, 2]),
    )
    ax = axes[0]
    positions = np.arange(len(rows))
    colours = [
        MUTED_TEXT_COLOUR if s == "All ages" else RECORDED_COLOUR for s in rows.scope
    ]
    for y, (_, row), colour in zip(positions, rows.iterrows(), colours, strict=True):
        ax.errorbar(
            row.annual_pct,
            y,
            xerr=[[row.annual_pct - row.lower_95], [row.upper_95 - row.annual_pct]],
            fmt="o",
            color=colour,
            capsize=3,
        )
        ax.text(
            4.2,
            y,
            f"{row.annual_pct:+.1f}%\n({row.lower_95:+.1f}% to {row.upper_95:+.1f}%)",
            va="center",
            fontsize=8,
            color=TEXT_COLOUR,
        )
    ax.axvline(0, color=MUTED_TEXT_COLOUR, linewidth=0.8)
    ax.set(
        yticks=positions,
        xticks=[-6, -4, -2, 0, 2, 4],
        xlabel="Age-adjusted change per year (%)",
        title="Annual change in recorded rate",
        xlim=(-7, 9),
    )
    ax.set_yticklabels(
        [
            f"{s}\n({int(c):,} records)"
            for s, c in zip(rows.scope, rows.cases, strict=True)
        ],
        fontsize=9,
    )
    ax.invert_yaxis()

    ax = axes[1]
    colours = ordered_palette(len(BANDS))
    for band, colour in zip(BANDS, colours, strict=True):
        part = band_rates.loc[band_rates.band == band].set_index("year")
        ax.plot(
            part.index,
            part.recorded_over_morris_expected,
            marker="o",
            markersize=3,
            color=colour,
            label=band,
        )
    ax.set(
        title="Recorded / age-expected\nwithout terminations",
        xlabel="Year",
        xticks=YEARS[::2],
        ylim=(0, None),
    )
    ax.legend(loc="lower left", frameon=False, fontsize=8)
    fig.text(
        0.02,
        -0.04,
        "Left: pointwise 95% Wald intervals from Poisson models with maternal-age "
        "strata; not adjusted for comparing bands.\nRight: recorded counts divided "
        "by Morris maternal-age expected counts. This ratio combines recording and "
        "prenatal reduction.",
        fontsize=9,
        color=MUTED_TEXT_COLOUR,
    )
    fig.subplots_adjust(left=0.17, right=0.98, top=0.9, bottom=0.12, wspace=0.3)
    save_figure(fig, out, "age_band_recorded_trend")


def _shade_eras(ax, label=True):
    spans = (
        (ERAS["1989–1995"], "Many statuses\nunknown"),
        (TRANSITION, "Certificate\ntransition"),
    )
    for (first, last), text in spans:
        ax.axvspan(
            first - 0.5, last + 0.5, color=MUTED_TEXT_COLOUR, alpha=0.08, linewidth=0
        )
        if label:
            ax.text(
                (first + last) / 2,
                0.98,
                text,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=8,
                color=MUTED_TEXT_COLOUR,
            )


def make_long_figures(out, summary, standardised, band_rates):
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_XL, layout="none")
    ax = axes[0]
    _shade_eras(ax)
    everything = standardised.loc[standardised.certificate == "all"].set_index("year")
    ax.plot(
        summary.year,
        summary.recorded_per_10000_known,
        linestyle="--",
        marker="o",
        markersize=3,
        markerfacecolor="white",
        color=RECORDED_COLOUR,
        label="Crude",
    )
    _errorbar(
        ax,
        everything,
        marker="o",
        markersize=3,
        color=RECORDED_COLOUR,
        capsize=2,
        label="Standardised to 2018 maternal ages",
    )
    ax.set(
        title="Recorded per 10,000 births with known status",
        xlabel="Year",
        ylim=(0, 10),
        xticks=range(1990, 2030, 10),
    )
    ax.legend(loc="lower left", frameon=False, fontsize=9)

    ax = axes[1]
    _shade_eras(ax, label=False)
    colours = ordered_palette(len(BANDS))
    for band, colour in zip(BANDS, colours, strict=True):
        part = band_rates.loc[band_rates.band == band].set_index("year")
        ax.plot(
            part.index,
            part.recorded_over_morris_expected,
            marker="o",
            markersize=2.5,
            color=colour,
            label=band,
        )
    ax.set(
        title="Recorded / age-expected without terminations",
        xlabel="Year",
        ylim=(0, None),
        xticks=range(1990, 2030, 10),
    )
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    fig.text(
        0.05,
        -0.05,
        "All US occurrence births with a known Down syndrome status. Recorded = yes on "
        "the 1989 item, or confirmed or pending on the revised item.\nBars are "
        "pointwise 95% Fay-Feuer intervals. Shading marks years with 5-18% unknown "
        "status, or with both certificate items. The ratio on the right\ncombines recording "
        "and prenatal reduction; it is not an estimate of either.",
        fontsize=9,
        color=MUTED_TEXT_COLOUR,
    )
    fig.subplots_adjust(left=0.07, right=0.98, top=0.9, bottom=0.12, wspace=0.22)
    save_figure(fig, out, "long_run_recorded_trend")

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_XL, layout="none")
    ax = axes[0]
    styles = {
        "all": dict(linestyle="-", marker="o"),
        "revised": dict(linestyle="--", marker="s"),
        "1989 item": dict(linestyle=":", marker="o", markerfacecolor="white"),
    }
    labels = {
        "all": "All certificates",
        "revised": "Revised certificate",
        "1989 item": "1989 certificate item",
    }
    transition = standardised.loc[standardised.year.between(*TRANSITION)].set_index(
        "year"
    )
    for certificate, style in styles.items():
        part = transition.loc[transition.certificate == certificate]
        _errorbar(
            ax,
            part,
            color=RECORDED_COLOUR,
            capsize=2,
            markersize=4,
            label=labels[certificate],
            **style,
        )
    ax.set(
        title="Age-standardised recorded rate per 10,000",
        xlabel="Year",
        ylim=(0, None),
        xticks=range(2004, 2016, 2),
    )
    ax.legend(loc="lower left", frameon=False, fontsize=8)

    ax = axes[1]
    part = summary.loc[summary.year.between(*TRANSITION)]
    ax.bar(part.year, 100 * part.revised_item_share_of_known, color=MUTED_TEXT_COLOUR)
    ax.set(
        title="Births on the revised certificate (%)",
        xlabel="Year",
        ylim=(0, 100),
        xticks=range(2004, 2016, 2),
    )
    fig.text(
        0.05,
        -0.04,
        "States adopted the revised certificate in different years, so the two "
        "certificate groups also differ by state.\nBars on the left are pointwise "
        "95% Fay-Feuer intervals.",
        fontsize=9,
        color=MUTED_TEXT_COLOUR,
    )
    fig.subplots_adjust(left=0.07, right=0.98, top=0.9, bottom=0.12, wspace=0.22)
    save_figure(fig, out, "certificate_transition")


def recent_analysis(cells):
    """2018-2025 analysis on all births, as in the first version of the note."""
    national = national_summary(cells)
    strata = strata_counts(cells)
    weights = standard_weights(strata)
    standardised = standardised_rates(strata, weights)
    trend_rows, year_rows = [], []
    for status in STATUSES:
        trend, ratios = age_adjusted_trend(strata[status], strata)
        trend_rows.append(dict(scope="All ages", status=status, **trend))
        year_rows += [
            dict(status=status, year=year, age_adjusted_rate_ratio_vs_2018=ratio)
            for year, ratio in ratios.items()
        ]
    band_rates = []
    for band in BANDS:
        part = cells.loc[cells.band == band]
        band_strata = strata_counts(part, ("births", "recorded", "morris_expected"))
        trend, _ = age_adjusted_trend(band_strata.recorded, band_strata)
        trend_rows.append(dict(scope=band, status="recorded", **trend))
        yearly = part.groupby("year", as_index=False)[
            ["births", "recorded", "morris_expected"]
        ].sum()
        yearly["band"] = band
        yearly["per_10000"] = 1e4 * yearly.recorded / yearly.births
        yearly["recorded_over_morris_expected"] = (
            yearly.recorded / yearly.morris_expected
        )
        band_rates.append(yearly)
    return dict(
        weights=weights,
        national=national,
        standardised=standardised,
        trends=pd.DataFrame(trend_rows),
        year_ratios=pd.DataFrame(year_rows),
        band_rates=pd.concat(band_rates, ignore_index=True),
        band_test=band_slope_test(cells),
        unknown=unknown_sensitivity(cells),
        external=external_comparison(national),
    )


def long_analysis(cells, weights):
    """1989-2025 analysis on the known-status denominator."""
    return dict(
        summary=long_summary(cells),
        standardised=long_standardised_rates(cells, weights),
        eras=era_trends(cells),
        era_band_tests=era_band_tests(cells),
        certificate_gap=certificate_gap(cells),
        band_rates=long_band_rates(cells),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--duckdb-path", type=Path, default=Path("data/us_births.db"))
    source.add_argument(
        "--counts-csv",
        type=Path,
        help="Saved year by single-age by certificate-type aggregate counts",
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    init_script()
    out = args.output_dir or Path("output/recorded_ds_national_trend") / datetime.now(
        UTC
    ).strftime("%Y%m%dT%H%M%SZ")
    out.mkdir(parents=True, exist_ok=False)
    source_path = args.counts_csv or args.duckdb_path
    before = source_path.stat()
    source_hash = sha256(source_path)
    if args.counts_csv:
        cells = pd.read_csv(args.counts_csv)
    else:
        with duckdb.connect(str(args.duckdb_path), read_only=True) as db:
            cells = db.execute(QUERY).df()
        if cells[["age_mismatches", "status_mismatches"]].to_numpy().sum() != 0:
            raise ValueError("Source age or status checks failed")
        cells = cells.drop(columns=["age_mismatches", "status_mismatches"])
    validate_cells(cells)
    keys = ["year", "age", "revised"]
    cells = cells.sort_values(keys).reset_index(drop=True)
    cells[[*keys, *COUNTS]] = cells[[*keys, *COUNTS]].astype("int64")
    cells = add_derived(cells)
    if "morris_expected_db" in cells:
        if not np.allclose(cells.morris_expected_db, cells.morris_expected, rtol=1e-9):
            raise ValueError("Stored p_ds_lb_nt disagrees with the Morris formula")
        cells = cells.drop(columns="morris_expected_db")

    recent_cells = collapse_certificates(cells.loc[cells.year.isin(YEARS)])
    recent = recent_analysis(recent_cells)
    long = long_analysis(cells, recent["weights"])

    save_csv(cells[keys + COUNTS], out, "year_age_certificate_counts.csv")
    save_csv(recent["weights"].rename("weight").reset_index(), out, "age_weights.csv")
    save_csv(recent["national"], out, "national_summary.csv")
    save_csv(recent["standardised"], out, "standardised_rates_95ci.csv")
    save_csv(recent["trends"], out, "age_adjusted_trends.csv")
    save_csv(recent["year_ratios"], out, "age_adjusted_year_ratios.csv")
    save_csv(recent["band_rates"], out, "age_band_rates.csv")
    save_csv(pd.DataFrame([recent["band_test"]]), out, "age_band_slope_test.csv")
    save_csv(recent["unknown"], out, "unknown_status_sensitivity.csv")
    save_csv(recent["external"], out, "external_comparison.csv")
    save_csv(long["summary"], out, "long_run_summary.csv")
    save_csv(long["standardised"], out, "long_run_standardised_rates_95ci.csv")
    save_csv(long["eras"], out, "long_run_period_trends.csv")
    save_csv(long["era_band_tests"], out, "long_run_period_band_tests.csv")
    gap_years, gap_test = long["certificate_gap"]
    save_csv(gap_years, out, "certificate_gap_by_year.csv")
    save_csv(pd.DataFrame([gap_test]), out, "certificate_gap_test.csv")
    save_csv(long["band_rates"], out, "long_run_age_band_rates.csv")
    make_recent_figures(
        out,
        recent["national"],
        recent["standardised"],
        recent["band_rates"],
        recent["trends"].loc[recent["trends"].status == "recorded"],
    )
    make_long_figures(out, long["summary"], long["standardised"], long["band_rates"])

    after = source_path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError("Input changed during analysis")
    snapshot = git_snapshot(Path.cwd())
    manifest = dict(
        ai_assisted="Claude Code (Opus 5.5)",
        generated_at=datetime.now(UTC).isoformat(),
        source=str(source_path.resolve()),
        source_sha256=source_hash,
        source_type="aggregate_csv" if args.counts_csv else "duckdb",
        source_stat_unchanged=True,
        reference_sha256={str(p): sha256(p) for p in (NBDPN_CSV, WONDER_CSV)},
        git_commit=snapshot.commit,
        git_dirty=snapshot.dirty,
        python=platform.python_version(),
        packages=package_versions(
            ["numpy", "pandas", "scipy", "statsmodels", "matplotlib", "duckdb"]
        ),
        code_sha256={
            str(Path(p).relative_to(Path.cwd())): sha256(Path(p))
            for p in [
                Path(__file__).resolve(),
                Path(recorded_trends.__file__).resolve(),
                Path(chance.__file__).resolve(),
                Path(recorded_trends.__file__).with_name("plot_colours.py").resolve(),
            ]
        },
        recorded_recent="2018-2025: confirmed or pending on the revised item (COALESCE(ca_down, ca_downs) in C or P).",
        recorded_long_run="1989-2025: revised item C or P, or 1989 item code 1 (COALESCE(uca_downs, downs) = 1) where the revised item is absent. Agrees with down_ind in every record. In 2003 revised-certificate births carry only the 1989-style item and count with it.",
        denominator_recent="All births occurring in the 50 states and DC in each year, including unknown status and foreign residents.",
        denominator_long_run="Births with a known status: recorded or not recorded. Excludes revised U, 1989-item codes 8 and 9, and records with no item.",
        standard="All 2018 occurrence births over 27 maternal-age strata (mage_c): under 20, each age 20-44, 45 and over.",
        primary_model="Poisson GLM: log mean = log(exposure) + stratum intercept + slope * centred year. Year-indicator model checks departures from a straight line.",
        periods={k: list(v) for k, v in ERAS.items()},
        certificate_term="2004-2015 all ages only: indicator for the revised certificate item within year and stratum. Its rate ratio compares certificate types, which also differ by state. certificate_gap_test checks whether one constant ratio fits all transition years.",
        morris_expected="Sum over births of the Morris maternal-age live-birth probability without terminations (chance.MORRIS_PARAMS). Describes age composition only.",
        intervals="Standardised rates: pointwise 95% Fay-Feuer gamma. Trends: pointwise 95% Wald, also scaled by Pearson dispersion when above one. None are simultaneous.",
        unknown_sensitivity="2018-2025 only. Unknown-status births assigned the known-status recorded rate of their year and stratum, times 1, 5 or 10.",
        limitations=[
            "Exploratory national summary; periods follow certificate and reporting changes, but the analysis was planned after inspecting these data.",
            "Describes certificate recording. Cannot separate true prevalence, prenatal reduction and recording completeness.",
            "Count variation conditional on fixed births; no allowance for diagnostic error, status changes after registration, state clustering or imputed age.",
            "No state identifier. Unknown-status shares fall sharply before 1996 and certificate type changes by state in 2004-2015; changes in which areas report or which states use each item can shift rates without any change in recording within a state.",
            "Maternal age top-coding differs by period (49 before 1997, 54 in 1997-2003, 50 from 2004); the pooled 45+ stratum absorbs this.",
        ],
        artefact_sha256={
            p.name: sha256(p)
            for p in sorted(out.iterdir())
            if p.suffix in (".csv", ".png", ".svg")
        },
    )
    write_text_atomically(out / "manifest.json", json.dumps(manifest, indent=2) + "\n")
    write_text_atomically(out / "query.sql", QUERY)
    print(long["summary"].round(3).to_string(index=False), flush=True)
    columns = ["era", "scope", "certificate_term", "cases", "annual_pct"]
    columns += ["lower_95", "upper_95", "pearson_dispersion", "yearly_vs_linear_p"]
    columns += ["revised_rate_ratio", "revised_lower_95", "revised_upper_95"]
    print(long["eras"][columns].round(3).to_string(index=False), flush=True)
    print(long["era_band_tests"].round(4).to_string(index=False), flush=True)
    print(gap_years.round(3).to_string(index=False), gap_test, flush=True)
    print(f"Output: {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
