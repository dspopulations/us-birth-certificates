# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Reproduce the age-dependent recorded-DS trend analysis for 2018-2025.

AI-assisted revision by Codex (GPT-6).
Run from the repository root. Reads the database without changing it, or uses
saved aggregate cells. All outputs concern certificate status, not prevalence.
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path

import duckdb
import matplotlib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from dse_research_utils.environment.setup import init_script
from dse_research_utils.metadata.provenance import git_snapshot, package_versions
from dse_research_utils.plot.io import save_styled_figure
from dse_research_utils.plot.styles import DPI_FILE, FIGSIZE_XXL, MUTED_TEXT_COLOUR
from scipy.stats import chi2

from dspopulations_us_birth_certificates import recorded_trends
from dspopulations_us_birth_certificates.file_io import (
    write_atomically,
    write_text_atomically,
)
from dspopulations_us_birth_certificates.ingest.layouts import sha256
from dspopulations_us_birth_certificates.plot_colours import (
    CONFIRMED_COLOUR,
    PENDING_COLOUR,
)
from dspopulations_us_birth_certificates.recorded_trends import (
    bootstrap_test,
    constrained_contrast,
    fit_trends,
    fitted_proportions,
    gamma_interval,
    holm,
    profile_contrast,
    trend_designs,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

YEARS = np.arange(2018, 2026)
TIMES = YEARS - 2021.5
AGES = np.arange(19, 46)
RACES = np.arange(1, 8)
STATUSES = ("confirmed", "pending")
COUNTS = ["births", "confirmed", "pending", "no", "unknown", "missing"]
GROUPS = {
    1: "Non-Hispanic White",
    2: "Non-Hispanic Black",
    3: "Non-Hispanic American Indian or Alaska Native",
    4: "Non-Hispanic Asian",
    5: "Non-Hispanic Native Hawaiian or Other Pacific Islander",
    6: "Non-Hispanic more than one race",
    7: "Hispanic, any race",
    8: "Hispanic origin unknown or not stated",
}
SHORT = {
    1: "NH White",
    2: "NH Black",
    3: "NH American Indian /\nAlaska Native",
    4: "NH Asian",
    5: "NH Native Hawaiian /\nOther Pacific Islander",
    6: "NH more than one race",
    7: "Hispanic, any race",
    8: "Hispanic origin unknown",
}
ORDER = [1, 2, 7, 4, 3, 6, 5]
BANDS = ["Under 20", "20–24", "25–29", "30–34", "35–39", "40–44", "45+"]
COLOURS = {"confirmed": CONFIRMED_COLOUR, "pending": PENDING_COLOUR}
QUERY = """SELECT year, mracehisp AS race_code,
       LEAST(45, GREATEST(19, mager)) AS age_stratum,
       COUNT(*) AS births,
       COUNT(*) FILTER (WHERE COALESCE(ca_down,ca_downs)='C') AS confirmed,
       COUNT(*) FILTER (WHERE COALESCE(ca_down,ca_downs)='P') AS pending,
       COUNT(*) FILTER (WHERE COALESCE(ca_down,ca_downs)='N') AS no,
       COUNT(*) FILTER (WHERE COALESCE(ca_down,ca_downs)='U') AS unknown,
       COUNT(*) FILTER (WHERE COALESCE(ca_down,ca_downs) IS NULL) AS missing,
       COUNT(*) FILTER (WHERE mager IS NULL OR mager NOT BETWEEN 12 AND 50) AS invalid_age,
       COUNT(*) FILTER (WHERE mracehisp IS DISTINCT FROM
           CASE WHEN mhisp_r=9 THEN 8 WHEN mhisp_r BETWEEN 1 AND 5 THEN 7
                WHEN mhisp_r=0 THEN mrace6 END) AS race_mismatches,
       COUNT(*) FILTER (WHERE COALESCE(ca_down,ca_downs) IS DISTINCT FROM ca_down_c) AS status_mismatches
FROM us_births WHERE year BETWEEN 2018 AND 2025
GROUP BY year, mracehisp, LEAST(45, GREATEST(19, mager))
ORDER BY year, race_code, age_stratum
"""


def save_csv(frame, out, name):
    write_atomically(out / name, lambda path: frame.to_csv(path, index=False))


def arrays(frame, status, ages=AGES):
    index = pd.MultiIndex.from_product(
        [RACES, ages, YEARS], names=["race_code", "age_stratum", "year"]
    )
    aligned = frame.set_index(index.names).reindex(index)
    if aligned[["births", status]].isna().any().any():
        raise ValueError("Missing race-age-year aggregate cell")
    shape = (7, len(ages), 8)
    return aligned[status].to_numpy().reshape(shape), aligned.births.to_numpy().reshape(
        shape
    )


def validate_cells(frame):
    keys = ["year", "race_code", "age_stratum"]
    expected = pd.MultiIndex.from_product([YEARS, range(1, 9), AGES], names=keys)
    actual = pd.MultiIndex.from_frame(frame[keys])
    if frame.duplicated(keys).any() or set(actual) != set(expected):
        raise ValueError("Expected one cell for each of 8 years, 8 groups and 27 ages")
    values = frame[COUNTS].to_numpy()
    if (
        not np.isfinite(values).all()
        or (values < 0).any()
        or (values != np.floor(values)).any()
    ):
        raise ValueError("Counts must be finite non-negative integers")
    if not (frame.births > 0).all() or not np.array_equal(
        frame.births, frame[COUNTS[1:]].sum(axis=1)
    ):
        raise ValueError(
            "Birth denominators must be positive and equal all status counts"
        )


def direct_rates(cells, weights):
    rows = []
    for (year, race), part in cells.groupby(["year", "race_code"]):
        part = part.set_index("age_stratum").reindex(AGES)
        for status in STATUSES:
            rate, lower, upper = gamma_interval(
                part[status].to_numpy(), part.births.to_numpy(), weights
            )
            rows.append(
                dict(
                    year=year,
                    race_code=race,
                    group=GROUPS[race],
                    status=status,
                    births=int(part.births.sum()),
                    records=int(part[status].sum()),
                    per_10000=rate,
                    lower_95=lower,
                    upper_95=upper,
                )
            )
    return pd.DataFrame(rows)


def library_check(y, n, z_low, z_high, fit, *, binomial=False, calendar=False):
    """Independently fit both joint GLMs, omitting zero-total boundary strata."""
    active = y.sum(-1) > 0
    intercepts = np.repeat(np.eye(active.sum()), 8, axis=0)
    design_low = (z_low[..., None, :] * TIMES[None, None, :, None])[active].reshape(
        -1, z_low.shape[-1]
    )
    design_high = (z_high[..., None, :] * TIMES[None, None, :, None])[active].reshape(
        -1, z_high.shape[-1]
    )
    if calendar:
        # Six calendar contrasts orthogonal to both a constant and linear year.
        from scipy.linalg import null_space

        curve = null_space(np.stack([np.ones(8), TIMES]))
        curve = np.tile(curve, (active.sum(), 1))
        intercepts = np.column_stack([intercepts, curve])
    results = []
    for trend in (design_low, design_high):
        x = np.column_stack([intercepts, trend])
        if binomial:
            model = sm.GLM(
                np.column_stack([y[active].ravel(), (n - y)[active].ravel()]),
                x,
                family=sm.families.Binomial(),
            )
        else:
            model = sm.GLM(
                y[active].ravel(),
                x,
                family=sm.families.Poisson(),
                offset=np.log(n[active].ravel()),
            )
        result = model.fit(maxiter=100, tol=1e-10, wls_method="qr")
        if not result.converged:
            raise RuntimeError("Independent GLM did not converge")
        results.append(result)
    statistic = 2 * (results[1].llf - results[0].llf)
    row = dict(lrt=float(statistic), df=6, p_value=float(chi2.sf(statistic, 6)))
    if not binomial and not calendar:
        expected = y.sum(-1)[..., None] * fit["p"][0]
        row["max_slope_error"] = float(
            np.max(np.abs(results[1].params[-13:] - fit["beta"][0]))
        )
        row["max_fitted_count_error"] = float(
            np.max(np.abs(results[1].fittedvalues - expected[active].ravel()))
        )
        cov = np.linalg.inv(fit["information"][0])
        row["max_covariance_error"] = float(
            np.max(np.abs(results[1].cov_params()[-13:, -13:] - cov))
        )
        if max(row[k] for k in row if k.startswith("max_")) > 1e-6:
            raise RuntimeError("Conditional fit disagrees with independent GLM")
    return row


def save_figure(fig, out, stem):
    save_styled_figure(
        out, stem, fig=fig, dpi=DPI_FILE, bbox_inches="tight", svg=True, close=True
    )


def make_figures(out, direct, fitted, slopes, pairs):
    for status in STATUSES:
        colour = COLOURS[status]
        fig, axes = plt.subplots(2, 4, figsize=FIGSIZE_XXL, layout="none")
        for ax, race in zip(axes.flat, [*ORDER, 8], strict=True):
            points = direct.loc[(direct.status == status) & (direct.race_code == race)]
            model = fitted.loc[(fitted.status == status) & (fitted.race_code == race)]
            ax.errorbar(
                points.year,
                points.per_10000,
                yerr=np.stack(
                    [
                        points.per_10000 - points.lower_95,
                        points.upper_95 - points.per_10000,
                    ]
                ),
                fmt="o",
                color=colour,
                markersize=3,
                elinewidth=0.8,
                capsize=2,
                label="Observed, age adjusted",
            )
            if len(model):
                ax.plot(
                    model.year,
                    model.per_10000,
                    color=colour,
                    linewidth=1.6,
                    label="Age-dependent model",
                )
            ax.set(
                title=SHORT[race],
                xlim=(2017.5, 2025.5),
                ylim=(0, None),
                xticks=[2018, 2021, 2025],
            )
            ax.tick_params(labelsize=8)
            if race == 8:
                ax.text(
                    0.04,
                    0.94,
                    "Descriptive only",
                    transform=ax.transAxes,
                    va="top",
                    fontsize=8,
                    color=MUTED_TEXT_COLOUR,
                )
        fig.suptitle(
            f"{status.title()} records per 10,000 births, 2018–2025", fontsize=15
        )
        axes[0, 0].legend(fontsize=8, frameon=False)
        fig.text(
            0.07,
            0.025,
            "Common 2018 US age distribution. Points and bars show direct estimates and pointwise 95% gamma intervals.\n"
            "Lines show fitted proportions; bars are not uncertainty bands for the lines. Panel scales differ. NH = non-Hispanic.",
            fontsize=9,
            color=MUTED_TEXT_COLOUR,
        )
        fig.subplots_adjust(
            left=0.07, right=0.98, top=0.86, bottom=0.15, hspace=0.5, wspace=0.38
        )
        save_figure(fig, out, f"{status}_standardised_trends")

        fig, axes = plt.subplots(
            2, 4, figsize=FIGSIZE_XXL, layout="none", sharex=True, sharey=True
        )
        selected = slopes.loc[slopes.status == status]
        for ax, race in zip(axes.flat[:7], ORDER, strict=True):
            part = selected.loc[selected.race_code == race].sort_values("age_band")
            ax.errorbar(
                part.annual_change_pct,
                np.arange(7),
                xerr=np.stack(
                    [
                        part.annual_change_pct - part.lower_95,
                        part.upper_95 - part.annual_change_pct,
                    ]
                ),
                fmt="o",
                color=colour,
                markersize=4,
                capsize=2,
            )
            ax.axvline(0, color=MUTED_TEXT_COLOUR, linewidth=0.8)
            ax.set(title=SHORT[race], yticks=range(7), yticklabels=BANDS)
            ax.tick_params(labelsize=8, labelleft=True)
        axes[0, 0].invert_yaxis()
        axes[1, 3].axis("off")
        axes[1, 3].text(
            0,
            0.9,
            "Change within each age band\n\nPoints use all eight years.\nBars are pointwise 95%\nprofile-likelihood intervals.\n\nRace differences in log trends\nare shared across age bands.\nNH = non-Hispanic.",
            va="top",
            fontsize=10,
            color=MUTED_TEXT_COLOUR,
        )
        fig.suptitle(
            f"{status.title()} records: annual relative change by maternal age",
            fontsize=15,
        )
        fig.supxlabel("Relative change per year (%)", y=0.04)
        fig.subplots_adjust(
            left=0.07, right=0.98, top=0.86, bottom=0.13, hspace=0.48, wspace=0.47
        )
        save_figure(fig, out, f"{status}_age_band_changes")

    fig, axes = plt.subplots(
        1, 2, figsize=FIGSIZE_XXL, layout="none", sharex=True, sharey=True
    )
    order = [r for r in ORDER if r != 1]
    for ax, status in zip(axes, STATUSES, strict=True):
        part = (
            pairs.loc[(pairs.status == status) & (pairs.group_1_code == 1)]
            .set_index("group_2_code")
            .loc[order]
        )
        ax.errorbar(
            part.annual_factor_ratio,
            np.arange(6),
            xerr=np.stack(
                [
                    part.annual_factor_ratio - part.lower_95,
                    part.upper_95 - part.annual_factor_ratio,
                ]
            ),
            fmt="o",
            color=COLOURS[status],
            capsize=3,
        )
        ax.axvline(1, color=MUTED_TEXT_COLOUR, linewidth=0.8)
        ax.set(
            title=status.title(),
            yticks=range(6),
            yticklabels=[SHORT[r] for r in order],
            xlabel="Annual multiplier / White annual multiplier",
        )
    axes[0].invert_yaxis()
    fig.suptitle("Relative trend differences at the same maternal ages", fontsize=15)
    fig.text(
        0.24,
        0.025,
        "Reference = non-Hispanic White. Values above 1 mean a more positive or less negative trend.\n"
        "Bars are pointwise 95% profile-likelihood intervals. They are not adjusted for multiple comparisons.\n"
        "The accompanying table adjusts tests across all 42 pairs. NH = non-Hispanic.",
        fontsize=9,
        color=MUTED_TEXT_COLOUR,
    )
    fig.subplots_adjust(left=0.24, right=0.98, top=0.88, bottom=0.2, wspace=0.12)
    save_figure(fig, out, "race_trend_contrasts")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--duckdb-path", type=Path, default=Path("data/us_births.db"))
    source.add_argument(
        "--counts-csv",
        type=Path,
        help="Saved 27-stratum aggregate counts, including origin unknown",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bootstrap-repetitions", type=int, default=19999)
    args = parser.parse_args(argv)
    if args.bootstrap_repetitions < 1:
        parser.error("--bootstrap-repetitions must be positive")
    init_script()
    out = args.output_dir or Path("output/recorded_ds_age_dependent") / datetime.now(
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
        checks = ["invalid_age", "race_mismatches", "status_mismatches"]
        if cells[checks].to_numpy().sum() != 0:
            raise ValueError("Source-code checks failed")
        cells = cells.drop(columns=checks)
    validate_cells(cells)
    cells = cells.sort_values(["year", "race_code", "age_stratum"]).reset_index(
        drop=True
    )
    cells[COUNTS] = cells[COUNTS].astype("int64")
    births2018 = (
        cells.loc[cells.year == 2018].groupby("age_stratum").births.sum().reindex(AGES)
    )
    weights = (births2018 / births2018.sum()).to_numpy()
    direct = direct_rates(cells, weights)
    national = cells.groupby("year", as_index=False)[COUNTS].sum()
    for status in STATUSES:
        national[f"{status}_per_10000"] = 10000 * national[status] / national.births
    low, high, race_design = trend_designs(RACES, AGES)
    fits, slopes, pairs, tests, sensitivity, validation, diagnostics = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for status_index, status in enumerate(STATUSES):
        print(f"Fitting {status}", flush=True)
        y, n = arrays(cells, status)
        fitted = fit_trends(y, n, TIMES, high)
        predicted = fitted_proportions(y, n, TIMES, high, fitted, weights)
        expected = y.sum(-1)[..., None] * fitted["p"][0]
        active = expected > 0
        residual_df = int(active.sum() - (y.sum(-1) > 0).sum() - high.shape[-1])
        diagnostics.append(
            dict(
                status=status,
                records=int(y.sum()),
                score_max=fitted["score_max"],
                pearson_dispersion=float(
                    np.sum((y[active] - expected[active]) ** 2 / expected[active])
                    / residual_df
                ),
                max_fitted_probability=float((expected / n).max()),
                zero_case_strata=int((y.sum(-1) == 0).sum()),
            )
        )
        for r, race in enumerate(RACES):
            for j, year in enumerate(YEARS):
                fits.append(
                    dict(
                        status=status,
                        race_code=race,
                        group=GROUPS[race],
                        year=year,
                        per_10000=predicted[r, j],
                    )
                )
            for band in range(7):
                contrast = np.zeros(13)
                contrast[band] = 1
                if r > 0:
                    contrast[6 + r] = 1
                lower, upper = profile_contrast(y, n, TIMES, high, fitted, contrast)
                slopes.append(
                    dict(
                        status=status,
                        race_code=race,
                        group=GROUPS[race],
                        age_band=band,
                        age_label=BANDS[band],
                        annual_change_pct=100 * np.expm1(contrast @ fitted["beta"][0]),
                        lower_95=100 * (lower - 1),
                        upper_95=100 * (upper - 1),
                    )
                )
        for a, b in combinations(range(7), 2):
            contrast = np.zeros(13)
            if a > 0:
                contrast[6 + a] -= 1
            contrast[6 + b] += 1
            result, _ = constrained_contrast(y, n, TIMES, high, fitted, contrast)
            lower, upper = profile_contrast(y, n, TIMES, high, fitted, contrast)
            pairs.append(
                dict(
                    status=status,
                    group_1_code=a + 1,
                    group_2_code=b + 1,
                    group_1=GROUPS[a + 1],
                    group_2=GROUPS[b + 1],
                    **result,
                    lower_95=lower,
                    upper_95=upper,
                )
            )
        for label, null, alternative, seed in [
            (
                "Race differences allowing age-specific trends",
                low,
                high,
                81011 + status_index,
            ),
            (
                "Age-specific versus constant-age trends",
                race_design,
                high,
                91011 + status_index,
            ),
        ]:
            print(f"Simulating {status}: {label}", flush=True)
            tests.append(
                dict(
                    status=status,
                    test=label,
                    **bootstrap_test(
                        y, n, TIMES, null, alternative, args.bootstrap_repetitions, seed
                    ),
                )
            )
        validation.append(dict(status=status, **library_check(y, n, low, high, fitted)))
        for label, binomial, calendar in [
            ("Binomial logit approximation check", True, False),
            ("Common unrestricted calendar effects", False, True),
        ]:
            sensitivity.append(
                dict(
                    status=status,
                    check=label,
                    **library_check(
                        y, n, low, high, fitted, binomial=binomial, calendar=calendar
                    ),
                )
            )
        for year in YEARS:
            keep = YEARS != year
            a = fit_trends(y[..., keep], n[..., keep], TIMES[keep], low)
            b = fit_trends(y[..., keep], n[..., keep], TIMES[keep], high)
            statistic = float(2 * (b["ll"] - a["ll"])[0])
            sensitivity.append(
                dict(
                    status=status,
                    check=f"Omit {year}",
                    lrt=statistic,
                    df=6,
                    p_value=float(chi2.sf(statistic, 6)),
                    black_white_annual_factor_ratio=float(np.exp(b["beta"][0, 7])),
                )
            )
        restricted = cells.loc[cells.age_stratum.between(20, 44)]
        grouped = cells.copy()
        grouped["age_stratum"] = np.where(
            grouped.age_stratum < 20, 19, (grouped.age_stratum // 5) * 5
        )
        grouped = grouped.groupby(["year", "race_code", "age_stratum"], as_index=False)[
            COUNTS
        ].sum()
        for label, data in [
            ("Individual ages 20-44", restricted),
            ("Seven age bands for intercepts", grouped),
        ]:
            ages = sorted(data.age_stratum.unique())
            sy, sn = arrays(data, status, ages=ages)
            sl, sh, _ = trend_designs(RACES, ages)
            a, b = fit_trends(sy, sn, TIMES, sl), fit_trends(sy, sn, TIMES, sh)
            statistic = float(2 * (b["ll"] - a["ll"])[0])
            sensitivity.append(
                dict(
                    status=status,
                    check=label,
                    lrt=statistic,
                    df=6,
                    p_value=float(chi2.sf(statistic, 6)),
                )
            )
    fits, slopes, pairs, tests, sensitivity = map(
        pd.DataFrame, (fits, slopes, pairs, tests, sensitivity)
    )
    for _, part in tests.groupby("test"):
        tests.loc[part.index, "holm_p_two_statuses"] = holm(part.bootstrap_p)
    pairs["holm_p_all_42_pairs"] = holm(pairs.p_value)
    for _, part in sensitivity.groupby("check"):
        sensitivity.loc[part.index, "holm_p_two_statuses"] = holm(part.p_value)
    endpoint = fits.pivot(
        index=["status", "race_code", "group"], columns="year", values="per_10000"
    )[[2018, 2025]].reset_index()
    endpoint.columns = [
        "status",
        "race_code",
        "group",
        "fitted_2018_per_10000",
        "fitted_2025_per_10000",
    ]
    endpoint["fitted_change_pct"] = 100 * (
        endpoint.fitted_2025_per_10000 / endpoint.fitted_2018_per_10000 - 1
    )
    for frame, name in [
        (cells, "aggregate_counts.csv"),
        (national, "national_counts_and_crude_proportions.csv"),
        (pd.DataFrame(dict(age_stratum=AGES, weight=weights)), "age_weights.csv"),
        (direct, "direct_proportions_95ci.csv"),
        (fits, "fitted_standardised_proportions.csv"),
        (endpoint, "fitted_endpoint_changes.csv"),
        (slopes, "age_band_trends_95ci.csv"),
        (pairs, "pairwise_trend_tests.csv"),
        (tests, "global_tests.csv"),
        (sensitivity, "sensitivity_tests.csv"),
        (pd.DataFrame(validation), "independent_glm_validation.csv"),
        (pd.DataFrame(diagnostics), "model_diagnostics.csv"),
    ]:
        save_csv(frame, out, name)
    make_figures(out, direct, fits, slopes, pairs)
    after = source_path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError("Input changed during analysis")
    snapshot = git_snapshot(Path.cwd())
    manifest = dict(
        ai_assisted="Codex (GPT-6)",
        generated_at=datetime.now(UTC).isoformat(),
        source=str(source_path.resolve()),
        source_sha256=source_hash,
        source_type="aggregate_csv" if args.counts_csv else "duckdb",
        source_stat_unchanged=True,
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
                Path(recorded_trends.__file__).with_name("plot_colours.py").resolve(),
            ]
        },
        primary_model="Separate C and P Poisson models: mean = births * exp(alpha[race,27-age-stratum] + (year-2021.5)*(age_band_slope[7 bands]+race_slope[race])). White race slope fixed at zero. The race deviation is constant across age bands; band slopes differ.",
        primary_test="Six race deviations equal zero after allowing age-specific trends. Two status-specific tests, Holm adjusted. Conditional parametric bootstrap, fixed race-age case totals and birth denominators, plus-one p-values.",
        bootstrap_repetitions=args.bootstrap_repetitions,
        group_coding=GROUPS,
        standard="All US occurrence births in 2018, including origin unknown; 27 age strata (under 20, each age 20-44, 45+).",
        denominator="All births occurring in the 50 states and DC, including unknown status and foreign residents. Origin unknown is descriptive only and excluded from trend fits.",
        intervals="Direct rates: pointwise 95% Fay-Feuer gamma. Slopes and pairwise annual factor ratios: pointwise 95% profile likelihood, chi-square(1) cutoff. Neither interval family is simultaneous. No confidence bands for fitted standardised curves are claimed.",
        multiplicity="Holm across two primary tests. All 42 pairs corrected together. Age-assumption tests form a separate two-status diagnostic family; each sensitivity also adjusts across two statuses. These adjustments do not cover data-driven period or model selection.",
        fitted_curves="Recover race-age intercepts from case totals and fitted slopes, then standardise to common 2018 age weights. Zero-total strata have fitted mean zero. Curves are point estimates, not true-prevalence estimates. No single constant annual multiplier for a group is assumed.",
        limitations=[
            "Exploratory selection of 2018 and the age-dependent model after seeing data.",
            "Relative-scale trends, constant within age bands and constant race differences across those bands.",
            "Count variation conditional on fixed births and weights; does not include diagnostic error, under-recording, imputation uncertainty, geographic recording differences or dependence between births.",
            "Sparse groups and pooled age tails limit precision and adjustment. Confirmed and pending are separate marginal models of mutually exclusive recorded statuses; no within-birth status transition is observed.",
        ],
        artefact_sha256={
            p.name: sha256(p)
            for p in sorted(out.iterdir())
            if p.suffix in (".csv", ".png", ".svg")
        },
    )
    write_text_atomically(out / "manifest.json", json.dumps(manifest, indent=2) + "\n")
    write_text_atomically(out / "query.sql", QUERY)
    print(tests.to_string(index=False), flush=True)
    print(endpoint.to_string(index=False), flush=True)
    print(f"Output: {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
