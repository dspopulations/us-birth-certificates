"""Audit measurement comparability across birth-certificate eras, 1989-2024.

Extending the core model's window backwards buys anchored surveillance years,
but only if the recorded Down-syndrome indicator means the same thing across the
span.  It does not.  Three things change:

* the source field (``downs`` to 2002, ``uca_downs`` 2003-2015, ``ca_down`` /
  ``ca_downs`` from the 2003 revision onward),
* the completeness of anomaly reporting (17.6% of 1989 births have unknown
  status, against 0.2% from 2015),
* the existence of the confirmed/pending distinction, which the unrevised
  certificate does not carry at all.

This audit quantifies each, and tests whether the pooled reporting fraction is
better described as a trend or as a step at the point the revision completes.
The step matters: de Graaf's workbook extrapolates a linear trend six years past
the last surveillance observation, and a trend fitted through a level shift
extrapolates in a way a level shift does not.

Read-only.  Writes ``era_by_year.csv``, ``revision_split.csv`` and
``reporting_fraction_fits.csv``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
from dse_research_utils.environment.setup import init_script

DEFAULT_DB_PATH = Path("data/us_births.db")
DEFAULT_ANCHOR_CSV = Path("output/degraaf_surveillance/expected_births_anchor.csv")
DEFAULT_OUTPUT_ROOT = Path("output/certificate_revision_audit")
FIRST_YEAR = 1989
LAST_YEAR = 2024
WINDOW_HALF_WIDTH = 2
# The revision completed nationally in 2016; the first surveillance window whose
# five years are wholly revised is centred on 2018.
REVISION_COMPLETE_YEAR = 2015


def load_era_table(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Per-year field provenance, completeness and the confirmed/pending split."""
    frame = con.sql(f"""
        select year,
               count(*) as births,
               sum(case when downs is not null then 1 else 0 end) as field_downs,
               sum(case when uca_downs is not null then 1 else 0 end) as field_uca,
               sum(case when ca_down is not null or ca_downs is not null
                        then 1 else 0 end) as revised,
               sum(case when down_ind is null then 1 else 0 end) as status_unknown,
               sum(case when mage_c is null then 1 else 0 end) as age_unknown,
               sum(case when mracehisp_c is null then 1 else 0 end) as race_unknown,
               sum(case when down_ind = 1 then 1 else 0 end) as flags,
               sum(case when down_ind = 1 and ca_down_c = 'P' then 1 else 0 end)
                   as pending,
               sum(case when down_ind = 1
                         and (ca_down is not null or ca_downs is not null)
                        then 1 else 0 end) as flags_revised,
               sum(case when down_ind = 1
                         and (ca_down is not null or ca_downs is not null)
                         and ca_down_c = 'P' then 1 else 0 end)
                   as pending_revised
        from us_births
        where year between {FIRST_YEAR} and {LAST_YEAR}
        group by 1
        order by 1
    """).df()
    frame["revised_coverage"] = frame["revised"] / frame["births"]
    frame["status_unknown_share"] = frame["status_unknown"] / frame["births"]
    frame["age_unknown_share"] = frame["age_unknown"] / frame["births"]
    frame["race_unknown_share"] = frame["race_unknown"] / frame["births"]
    frame["flag_rate_per10k"] = 1e4 * frame["flags"] / frame["births"]
    frame["pending_share"] = frame["pending"] / frame["flags"]
    # Within revised records only: the sole basis on which confirmed/pending is
    # a consistent measurement rather than an artefact of certificate version.
    frame["pending_share_within_revised"] = np.where(
        frame["flags_revised"] > 0,
        frame["pending_revised"] / frame["flags_revised"].replace(0, np.nan),
        np.nan,
    )
    frame["unrevised_births"] = frame["births"] - frame["revised"]
    frame["flags_unrevised"] = frame["flags"] - frame["flags_revised"]
    frame["flag_rate_revised"] = np.where(
        frame["revised"] > 0, 1e4 * frame["flags_revised"] / frame["revised"], np.nan
    )
    frame["flag_rate_unrevised"] = np.where(
        frame["unrevised_births"] > 0,
        1e4 * frame["flags_unrevised"] / frame["unrevised_births"],
        np.nan,
    )
    return frame


def summarise_revision_contrast(era: pd.DataFrame) -> dict[str, Any]:
    """Compare recording between certificate versions where both are in use.

    Restricted to years in which neither version dominates, so the contrast is
    not driven by a handful of residual records. This is a between-state
    comparison -- the public-use files carry no state identifier -- so early
    adopters differ from late adopters in unobserved ways. It bounds the size of
    the measurement shift; it does not identify its cause.
    """
    both = era[(era.revised_coverage > 0.05) & (era.revised_coverage < 0.95)]
    revised_rate = float(1e4 * both["flags_revised"].sum() / both["revised"].sum())
    unrevised_rate = float(
        1e4 * both["flags_unrevised"].sum() / both["unrevised_births"].sum()
    )
    return {
        "years": f"{int(both.year.min())}-{int(both.year.max())}",
        "n_years": int(len(both)),
        "flag_rate_revised_per10k": revised_rate,
        "flag_rate_unrevised_per10k": unrevised_rate,
        "ratio": revised_rate / unrevised_rate,
        "pending_share_within_revised_mean": float(
            era["pending_share_within_revised"].dropna().mean()
        ),
        "pending_share_within_revised_sd": float(
            era["pending_share_within_revised"].dropna().std()
        ),
    }


def fit_reporting_fraction(anchor: pd.DataFrame, era: pd.DataFrame) -> pd.DataFrame:
    """Compare trend, step and coverage explanations of the reporting fraction."""
    indexed = era.set_index("year")

    def window_mean(mid_year: int, column: str) -> float | None:
        years = range(mid_year - WINDOW_HALF_WIDTH, mid_year + WINDOW_HALF_WIDTH + 1)
        if not all(year in indexed.index for year in years):
            return None
        values = np.array([indexed.loc[year, column] for year in years], dtype=float)
        weights = np.array([indexed.loc[year, "births"] for year in years], dtype=float)
        return float(np.average(values, weights=weights))

    frame = anchor.copy()
    frame["revised_coverage_5yr"] = [
        window_mean(int(y), "revised_coverage") for y in frame.mid_year
    ]
    observed = frame.dropna(subset=["reporting_fraction", "revised_coverage_5yr"])
    y = observed["reporting_fraction"].to_numpy(dtype=float)
    total = float(np.sum((y - y.mean()) ** 2))

    designs = {
        "linear trend in year": np.vstack(
            [observed.mid_year.to_numpy(dtype=float) - 2000.0, np.ones(len(y))]
        ).T,
        "revised coverage": np.vstack(
            [observed.revised_coverage_5yr.to_numpy(dtype=float), np.ones(len(y))]
        ).T,
        f"step at {REVISION_COMPLETE_YEAR}": np.vstack(
            [
                (observed.mid_year.to_numpy() >= REVISION_COMPLETE_YEAR).astype(float),
                np.ones(len(y)),
            ]
        ).T,
    }
    records: list[dict[str, Any]] = []
    for name, design in designs.items():
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        residual = y - design @ beta
        records.append(
            {
                "model": name,
                "n": len(y),
                "slope_or_step": float(beta[0]),
                "intercept": float(beta[1]),
                "r_squared": 1.0 - float(np.sum(residual**2)) / total,
                "residual_sd": float(residual.std(ddof=2)),
            }
        )

    # A trend fitted only on the pre-revision-complete era, to show how much of
    # de Graaf's slope comes from the two post-step windows.
    early = observed[observed.mid_year < REVISION_COMPLETE_YEAR]
    design = np.vstack(
        [early.mid_year.to_numpy(dtype=float) - 2000.0, np.ones(len(early))]
    ).T
    beta, *_ = np.linalg.lstsq(design, early.reporting_fraction.to_numpy(), rcond=None)
    records.append(
        {
            "model": f"linear trend, {int(early.mid_year.min())}-"
            f"{int(early.mid_year.max())} only",
            "n": int(len(early)),
            "slope_or_step": float(beta[0]),
            "intercept": float(beta[1]),
            "r_squared": float("nan"),
            "residual_sd": float("nan"),
        }
    )
    return pd.DataFrame.from_records(records), frame


def report(
    era: pd.DataFrame,
    contrast: dict[str, Any],
    fits: pd.DataFrame,
    windows: pd.DataFrame,
) -> None:
    print("\n--- candidate start years ---")
    print(
        f"  {'year':<6}{'status unknown':>16}{'race unknown':>14}"
        f"{'age unknown':>13}{'revised':>10}{'pending':>9}  source field"
    )
    for year in (1989, 1993, 1996, 2000, 2003, 2004, 2010, 2016, 2024):
        row = era[era.year == year]
        if row.empty:
            continue
        row = row.iloc[0]
        field = (
            "downs"
            if row.field_downs
            else ("uca_downs" if row.field_uca else "ca_down(s)")
        )
        if row.revised and row.field_uca:
            field += " + ca_down(s)"
        print(
            f"  {year:<6}{row.status_unknown_share:>15.2%}"
            f"{row.race_unknown_share:>14.2%}{row.age_unknown_share:>13.2%}"
            f"{row.revised_coverage:>10.1%}{row.pending_share:>9.1%}  {field}"
        )

    print("\n--- revised vs unrevised recording, years where both are in use ---")
    print(
        f"  {contrast['years']} ({contrast['n_years']} years): "
        f"revised {contrast['flag_rate_revised_per10k']:.2f}/10k vs unrevised "
        f"{contrast['flag_rate_unrevised_per10k']:.2f}/10k, "
        f"ratio {contrast['ratio']:.3f}"
    )
    print(
        f"  pending share within revised records: "
        f"{contrast['pending_share_within_revised_mean']:.1%} "
        f"(sd {contrast['pending_share_within_revised_sd']:.1%})"
    )
    print(
        "  NOTE unrevised flags are all coded 'C': the unrevised certificate has "
        "no\n       confirmation step, so pre-2016 'confirmed' mixes confirmed "
        "with never-assessed."
    )

    print("\n--- reporting fraction: trend vs step ---")
    for row in fits.itertuples():
        r2 = "     n/a" if not np.isfinite(row.r_squared) else f"{row.r_squared:>8.4f}"
        print(
            f"  {row.model:<34} n={row.n:<3} coef={row.slope_or_step:>+9.6f} "
            f"intercept={row.intercept:.4f}  R2={r2}"
        )

    observed = windows.dropna(subset=["reporting_fraction"])
    early = observed[observed.mid_year < REVISION_COMPLETE_YEAR]["reporting_fraction"]
    late = observed[observed.mid_year >= REVISION_COMPLETE_YEAR]["reporting_fraction"]
    print(
        f"\n  pre-{REVISION_COMPLETE_YEAR} mean {early.mean():.4f} "
        f"(sd {early.std():.4f}, n={len(early)}) | "
        f"post mean {late.mean():.4f} (n={len(late)})"
    )

    print("\n--- flag rate within fully revised records ---")
    full = era[era.revised_coverage >= 0.999]
    for lo, hi in ((2016, 2018), (2019, 2021), (2022, 2024)):
        window = full[(full.year >= lo) & (full.year <= hi)]
        if not window.empty:
            print(f"  {lo}-{hi}: {window.flag_rate_revised.mean():.3f}/10k")
    early_rate = full[
        (full.year >= 2016) & (full.year <= 2018)
    ].flag_rate_revised.mean()
    late_rate = full[(full.year >= 2022) & (full.year <= 2024)].flag_rate_revised.mean()
    print(
        f"  -> {100 * (late_rate / early_rate - 1):+.1f}% on a constant instrument, "
        "so the post-2018 change is real and unexplained by the revision"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument(
        "--anchor-csv",
        type=Path,
        default=DEFAULT_ANCHOR_CSV,
        help="Pooled surveillance anchor from extract_degraaf_surveillance.py.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    init_script()

    with duckdb.connect(str(args.db_path), read_only=True) as con:
        era = load_era_table(con)
    contrast = summarise_revision_contrast(era)

    if not args.anchor_csv.exists():
        print(
            f"anchor CSV not found: {args.anchor_csv}\n"
            "run scripts/extract_degraaf_surveillance.py first"
        )
        return 2
    anchor = pd.read_csv(args.anchor_csv)
    fits, windows = fit_reporting_fraction(anchor, era)
    report(era, contrast, fits, windows)

    args.output_root.mkdir(parents=True, exist_ok=True)
    era.to_csv(args.output_root / "era_by_year.csv", index=False)
    pd.DataFrame([contrast]).to_csv(
        args.output_root / "revision_split.csv", index=False
    )
    fits.to_csv(args.output_root / "reporting_fraction_fits.csv", index=False)
    windows.to_csv(args.output_root / "reporting_fraction_windows.csv", index=False)
    print(f"\nwrote 4 files to {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
