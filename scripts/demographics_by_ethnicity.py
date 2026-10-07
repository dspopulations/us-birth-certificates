"""AI-assisted documentation revision by Codex (GPT-6).

Describe model-based ethnicity contrasts from a saved selection fit.

Reports crude and maternal-age-standardised rates, reduction and recording
summaries. Standardisation uses a common national age distribution within the
fitted model. It does not identify a causal race effect or separate recording
from prenatal reduction.

The default input is the latest reporting variant C. A reporting profile does not
establish convergence. Check the fit's diagnostics and priors before using results.
Current recording priors are race-by-year surfaces, not the old global pin.

Usage:
    uv run python scripts/demographics_by_ethnicity.py [FIT_DIR]
    uv run python scripts/demographics_by_ethnicity.py --variant A
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from dspopulations_us_birth_certificates.intervals import equal_tail_interval
from dspopulations_us_birth_certificates.selection import (
    AGE_LEVELS,
    RACE_LEVELS,
    inv_logit,
    latest_fit_dir,
)

# Age-band midpoints used only to report a mean maternal age (45+ -> 47).
AGE_MID = np.array([18, 22, 27, 32, 37, 42, 47.0])


def _coeff_table(post: xr.Dataset, name: str) -> pd.DataFrame:
    """Posterior mean and project-standard ETI for a race-dimensioned coefficient."""
    arr = post[name].values.reshape(-1, len(RACE_LEVELS))
    lo, hi = equal_tail_interval(arr, axis=0)
    return pd.DataFrame(
        {
            "ethnicity": RACE_LEVELS,
            "mean": arr.mean(0),
            "lo": lo,
            "hi": hi,
        }
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("fit_dir", nargs="?", default=None, help="explicit fit directory")
    ap.add_argument(
        "--variant", default="C", help="variant to auto-pick latest of (default C)"
    )
    ns = ap.parse_args(argv)

    fit_dir = Path(ns.fit_dir) if ns.fit_dir else latest_fit_dir(ns.variant)
    print(f"fit: {fit_dir}\n")

    cells = pd.read_parquet(fit_dir / "cells.parquet")
    n_cell = cells["N_cell"].to_numpy(float)
    r_cell = cells["R_cell"].to_numpy(float)
    race = cells["race_idx"].to_numpy()
    age = cells["age_idx"].to_numpy()
    with open(fit_dir / "config.json") as fh:
        fpr = float(json.load(fh)["priors"]["false_positive_rate"])

    with xr.open_dataset(fit_dir / "idata.nc", group="posterior") as post:
        # Posterior-mean true-DS-livebirth probability per cell (theta * eta).
        p = post["p_ds_lb"].values.reshape(-1, len(cells)).mean(0)
        theta = inv_logit(post["theta_lb_age"].values.reshape(-1, 7).mean(0))[age]
        eta_term_race = _coeff_table(post, "eta_term_race")
        s_race = _coeff_table(post, "s_race")

    n_age, n_race = len(AGE_LEVELS), len(RACE_LEVELS)

    # Per (race, age) block averages of theta*eta and theta, weighted by births.
    # These average over the within-block edu/payer/year mix using actual weights.
    w = np.zeros((n_race, n_age))
    pe = np.zeros((n_race, n_age))  # theta*eta block mean
    th = np.zeros((n_race, n_age))  # theta block mean
    for rr in range(n_race):
        for aa in range(n_age):
            m = (race == rr) & (age == aa)
            wt = n_cell[m].sum()
            w[rr, aa] = wt
            if wt > 0:
                pe[rr, aa] = (p[m] * n_cell[m]).sum() / wt
                th[rr, aa] = (theta[m] * n_cell[m]).sum() / wt

    # National maternal-age distribution = the age-standard weights.
    std_w = w.sum(0)
    std_w = std_w / std_w.sum()

    # Fall back to the race-marginal block mean for any empty (race, age) cell so
    # standardisation has an eta estimate at every age.
    marg_pe = (w * pe).sum(0) / np.where(w.sum(0) > 0, w.sum(0), 1.0)
    marg_th = (w * th).sum(0) / np.where(w.sum(0) > 0, w.sum(0), 1.0)
    for rr in range(n_race):
        empty = w[rr] == 0
        pe[rr, empty] = marg_pe[empty]
        th[rr, empty] = marg_th[empty]

    rows = []
    for rr in range(n_race):
        m = race == rr
        if not m.any():
            continue
        nr = n_cell[m].sum()
        rr_rec = r_cell[m].sum()
        true_ct = (p[m] * n_cell[m]).sum()
        fp = fpr * ((1.0 - p[m]) * n_cell[m]).sum()
        own_w = w[rr] / w[rr].sum()
        crude_red = 1.0 - (own_w * pe[rr]).sum() / (own_w * th[rr]).sum()
        std_red = 1.0 - (std_w * pe[rr]).sum() / (std_w * th[rr]).sum()
        rows.append(
            {
                "ethnicity": RACE_LEVELS[rr],
                "births_%": nr / n_cell.sum() * 100,
                "mean_age": (AGE_MID[age[m]] * n_cell[m]).sum() / nr,
                "rec_/10k": rr_rec / nr * 1e4,
                "true_/10k": true_ct / nr * 1e4,
                "reduction_crude": crude_red,
                "reduction_agestd": std_red,
                "recording_s": (rr_rec - fp) / true_ct,
            }
        )
    df = pd.DataFrame(rows)

    # Maternal-age distribution by ethnicity (fraction of each group's births).
    dist = w / w.sum(1, keepdims=True)
    adf = pd.DataFrame(dist, index=RACE_LEVELS, columns=AGE_LEVELS)

    pd.set_option("display.width", 240)
    fmt = "{:.2f}".format
    print("=== DS by ethnicity ===")
    print(df.to_string(index=False, float_format=fmt))
    print(
        "\nreduction_crude reflects each group's own age mix; reduction_agestd "
        "re-weights every\ngroup to the national maternal-age distribution, so it "
        "isolates the race effect net of age."
    )

    print("\n=== Maternal-age distribution by ethnicity (fraction of births) ===")
    print(adf.to_string(float_format="{:.3f}".format))

    print("\n=== eta_term_race (termination log-odds offset; +=more termination) ===")
    print(eta_term_race.to_string(index=False, float_format=fmt))
    print("\n=== s_race (year-averaged recording logit; externally anchored) ===")
    print(s_race.to_string(index=False, float_format=fmt))

    print(
        "\nNOTE: reduction and recording remain conditional on the fitted priors. "
        "The recording surface uses surveillance and recorded counts. "
        "A/B/C comparisons test selected prior restrictions; their range is not "
        "an identified bound on subgroup effects."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
