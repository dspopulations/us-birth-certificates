"""How prenatal screening and termination of DS pregnancies vary over 2016-2024.

Reads the converged reporting fits (C primary; B for the recording bound) and
reports, by calendar year, the population (DS-pregnancy-weighted) values of:

  - eta_detect : prenatal screening detection reach,
  - eta_term   : termination given detection,
  - reduction  = eta_detect * eta_term = 1 - eta : DS pregnancies not born alive,
    with a per-year 95% credible interval (variant C),

plus the recorded vs model true DS livebirth rate per 10,000, the raw
eta_detect_year / eta_term_year log-odds offsets, and a raw recorded-DS-rate-by-age
trend used to probe whether screening expanded fastest in older mothers.

Identifiability / structure notes (printed):
  - recording s has NO year term, so the year TREND in the combined reduction is
    data-identified; the detection-vs-termination split of it is prior-driven
    (eta_detect_year pinned to the NIPT-adoption curve, eta_term_year free residual).
  - eta_detect includes a year-by-age interaction. This lets the model express
    age-specific screening expansion, while the detection-vs-termination split still
    depends strongly on the structural priors.

Figures (-> notes/figures/): year_detection_termination, recorded_rate_by_age_change.

Usage:
    python scripts/year_trends.py
"""

from __future__ import annotations  # noqa: I001

import dspopulations_us_birth_certificates.env_guard  # noqa: F401

import json  # noqa: E402
import os  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402
from dse_research_utils.environment import setup  # noqa: E402
from dse_research_utils.plot import styles  # noqa: E402

from dspopulations_us_birth_certificates.plot_utils import save_fig  # noqa: E402
from dspopulations_us_birth_certificates.selection import (  # noqa: E402
    AGE_LEVELS,
    inv_logit,
    latest_fit_dir,
)

OUTPUT_DIR = "notes/figures"
RECON_TOL = 1e-8


def load_variant(variant: str, with_ci: bool = False) -> dict:
    fit_dir = latest_fit_dir(variant)
    cells = pd.read_parquet(fit_dir / "cells.parquet")
    with open(fit_dir / "config.json") as fh:
        cfg = json.load(fh)
    y0 = int(cfg["year_range"][0])
    n = cells["N_cell"].to_numpy(float)
    r = cells["R_cell"].to_numpy(float)
    idx = {
        k: cells[f"{k}_idx"].to_numpy() for k in ("year", "age", "race", "edu", "payer")
    }

    with xr.open_dataset(fit_dir / "idata.nc", group="posterior") as post:
        years = post["year"].values
        n_year, n_age = len(years), post.sizes["age"]
        p_draws = post["p_ds_lb"].values.reshape(-1, len(cells))  # (draws, cells)
        n_draws = p_draws.shape[0]
        p = p_draws.mean(0)
        tla_draws = inv_logit(post["theta_lb_age"].values.reshape(n_draws, n_age))
        theta = tla_draws.mean(0)[idx["age"]]

        def draw_vector(name: str) -> np.ndarray:
            return post[name].values.reshape(n_draws, post[name].shape[-1])

        def draw_scalar(name: str) -> np.ndarray:
            return post[name].values.reshape(n_draws)

        eta_detect_year = draw_vector("eta_detect_year")
        eta_detect_age = draw_vector("eta_detect_age")
        eta_detect_year_age = post["eta_detect_year_age"].values.reshape(
            n_draws, n_year, n_age
        )
        eta_detect_race = draw_vector("eta_detect_race")
        eta_detect_edu = draw_vector("eta_detect_edu")
        eta_detect_payer = draw_vector("eta_detect_payer")
        eta_term_year = draw_vector("eta_term_year")
        eta_term_age = draw_vector("eta_term_age")
        eta_term_race = draw_vector("eta_term_race")
        eta_term_edu = draw_vector("eta_term_edu")

        det_draws = inv_logit(
            draw_scalar("eta_detect_int")[:, None]
            + eta_detect_year[:, idx["year"]]
            + eta_detect_age[:, idx["age"]]
            + eta_detect_year_age[:, idx["year"], idx["age"]]
            + eta_detect_race[:, idx["race"]]
            + eta_detect_edu[:, idx["edu"]]
            + eta_detect_payer[:, idx["payer"]]
        )
        term_draws = inv_logit(
            draw_scalar("eta_term_int")[:, None]
            + eta_term_year[:, idx["year"]]
            + eta_term_age[:, idx["age"]]
            + eta_term_race[:, idx["race"]]
            + eta_term_edu[:, idx["edu"]]
        )
        det = det_draws.mean(0)
        term = term_draws.mean(0)
        edy = eta_detect_year
        ety = eta_term_year
        recon_draws = tla_draws[:, idx["age"]] * (1.0 - det_draws * term_draws)
        recon_max_err = float(np.abs(recon_draws - p_draws).max())

    # Births and recorded DS by (age, year) -- raw data for the age-split probe.
    n_ay = np.zeros((n_age, n_year))
    r_ay = np.zeros((n_age, n_year))
    for a in range(n_age):
        for y in range(n_year):
            m = (idx["age"] == a) & (idx["year"] == y)
            n_ay[a, y] = n[m].sum()
            r_ay[a, y] = r[m].sum()

    w = theta * n
    rows = {}
    red_lo = np.full(n_year, np.nan)
    red_hi = np.full(n_year, np.nan)
    for y in range(n_year):
        m = idx["year"] == y
        wy = w[m].sum()
        rows[y0 + y] = {
            "recorded_per10k": r[m].sum() / n[m].sum() * 1e4,
            "true_per10k": (p[m] * n[m]).sum() / n[m].sum() * 1e4,
            "reduction": 1.0 - (p[m] * n[m]).sum() / (theta[m] * n[m]).sum(),
            "eta_detect": (det[m] * w[m]).sum() / wy,
            "eta_term": (term[m] * w[m]).sum() / wy,
        }
        if with_ci:
            # Per-draw reduction in year y: 1 - sum(p*N) / sum(theta*N).
            tot = (p_draws[:, m] * n[m]).sum(1)  # (draws,)
            nat = tla_draws[:, idx["age"][m]] @ n[m]  # (draws,)
            rd = 1.0 - tot / nat
            red_lo[y] = np.quantile(rd, 0.025)
            red_hi[y] = np.quantile(rd, 0.975)
    df = pd.DataFrame(rows).T
    df.index.name = "year"

    return {
        "variant": variant,
        "fit_dir": fit_dir,
        "df": df,
        "years": (y0 + np.arange(n_year)),
        "red_lo": red_lo,
        "red_hi": red_hi,
        "rec_rate_ay": r_ay / n_ay * 1e4,  # (age, year) recorded DS per 10k
        "edy": (edy.mean(0), np.quantile(edy, 0.025, 0), np.quantile(edy, 0.975, 0)),
        "ety": (ety.mean(0), np.quantile(ety, 0.025, 0), np.quantile(ety, 0.975, 0)),
        "recon_max_err": recon_max_err,
    }


def fig_year(c: dict, b: dict) -> None:
    fig, ax = plt.subplots(figsize=styles.FIGSIZE_LG)
    yr = c["years"]
    ax.plot(
        yr,
        c["df"]["eta_detect"],
        "-o",
        color=styles.COLOUR_BLUE,
        label="Screening detection",
    )
    ax.plot(
        yr,
        c["df"]["eta_term"],
        "-^",
        color=styles.COLOUR_GREEN,
        label="Termination if detected",
    )
    ax.plot(
        yr,
        c["df"]["reduction"],
        "-s",
        color=styles.COLOUR_RED,
        label="Reduction = not born alive (variant C)",
    )
    ax.fill_between(
        yr,
        c["red_lo"],
        c["red_hi"],
        color=styles.COLOUR_RED,
        alpha=0.2,
        label="Reduction 95% CI (C)",
    )
    ax.plot(
        yr,
        b["df"]["reduction"],
        "--s",
        color=styles.COLOUR_ORANGE,
        alpha=0.8,
        label="Reduction (variant B)",
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)
    ax.set_title("Prenatal screening and termination of DS pregnancies, 2016–2024")
    ax.legend(fontsize=7, loc="center left")
    data = c["df"].reset_index()[["year", "eta_detect", "eta_term", "reduction"]].copy()
    data["reduction_lo95"] = c["red_lo"]
    data["reduction_hi95"] = c["red_hi"]
    data["reduction_B"] = b["df"]["reduction"].to_numpy()
    save_fig(fig, OUTPUT_DIR, "year_detection_termination", data=data)
    plt.close(fig)


def fig_age_split(c: dict) -> None:
    # Pool 3-year periods so small-count extreme bands are not dominated by noise.
    fig, ax = plt.subplots(figsize=styles.FIGSIZE_MD)
    rate = c["rec_rate_ay"]  # (age, year)
    early = rate[:, :3].mean(1)  # 2016-2018
    late = rate[:, -3:].mean(1)  # 2022-2024
    pct = (late / early - 1.0) * 100.0
    x = np.arange(len(AGE_LEVELS))
    ax.bar(
        x,
        pct,
        color=[styles.COLOUR_RED if v < 0 else styles.COLOUR_BLUE for v in pct],
        alpha=0.85,
    )
    ax.axhline(0, color=styles.TEXT_COLOUR, lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(AGE_LEVELS, rotation=45, ha="right")
    ax.set_ylabel("Recorded DS rate change (%)\n2016–18 vs 2022–24")
    ax.set_xlabel("Maternal age band")
    ax.set_title("Recorded DS rate fell more in older mothers (raw data)")
    data = pd.DataFrame(
        {
            "age_band": AGE_LEVELS,
            "early_2016_18": early,
            "late_2022_24": late,
            "pct_change": pct,
        }
    )
    save_fig(fig, OUTPUT_DIR, "recorded_rate_by_age_change", data=data)
    plt.close(fig)


def main() -> int:
    setup.init_script()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    c = load_variant("C", with_ci=True)
    b = load_variant("B", with_ci=False)
    print(f"C: {c['fit_dir']}\nB: {b['fit_dir']}")
    print(f"reconstruction max |recon_p - p_ds_lb| = {c['recon_max_err']:.2e} (C)\n")
    for variant in (c, b):
        if variant["recon_max_err"] > RECON_TOL:
            raise RuntimeError(
                f"Variant {variant['variant']} reconstruction mismatch: "
                f"{variant['recon_max_err']:.2e} > {RECON_TOL:.1e}"
            )

    pd.set_option("display.width", 200)
    show = c["df"].copy()
    show["red_lo95"] = c["red_lo"]
    show["red_hi95"] = c["red_hi"]
    show["reduction_B"] = b["df"]["reduction"]
    print("=== By year (variant C; reduction 95% CI; reduction_B = variant B) ===")
    print(show.to_string(float_format="{:.3f}".format))

    print("\n=== Recorded DS rate per 10k by maternal-age band x year (raw data) ===")
    rec = pd.DataFrame(c["rec_rate_ay"], index=AGE_LEVELS, columns=c["years"])
    print(rec.to_string(float_format="{:.2f}".format))
    chg = (c["rec_rate_ay"][:, -1] / c["rec_rate_ay"][:, 0] - 1.0) * 100
    print("\n2016->2024 change in recorded DS rate (%), by age band:")
    print(pd.Series(chg, index=AGE_LEVELS).to_string(float_format="{:+.1f}".format))

    fig_year(c, b)
    fig_age_split(c)
    print(
        f"\nwrote year_detection_termination, recorded_rate_by_age_change to {OUTPUT_DIR}/"
    )
    print(
        "\nNote: the combined-reduction TREND is data-identified (s has no year term);\n"
        "the detection-vs-termination split is prior-driven (detection pinned to NIPT).\n"
        "eta_detect includes a year-by-age interaction, so age-specific screening expansion\n"
        "can appear in the model; the raw recorded-rate-by-age trend remains a useful\n"
        "empirical check on that structure."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
