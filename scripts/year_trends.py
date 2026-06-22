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
  - eta_detect has NO year-by-age interaction (it is additive in year and age), so
    the model cannot itself resolve whether screening reached older mothers first.
    The raw recorded-DS-rate-by-age trend is the empirical signal for that question.

Figures (-> notes/figures/): year_detection_termination, recorded_rate_by_age_change.

Usage:
    python scripts/year_trends.py
"""

from __future__ import annotations

import os

# This Windows/conda environment aborts inside MKL's threadpool (OSError WinError
# 0xc06d007f) on numpy paths unless MKL threading is tamed; must precede numpy.
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import glob  # noqa: E402
import json  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402
from dse_research_utils.environment import setup  # noqa: E402
from dse_research_utils.plot import styles  # noqa: E402

from dspopulations_us_birth_certificates.plot_utils import _save_fig  # noqa: E402
from dspopulations_us_birth_certificates.selection import AGE_LEVELS  # noqa: E402

OUTPUT_DIR = "notes/figures"


def _inv_logit(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _latest_fit(variant: str) -> str:
    runs = sorted(
        (d for d in glob.glob(f"output/selection/{variant}/full/*") if os.path.isfile(f"{d}/idata.nc")),
        key=os.path.getmtime,
    )
    if not runs:
        raise SystemExit(f"no converged fit for variant {variant}")
    return runs[-1]


def load_variant(variant: str, with_ci: bool = False) -> dict:
    fit_dir = _latest_fit(variant)
    cells = pd.read_parquet(f"{fit_dir}/cells.parquet")
    cfg = json.load(open(f"{fit_dir}/config.json"))
    y0 = int(cfg["year_range"][0])
    n = cells["N_cell"].to_numpy(float)
    r = cells["R_cell"].to_numpy(float)
    idx = {k: cells[f"{k}_idx"].to_numpy() for k in ("year", "age", "race", "edu", "payer")}

    post = xr.open_dataset(f"{fit_dir}/idata.nc", group="posterior")
    years = post["year"].values
    n_year, n_age = len(years), post.sizes["age"]
    p_draws = post["p_ds_lb"].values.reshape(-1, len(cells))  # (draws, cells)
    p = p_draws.mean(0)
    tla_draws = _inv_logit(post["theta_lb_age"].values.reshape(-1, n_age))  # (draws, age)
    theta = tla_draws.mean(0)[idx["age"]]

    def cmean(name: str) -> np.ndarray:
        return post[name].values.reshape(-1, post[name].shape[-1]).mean(0)

    def cscalar(name: str) -> float:
        return float(post[name].values.mean())

    det = _inv_logit(
        cscalar("eta_detect_int")
        + cmean("eta_detect_year")[idx["year"]]
        + cmean("eta_detect_age")[idx["age"]]
        + cmean("eta_detect_race")[idx["race"]]
        + cmean("eta_detect_edu")[idx["edu"]]
        + cmean("eta_detect_payer")[idx["payer"]]
    )
    term = _inv_logit(
        cscalar("eta_term_int")
        + cmean("eta_term_year")[idx["year"]]
        + cmean("eta_term_age")[idx["age"]]
        + cmean("eta_term_race")[idx["race"]]
        + cmean("eta_term_edu")[idx["edu"]]
    )
    edy = post["eta_detect_year"].values.reshape(-1, n_year)
    ety = post["eta_term_year"].values.reshape(-1, n_year)
    post.close()

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

    recon_max_err = float(np.abs(theta * (1.0 - det * term) - p).max())

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
    ax.plot(yr, c["df"]["eta_detect"], "-o", color=styles.COLOUR_BLUE, label="Screening detection")
    ax.plot(yr, c["df"]["eta_term"], "-^", color=styles.COLOUR_GREEN, label="Termination if detected")
    ax.plot(yr, c["df"]["reduction"], "-s", color=styles.COLOUR_RED, label="Reduction = not born alive (variant C)")
    ax.fill_between(yr, c["red_lo"], c["red_hi"], color=styles.COLOUR_RED, alpha=0.2, label="Reduction 95% CI (C)")
    ax.plot(yr, b["df"]["reduction"], "--s", color=styles.COLOUR_ORANGE, alpha=0.8, label="Reduction (variant B)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)
    ax.set_title("Prenatal screening and termination of DS pregnancies, 2016–2024")
    ax.legend(fontsize=7, loc="center left")
    data = c["df"].reset_index()[["year", "eta_detect", "eta_term", "reduction"]].copy()
    data["reduction_lo95"] = c["red_lo"]
    data["reduction_hi95"] = c["red_hi"]
    data["reduction_B"] = b["df"]["reduction"].to_numpy()
    _save_fig(fig, OUTPUT_DIR, "year_detection_termination", data=data)
    plt.close(fig)


def fig_age_split(c: dict) -> None:
    # Pool 3-year periods so small-count extreme bands are not dominated by noise.
    fig, ax = plt.subplots(figsize=styles.FIGSIZE_MD)
    rate = c["rec_rate_ay"]  # (age, year)
    early = rate[:, :3].mean(1)  # 2016-2018
    late = rate[:, -3:].mean(1)  # 2022-2024
    pct = (late / early - 1.0) * 100.0
    x = np.arange(len(AGE_LEVELS))
    ax.bar(x, pct, color=[styles.COLOUR_RED if v < 0 else styles.COLOUR_BLUE for v in pct], alpha=0.85)
    ax.axhline(0, color=styles.TEXT_COLOUR, lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(AGE_LEVELS, rotation=45, ha="right")
    ax.set_ylabel("Recorded DS rate change (%)\n2016–18 vs 2022–24")
    ax.set_xlabel("Maternal age band")
    ax.set_title("Recorded DS rate fell more in older mothers (raw data)")
    data = pd.DataFrame(
        {"age_band": AGE_LEVELS, "early_2016_18": early, "late_2022_24": late, "pct_change": pct}
    )
    _save_fig(fig, OUTPUT_DIR, "recorded_rate_by_age_change", data=data)
    plt.close(fig)


def main() -> int:
    setup.init_script()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    c = load_variant("C", with_ci=True)
    b = load_variant("B", with_ci=False)
    print(f"C: {c['fit_dir']}\nB: {b['fit_dir']}")
    print(f"reconstruction max |recon_p - p_ds_lb| = {c['recon_max_err']:.2e} (C)\n")

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
    print(f"\nwrote year_detection_termination, recorded_rate_by_age_change to {OUTPUT_DIR}/")
    print(
        "\nNote: the combined-reduction TREND is data-identified (s has no year term);\n"
        "the detection-vs-termination split is prior-driven (detection pinned to NIPT).\n"
        "eta_detect has no year-by-age interaction, so the model cannot itself say whether\n"
        "screening reached older mothers first -- the raw recorded-rate-by-age trend above\n"
        "is the empirical signal, and would need an interaction term to model explicitly."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
