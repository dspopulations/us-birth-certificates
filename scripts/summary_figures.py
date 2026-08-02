"""Render the figures used in the plain-language summary.

Self-contained off the converged reporting fits (variants C and B). Produces,
into ``notes/figures/`` (PNG + SVG + companion CSV per figure):

  1. ascertainment_funnel   -- natural -> born-alive (C..B bound) -> recorded.
  2. age_termination_reduction -- termination share by maternal-age band (C, B).
  3. ds_rate_by_ethnicity   -- recorded vs true DS livebirth rate /10k, C..B bound.
  4. education_termination_gradient -- eta_term_edu posterior (mean + 89% ETI) vs prior.

Variant C pins recording (~0.40); variant B frees it (~0.32). The C..B spread is
the recording-vs-termination sensitivity bound, not sampling noise.

Usage:
    python scripts/summary_figures.py
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

from dspopulations_us_birth_certificates.intervals import equal_tail_interval  # noqa: E402
from dspopulations_us_birth_certificates.plot_utils import save_fig  # noqa: E402
from dspopulations_us_birth_certificates.selection import (  # noqa: E402
    AGE_LEVELS,
    EDU_LEVELS,
    RACE_LEVELS,
    inv_logit,
    latest_fit_dir,
)

OUTPUT_DIR = "notes/figures"
# Short ethnicity labels for axis ticks.
RACE_SHORT_BY_LEVEL = {
    "NH White": "White",
    "NH Black": "Black",
    "NH AIAN": "AIAN",
    "NH Asian/Pacific Islander": "Asian/PI",
    "Hispanic": "Hispanic",
    "Unknown": "Unknown",
    "NH Multi-race": "Multi-race",
}
missing_race_labels = set(RACE_LEVELS) - set(RACE_SHORT_BY_LEVEL)
if missing_race_labels:
    raise ValueError(f"Missing short race labels: {sorted(missing_race_labels)!r}")
RACE_SHORT = [RACE_SHORT_BY_LEVEL[level] for level in RACE_LEVELS]


def load_variant(variant: str) -> dict:
    """Posterior-mean per-cell quantities + aggregates for one converged fit."""
    fit_dir = latest_fit_dir(variant)
    cells = pd.read_parquet(fit_dir / "cells.parquet")
    n = cells["N_cell"].to_numpy(float)
    r = cells["R_cell"].to_numpy(float)
    race = cells["race_idx"].to_numpy()
    age = cells["age_idx"].to_numpy()
    with open(fit_dir / "config.json") as fh:
        config = json.load(fh)
    fpr = float(config["priors"]["false_positive_rate"])

    with xr.open_dataset(fit_dir / "idata.nc", group="posterior") as post:
        p = post["p_ds_lb"].values.reshape(-1, len(cells)).mean(0)
        theta = inv_logit(
            post["theta_lb_age"].values.reshape(-1, len(AGE_LEVELS)).mean(0)
        )
        ete = post["eta_term_edu"].values.reshape(-1, len(EDU_LEVELS))
        ete_lo, ete_hi = equal_tail_interval(ete, axis=0)

    theta_cell = theta[age]
    # Per maternal-age termination reduction (1 - eta).
    age_red = np.array(
        [
            1.0
            - (p[age == a] * n[age == a]).sum()
            / (theta_cell[age == a] * n[age == a]).sum()
            for a in range(len(AGE_LEVELS))
        ]
    )
    # Per ethnicity recorded / true rate per 10k.
    rec10k = np.array(
        [r[race == j].sum() / n[race == j].sum() * 1e4 for j in range(len(RACE_LEVELS))]
    )
    true10k = np.array(
        [
            (p[race == j] * n[race == j]).sum() / n[race == j].sum() * 1e4
            for j in range(len(RACE_LEVELS))
        ]
    )
    return {
        "variant": variant,
        "fit_dir": fit_dir,
        "natural": float((theta_cell * n).sum()),
        "total_true": float((p * n).sum()),
        "recorded_corrected": float(r.sum() - fpr * ((1.0 - p) * n).sum()),
        "recorded_raw": float(r.sum()),
        "age_reduction": age_red,
        "rec10k": rec10k,
        "true10k": true10k,
        "ete_mean": ete.mean(0),
        "ete_lo": ete_lo,
        "ete_hi": ete_hi,
        "ete_prior": np.asarray(config["priors"]["eta_term_edu"], float),
    }


def fig_funnel(c: dict, b: dict) -> None:
    fig, ax = plt.subplots(figsize=styles.FIGSIZE_LG)
    stages = [
        "Natural\n(no screening)",
        "Born alive\nwith DS",
        "Recorded on\ncertificate",
    ]
    y = [2, 1, 0]
    natural = c["natural"]
    ax.barh(y[0], natural, color=styles.COLOUR_BLUE, height=0.6)
    # Born: solid to C (lower), hatched extension to B (upper bound).
    ax.barh(y[1], c["total_true"], color=styles.COLOUR_ORANGE, height=0.6)
    ax.barh(
        y[1],
        b["total_true"] - c["total_true"],
        left=c["total_true"],
        color=styles.COLOUR_ORANGE,
        alpha=0.35,
        hatch="//",
        height=0.6,
    )
    ax.barh(y[2], c["recorded_corrected"], color=styles.COLOUR_RED, height=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(stages)
    ax.set_xlabel("DS livebirths, 2016-2024")
    ax.set_title(
        "Ascertainment funnel: most true DS births never reach the certificate"
    )
    for yi, val in [(y[0], natural), (y[2], c["recorded_corrected"])]:
        ax.text(val + natural * 0.01, yi, f"{val:,.0f}", va="center", fontsize=9)
    ax.text(
        b["total_true"] + natural * 0.01,
        y[1],
        f"{c['total_true']:,.0f}-{b['total_true']:,.0f}",
        va="center",
        fontsize=9,
    )
    ax.annotate(
        f"-{(1 - b['total_true'] / natural) * 100:.0f}% to "
        f"-{(1 - c['total_true'] / natural) * 100:.0f}%\nterminated",
        xy=(natural * 0.5, 1.5),
        ha="center",
        va="center",
        fontsize=8,
        color=styles.TEXT_COLOUR,
    )
    ax.annotate(
        "only ~32-38% recorded\n=> ~62-68% missed",
        xy=(c["total_true"] * 0.4, 0.5),
        ha="center",
        va="center",
        fontsize=8,
        color=styles.TEXT_COLOUR,
    )
    ax.margins(x=0.18)
    data = pd.DataFrame(
        {
            "stage": [
                "natural",
                "born_C",
                "born_B",
                "recorded_corrected",
                "recorded_raw",
            ],
            "count": [
                natural,
                c["total_true"],
                b["total_true"],
                c["recorded_corrected"],
                c["recorded_raw"],
            ],
        }
    )
    save_fig(fig, OUTPUT_DIR, "ascertainment_funnel", data=data)
    plt.close(fig)


def fig_age_reduction(c: dict, b: dict) -> None:
    fig, ax = plt.subplots(figsize=styles.FIGSIZE_MD)
    x = np.arange(len(AGE_LEVELS))
    ax.plot(
        x,
        c["age_reduction"],
        "-o",
        color=styles.COLOUR_BLUE,
        label="Variant C (pin recording)",
    )
    ax.plot(
        x,
        b["age_reduction"],
        "--s",
        color=styles.COLOUR_ORANGE,
        label="Variant B (free recording)",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(AGE_LEVELS, rotation=45, ha="right")
    ax.set_xlabel("Maternal age band")
    ax.set_ylabel("Reduction: DS pregnancies not born alive")
    ax.set_title(
        "Elective termination of DS pregnancies rises steeply with maternal age"
    )
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    data = pd.DataFrame(
        {
            "age_band": AGE_LEVELS,
            "reduction_C": c["age_reduction"],
            "reduction_B": b["age_reduction"],
        }
    )
    save_fig(fig, OUTPUT_DIR, "age_termination_reduction", data=data)
    plt.close(fig)


def fig_ethnicity(c: dict, b: dict) -> None:
    fig, ax = plt.subplots(figsize=styles.FIGSIZE_LG)
    x = np.arange(len(RACE_LEVELS))
    # True-rate bound C..B drawn as a vertical range bar; recorded as a marker.
    lo = np.minimum(c["true10k"], b["true10k"])
    hi = np.maximum(c["true10k"], b["true10k"])
    ax.bar(
        x,
        hi - lo,
        bottom=lo,
        width=0.5,
        color=styles.COLOUR_ORANGE,
        alpha=0.55,
        label="True (model, C..B)",
    )
    ax.plot(
        x, c["rec10k"], "o", color=styles.COLOUR_RED, label="Recorded (certificate)"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(RACE_SHORT, rotation=30, ha="right")
    ax.set_ylabel("DS livebirths per 10,000")
    ax.set_title("True vs recorded DS livebirth rate by ethnicity")
    ax.legend(fontsize=8)
    data = pd.DataFrame(
        {
            "ethnicity": RACE_LEVELS,
            "recorded_per10k": c["rec10k"],
            "true_per10k_C": c["true10k"],
            "true_per10k_B": b["true10k"],
        }
    )
    save_fig(fig, OUTPUT_DIR, "ds_rate_by_ethnicity", data=data)
    plt.close(fig)


def fig_education(c: dict) -> None:
    fig, ax = plt.subplots(figsize=styles.FIGSIZE_MD)
    x = np.arange(len(EDU_LEVELS))
    err = np.vstack([c["ete_mean"] - c["ete_lo"], c["ete_hi"] - c["ete_mean"]])
    ax.errorbar(
        x,
        c["ete_mean"],
        yerr=err,
        fmt="o",
        color=styles.COLOUR_BLUE,
        capsize=3,
        label="Posterior (data)",
    )
    ax.plot(x, c["ete_prior"], "x", color=styles.TEXT_COLOUR, alpha=0.6, label="Prior")
    ax.axhline(0, color=styles.TEXT_COLOUR, lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(EDU_LEVELS, rotation=45, ha="right")
    ax.set_ylabel("Termination log-odds offset")
    ax.set_title(
        "Higher education -> much higher termination (data overrides the prior)"
    )
    ax.legend(fontsize=8)
    data = pd.DataFrame(
        {
            "education": EDU_LEVELS,
            "prior": c["ete_prior"],
            "post_mean": c["ete_mean"],
            "lo89": c["ete_lo"],
            "hi89": c["ete_hi"],
        }
    )
    save_fig(fig, OUTPUT_DIR, "education_termination_gradient", data=data)
    plt.close(fig)


def main() -> int:
    setup.init_script()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    c = load_variant("C")
    b = load_variant("B")
    print(f"C: {c['fit_dir']}\nB: {b['fit_dir']}")
    print(
        f"natural={c['natural']:,.0f}  born C..B={c['total_true']:,.0f}.."
        f"{b['total_true']:,.0f}  recorded(corr)={c['recorded_corrected']:,.0f}"
    )
    fig_funnel(c, b)
    fig_age_reduction(c, b)
    fig_ethnicity(c, b)
    fig_education(c)
    print(f"wrote 4 figures (png/svg/csv) to {OUTPUT_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
