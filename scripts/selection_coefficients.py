"""Surface the demographic coefficients of a converged selection fit.

For each stage (detection / termination / recording) and demographic dimension
(maternal race, education, insurance payer) prints the prior mean, the posterior
mean and 95 % credible interval (log-odds offset), and flags whether the data
moved the coefficient off its prior (prior mean outside the 95 % CI).

How to read it (also printed):
  - eta_detect_* (race, education, PAYER) are PINNED at sigma 0.20 and are NOT
    adjusted by variant: they encode the screening-access assumption. Inputs, not
    findings. Payer (insurance) enters the model ONLY here.
  - eta_term_* (race, education) are the data-identified residual on eta -- loosest
    in variant A, tightest in B.
  - s_* (race, education) are PINNED (sigma 0.05) in variants A and C; only variant
    B (sigma 0.10) lets the data move recording by subgroup.
The detection-vs-termination split inside eta is NOT identified; only the combined
effect on eta is. A "moved" flag on a pinned (eta_detect_* / s_* in A,C) coefficient
means the strong likelihood overrode even a tight prior -- a real data signal; on a
loose coefficient it just means the data were informative.

Usage:
    python scripts/selection_coefficients.py [FIT_DIR]
    python scripts/selection_coefficients.py --variant B
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import xarray as xr

from dspopulations_us_birth_certificates.selection import (
    EDU_LEVELS,
    PAYER_LEVELS,
    RACE_LEVELS,
    latest_fit_dir,
)

# (heading, [(posterior var name, level labels)])
STAGES = [
    (
        "DETECTION  eta_detect_*  (screening reach; PINNED sigma~0.20 assumption)",
        [
            ("eta_detect_race", RACE_LEVELS),
            ("eta_detect_edu", EDU_LEVELS),
            ("eta_detect_payer", PAYER_LEVELS),
        ],
    ),
    (
        "TERMINATION  eta_term_*  (data-identified residual on eta)",
        [
            ("eta_term_race", RACE_LEVELS),
            ("eta_term_edu", EDU_LEVELS),
        ],
    ),
    (
        "RECORDING  s_*  (PINNED sigma~0.05 in A/C; free sigma~0.10 in B)",
        [
            ("s_race", RACE_LEVELS),
            ("s_edu", EDU_LEVELS),
        ],
    ),
]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("fit_dir", nargs="?", default=None, help="explicit fit directory")
    ap.add_argument("--variant", default="C", help="variant to auto-pick latest of (default C)")
    ns = ap.parse_args(argv)

    fit_dir = ns.fit_dir or latest_fit_dir(ns.variant)
    print(f"fit: {fit_dir}\n")
    with open(f"{fit_dir}/config.json") as fh:
        priors = json.load(fh)["priors"]

    pd.set_option("display.width", 240)
    with xr.open_dataset(f"{fit_dir}/idata.nc", group="posterior") as post:
        for heading, coeffs in STAGES:
            print(f"=== {heading} ===")
            for name, levels in coeffs:
                arr = post[name].values.reshape(-1, len(levels))
                prior = np.asarray(priors[name], float)
                sigma = float(priors[f"{name}_sigma"])
                mean = arr.mean(0)
                lo = np.quantile(arr, 0.025, axis=0)
                hi = np.quantile(arr, 0.975, axis=0)
                moved = (prior < lo) | (prior > hi)
                tab = pd.DataFrame(
                    {
                        "level": levels,
                        "prior": prior,
                        "post_mean": mean,
                        "lo95": lo,
                        "hi95": hi,
                        "moved": ["yes" if m else "" for m in moved],
                    }
                )
                print(f"\n{name}  (prior sigma {sigma:g})")
                print(tab.to_string(index=False, float_format="{:+.2f}".format))
            print()

    print(
        "moved = prior mean lies outside the 95% CI (data overrode the prior).\n"
        "Positive offset => higher probability at that stage for that level.\n"
        "Detection offsets are pinned screening-access inputs; payer is detection-"
        "only. Termination\noffsets carry the data-identified residual on eta. "
        "Recording offsets are pinned in A/C and\nonly move in B -- the "
        "detection-vs-termination and recording-vs-termination splits are not\n"
        "identified from birth-certificate data alone."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
