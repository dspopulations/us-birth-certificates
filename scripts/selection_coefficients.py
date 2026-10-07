"""AI-assisted documentation revision by Codex (GPT-6).

Report demographic coefficients from a saved selection fit.

For each available detection, termination and recording coefficient, prints its
prior centre, posterior mean and 89% equal-tail interval. A moved flag means the
prior centre is outside that interval. It does not by itself establish a causal
effect, identification or which evidence source caused the shift.

Detection and termination act through a product. Current recording uses an
externally derived race-by-year surface, with an education residual. Inspect the
saved prior configuration and diagnostic tables before interpreting a coefficient.

Usage:
    uv run python scripts/selection_coefficients.py [FIT_DIR]
    uv run python scripts/selection_coefficients.py --variant B
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import xarray as xr

from dspopulations_us_birth_certificates.intervals import equal_tail_interval
from dspopulations_us_birth_certificates.selection import (
    EDU_LEVELS,
    PAYER_LEVELS,
    RACE_LEVELS,
    latest_fit_dir,
)

# (heading, [(posterior var name, level labels)])
STAGES = [
    (
        "DETECTION  eta_detect_*  (screening reach under supplied priors)",
        [
            ("eta_detect_race", RACE_LEVELS),
            ("eta_detect_edu", EDU_LEVELS),
            ("eta_detect_payer", PAYER_LEVELS),
        ],
    ),
    (
        "TERMINATION  eta_term_*  (conditional termination offsets)",
        [
            ("eta_term_race", RACE_LEVELS),
            ("eta_term_edu", EDU_LEVELS),
        ],
    ),
    (
        "RECORDING  s_*  (education offsets; race uses an anchor surface)",
        [
            ("s_race", RACE_LEVELS),
            ("s_edu", EDU_LEVELS),
        ],
    ),
]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("fit_dir", nargs="?", default=None, help="explicit fit directory")
    ap.add_argument(
        "--variant", default="C", help="variant to auto-pick latest of (default C)"
    )
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
                lo, hi = equal_tail_interval(arr, axis=0)
                moved = (prior < lo) | (prior > hi)
                tab = pd.DataFrame(
                    {
                        "level": levels,
                        "prior": prior,
                        "post_mean": mean,
                        "lo89": lo,
                        "hi89": hi,
                        "moved": ["yes" if m else "" for m in moved],
                    }
                )
                print(f"\n{name}  (prior sigma {sigma:g})")
                print(tab.to_string(index=False, float_format="{:+.2f}".format))
            print()

    print(
        "moved = prior centre lies outside the 89% equal-tail interval.\n"
        "Positive offsets increase that stage's probability relative to its reference.\n"
        "The screening/termination and recording/prevalence splits depend on priors "
        "and external evidence; these coefficient tables do not establish identification."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
