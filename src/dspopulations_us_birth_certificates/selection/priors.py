# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""AI-assisted documentation revision by Codex (GPT-6).

Prior inputs for the three-stage selection model.

The model uses a Morris counterfactual live-birth rate, separate screening and
termination priors, a surveillance-derived recording surface, and a fixed
false-positive probability. Their products enter the recorded-count likelihood.
The priors constrain a decomposition the certificate counts cannot identify alone.

The recording surface reuses the fitted recorded counts. It is not an independent
validation-study measurement. Unknown and multi-race have weak fallback priors.
See the selection README and notes/20260707-s-anchor-and-identifiability-diagnostic.md.

Literature context:
- Morris et al. (2002), DOI 10.1136/jms.9.1.2.
- Natoli et al. (2012), DOI 10.1002/pd.2910.
- Boulet et al. (2011), DOI 10.1177/003335491112600209.
- Salemi et al. (2017), DOI 10.1111/ppe.12326.

These studies describe different populations and periods. Their estimates are
not interchangeable national rates for 2016 to 2024.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from dse_research_utils.statistics.transforms import invlogit
from dse_research_utils.statistics.transforms import logit as shared_logit

from dspopulations_us_birth_certificates.selection.recording_anchor import (
    S_RACE_YEAR_LOGIT,
    S_RACE_YEAR_SIGMA,
)


def logit(p):
    """Logit transform, retaining list-to-array conversion."""
    return shared_logit(np.asarray(p, dtype=float))


def inv_logit(x):
    """Inverse logit with stable values for large positive or negative inputs."""
    return invlogit(np.asarray(x, dtype=float))


# --------------------------------------------------------------------------- #
# Factor-level vocabularies                                                   #
# --------------------------------------------------------------------------- #

AGE_LEVELS = ["<20", "20-24", "25-29", "30-34", "35-39", "40-44", "45+"]
# Race vocabulary, aligned to the mracehisp_c coding produced by
# duckdb_prepare.py and consumed by selection.data.RACE_MAP. The codes
# are: 1=NH White (idx 0), 2=NH Black (1), 3=NH AIAN only (2), 4=NH
# Asian/Pacific Islander (broad bucket — Asian + NHOPI + Other; 3),
# 5=Hispanic (4), NULL=Unknown (5), 6=NH more than one race (idx 6).
# Unknown stays at idx 5; multi-race is appended at idx 6 (de Graaf has
# no multi-race anchor, so its s(race, year) is the same weak fallback
# as Unknown). The prior arrays below are indexed in this order —
# re-derive ETA_TERM_RACE / S_RACE etc. against the published literature
# if any single demographic's prior magnitude looks off; an earlier
# version of this label list swapped positions 2 and 3 and the values
# may need a second look. The NH Multi-race offsets are unpinned
# placeholders (no external evidence; neutral, like Unknown).
RACE_LEVELS = [
    "NH White",
    "NH Black",
    "NH AIAN",
    "NH Asian/Pacific Islander",
    "Hispanic",
    "Unknown",
    "NH Multi-race",
]
EDU_LEVELS = [
    "<HS",
    "HS/GED",
    "Some college",
    "Bachelor's",
    "Master's+",
    "Unknown",
]
PAYER_LEVELS = ["Medicaid", "Private", "Self-pay/Other", "Unknown"]

N_AGE = len(AGE_LEVELS)
N_RACE = len(RACE_LEVELS)
N_EDU = len(EDU_LEVELS)
N_PAYER = len(PAYER_LEVELS)


# --------------------------------------------------------------------------- #
# Stage 1: baseline livebirth rate theta_LB (Morris / de Graaf)               #
# --------------------------------------------------------------------------- #

MORRIS_THETA_LB_PER_1000 = np.array(
    [
        0.66,  # <20
        0.70,  # 20-24
        0.84,  # 25-29
        1.48,  # 30-34
        4.72,  # 35-39
        15.22,  # 40-44
        30.71,  # 45+
    ]
)

MORRIS_THETA_LB = MORRIS_THETA_LB_PER_1000 / 1000.0
MORRIS_LOGIT = logit(MORRIS_THETA_LB)
# Tightened in June after a natural-rate prior conflict. This holds the rate
# near the benchmark and transfers the age pattern to other model terms.
# It does not validate that allocation. See the dated age-gradient note.
MORRIS_SIGMA = 0.001


# --------------------------------------------------------------------------- #
# Stage 2a: detection eta_detect                                              #
# --------------------------------------------------------------------------- #

ETA_DETECT_BASELINE = 0.70
ETA_DETECT_LOGIT = logit(ETA_DETECT_BASELINE)
ETA_DETECT_SIGMA = 0.30

# Year effects covering 2016-2024 — re-anchored (2026-06-21) to the
# serum->NIPS transition (cfDNA average-risk validation ~2014-15; ACOG
# "for all patients" Sept 2020). Shape is a logistic adoption S-curve:
# reference-level effective detection ~62% in the serum-dominant era
# rising to ~81% in the NIPS-dominant years, steepest 2019-2021. This is
# the externally-anchored time structure that identifies eta vs s (s has
# no reason to track NIPS penetration), so the *shape* is kept informative
# while the overall level is loose. Implied reference-level detection shown
# inline. See notes/20260621-screening-cascade-eta-reanchoring.md.
ETA_DETECT_YEAR_OFFSETS = np.array(
    [
        -0.35,  # 2016  ~62%
        -0.25,  # 2017  ~65%
        -0.10,  # 2018  ~68%
        0.10,  # 2019  ~72%
        0.30,  # 2020  ~76%
        0.45,  # 2021  ~79%
        0.55,  # 2022  ~80%
        0.60,  # 2023  ~81%
        0.63,  # 2024  ~81%
    ]
)
ETA_DETECT_YEAR_SIGMA = 0.15

# Race. Reference = NH White.
ETA_DETECT_RACE = np.array(
    [
        0.00,  # NH White (reference)
        -0.30,  # NH Black
        -0.40,  # NH AIAN
        -0.10,  # NH Asian/Pacific Islander
        -0.25,  # Hispanic
        0.00,  # Unknown
        0.00,  # NH Multi-race (placeholder - no external evidence; neutral)
    ]
)
ETA_DETECT_RACE_SIGMA = 0.20

# Education. Reference = some college.
ETA_DETECT_EDU = np.array(
    [
        -0.45,  # <HS
        -0.20,  # HS
        0.00,  # Some college (reference)
        0.15,  # Bachelor's
        0.25,  # Master's+
        0.00,  # Unknown
    ]
)
ETA_DETECT_EDU_SIGMA = 0.20

# Payer. Reference = Private.
ETA_DETECT_PAYER = np.array(
    [
        -0.20,  # Medicaid
        0.00,  # Private (reference)
        -0.15,  # Self-pay/Other
        0.00,  # Unknown
    ]
)
ETA_DETECT_PAYER_SIGMA = 0.20

# Age effects on detection/access (2026-06-21): older mothers reach screening and
# diagnostic testing far more (the AMA trigger), so detection rises steeply with
# maternal age. Informative INCREASING prior (was a zero-mean wiggle, mu=0 / sigma
# 0.20); eta_detect now carries the dominant age gradient. Offsets on the logit,
# added to the eta_detect baseline. See
# notes/20260621-theta-lb-escape-age-gradient.md.
ETA_DETECT_AGE = np.array(
    [
        -1.5,  # <20
        -1.0,  # 20-24
        -0.4,  # 25-29
        0.3,  # 30-34
        1.0,  # 35-39
        1.6,  # 40-44
        1.9,  # 45+
    ]
)
# Tightened in June to restrict the screening/termination age trade-off.
# This constrains the allocation; it does not independently measure either stage.
# See notes/20260621-theta-lb-escape-age-gradient.md.
ETA_DETECT_AGE_SIGMA = 0.1

# Year-by-age interaction on detection (2026-06-22): lets the NIPT-era screening
# rollout differ by maternal age — the "did screening reach older mothers first?"
# question that the additive year+age structure could not express. Modelled as a
# ZERO-SUM interaction (orthogonal to the pinned year and age main effects), so it
# captures only the differential, not a shift in either margin. Sigma 0.35 is
# weakly-informative: wide enough for the ~0.1-0.3 logit age-differentials the raw
# recorded-rate trend suggests, tight enough to regularise the 9x7 cells with little
# data. The recording surface also has a year dimension. This interaction's
# interpretation is conditional on that surface and the other stage priors.
# See notes/20260622-predictors-bayesian-model.md.
ETA_DETECT_YEAR_AGE_SIGMA = 0.35


# --------------------------------------------------------------------------- #
# Stage 2b: termination eta_term                                              #
# --------------------------------------------------------------------------- #

ETA_TERM_BASELINE = 0.67  # Natoli 2012 US population-based weighted mean
ETA_TERM_LOGIT = logit(ETA_TERM_BASELINE)
# A wider termination-level prior allows more movement within the constrained
# decomposition. It does not make the level independently identified by the data.
ETA_TERM_SIGMA = 0.60

ETA_TERM_RACE = np.array(
    [
        0.00,  # NH White (reference)
        -0.70,  # NH Black
        -0.30,  # NH AIAN
        -0.15,  # NH Asian/Pacific Islander
        -0.40,  # Hispanic
        0.00,  # Unknown
        0.00,  # NH Multi-race (placeholder - no external evidence; neutral)
    ]
)
ETA_TERM_RACE_SIGMA = 0.20

ETA_TERM_EDU = np.array(
    [
        -0.30,  # <HS
        -0.10,  # HS
        0.00,  # Some college (reference)
        0.10,  # Bachelor's
        0.20,  # Master's+
        0.00,  # Unknown
    ]
)
ETA_TERM_EDU_SIGMA = 0.20

# Age effects on termination choice (2026-06-21, NEW). Termination given a
# confirmed diagnosis varies with maternal age (Natoli 2012 noted age variation).
# Modest INCREASING prior — the softest piece (the US direction is genuinely
# uncertain), wide enough for the data to refine. NB: only the COMBINED
# eta_detect*eta_term age effect is conditional on natural-rate and recording
# restrictions; the access-vs-choice split depends on separate priors. See notes/20260621-theta-lb-escape-age-gradient.md.
ETA_TERM_AGE = np.array(
    [
        -0.4,  # <20
        -0.2,  # 20-24
        -0.1,  # 25-29
        0.1,  # 30-34
        0.3,  # 35-39
        0.4,  # 40-44
        0.5,  # 45+
    ]
)
ETA_TERM_AGE_SIGMA = 0.4

# Year effect on termination: a single homoscedastic sigma absorbing
# mild year-over-year drift in termination rates. Without a separate
# policy shock to identify, we expect US termination rates conditional
# on detection to be approximately stable around the Natoli anchor.
ETA_TERM_YEAR_SIGMA = 0.15


# --------------------------------------------------------------------------- #
# Stage 3: recording sensitivity from a derived surveillance surface                                 #
# --------------------------------------------------------------------------- #

# Recording priors are derived from surveillance prevalence and the same
# recorded counts used in the likelihood. They replaced the old global pin.
# Unknown (index 5) and multi-race (index 6) lack matching surveillance inputs.
# The generated recording_anchor.py gives both weak fallback priors.
# Extrapolation uncertainty is supplied rather than measured from the tail.
# S_EDU is a within-cell residual. Clinical flags are excluded from recording
# effects because they can also relate to true disease status.
S_EDU = np.array(
    [
        -0.30,  # <HS
        -0.10,  # HS
        0.00,  # Some college (reference)
        0.10,  # Bachelor's
        0.20,  # Master's+
        0.00,  # Unknown
    ]
)
S_EDU_SIGMA = 0.05  # tightened (2026-06-21): keep s_edu from absorbing the ridge


# --------------------------------------------------------------------------- #
# Working false-positive probability per birth without Down syndrome.                                        #
# --------------------------------------------------------------------------- #

FALSE_POSITIVE_RATE = 7.8e-5


# --------------------------------------------------------------------------- #
# Bundled prior container                                                     #
# --------------------------------------------------------------------------- #


@dataclass
class ModelPriors:
    """All priors bundled for ``build_model``."""

    # Stage 1
    theta_lb_logit: np.ndarray = field(default_factory=lambda: MORRIS_LOGIT.copy())
    theta_lb_sigma: float = MORRIS_SIGMA

    # Stage 2a
    eta_detect_logit: float = ETA_DETECT_LOGIT
    eta_detect_sigma: float = ETA_DETECT_SIGMA
    eta_detect_year_offsets: np.ndarray = field(
        default_factory=lambda: ETA_DETECT_YEAR_OFFSETS.copy()
    )
    eta_detect_year_sigma: float = ETA_DETECT_YEAR_SIGMA
    eta_detect_race: np.ndarray = field(default_factory=lambda: ETA_DETECT_RACE.copy())
    eta_detect_race_sigma: float = ETA_DETECT_RACE_SIGMA
    eta_detect_edu: np.ndarray = field(default_factory=lambda: ETA_DETECT_EDU.copy())
    eta_detect_edu_sigma: float = ETA_DETECT_EDU_SIGMA
    eta_detect_payer: np.ndarray = field(
        default_factory=lambda: ETA_DETECT_PAYER.copy()
    )
    eta_detect_payer_sigma: float = ETA_DETECT_PAYER_SIGMA
    eta_detect_age: np.ndarray = field(default_factory=lambda: ETA_DETECT_AGE.copy())
    eta_detect_age_sigma: float = ETA_DETECT_AGE_SIGMA
    eta_detect_year_age_sigma: float = ETA_DETECT_YEAR_AGE_SIGMA

    # Stage 2b
    eta_term_logit: float = ETA_TERM_LOGIT
    eta_term_sigma: float = ETA_TERM_SIGMA
    eta_term_race: np.ndarray = field(default_factory=lambda: ETA_TERM_RACE.copy())
    eta_term_race_sigma: float = ETA_TERM_RACE_SIGMA
    eta_term_edu: np.ndarray = field(default_factory=lambda: ETA_TERM_EDU.copy())
    eta_term_edu_sigma: float = ETA_TERM_EDU_SIGMA
    eta_term_age: np.ndarray = field(default_factory=lambda: ETA_TERM_AGE.copy())
    eta_term_age_sigma: float = ETA_TERM_AGE_SIGMA
    eta_term_year_sigma: float = ETA_TERM_YEAR_SIGMA

    # Stage 3 (de Graaf surveillance anchor; see recording_anchor.py).
    # s_race_year_* are [N_RACE, n_year] logit mean/sigma; the model slices [:, :n_year].
    s_race_year_logit: np.ndarray = field(
        default_factory=lambda: S_RACE_YEAR_LOGIT.copy()
    )
    s_race_year_sigma: np.ndarray = field(
        default_factory=lambda: S_RACE_YEAR_SIGMA.copy()
    )
    s_edu: np.ndarray = field(default_factory=lambda: S_EDU.copy())
    s_edu_sigma: float = S_EDU_SIGMA

    # False positives
    false_positive_rate: float = FALSE_POSITIVE_RATE


# --------------------------------------------------------------------------- #
# Sensitivity-analysis variants                                                #
# --------------------------------------------------------------------------- #


def variant_A_tight_s() -> ModelPriors:
    """Tight sensitivity priors, weak termination priors."""
    p = ModelPriors()
    p.s_race_year_sigma = p.s_race_year_sigma * 0.5
    p.s_edu_sigma = S_EDU_SIGMA / 2
    p.eta_term_race_sigma = ETA_TERM_RACE_SIGMA * 2
    p.eta_term_edu_sigma = ETA_TERM_EDU_SIGMA * 2
    return p


def variant_B_tight_eta_term() -> ModelPriors:
    """Tight termination priors, weak sensitivity priors."""
    p = ModelPriors()
    p.eta_term_race_sigma = ETA_TERM_RACE_SIGMA / 2
    p.eta_term_edu_sigma = ETA_TERM_EDU_SIGMA / 2
    p.s_race_year_sigma = p.s_race_year_sigma * 2.0
    p.s_edu_sigma = S_EDU_SIGMA * 2
    return p


def variant_C_default() -> ModelPriors:
    """Main specification — both priors informative."""
    return ModelPriors()


def variant_D_recording_off() -> ModelPriors:
    """Diagnostic variant with recording near one and a classifier-based target.

    The aggregation can use C-only USBC11_M1_CN scores or quota flags. Neither
    provides a validated count of true DS cases. Scores estimate recorded status;
    probability calibration against that label does not correct missed diagnoses.
    This variant also fixes false positives at zero. Use it to compare assumptions,
    rather than as an independently corrected prevalence estimate.
    """
    p = ModelPriors()
    p.s_race_year_logit = np.full_like(S_RACE_YEAR_LOGIT, logit(0.999))
    p.s_race_year_sigma = np.full_like(S_RACE_YEAR_SIGMA, 0.001)
    p.s_edu = np.zeros(N_EDU)
    p.s_edu_sigma = 0.001
    p.false_positive_rate = 0.0
    return p


VARIANTS = {
    "A": variant_A_tight_s,
    "B": variant_B_tight_eta_term,
    "C": variant_C_default,
    "D": variant_D_recording_off,
}
