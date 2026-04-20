"""Published-literature priors for the three-stage selection model.

Stage 1 — baseline DS livebirth rate ``theta_lb(age)`` from Morris/de Graaf.
Stage 2a — prenatal detection ``eta_detect`` from Kuppermann (2006) and
    NIPT rollout evidence.
Stage 2b — termination given diagnosis ``eta_term`` from Natoli (2012) and
    Kuppermann (2006), with a heterogeneous region×year sigma pinned at the
    Dobbs boundary (mid-2022).
Stage 3 — birth-certificate sensitivity ``s`` from Boulet (2011) and Salemi
    (2017).

Each prior is an informative Normal on the logit scale. Sigmas reflect the
precision of the external evidence. Four sensitivity variants follow
(A/B/C/D) — see ``docs/bayesian_selection_model.md`` section 8.

References
----------
- Morris, J.K., Mutton, D.E. & Alberman, E. (2002). J Med Screen 9:2-6.
- de Graaf, G., Buckley, F. & Skotko, B.G. (2015). EJHG 23:1140 (corrigendum).
- Cuckle, H. (2021). Prenat Diagn 41:621-629.
- Natoli, J.L., et al. (2012). Prenat Diagn 32:142-153.
- Kuppermann, M., et al. (2006). Obstet Gynecol 107:1087-1097.
- de Graaf, G., Buckley, F. & Skotko, B.G. (2017). Genet Med 19:439-447.
- Chaiken, S.R., et al. (2023). JAMA Netw Open 6:e233684.
- Boulet, S.L., et al. (2011). Public Health Rep 126:186-194.
- Salemi, J.L., et al. (2017). Paediatr Perinat Epidemiol 31:67-75.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def logit(p):
    """Logit transform, safe for arrays."""
    p = np.asarray(p, dtype=float)
    return np.log(p / (1.0 - p))


def inv_logit(x):
    """Inverse logit (sigmoid)."""
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


# --------------------------------------------------------------------------- #
# Factor-level vocabularies                                                   #
# --------------------------------------------------------------------------- #

AGE_LEVELS = ["<20", "20-24", "25-29", "30-34", "35-39", "40-44", "45+"]
RACE_LEVELS = [
    "NH White",
    "NH Black",
    "NH AIAN_NHOPI_Other",
    "NH Asian",
    "Hispanic",
    "Unknown",
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
MORRIS_SIGMA = 0.10  # tight: biology + large dataset


# --------------------------------------------------------------------------- #
# Stage 2a: detection eta_detect                                              #
# --------------------------------------------------------------------------- #

ETA_DETECT_BASELINE = 0.70
ETA_DETECT_LOGIT = logit(ETA_DETECT_BASELINE)
ETA_DETECT_SIGMA = 0.30

# Year effects covering 2016-2024 (centred around 2020; NIPT rollout).
ETA_DETECT_YEAR_OFFSETS = np.array(
    [
        -0.25,  # 2016
        -0.15,  # 2017
        -0.05,  # 2018
        0.05,  # 2019
        0.15,  # 2020
        0.20,  # 2021
        0.25,  # 2022
        0.28,  # 2023
        0.30,  # 2024
    ]
)
ETA_DETECT_YEAR_SIGMA = 0.15

# Race. Reference = NH White.
ETA_DETECT_RACE = np.array(
    [
        0.00,  # NH White (reference)
        -0.30,  # NH Black
        -0.40,  # AIAN_NHOPI_Other
        -0.10,  # NH Asian
        -0.25,  # Hispanic
        0.00,  # Unknown
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

# Residual age effects on detection (older mothers more likely to access
# screening historically).
ETA_DETECT_AGE_SIGMA = 0.20


# --------------------------------------------------------------------------- #
# Stage 2b: termination eta_term                                              #
# --------------------------------------------------------------------------- #

ETA_TERM_BASELINE = 0.67
ETA_TERM_LOGIT = logit(ETA_TERM_BASELINE)
ETA_TERM_SIGMA = 0.25

ETA_TERM_RACE = np.array(
    [
        0.00,  # NH White (reference)
        -0.70,  # NH Black
        -0.30,  # AIAN_NHOPI_Other
        -0.15,  # NH Asian
        -0.40,  # Hispanic
        0.00,  # Unknown
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

# Year effects on termination: pre-Dobbs tight, post-Dobbs (mid-2022+)
# wider. Originally specified as a region×year interaction, but the
# project DuckDB has no state-level column so the model uses a year-only
# term. Dobbs identification is weaker as a result — the contrast is
# between the post-2022 national mean and the pre-2022 national mean
# rather than treated-vs-untreated-state differential.
ETA_TERM_YEAR_SIGMA_PRE_DOBBS = 0.15
ETA_TERM_YEAR_SIGMA_POST_DOBBS = 0.40


# --------------------------------------------------------------------------- #
# Stage 3: BC sensitivity s (Boulet / Salemi)                                 #
# --------------------------------------------------------------------------- #

S_BASELINE = 0.40
S_LOGIT = logit(S_BASELINE)
S_SIGMA = 0.30

S_RACE = np.array(
    [
        0.00,  # NH White (reference)
        -0.40,  # NH Black
        -0.30,  # AIAN_NHOPI_Other
        -0.10,  # NH Asian
        -0.20,  # Hispanic
        0.00,  # Unknown
    ]
)
S_RACE_SIGMA = 0.25

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
S_EDU_SIGMA = 0.20

S_PRETERM_MU = -0.40
S_PRETERM_SIGMA = 0.20

S_CCHD_MU = -0.50
S_CCHD_SIGMA = 0.40

S_NICU_MU = -0.50
S_NICU_SIGMA = 0.40

S_AVEN_MU = -0.40
S_AVEN_SIGMA = 0.40


# --------------------------------------------------------------------------- #
# False-positive rate (Ohio/NY study).                                        #
# --------------------------------------------------------------------------- #

FALSE_POSITIVE_RATE = 7.8e-5


# --------------------------------------------------------------------------- #
# Bundled prior container                                                     #
# --------------------------------------------------------------------------- #


@dataclass
class ModelPriors:
    """All priors bundled for ``build_model``."""

    # Stage 1
    theta_lb_logit: np.ndarray = field(
        default_factory=lambda: MORRIS_LOGIT.copy()
    )
    theta_lb_sigma: float = MORRIS_SIGMA

    # Stage 2a
    eta_detect_logit: float = ETA_DETECT_LOGIT
    eta_detect_sigma: float = ETA_DETECT_SIGMA
    eta_detect_year_offsets: np.ndarray = field(
        default_factory=lambda: ETA_DETECT_YEAR_OFFSETS.copy()
    )
    eta_detect_year_sigma: float = ETA_DETECT_YEAR_SIGMA
    eta_detect_race: np.ndarray = field(
        default_factory=lambda: ETA_DETECT_RACE.copy()
    )
    eta_detect_race_sigma: float = ETA_DETECT_RACE_SIGMA
    eta_detect_edu: np.ndarray = field(
        default_factory=lambda: ETA_DETECT_EDU.copy()
    )
    eta_detect_edu_sigma: float = ETA_DETECT_EDU_SIGMA
    eta_detect_payer: np.ndarray = field(
        default_factory=lambda: ETA_DETECT_PAYER.copy()
    )
    eta_detect_payer_sigma: float = ETA_DETECT_PAYER_SIGMA
    eta_detect_age_sigma: float = ETA_DETECT_AGE_SIGMA

    # Stage 2b
    eta_term_logit: float = ETA_TERM_LOGIT
    eta_term_sigma: float = ETA_TERM_SIGMA
    eta_term_race: np.ndarray = field(
        default_factory=lambda: ETA_TERM_RACE.copy()
    )
    eta_term_race_sigma: float = ETA_TERM_RACE_SIGMA
    eta_term_edu: np.ndarray = field(
        default_factory=lambda: ETA_TERM_EDU.copy()
    )
    eta_term_edu_sigma: float = ETA_TERM_EDU_SIGMA
    eta_term_year_sigma_pre_dobbs: float = ETA_TERM_YEAR_SIGMA_PRE_DOBBS
    eta_term_year_sigma_post_dobbs: float = ETA_TERM_YEAR_SIGMA_POST_DOBBS

    # Stage 3
    s_logit: float = S_LOGIT
    s_sigma: float = S_SIGMA
    s_race: np.ndarray = field(default_factory=lambda: S_RACE.copy())
    s_race_sigma: float = S_RACE_SIGMA
    s_edu: np.ndarray = field(default_factory=lambda: S_EDU.copy())
    s_edu_sigma: float = S_EDU_SIGMA
    s_preterm_mu: float = S_PRETERM_MU
    s_preterm_sigma: float = S_PRETERM_SIGMA
    s_cchd_mu: float = S_CCHD_MU
    s_cchd_sigma: float = S_CCHD_SIGMA
    s_nicu_mu: float = S_NICU_MU
    s_nicu_sigma: float = S_NICU_SIGMA
    s_aven_mu: float = S_AVEN_MU
    s_aven_sigma: float = S_AVEN_SIGMA

    # False positives
    false_positive_rate: float = FALSE_POSITIVE_RATE


# --------------------------------------------------------------------------- #
# Sensitivity-analysis variants                                                #
# --------------------------------------------------------------------------- #


def variant_A_tight_s() -> ModelPriors:
    """Tight sensitivity priors, weak termination priors."""
    p = ModelPriors()
    p.s_race_sigma = S_RACE_SIGMA / 2
    p.s_edu_sigma = S_EDU_SIGMA / 2
    p.eta_term_race_sigma = ETA_TERM_RACE_SIGMA * 2
    p.eta_term_edu_sigma = ETA_TERM_EDU_SIGMA * 2
    return p


def variant_B_tight_eta_term() -> ModelPriors:
    """Tight termination priors, weak sensitivity priors."""
    p = ModelPriors()
    p.eta_term_race_sigma = ETA_TERM_RACE_SIGMA / 2
    p.eta_term_edu_sigma = ETA_TERM_EDU_SIGMA / 2
    p.s_race_sigma = S_RACE_SIGMA * 2
    p.s_edu_sigma = S_EDU_SIGMA * 2
    return p


def variant_C_default() -> ModelPriors:
    """Main specification — both priors informative."""
    return ModelPriors()


def variant_D_dobbs_only() -> ModelPriors:
    """Uninformative race/education priors on termination.

    Termination race/education effects get shrunk to zero with wide
    sigma; Dobbs identification rests on the pre/post-2022 national
    year shift alone. Agreement with Variant C is evidence of
    data-driven identification rather than prior-driven decomposition
    — though without state-level contrast the test is weaker than in
    the original plan (§4.3, §10 #5).
    """
    p = ModelPriors()
    p.eta_term_race = np.zeros_like(p.eta_term_race)
    p.eta_term_race_sigma = 1.0
    p.eta_term_edu = np.zeros_like(p.eta_term_edu)
    p.eta_term_edu_sigma = 1.0
    p.eta_term_year_sigma_post_dobbs = 0.60
    return p


VARIANTS = {
    "A": variant_A_tight_s,
    "B": variant_B_tight_eta_term,
    "C": variant_C_default,
    "D": variant_D_dobbs_only,
}
