"""Published-literature priors for the three-stage selection model.

Stage 1 — baseline DS livebirth rate ``theta_lb(age)`` from Morris/de Graaf.
Stage 2a — prenatal detection ``eta_detect`` from Kuppermann (2006) and
    NIPT rollout evidence.
Stage 2b — termination given diagnosis ``eta_term`` from Natoli (2012) and
    Kuppermann (2006), with a homoscedastic year sigma to absorb mild
    year-over-year drift.
Stage 3 — birth-certificate sensitivity ``s`` from Boulet (2011) and Salemi
    (2017).

Each prior is an informative Normal on the logit scale. Sigmas reflect the
precision of the external evidence. Three sensitivity variants follow
(A/B/C) — see ``docs/bayesian_selection_model.md`` section 8.

References
----------
- Morris, J.K., Mutton, D.E. & Alberman, E. (2002). J Med Screen 9:2-6.
- de Graaf, G., Buckley, F. & Skotko, B.G. (2015). EJHG 23:1140 (corrigendum).
- Cuckle, H. (2021). Prenat Diagn 41:621-629.
- Natoli, J.L., et al. (2012). Prenat Diagn 32:142-153.
- Kuppermann, M., et al. (2006). Obstet Gynecol 107:1087-1097.
- de Graaf, G., Buckley, F. & Skotko, B.G. (2017). Genet Med 19:439-447.
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
# Race vocabulary, aligned to the mracehisp_c coding produced by
# duckdb_prepare.py and consumed by selection.data.RACE_MAP. The codes
# are: 1=NH White (idx 0), 2=NH Black (1), 3=NH AIAN only (2), 4=NH
# Asian/Pacific Islander (broad bucket — Asian + NHOPI + Other; 3),
# 5=Hispanic (4), NULL=Unknown (5). The prior arrays below are indexed
# in this order — re-derive ETA_TERM_RACE / S_RACE etc. against the
# published literature if any single demographic's prior magnitude looks
# off; an earlier version of this label list swapped positions 2 and 3
# and the values may need a second look.
RACE_LEVELS = [
    "NH White",
    "NH Black",
    "NH AIAN",
    "NH Asian/Pacific Islander",
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
# Pinned hard (2026-06-21). At 33.5M rows a sigma=0.10 logit prior behaves like
# data, letting theta_LB drift 11-15 sigma to absorb the screening/termination
# age signal (total inflated ~5x). Morris is the EXTERNAL conception-rate anchor;
# pin it so the maternal-age gradient lands in eta. See
# notes/20260621-theta-lb-escape-age-gradient.md.
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
# Tightened 0.5 -> 0.1 (2026-06-22): with both eta_detect_age and eta_term_age
# loose, only their PRODUCT (the combined age effect on eta) is identified, so the
# sampler wandered the ridge and variant A failed to converge (r-hat 1.73, ESS 6 at
# the 25-29 band). Pin the screening-access age effect (well anchored to AMA uptake)
# and let eta_term_age carry the data-identified residual. See
# notes/20260621-theta-lb-escape-age-gradient.md.
ETA_DETECT_AGE_SIGMA = 0.1

# Year-by-age interaction on detection (2026-06-22): lets the NIPT-era screening
# rollout differ by maternal age — the "did screening reach older mothers first?"
# question that the additive year+age structure could not express. Modelled as a
# ZERO-SUM interaction (orthogonal to the pinned year and age main effects), so it
# captures only the differential, not a shift in either margin. Sigma 0.35 is
# weakly-informative: wide enough for the ~0.1-0.3 logit age-differentials the raw
# recorded-rate trend suggests, tight enough to regularise the 9x7 cells with little
# data. The year dimension is clean (s has no year term) so the interaction is
# data-identified. See notes/20260622-predictors-bayesian-model.md sec. 8.
ETA_DETECT_YEAR_AGE_SIGMA = 0.35


# --------------------------------------------------------------------------- #
# Stage 2b: termination eta_term                                              #
# --------------------------------------------------------------------------- #

ETA_TERM_BASELINE = 0.67  # Natoli 2012 US population-based weighted mean
ETA_TERM_LOGIT = logit(ETA_TERM_BASELINE)
# Data-identified level (2026-06-21): widened 0.25 -> 0.60 so the US
# termination-given-diagnosis *level* is set by the data (under the pin-s
# identification), not by the prior. Centre stays at Natoli's 67% (US,
# population-based, heterogeneous); the ~90% in the literature is European/
# hospital-based and is deliberately NOT imported. The time-varying engine
# is eta_detect (NIPS detection), NOT eta_term — evidence shows
# termination|diagnosis is flat-to-declining, not rising, with NIPS
# (Lund 2021, Miltoft 2018), so the year effect below stays a zero-mean
# drift. See notes/20260621-screening-cascade-eta-reanchoring.md.
ETA_TERM_SIGMA = 0.60

ETA_TERM_RACE = np.array(
    [
        0.00,  # NH White (reference)
        -0.70,  # NH Black
        -0.30,  # NH AIAN
        -0.15,  # NH Asian/Pacific Islander
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

# Age effects on termination choice (2026-06-21, NEW). Termination given a
# confirmed diagnosis varies with maternal age (Natoli 2012 noted age variation).
# Modest INCREASING prior — the softest piece (the US direction is genuinely
# uncertain), wide enough for the data to refine. NB: only the COMBINED
# eta_detect*eta_term age effect is data-identified; the access-vs-choice split is
# prior-driven. See notes/20260621-theta-lb-escape-age-gradient.md.
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
# Stage 3: BC sensitivity s (Boulet / Salemi)                                 #
# --------------------------------------------------------------------------- #

S_BASELINE = 0.40
S_LOGIT = logit(S_BASELINE)
# Pinned HARD (2026-06-21): sigma=0.10 was still overwhelmed (s_int escaped ~7 sigma
# to ~0.24 along the eta*s ridge), so pin the recording LEVEL at the Boulet/Salemi
# validation value (~0.40) with sigma=0.001, like theta_LB. With s pinned the age-
# graded reduction lands in eta. The level trades off validation (s~0.40 -> total
# ~38k) vs surveillance (total ~45k -> s~0.34). See
# notes/20260621-theta-lb-escape-age-gradient.md.
S_SIGMA = 0.001

S_RACE = np.array(
    [
        0.00,  # NH White (reference)
        -0.40,  # NH Black
        -0.30,  # NH AIAN
        -0.10,  # NH Asian/Pacific Islander
        -0.20,  # Hispanic
        0.00,  # Unknown
    ]
)
S_RACE_SIGMA = 0.05  # tightened (2026-06-21): keep s_race from absorbing the ridge

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

# Clinical-flag recording effects (preterm/CCHD/NICU/Aven) were DROPPED from s on
# 2026-06-21. They blew up positive (s_nicu +5.8) because CCHD/NICU correlate with
# true DS prevalence (invariant #2 is backwards) and the model could only express
# that through recording. The flags remain available as the Aim-4 co-occurring-
# conditions analysis (diagnostics.cchd_consistency_*), where they are an OUTCOME of
# true DS, not a recording covariate. See notes/20260621-theta-lb-escape-age-gradient.md.


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
    eta_detect_age: np.ndarray = field(
        default_factory=lambda: ETA_DETECT_AGE.copy()
    )
    eta_detect_age_sigma: float = ETA_DETECT_AGE_SIGMA
    eta_detect_year_age_sigma: float = ETA_DETECT_YEAR_AGE_SIGMA

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
    eta_term_age: np.ndarray = field(
        default_factory=lambda: ETA_TERM_AGE.copy()
    )
    eta_term_age_sigma: float = ETA_TERM_AGE_SIGMA
    eta_term_year_sigma: float = ETA_TERM_YEAR_SIGMA

    # Stage 3
    s_logit: float = S_LOGIT
    s_sigma: float = S_SIGMA
    s_race: np.ndarray = field(default_factory=lambda: S_RACE.copy())
    s_race_sigma: float = S_RACE_SIGMA
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


def variant_D_rprime() -> ModelPriors:
    """Comparative track: fit R' = recorded + predicted-missing, recording off.

    R' is the project's ``down_ind = 1 OR ds_pred_missing`` union -- recorded cases
    plus the likely-missed cases flagged by the **C-only-trained, demographically
    blind** USBC11_M1_CN model (``ds_pred_missing_14``). Note: "predicted" refers to
    the predicted-missing *flag*, NOT a "C+P" training label -- the underlying GB is
    trained confirmed-only, and drops race/education/payer as features.

    The recording stage is pinned to ~1 (s_int -> logit(0.999), no demographic
    offsets) and the false-positive rate to 0, so the model decomposes R' directly
    into natural rate x survival -- the recording-vs-termination non-identifiability
    that needs the A/B/C bound does not arise here. Pair with
    ``prepare_cells(missing_flag_column="ds_pred_missing_14")``. See
    notes/20260622-predictors-bayesian-model.md.
    """
    p = ModelPriors()
    p.s_logit = logit(0.999)
    p.s_sigma = 0.001
    p.s_race = np.zeros(N_RACE)
    p.s_race_sigma = 0.001
    p.s_edu = np.zeros(N_EDU)
    p.s_edu_sigma = 0.001
    p.false_positive_rate = 0.0
    return p


VARIANTS = {
    "A": variant_A_tight_s,
    "B": variant_B_tight_eta_term,
    "C": variant_C_default,
    "D": variant_D_rprime,
}
