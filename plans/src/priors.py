"""
priors.py
=========

Published-literature values for the priors in the three-stage Bayesian
selection model for Down syndrome livebirth ascertainment.

Stages
------
    Stage 1  baseline livebirth rate θ_LB(age) — Morris / de Graaf
    Stage 2  detection η_detect, termination η_term — Natoli / Kuppermann / Chaiken
    Stage 3  BC sensitivity s — Boulet / Salemi

Each prior is an informative Normal distribution on the logit scale.
Citations are inline.  Sigmas reflect the precision of the external
evidence; wider sigma = more flex for the data to update.

References
----------
- Morris, J.K., Mutton, D.E. & Alberman, E. (2002). Revised estimates of the
  maternal age specific live birth prevalence of Down's syndrome.
  J Med Screen 9:2-6. [Primary reference for θ_LB; supersedes Hook 1981.]
- de Graaf, G., Buckley, F. & Skotko, B.G. (2015). Erratum: Trends in
  maternal age distribution and the live birth prevalence of Down's
  syndrome in England and Wales: 1938-2010. Eur J Hum Genet 23:1140.
  [Provides age-banded numerical values derived from Morris's formula.]
- Cuckle, H. (2021). Maternal age in the epidemiology of common autosomal
  trisomies. Prenat Diagn 41:621-629. [Confirms Morris as current standard.]
- Natoli, J.L., et al. (2012). Prenatal diagnosis of Down syndrome:
  A systematic review of termination rates (1995-2011).
  Prenat Diagn 32:142-153. [US weighted average 67%.]
- Kuppermann, M., et al. (2006). Beyond race or ethnicity and socioeconomic
  status: Predictors of prenatal testing for Down syndrome.
  Obstet Gynecol 107:1087-1097.
- de Graaf, G., Buckley, F. & Skotko, B.G. (2017). Estimation of the number
  of people with Down syndrome in the United States. Genet Med 19:439-447.
- Chaiken, S.R., et al. (2023). Association Between Rates of Down Syndrome
  Diagnosis in States With vs Without 20-Week Abortion Bans From 2011 to 2018.
  JAMA Netw Open 6:e233684.
- Boulet, S.L., et al. (2011). Sensitivity of birth certificate reports of
  birth defects in Atlanta, 1995-2005. Public Health Rep 126:186-194.
- Salemi, J.L., et al. (2017). Evaluation of the sensitivity and accuracy of
  birth defects indicators on the 2003 revision of the U.S. birth certificate.
  Paediatr Perinat Epidemiol 31:67-75.
- Egan, J.F.X., et al. (2004, 2011). Down syndrome livebirths models.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# --------------------------------------------------------------------------- #
# Utility                                                                     #
# --------------------------------------------------------------------------- #

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
#
# The model uses integer-coded categorical factors.  These lists define the
# level ordering; every prior array below is indexed to match them.  If you
# change the coding in the data pipeline, change these in lockstep.

AGE_LEVELS = ["<20", "20-24", "25-29", "30-34", "35-39", "40-44", "45+"]
RACE_LEVELS = ["NH White", "NH Black", "NH AIAN_NHOPI_Other",
               "NH Asian", "Hispanic", "Unknown"]
EDU_LEVELS = ["<HS", "HS/GED", "Some college",
              "Bachelor's", "Master's+", "Unknown"]
PAYER_LEVELS = ["Medicaid", "Private", "Self-pay/Other", "Unknown"]

N_AGE = len(AGE_LEVELS)
N_RACE = len(RACE_LEVELS)
N_EDU = len(EDU_LEVELS)
N_PAYER = len(PAYER_LEVELS)


# --------------------------------------------------------------------------- #
# Stage 1: baseline livebirth rate θ_LB (Morris / de Graaf)                   #
# --------------------------------------------------------------------------- #
#
# DS livebirth rate per 1000 livebirths IN THE ABSENCE OF SCREENING, by
# maternal age band.  Values from de Graaf, Buckley & Skotko (2015)
# corrigendum, derived from Morris et al. (2002).  These represent the
# empirical livebirth rate after natural fetal loss but before any
# screening-and-termination selection.
#
# The prior is TIGHT because Morris is a large, well-validated external
# measurement.  Letting θ_LB flex would absorb variation that properly
# belongs to η or s.

MORRIS_THETA_LB_PER_1000 = np.array([
     0.66,  # <20
     0.70,  # 20-24
     0.84,  # 25-29
     1.48,  # 30-34
     4.72,  # 35-39
    15.22,  # 40-44
    30.71,  # 45+
])

MORRIS_THETA_LB = MORRIS_THETA_LB_PER_1000 / 1000.0
MORRIS_LOGIT = logit(MORRIS_THETA_LB)
MORRIS_SIGMA = 0.10          # tight: biology + large dataset


# --------------------------------------------------------------------------- #
# Stage 2a: detection η_detect                                                #
# --------------------------------------------------------------------------- #
#
# Prenatal detection probability.  Baseline ~70% in the 2016-2024 window,
# rising with the NIPT rollout: ACOG recommended NIPT for high-risk
# women in 2012 and for all women in 2020.

ETA_DETECT_BASELINE = 0.70
ETA_DETECT_LOGIT = logit(ETA_DETECT_BASELINE)
ETA_DETECT_SIGMA = 0.30

# Year effects covering 2016-2024.  Positive = higher detection.  Centred
# around the midpoint of the window (2020) with mild monotone rise.
ETA_DETECT_YEAR_OFFSETS = np.array([
    -0.25,  # 2016
    -0.15,  # 2017
    -0.05,  # 2018
     0.05,  # 2019
     0.15,  # 2020  (ACOG recommendation extended to all women)
     0.20,  # 2021
     0.25,  # 2022
     0.28,  # 2023
     0.30,  # 2024
])
ETA_DETECT_YEAR_SIGMA = 0.15

# Race effects on detection access.  Reference = NH White.
# Negative = lower detection.  Based on Kuppermann (2006) and de Graaf
# (2017) on NIPT access differentials.
# Order: NH White, NH Black, AIAN_NHOPI_Other, NH Asian, Hispanic, Unknown
ETA_DETECT_RACE = np.array([
     0.00,  # NH White (reference)
    -0.30,  # NH Black
    -0.40,  # AIAN_NHOPI_Other
    -0.10,  # NH Asian
    -0.25,  # Hispanic
     0.00,  # Unknown
])
ETA_DETECT_RACE_SIGMA = 0.20

# Education effects on detection access.  Reference = some college.
# Order: <HS, HS, Some college, Bachelor's, Master's+, Unknown
ETA_DETECT_EDU = np.array([
    -0.45,  # <HS
    -0.20,  # HS
     0.00,  # Some college (reference)
     0.15,  # Bachelor's
     0.25,  # Master's+
     0.00,  # Unknown
])
ETA_DETECT_EDU_SIGMA = 0.20

# Payer effects.  Medicaid enrolees historically had lower NIPT uptake,
# though the gap has narrowed post-2020.  Reference = Private.
# Order: Medicaid, Private, Self-pay/Other, Unknown
ETA_DETECT_PAYER = np.array([
    -0.20,  # Medicaid
     0.00,  # Private (reference)
    -0.15,  # Self-pay/Other
     0.00,  # Unknown
])
ETA_DETECT_PAYER_SIGMA = 0.20

# Residual age effects on detection (older mothers more likely to access
# screening historically).  Centred on 0, mild prior.
ETA_DETECT_AGE_SIGMA = 0.20


# --------------------------------------------------------------------------- #
# Stage 2b: termination η_term (Natoli / Kuppermann / de Graaf / Chaiken)     #
# --------------------------------------------------------------------------- #
#
# Probability of pregnancy termination given a prenatal DS diagnosis.
# Natoli (2012) systematic review gives a US weighted average of 67%.

ETA_TERM_BASELINE = 0.67
ETA_TERM_LOGIT = logit(ETA_TERM_BASELINE)
ETA_TERM_SIGMA = 0.25

# Race effects: Kuppermann (2006) finds substantially lower termination
# among NH Black and Hispanic mothers.
ETA_TERM_RACE = np.array([
     0.00,  # NH White (reference)
    -0.70,  # NH Black  (substantially lower termination)
    -0.30,  # AIAN_NHOPI_Other
    -0.15,  # NH Asian
    -0.40,  # Hispanic
     0.00,  # Unknown
])
ETA_TERM_RACE_SIGMA = 0.20

ETA_TERM_EDU = np.array([
    -0.30,  # <HS
    -0.10,  # HS
     0.00,  # Some college (reference)
     0.10,  # Bachelor's
     0.20,  # Master's+
     0.00,  # Unknown
])
ETA_TERM_EDU_SIGMA = 0.20

# Region-year effects: pre-Dobbs variation constrained tight around zero;
# post-Dobbs (mid-2022 onwards) allowed wider variation to accommodate
# the shock for states that restricted abortion access.
ETA_TERM_REGION_YEAR_SIGMA_PRE_DOBBS = 0.15
ETA_TERM_REGION_YEAR_SIGMA_POST_DOBBS = 0.40


# --------------------------------------------------------------------------- #
# Stage 3: BC sensitivity s (Boulet / Salemi)                                 #
# --------------------------------------------------------------------------- #
#
# DS-specific sensitivity anchored midway between Boulet (~23% for combined
# defects) and Egan's back-calculated ~53% for DS specifically.  We use
# 40% as the intercept with a moderately wide prior.

S_BASELINE = 0.40
S_LOGIT = logit(S_BASELINE)
S_SIGMA = 0.30

# Race effects: Boulet found lower sensitivity for NH Black and "other
# non-Hispanic" mothers.  AORs ~0.4 -> log-odds ~-0.9, shrunk somewhat to
# account for generalisability beyond Atlanta 1995-2005.
S_RACE = np.array([
     0.00,  # NH White (reference)
    -0.40,  # NH Black
    -0.30,  # AIAN_NHOPI_Other
    -0.10,  # NH Asian
    -0.20,  # Hispanic
     0.00,  # Unknown
])
S_RACE_SIGMA = 0.25

S_EDU = np.array([
    -0.30,  # <HS
    -0.10,  # HS
     0.00,  # Some college (reference)
     0.10,  # Bachelor's
     0.20,  # Master's+
     0.00,  # Unknown
])
S_EDU_SIGMA = 0.20

# Clinical-marker effects on s.  All negative (lower recording) via the
# "workup not complete at certificate submission" mechanism.  Priors are
# wider than the demographic effects because the effect sizes are less
# directly pinned in the validation literature and may update substantially.
S_PRETERM_MU = -0.40
S_PRETERM_SIGMA = 0.20

S_CCHD_MU = -0.50
S_CCHD_SIGMA = 0.40

S_NICU_MU = -0.50
S_NICU_SIGMA = 0.40

S_AVEN_MU = -0.40
S_AVEN_SIGMA = 0.40


# --------------------------------------------------------------------------- #
# False-positive rate                                                         #
# --------------------------------------------------------------------------- #
#
# Ohio/NY false-positive study: ~7.8% of recorded DS cases were not true DS.
# Derivation: recorded rate ≈ 10 per 10,000 livebirths = 1e-3.  If 7.8% of
# these are FPs, FP rate per livebirth ≈ 0.078 * 1e-3 ≈ 7.8e-5.

FALSE_POSITIVE_RATE = 7.8e-5


# --------------------------------------------------------------------------- #
# Bundled prior container                                                     #
# --------------------------------------------------------------------------- #

@dataclass
class ModelPriors:
    """All priors bundled for passing into model.build_model()."""

    # Stage 1
    theta_lb_logit: np.ndarray = field(
        default_factory=lambda: MORRIS_LOGIT.copy())
    theta_lb_sigma: float = MORRIS_SIGMA

    # Stage 2a: detection
    eta_detect_logit: float = ETA_DETECT_LOGIT
    eta_detect_sigma: float = ETA_DETECT_SIGMA
    eta_detect_year_offsets: np.ndarray = field(
        default_factory=lambda: ETA_DETECT_YEAR_OFFSETS.copy())
    eta_detect_year_sigma: float = ETA_DETECT_YEAR_SIGMA
    eta_detect_race: np.ndarray = field(
        default_factory=lambda: ETA_DETECT_RACE.copy())
    eta_detect_race_sigma: float = ETA_DETECT_RACE_SIGMA
    eta_detect_edu: np.ndarray = field(
        default_factory=lambda: ETA_DETECT_EDU.copy())
    eta_detect_edu_sigma: float = ETA_DETECT_EDU_SIGMA
    eta_detect_payer: np.ndarray = field(
        default_factory=lambda: ETA_DETECT_PAYER.copy())
    eta_detect_payer_sigma: float = ETA_DETECT_PAYER_SIGMA
    eta_detect_age_sigma: float = ETA_DETECT_AGE_SIGMA

    # Stage 2b: termination
    eta_term_logit: float = ETA_TERM_LOGIT
    eta_term_sigma: float = ETA_TERM_SIGMA
    eta_term_race: np.ndarray = field(
        default_factory=lambda: ETA_TERM_RACE.copy())
    eta_term_race_sigma: float = ETA_TERM_RACE_SIGMA
    eta_term_edu: np.ndarray = field(
        default_factory=lambda: ETA_TERM_EDU.copy())
    eta_term_edu_sigma: float = ETA_TERM_EDU_SIGMA
    eta_term_ry_sigma_pre_dobbs: float = ETA_TERM_REGION_YEAR_SIGMA_PRE_DOBBS
    eta_term_ry_sigma_post_dobbs: float = ETA_TERM_REGION_YEAR_SIGMA_POST_DOBBS

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
# Sensitivity-analysis variants (§8 of the report)                            #
# --------------------------------------------------------------------------- #

def variant_A_tight_s() -> ModelPriors:
    """
    Variant A: tight sensitivity priors, weak termination priors.

    Tests whether BC sensitivity drives the observed demographic differential.
    If posterior demographic contrasts load onto s under this variant, Boulet
    is doing most of the work in the decomposition.
    """
    p = ModelPriors()
    p.s_race_sigma = S_RACE_SIGMA / 2
    p.s_edu_sigma = S_EDU_SIGMA / 2
    p.eta_term_race_sigma = ETA_TERM_RACE_SIGMA * 2
    p.eta_term_edu_sigma = ETA_TERM_EDU_SIGMA * 2
    return p


def variant_B_tight_eta_term() -> ModelPriors:
    """
    Variant B: tight termination priors, weak sensitivity priors.

    Tests whether termination-rate variation drives the observed demographic
    differential.
    """
    p = ModelPriors()
    p.eta_term_race_sigma = ETA_TERM_RACE_SIGMA / 2
    p.eta_term_edu_sigma = ETA_TERM_EDU_SIGMA / 2
    p.s_race_sigma = S_RACE_SIGMA * 2
    p.s_edu_sigma = S_EDU_SIGMA * 2
    return p


def variant_C_default() -> ModelPriors:
    """Variant C: the main specification with both priors informative."""
    return ModelPriors()


def variant_D_dobbs_only() -> ModelPriors:
    """
    Variant D: uninformative race/education priors on termination.

    The model must identify termination effects through the Dobbs region x
    year interaction alone.  If posterior race/education effects under this
    variant agree with variant C, that is genuine data-driven identification
    rather than prior-driven.
    """
    p = ModelPriors()
    # Shrink race/edu prior means to zero with wide sigmas
    p.eta_term_race = np.zeros_like(p.eta_term_race)
    p.eta_term_race_sigma = 1.0
    p.eta_term_edu = np.zeros_like(p.eta_term_edu)
    p.eta_term_edu_sigma = 1.0
    # Widen the post-Dobbs region-year allowance to capture the shock
    p.eta_term_ry_sigma_post_dobbs = 0.60
    return p


VARIANTS = {
    "A": variant_A_tight_s,
    "B": variant_B_tight_eta_term,
    "C": variant_C_default,
    "D": variant_D_dobbs_only,
}


if __name__ == "__main__":
    # Quick print of the prior-implied baseline rates for sanity checking
    print("Morris / de Graaf θ_LB age-specific rates (per 1,000):")
    for age, rate in zip(AGE_LEVELS, MORRIS_THETA_LB_PER_1000):
        print(f"  {age:>7s}: {rate:6.2f}")
    print()
    print(f"Natoli termination baseline: {ETA_TERM_BASELINE:.2%}")
    print(f"Egan-anchored BC sensitivity: {S_BASELINE:.2%}")
    print(f"Ohio/NY false-positive rate:  {FALSE_POSITIVE_RATE:.2e}")
