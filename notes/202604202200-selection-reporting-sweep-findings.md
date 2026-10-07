> [!NOTE]
> AI-assisted revision by Codex (GPT-6), 7 October 2026.

> [!WARNING]
> This note was drafted by an AI coding assistant (Claude) from an
> autonomous session on 2026-04-20. Figures below reflect the four
> reporting-profile fits completed at 18:27–21:42 on 2026-04-20. A
> human reviewer should verify the interpretation and decide on the
> next design iteration before these numbers are cited anywhere.

> [!NOTE]
> This sweep ran against an earlier model specification that carried a
> pre/post-2022 asymmetric sigma on `eta_term_year` and a fourth
> variant (D) built around that mechanism. That year asymmetry has
> since been removed; the fourth variant no longer exists. The intercept-level
> findings below, θ_LB above Morris, s below the external sensitivity
> evidence, and the resulting incompatibility with surveillance, are
> not specific to that removed mechanism and still apply to the current
> A/B/C variants.

> [!NOTE]
> **Correction (2026-08-04).** This note graded `s_int` against "Boulet ≈ 40%
> sensitivity". That figure appears nowhere in Boulet, who reports **18%** for
> Down syndrome in metropolitan Atlanta, and is withdrawn. The comparison itself
> survives: both validation studies were run in low-recording areas, and
> transported to national recording level they give `0.374` and `0.319`. A
> posterior sensitivity of 4–7% remains incompatible with the external evidence
> by a wide margin, so every conclusion here stands, only the attribution
> changes. See
> [the study-area transport note](20260804-salemi-boulet-study-area-transport.md).

# April selection-model reporting sweep

**Original date:** 20 April 2026. Historical research note, revised for clarity in October 2026.

This sweep compared the early A, B and C prior variants. The following tables preserve the run record. Later changes to the natural-rate prior, age terms and recording anchor supersede these estimates.

## Original run evidence

| variant | output dir | runtime |
|---|---|---:|
| A | `output/selection/A/full/20260420-182719/` | ~52 min |
| B | `output/selection/B/full/20260420-191926/` | ~49 min |
| C | `output/selection/C/full/20260420-200804/` | ~47 min |

| variant | n RVs | max R̂ | min ESS bulk | min ESS tail | gates |
|---|---:|---:|---:|---:|---|
| A | 79 | 1.000 | 806 | 1559 | **PASS** |
| B | 79 | 1.030 | 201 | 298 | **FAIL** (R̂ > 1.01, ESS < 400) |
| C | 79 | 1.020 | 491 | 973 | **FAIL** (R̂ > 1.01) |

|   | A | B | C |
|---|---:|---:|---:|
| NH White | 0.160 | 0.019 | 0.068 |
| NH Black | 0.202 | 0.036 | 0.081 |
| NH AIAN/NHOPI/Other | 0.103 | 0.012 | 0.085 |
| NH Asian | 0.280 | 0.055 | 0.147 |
| Hispanic | 0.164 | 0.018 | 0.049 |
| Unknown | 0.230 | 0.047 | 0.128 |

|   | A (tight s) | B (tight η_term) | C (default) |
|---|---:|---:|---:|
| NH White | −0.74 | −0.07 | −0.27 |
| NH Black | −1.04 | −0.73 | −0.80 |
| NH AIAN/NHOPI/Other | −0.56 | −0.30 | −0.32 |
| NH Asian | +0.53 | −0.12 | +0.01 |
| Hispanic | −0.57 | −0.42 | −0.46 |
| Unknown | −0.21 | −0.01 | −0.02 |

|   | A | B | C |
|---|---:|---:|---:|
| NH White | −0.03 | −0.75 | −0.23 |
| NH Black | −0.76 | −1.53 | −0.99 |
| NH AIAN/NHOPI/Other | +0.05 | −0.30 | +0.13 |
| NH Asian | −0.66 | −1.71 | −1.11 |
| Hispanic | +0.01 | −0.76 | −0.23 |
| Unknown | −0.13 | −0.98 | −0.41 |

| RV | prior μ | prior σ | A | B | C | max |z| |
|---|---:|---:|---:|---:|---:|---:|
| `eta_detect_int` | +0.85 | 0.30 | −0.53 | −0.93 | −0.84 | **5.9** |
| `eta_term_int` | +0.71 | 0.25 | +0.30 | +0.17 | +0.21 | **2.0** |
| `s_int` | −0.41 | 0.30 | **−3.40** | −2.23 | −3.08 | **10.0** |

The named variants change which part of a product the model constrains most tightly. Similar totals across variants do not establish external accuracy when variants share the same misspecified structure.

The original interpretation graded recording estimates against a supposed Boulet estimate near 0.40. That attribution was wrong. [Boulet's Atlanta study](https://pmc.ncbi.nlm.nih.gov/articles/PMC3056031/) reports 18.1% sensitivity for Down syndrome. The [study-area note](20260804-salemi-boulet-study-area-transport.md) distinguishes the measured local rate from a conditional national rescaling.

The current variant D is a diagnostic based on classifier probabilities. It is not a validated count of true cases. See the [current variant definitions](../src/dspopulations_us_birth_certificates/selection/README.md).
