> [!NOTE]
> AI-assisted revision by Codex (GPT-6), 7 October 2026.

# Screening and termination priors

**Original date:** 21 June 2026. Historical research note, revised for clarity in October 2026.

This note proposed replacing a single reduction parameter with a screening and termination cascade. The selection model retains that structure. The core DSP models instead estimate combined reduction or a surveillance-anchored prevalence trajectory.

## Definitions

Let `d` be the probability that prenatal screening and diagnosis identify a Down syndrome pregnancy, conditional on the relevant pregnancy population. Let `t` be the probability of termination after diagnosis. The retained fraction is `eta = 1 - d * t` in this simplified model.

`eta` is a reduction multiplier applied to a counterfactual live-birth rate. It is not survival from conception, because the counterfactual live-birth rate already includes spontaneous fetal loss. Screening uptake, test sensitivity, diagnostic follow-up and termination must have compatible denominators before they are multiplied.

## Original source summary

| Quantity (T21) | Serum screening | cfDNA / NIPS |
|---|---|---|
| Detection rate (sensitivity) | combined FTS ~85%; quad ~70–81% at 5% FPR (standard screening lit; see §6) | **99.2%** (95% CI 98.5–99.6), Gil 2015; **99.4%**, Mackie 2017; **99.3%** general-population, Iwarsson 2017 |
| False-positive rate | ~5% (by design of the risk cut-off) | **0.09%**, Gil 2015 |
| **PPV (true positives / screen positives)** | **~0.78%** head-to-head, Xiao 2022; generally low single digits | **81.5%** head-to-head, Xiao 2022; **85.7–90.8%** SNP-NIPT high-risk, Verma 2018 / Eiben 2015 |

These source summaries describe different tests and study populations. They are prior inputs, not measurements of the national 2016 to 2024 cascade. Positive predictive value describes how often a positive test corresponds to the condition. It is not the detection probability among affected pregnancies.

Recorded certificate counts cannot separate screening from termination: their likelihood contains their product within `eta`, multiplied by recording sensitivity. Age and demographic terms do not remove that limit by themselves.

Use the source citations in [selection priors](../src/dspopulations_us_birth_certificates/selection/priors.py), and check population, year and outcome before changing a prior. The [current model guide](../docs/models/README.md) describes which quantities each family estimates. The original implementation task list is superseded.
