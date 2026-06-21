# θ_LB escapes its prior — the recorded-DS age gradient has nowhere to go

**Date:** 2026-06-21
**Status:** Diagnosis (April pathology reproduced + extended). Candidate fixes are
decisions for FB — they touch README invariant #1 and the age structure.
**Relates to:** `202604202200-selection-reporting-sweep-findings.md` (original
intercept-level diagnosis); `20260621-screening-cascade-eta-reanchoring.md` (the η
re-anchoring, which is orthogonal to this and not the cause).

## TL;DR

The A/B/C **dev** sweep (full spec, with the re-anchored η priors) reproduces the
April pathology: posterior **θ_LB sits 3–4× above the Morris prior (11–15σ on the
logit)**, collapsing aggregate **s to ~6.7%** and inflating **total DS livebirths to
~226–247k** (2016–2024) — about **5× the surveillance figure** (~42–51k; ~1.3e-3
per birth). The η re-anchoring did **not** cause this (the April reporting sweep got
~231k *before* any re-anchoring); it is structural.

Per-variant dev totals (posterior mean [95% CI]) and aggregates:

| Variant | total_true | agg η | reduction (1−η) | agg s | max r̂ | min ESS |
|---|---|---|---|---|---|---|
| A (tight s) | 225,999 [208,870–242,706] | 0.855 | 0.145 | 0.067 | 1.01 | 417 |
| B (tight η_term) | 247,382 [225,508–269,935] | 0.936 | 0.064 | 0.062 | 1.02 | 82 |
| C (default) | 235,912 [219,586–254,246] | 0.892 | 0.108 | 0.064 | 1.01 | 215 |

All three share the θ_LB escape, so the pin-s A-vs-B contrast is **not yet
testable** — the headline is dominated by the pathology, not the prior weighting.

## New this round: the age-gradient mechanism

θ_LB posterior vs Morris prior (variant A; invlogit of `theta_lb_age`):

| age | Morris prior | posterior | (post−prior)/σ |
|---|---|---|---|
| <20 | 0.00066 | 0.00197 | +10.9σ |
| 20–24 | 0.00070 | 0.00301 | +14.6σ |
| 25–29 | 0.00084 | 0.00350 | +14.3σ |
| 30–34 | 0.00148 | 0.00548 | +13.1σ |
| 35–39 | 0.00472 | 0.01749 | +13.2σ |
| 40–44 | 0.01522 | 0.05249 | +12.8σ |
| 45+ | 0.03071 | 0.05324 | +5.7σ |

Observed recorded-DS rate by age vs Morris natural rate:

| age | N frac | R | R/N | Morris | (R/N)/Morris = η·s |
|---|---|---|---|---|---|
| <20 | 0.045 | 433 | 2.89e-4 | 6.6e-4 | 0.438 |
| 20–24 | 0.184 | 1638 | 2.65e-4 | 7.0e-4 | 0.379 |
| 25–29 | 0.283 | 2613 | 2.76e-4 | 8.4e-4 | 0.328 |
| 30–34 | 0.295 | 3652 | 3.69e-4 | 1.48e-3 | 0.249 |
| 35–39 | 0.156 | 5326 | 1.02e-3 | 4.72e-3 | 0.216 |
| 40–44 | 0.034 | 3711 | 3.30e-3 | 1.52e-2 | 0.217 |
| 45+ | 0.003 | 403 | 4.54e-3 | 3.07e-2 | 0.148 |

- **Observed recorded gradient: 15.7×** (45+/<20). **Morris natural: 46.5×.**
- `(R/N)/Morris = η·s` falls monotonically **0.44 → 0.15** with maternal age.

The data demand a steep decline in η·s with maternal age — older mothers
screen/terminate far more, so a smaller fraction of their (much higher) natural DS
risk reaches a recorded livebirth. But the model has nowhere to put that decline:

- **`s` has no age term** (no `s_age`; clinical flags are barred from π by invariant #2).
- **`η_term` has no age term**; **`η_detect_age` is σ=0.20** (range only ~±0.4 logit).

So the only per-age knob with real range is **θ_LB(age)** — and with N in the
millions per age band, the per-age likelihood (Fisher information ~10⁵ on logit)
crushes the σ=0.10 Morris prior. θ_LB(age) bends to fit the observed gradient
(flattening from 46.5× toward ~27×, rising ~3× at young ages), and s collapses to
hold the product at R/N.

## What a correctly-pinned model needs

With θ_LB pinned at Morris and s roughly age-flat, the η-by-age implied by
`θ_LB·η·s = R/N` (requiring η ≤ 1 everywhere ⇒ s ≳ 0.44) is roughly:

| <20 | 20–24 | 25–29 | 30–34 | 35–39 | 40–44 | 45+ |
|---|---|---|---|---|---|---|
| ~0.99 | ~0.86 | ~0.75 | ~0.57 | ~0.49 | ~0.49 | ~0.34 |

— a **~3-logit decline** in η across age. The current η age channel cannot
represent this, so a hard θ_LB pin *alone* would just push the misfit into bad
age-PPC. **η needs real age range too.**

## Candidate fixes (decisions for FB)

1. **Pin θ_LB hard** — σ 0.10 → ~0.001, or a deterministic-at-Morris with a small
   log-deviation. (April rec #1; touches **invariant #1**.) Necessary but not
   sufficient on its own.
2. **Give η a real maternal-age gradient** — add an `eta_term_age` term
   (termination rises steeply with age) and/or widen `eta_detect_age` to σ≈0.5–1.0,
   enough for a ~3-logit decline. The age signal then lands in **termination**,
   where the epidemiology puts it — and the age gradient of recorded DS becomes
   informative about η(age) rather than corrupting θ_LB.
3. **Re-run the A/B/C sweep.** Only after (1)+(2) is the pin-s headline meaningful;
   target total should land near the surveillance ~42–51k, agg s ~0.4.

The intercept-level θ_LB↔s indeterminacy (April) and this age-gradient
indeterminacy are the same disease in two dimensions: both resolve by pinning θ_LB
hard and giving the selection channel enough freedom — **level via `s`, age via
`η`.**
