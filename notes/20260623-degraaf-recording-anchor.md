# De Graaf surveillance anchor for recording (`s`) and the full-margin total

> [!WARNING]
> Work in progress. All data and models are preliminary. Totals quoted here are from the
> **dev** sampler preset (2 chains, 1000+1000); re-run at the `reporting` preset before citing.

**Date:** 2026-06-23 · **Branch:** `dev/frank/eta-reanchor`

## Why

The selection model's headline total was effectively *chosen* by where we pinned the
recording rate `s`: `s_int` and `theta_LB` were both hard-pinned (sigma 0.001) to break the
eta x s ridge by fiat, and the A/B/C variants existed only to bound a ridge we had no
external data to resolve (pin `s≈0.40 → ~38–40k`; `s≈0.34 → ~45k`). The gradient-boosting
(GB) models can't fix this — they're an uncorrected positive-unlabelled classifier that
over-medicalises (see [the predictors note](20260622-predictors-bayesian-model.md)), so they
are now **retired as a total estimator** and kept only as a predictor-screening / propensity
diagnostic.

The missing ingredient was an **external anchor**. Frank supplied de Graaf surveillance-based
DS prevalence by ethnicity × year (`data/reference/ds_prevalence_ethnicity_2000_2023.csv`,
per 10k livebirths), which lets us work back:

```
true(race, year) = prevalence/1e4 * births        s(race, year) = recorded_DS / true
```

This breaks the ridge with *data* instead of a pin.

## What we built

### 1. Derivation pipeline — `scripts/derive_recording_rates.py`

- **Coverage.** De Graaf prevalence is observed for 2000–2014, 2016, 2018; **2015/2017 are
  being chased and 2019–2024 do not exist yet** (surveillance stops at 2018). So 7 of 9 study
  years are imputed — imputation is central, not incidental.
- **Indirect standardisation.** Decompose `prevalence = exp_prev(age structure, KNOWN every
  year) × survival_ratio`, where `exp_prev = Σ_age share(age|race,year)·Morris θ_LB(age)` and
  `survival_ratio = prevalence / exp_prev` (net screening-termination survival). Only the
  smooth survival ratio is imputed; the known age structure carries every year.
- **Backtested extrapolation.** Hold out 2011–2014, fit ≤2010, compare constant vs linear:
  **constant wins out-of-sample for every stable race** (RMSE 0.01–0.04 vs 0.03–0.08). So the
  tail holds each group's 2018 survival level; 2017 is interpolated; prior σ widens with the
  extrapolation horizon. **Caveat:** the backtest window predates the post-2020 "NIPS-for-all"
  surge, so holding flat may overstate true DS in 2022–2024 (some recorded-count decline that
  the model then attributes to recording is probably extra terminations).
- **AIAN is unreliable** — tiny counts, survival ratio exceeds 1.0 historically (mechanically
  impossible). Held flat at a robust recent level with a deliberately wide σ; no trend.
- **Output:** the committed `src/.../selection/recording_anchor.py` (generated) with four
  `[6 race, 9 year]` surfaces: `S_RACE_YEAR_LOGIT/_SIGMA` (recording prior) and
  `PREV_RACE_YEAR/_SIGMA` (de Graaf true-prevalence margin target; Unknown row = NaN). Plus
  `notes/figures/recording_rates_anchor.*`.

Empirical recording rate (pooled over the two clean years, 2016 & 2018):

| race | de-Graaf `s` | old hand-set prior |
|------|------|------|
| NH White | 0.46 | 0.40 |
| NH Black | 0.33 | 0.31 |
| NH AIAN | 0.71 (noisy) | 0.33 |
| NH Asian/PI | 0.37 | 0.38 |
| Hispanic | 0.38 | 0.35 |

The hand-set priors were close except AIAN (noisy) and the overall level (~0.42, not 0.40).

### 2. Recording anchor (`s`) — replaces the 0.40 pin

`model.py` stage 3 now uses `s = invlogit(s_race_year[race, year] + s_edu[edu])`, with
`s_race_year` an informative prior from the anchor (measured level + racial gradient + year
dimension), σ widening across the imputed tail. A year-averaged `s_race` `Deterministic` is
retained so the identifiability / coefficient / recovery diagnostics keep working unchanged.
`s_int` / the guessed `S_RACE` offsets are gone; `s_edu` stays a small within-cell residual.

### 3. Full-margin anchor — `build_model(..., prev_margin=)`, `fit_selection_model.py --anchor-margin`

A soft Normal observation ties the model's N-weighted **marginal `p_ds_lb` per race × year**
(45 named groups) to de Graaf's prevalence. With θ pinned to Morris, this pins **η's race×year
level** to surveillance, so the literature termination/detection priors can no longer drag the
total below de Graaf. By construction it is consistent with the recorded Binomial (the `s`
anchor was derived as `recorded/(prev·N)`). Off by default; on with `--anchor-margin`; skipped
for variant D.

## Results (dev preset)

**Convergence fixed.** Max R-hat **1.010**, min ESS ~340–380 — versus the old pinned-ridge
fits that hit R-hat ~1.7 / ESS single digits. The anchor broke the eta x s ridge, which was
the main structural pathology.

| | s-only anchor | full-margin | de Graaf target |
|---|---|---|---|
| Total true DS 2016–2024 | 36,930 (95% 31.6–42.4k) | **40,718 (95% 35.5–45.9k)** | 45,928 |
| Overall `s` | 0.481 | 0.437 | ~0.38 |

- **s-only:** posterior `s` drifted *above* the anchor (0.48 vs 0.38) → the model's η priors
  overrode the recording anchor and pulled the total *down* to 37k. This is the consistency
  failure that motivated full-margin: anchoring recording alone lets η undercut surveillance.
- **full-margin:** total rose to ~41k and per-race `s` moved *back toward* the anchor
  (White 0.457→0.425, Hispanic 0.422→0.392). But the fitted margins land **~1σ below** the de
  Graaf targets (e.g. White 2016 11.75 vs 12.64 per 10k; gap widens in later years), because
  the soft potential is outweighed by the recorded Binomial (33M births) + the η priors.

**Interpretation.** De Graaf's prevalence is **~12% higher** than what the recorded counts +
Morris θ + the termination/detection literature jointly support. 41k is the Bayesian
reconciliation of all the evidence, with de Graaf's 46k at the top of the 95% CI. The model
isn't ignoring surveillance — it's surfacing a real disagreement between top-down surveillance
and bottom-up recording+literature.

## Open decision — how hard to push de Graaf

The prevalence potential is currently **soft** (observed-year σ ≈ 7% of prevalence). Options:

1. **Accept ~41k** as the honest reconciliation (recommended default). Most defensible single
   number; above old structural C; consistent on `s`.
2. **Tighten the prevalence σ** (≈7% → 2–3%) so the margins snap to de Graaf → total ~45–46k.
   Says "trust de Graaf's survival over the η literature" (the original spirit of full-margin).
3. **Let de Graaf fully own η's race×year level** (loosen the η race/year priors so the
   potential wins outright). A larger reparameterisation; cleanest "surveillance is authority".

The question underneath: how much more do we trust de Graaf's live-birth prevalence than our
termination/detection priors?

## Next steps

- [ ] **Pick the tightness option above**, then run the **`reporting`** preset (1500×4,
      target_accept 0.95) for citable convergence + totals.
- [ ] **Drop in 2015/2017** when Frank's surveillance numbers arrive (shrinks the imputation;
      regenerate `recording_anchor.py`). 2019–2024 stay extrapolated until surveillance extends.
- [ ] **Revisit A/B variants** — with the anchor breaking the ridge, the A/B bound may be
      redundant; consider collapsing to a single anchored spec.
- [ ] **Post-2020 NIPS surge** — the held-flat survival tail can't see it; consider a small
      downward survival drift for 2021–2024 as a sensitivity.
- [ ] **Reproducibility** — the input prevalence CSV lives in gitignored `data/reference/`.
      It is de Graaf *published* surveillance (not NCHS microdata, so not DUA-restricted);
      consider a committed home so `derive_recording_rates.py` reproduces on a clean checkout.
- [ ] **Full-margin "hard" variant** — option 3 above, if we decide surveillance is authoritative.

## Files

- `scripts/derive_recording_rates.py` — derivation + backtest + figure + generates the module
- `src/.../selection/recording_anchor.py` — **generated**, committed anchor surfaces
- `src/.../selection/priors.py`, `model.py`, `simulate.py` — `s_race_year` reparameterisation,
  year-averaged `s_race` Deterministic, full-margin `prev_margin` observation
- `scripts/fit_selection_model.py` — `--anchor-margin` flag
- `notes/figures/recording_rates_anchor.*` — survival-ratio fan + `s(race,year)` surface
