# Modelling review and write-up options: what the GB and selection models can support

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Fable 5).

> [!WARNING]
> Work in progress. All data and models are preliminary. This note is a
> review-and-scoping document, not a result; numbers quoted are lifted from the
> existing `notes/` corpus (chiefly the 2026-06-22 predictors note) and should be
> re-confirmed against a `reporting`-preset fit before any external use.

**Date:** 2026-07-07
**Status:** Review for further discussion. No code changed. Synthesises the existing
`notes/` corpus, the two Quarto analysis templates, and a fresh file-level read of the
`selection/` and `models/` packages.
**Scope:** the whole modelling approach (GB classifier + Bayesian selection model +
the earlier prevalence model), the bias question, and what is defensibly publishable.
**Standing constraint (new, load-bearing):** **we do not have access to state/region
data.** This is treated here as permanent, not a future phase — it forecloses the
geographic identification route entirely and downgrades every demographic claim to a
national association. See §6.

---

## 1. Why this review

Two questions were on the table:

1. Can the original Aim-3 goal — using the gradient-boosting (GB) classifier to
   identify individual false-negative (missed) DS cases in birth-certificate data — be
   achieved without bias?
2. If not, and absent a record-level de-biasing of recorded births, what value do the
   models have, and how should they evolve?

The short answers: **(1) No — and this is already proven inside the repo.** **(2) The
value is real but lives almost entirely in the structural selection model as an
_aggregate_ reconstruction, plus a narrower legitimate role for the GB work.** The
project has already made this pivot _de facto_ (the 2026-06-23 note "retires the GB
models as a total estimator"); it has not yet been ratified in `plans/readme.md`, the
README, or the stated aims. That documentation gap is the first thing to close.

## 2. The three strands, briefly

- **GB classifier** (`models/`, driver `scripts/fit_model.py`). LightGBM binary
  classifier. Nine registered variants across two families (`usbc10*` full features,
  `usbc11*` clinical+age only) × two labels (C+P, C-only `_cn`) × prune stages
  (M0/M1/M2). Writes per-variant prediction + `ds_pred_missing` flag columns back to
  DuckDB.
- **Bayesian selection model** (`selection/`, driver `scripts/fit_selection_model.py`).
  Three-stage decomposition `P(R=1|X) = θ_LB·η·s + (1−θ_LB·η)·f`
  (`selection/model.py:291-305`), binomial per demographic cell. Variants A/B/C (prior
  sensitivity bound) + D (GB-corrected cross-check).
- **Earlier prevalence model** (`scripts/previous_model_yearly.py`, DuckDB
  `prevalence_year`/`reduction_rate_year`). Longer-horizon 1989–2024 context; prevalence
  held flat from 2018, so 2019+ is extrapolation.

## 3. Core finding — the GB goal cannot be de-biased

### 3.1 The classifier's target *is* the recorded flag

The C+P training label `ca_down_c_p_n` (`data_utils.py:44-45`) is defined identically to
the "recorded" indicator `down_ind` (`scripts/duckdb_prepare.py:250-262`): both are
`ca_down/ca_downs ∈ {C, P}`. The model is trained and scored on the **same** 2016–2024
population it is meant to correct (`scripts/fit_model.py:650-684`) — there is no external
or held-out population. So the classifier estimates **`P(recorded DS | X)`**, not
`P(true DS | X)`. That is what the object is; it is not a tuning problem.

### 3.2 Why removing demographic features did not help

`usbc11` drops race/education/payer to avoid entrenching the recording-sensitivity
signal (`usbc11.py:157-179`). It did not de-bias the cohort — it promoted **clinical
severity** to be the entire remaining propensity signal. The predicted-missing cohort is
~42% cyanotic CHD / ~83% NICU. You cannot feature-select your way out: the classifier
always over-weights whatever drives recording; dropping some drivers hands the weight to
the rest.

### 3.3 The mechanism, stated precisely (and what it leaves recoverable)

A DS birth is missed on the certificate for one of two reasons:

- **(a) paperwork/timing** — karyotype not back inside the ~5-day filing window, or the
  box simply not ticked (the Hamilton-County "communications error"). The near-tautology
  subset — `ca_disor` (chromosomal disorder) ticked Confirmed/Pending but the DS box not
  ticked — is individually recoverable.
- **(b) clinically unremarkable** — most DS babies have Apgars ≥ 7 and no CCHD. In
  certificate features these are statistically indistinguishable from ordinary births, so
  they sit at the bottom of any ranking and **cannot** be recovered.

Case (b) is exactly the demographically-under-reported subset the epidemiology
literature (Boulet 2011) cares about. The classifier can find (a), not (b).

### 3.4 The correct framing: positive-unlabelled learning under SAR

This is worked all the way through in `notes/20260622-predictors-bayesian-model.md`. The
recorded births are labelled positives; all other births are unlabelled. Recording
propensity `e(x) = P(recorded | x, DS)` depends on severity, SES, and race, so labelling
is **Selected At Random (SAR)**, not completely at random. The GB, trained by treating
every unrecorded birth as negative, is a textbook **non-traditional classifier**: it
predicts `P(recorded | x)`. The lesson — **estimate the class prior, not the individual
cases**: you can know _how many_ you are missing without knowing _which_. The aggregate
is recoverable where the individuals are not.

This framing is the most academically distinctive thing in the project (see §9).

## 4. What each model is actually worth

The worry "without de-biasing recorded births the models have no value" conflates two
things. Nothing corrects each record — that is not achievable. What exists is an
**aggregate** de-biasing, which is the real output.

- **Selection model = the de-biasing**, as an aggregate reconstruction. It is the
  principled inverse-propensity estimator (`total ≈ Σ_recorded 1/s(x)`): a defensible
  total (~40–41k for 2016–2024), an age gradient, subgroup orderings, and a year trend,
  each with honest uncertainty. Corroborated from two independent directions (earlier
  prevalence model ~44k; de Graaf surveillance ~46–48k). This is the scientific product.
- **GB models retain a narrower legitimate role** (not population estimation): (1) a
  _description of the recording propensity_ (SHAP/importance → what drives recording,
  which informs the selection model's covariates); (2) the communications-error recovery
  subset (§3.3a), a bounded, individually-valid correction; (3) **Variant D** — feeding
  the GB-corrected total into the structural model with recording pinned off — whose
  _failure_ (undercounts to ~25.7k, forcing implausible 38% under-20 termination) is
  itself evidence the recording-pin route is right.

Value is conditional: "_if_ certificates record ~40% of DS births, then ~40,000." The
"if" is load-bearing. That honesty is the model's main asset and main liability at once.

## 5. Where the selection model is fragile (grounded)

Ranked by how much they threaten a citable claim. This is the pre-publication scrutiny
list.

1. **The identifiability diagnostic may now reassure for the wrong reason.**
   `identifiability_table` flags `|r| < 0.7` between `η_term` and `s` race effects as
   "data-informed" (`selection/diagnostics.py:204-222`). But `s` is now tightly
   _anchored_ to de Graaf (`selection/model.py:263-270`), so low correlation can mean "s
   is externally pinned," not "the data separated termination from recording." The
   diagnostic predates the anchor and has not been re-reasoned for it. **Highest
   priority** — it underwrites the claim that the decomposition is data-identified.
2. **Seven of nine study years are imputed in the recording anchor.** De Graaf
   surveillance stops at 2018; `s(race, year)` for 2019–2024 holds each group's survival
   ratio flat and σ balloons across the tail (`selection/recording_anchor.py:15-17`;
   `derive_recording_rates.py`). The held-flat tail is blind to the post-2020
   "NIPT-for-all" surge, so a recorded-count decline that is really _more terminations_
   is attributed to _recording_. The 2020–2024 trend (a headline) rests on extrapolation.
   `--degraaf-tail` brackets this, but the headline is robust to it _only because the
   anchor is soft_ — a fragile robustness.
3. **A self-flagged correctness risk is still open.** `selection/priors.py:60-68` warns
   an earlier version swapped race positions 2↔3 (NH AIAN vs NH Asian/PI) and the values
   "may need a second look." Every race-indexed prior depends on this ordering; a mismatch
   attaches the wrong prior to the wrong group **silently**. Close it, don't carry it.
4. **The access-vs-choice split is entirely prior-asserted.** Only the product
   `η_detect·η_term` is identified; `η_detect_age` was pinned tight (σ 0.5→0.1) purely to
   make variant A converge (`selection/priors.py:207-213`). Any decomposition of the age
   gradient into "screening access" vs "termination choice" is a modelling choice, not a
   finding.
5. **No gold standard; ML validation optimistic.** No linked registry to check missed
   cases against — validation is internal only (age-PPC ~5%, self-simulated parameter
   recovery). Issue #67: GB metrics use one split for tuning and reporting
   (`cross_validate` unimplemented), so AP/Brier are tuning-set-optimistic; and
   `scripts/year_trends.py` reconstructs the year decomposition **without** the
   `eta_detect_year_age` interaction the fitted model contains — a live reporting bug
   (model right, reported trend wrong). These were fixed in #68 (merged 2026-07-08).

## 6. The state-data constraint and what it forecloses

We have no state/region column (`selection/data.py:22-26`), and no route to one. In the
US, prenatal screening/termination **access** and birth-certificate **recording
practice** both vary strongly by state, and demographic groups are geographically
clustered. With no region term, `η_term` absorbs geographic heterogeneity into a flat
national drift plus demographic offsets it cannot properly attribute. Consequences:

- **Any demographic claim (race, SES) carries a standing, unfixable caveat:** part of what
  reads as a group effect may be a state-access effect the model cannot see. Demote the
  demographic decomposition from "finding" to **"national association, not separable from
  geography."**
- This **raises** the relative value of claims that don't route through geography — the
  age gradient, the year trend, and the methods result — because they are unaffected.

The earlier roadmap's "pursue state-level restricted data" phase is **dead**. Everything
below is scoped to what national certificate data + external anchors can yield.

## 7. Write-up options — GB models

Say the "not this" once, clearly: the predicted-missing cohort is **not** a population
estimate, and nothing built by summing it (counts by group, co-occurrence rates) should
be published as one. Settled in `docs/analysis/predicted.qmd` and the 2026-06-22 note.

What the GB work _can_ support, most-citable first:

1. **A cautionary methods result, publishable on its own.** "A within-vital-records
   classifier trained on the recorded flag learns the recording propensity, not the
   condition, and cannot recover demographically-missed cases — and removing demographic
   features does not de-bias it, it just promotes clinical severity to carry the entire
   propensity." Vital-records + ML is an increasingly common move; this is a clean worked
   example of why the naive version fails. The 42% CCHD / 83% NICU cohort profile is the
   memorable evidence. Arguably the single most novel GB-side finding.
2. **A characterisation of the recording process.** SHAP/importance over the propensity
   model → _what makes a true DS birth get recorded_: confirmatory-workup timing signals
   (NICU, CCHD, assisted ventilation), the near-tautological `ca_disor` box, maternal age.
   A data-quality finding about NVSS DS ascertainment; motivates the selection model's
   covariate choices.
3. **The communications-error recovery subset** (`ca_disor` ticked, DS box not) — a
   bounded, individually-valid correction whose size can be quantified. Small, and _not_
   the demographically-missed population, but a real, defensible recovery.
4. **Variant D as triangulation.** Publish the _failure_ (~25,700; self-refuting via 38%
   under-20 termination) as positive evidence the recording-pin route is right and that
   the alternative was tested honestly.

## 8. Write-up options — statistical models (tiered by identifiability)

Tier explicitly — the defensibility gap is wide and a reader must see it. **A** =
data-identified (robust); **B** = assumption-conditional (report the "if" / the range);
**C** = national association, geography-confounded (report as association, never
mechanism).

| Claim | Tier | Basis / caveat |
|---|---|---|
| Rising reduction 2016–2024 (~38%→50%, plateau ~2022) | **A** | Recording carries no year term, so year movement in recorded rate maps onto survival. Cleanest result in the model. |
| "Older-mothers-first screening" is a compositional artefact | **A** | The year×age interaction is the best-sampled parameter (ESS 7–16k) and is flat; the raw pattern does not survive adjustment. Counterintuitive and solid. |
| Age gradient in termination (~5%→~64% across age) | **A (shape) / C (level split)** | Shape robust; access-vs-choice split prior-asserted (§5.4). |
| Total DS livebirths 2016–24 ≈ 40–41k (up to ~48k if s freed) | **B** | "If certificates record ~40%, then ~40,000." Corroborated by prevalence model (~44k) and de Graaf (~46–48k). Report the range, not a point. |
| Bottom-up sits ~12% below top-down surveillance | **B (finding in itself)** | Don't hide it by tightening the anchor — the method disagreement is substantive. |
| Co-occurring rates (CCHD ~5.6%, NICU ~58% ≈ recorded rates) | **B (conditional on R≈1)** | Class-prior method, recording-ratio swept as sensitivity. Main value: _inverts the GB artefact_ (which put CCHD ~20%). Lead with the inversion. |
| Ethnicity orderings (NH Asian/PI lowest true rate despite oldest; NH Black robust under-recording) | **C** | Survives age-standardisation, but access-vs-recording split _and_ geographic clustering uncontrolled (§6). |
| SES/education termination gradient (~2.4 log-odds, monotonic) | **C** | Data overrode the prior (real signal), but education/insurance near-collinear and geography uncontrolled. |
| Longitudinal 1989–2024 trajectory (peak ~5,450 in 2007 → ~4,800 by 2024) | **B/C** | Separate prevalence model; 2019+ extrapolated. Long-horizon context, flagged. |

Build a paper's spine around the **Tier-A** results: they depend on neither the recording
pin nor geography, and two of them are counterintuitive-but-solid. Do **not** headline the
demographic (race/SES) material anywhere — with no state data it can only be a
well-caveated associations section, and over-claiming there is the fastest way for a
reviewer to discount the solid parts.

## 9. The connective tissue — the PU/propensity framing as flagship

The GB failure and the structural success are the same story told twice: the GB is the
_uncorrected_ estimator (`Σ P(recorded|x)`), the structural model the
_inverse-propensity-corrected_ one (`Σ 1/s(x)`), and they differ by exactly the recording
rate. That framing (2026-06-22 note) is what elevates "we counted DS births" into a
methods contribution about ascertainment correction in biased vital-records data. Make it
explicit and central, not a footnote.

## 10. Suggested output clustering

Three units rather than one omnibus paper:

1. **Methods/epidemiology (flagship).** PU framing; why the GB approach fails; the
   structural selection model as the principled alternative; total + funnel + surveillance
   discrepancy. Most novel.
2. **Substantive findings.** Tier-A time story (rising reduction is a screening-_access_
   phenomenon, not changed family decisions; the "older-mothers-first" artefact) + the age
   gradient. Most mission-relevant and most defensible.
3. **Descriptive + co-occurring conditions.** Recorded population 1989–2024 (Aim 1);
   corrected co-occurrence rates via class priors; explicit warning that GB-style cohorts
   invert those rates. Directly useful for service planning.

## 11. Open decisions / forks (for FB)

1. **Ratify the pivot (Phase 0, docs only).** Rewrite Aim 3 in `plans/readme.md` / README
   from "identify the most likely missing cases" to "estimate the missed _count_ and its
   distribution (the class prior)." State plainly that individual missed-case
   identification is not achievable without bias; cite the PU framing; define the GB
   models' residual roles (§4). _Recommend: do this first — it reframes everything else._
2. **GB models' future role.** _Recommend:_ keep as propensity/predictor-screening
   diagnostic + communications-error recovery; formally drop as a total/population
   estimator.
3. **How hard to push de Graaf surveillance** (open since `notes/20260623-*`).
   _Recommend:_ keep the margin anchor _soft_ and report the ~41k-vs-46k gap as a
   substantive finding, rather than tightening σ to force the total onto the surveillance
   number.
4. **Anchor tail (post-2020).** With no state data and surveillance ending 2018, the
   2019–2024 trend is the most exposed headline. Either chase the missing de Graaf years
   (2015/2017/2019+) or model an explicit NIPT-era survival drift for 2021–2024 rather
   than holding flat. At minimum, foreground the extrapolation in any 2020–2024 claim.

## 12. Standing caveats (attach to every output)

- The total is **conditional on the pinned recording rate** (`s ≈ 0.40 → ~40k`;
  `s ≈ 0.32 → ~48k`); it is roughly inversely proportional to assumed recording.
- There is **no gold-standard registry** to validate the _missed_ cases — internal
  consistency only. "Fits the data" ≠ "correct."
- Demographic/subgroup effects **cannot be separated from geographic access variation**
  (no state data, §6). Report as national associations, not mechanisms.
- The access-vs-choice (η_detect vs η_term) split within any gap is **prior-driven**;
  only the combined effect on survival is identified.

## 13. Related internal notes

- `notes/20260622-predictors-bayesian-model.md` — the PU framing + current model + results
  (most important precursor).
- `notes/20260621-theta-lb-escape-age-gradient.md` — the θ_LB escape and its fix.
- `notes/20260621-screening-cascade-eta-reanchoring.md` — η re-anchoring (NIPS transition).
- `notes/20260623-degraaf-recording-anchor.md` — the de Graaf `s` anchor + "how hard to
  push de Graaf" decision.
- `notes/20260628-degraaf-corrected-prevalence-extraction.md` — corrected surveillance
  extraction + `--degraaf-tail` sensitivity.
- `notes/20260514-status-review.md` — prior status review (pre-anchor).
- GitHub issue #67 / PR #68 (`dev/codex/issue-67-model-review-fixes`, merged 2026-07-08) —
  model/methodology/reporting review follow-ups.
