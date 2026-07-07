# The `s(race, year)` anchor derivation and the stale identifiability diagnostic

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Fable 5).

> [!WARNING]
> Work in progress. All data and models are preliminary. This is a code-level
> walkthrough and a methodological finding for review, not a fitted result.

**Date:** 2026-07-07
**Status:** Review. Actionable findings: §2.3 (identifiability diagnostic's
interpretation is stale relative to the anchored `s`); §3 (parameter recovery validates
the machinery, not the science — its `θ`/`s` checks are near-vacuous); §5 (`year_trends.py`
reconstruction omits the year×age interaction and its docstring contradicts the model —
**already fixed** on the `dev/codex/issue-67-model-review-fixes` branch).
**Scope:** `scripts/derive_recording_rates.py` (the anchor generator), the
`identifiability_*` diagnostics in `selection/diagnostics.py`, the parameter-recovery test
(`tests/test_selection_parameter_recovery.py` + `selection/simulate.py`), the sampling/run
config (`selection/config.py`, `selection/sampling.py`), and the trend-reconstruction
scripts (`year_trends.py`, `year_standardised.py`, `year_age_interaction.py`) that replaced
the removed HSGP outcomes model.
**Relates to:** `notes/20260623-degraaf-recording-anchor.md` (the anchor's motivation and
results), `notes/20260707-modelling-review-and-writeup-options.md` §5.1 (where this was
first flagged).

---

## 1. How `derive_recording_rates.py` builds `s(race, year)`

`s` (birth-certificate recording sensitivity) is not observable directly, and neither is
the true DS livebirth count. The script inverts `s = recorded / true` by reconstructing
`true` from de Graaf surveillance prevalence. Surveillance exists only for 2000–2014,
2016, 2018, so most of the 2016–2024 window is imputed — and the design is built around
imputing the *right* quantity.

### 1.1 Pipeline

1. **Load** (`_load`, `derive_recording_rates.py:105-131`): the **known age structure**
   (`year × race × age` counts, 2000–2024, every year); **real recorded counts** (`N`, `R`
   by `year × race`, 2016–2024); and de Graaf **prevalence** per 10k, mapped to the five
   named ethnicities only (`ETH_TO_RACE`, `:81-87`), observed years only.
2. **Age-only expected prevalence** (`_exp_prev`, `:134-143`):
   `exp_prev(race,year) = Σ_age share(age|race,year)·Morris_θ_per_10k(age)` — what
   prevalence would be from the maternal-age mix alone, absent screening. Exists every year.
3. **Survival ratio** (`:350-351`): `surv_ratio = deGraaf_prev / exp_prev` — the smooth net
   screening-and-termination multiplier. **This is the quantity imputed** (not prevalence
   directly), because the age structure carries most of the year-to-year variation and is
   known.
4. **Backtest the extrapolation rule** (`_backtest`, `:151-173`): hold out 2011–2014, fit
   ≤2010, compare constant vs linear by hold-out RMSE on the four stable races.
   **Constant won.**
5. **Fill study years** (`_extrapolate`, `:176-201`): observed kept; 2017 = midpoint of
   2016/2018; 2019–2024 = held flat at 2018 (constant rule); **AIAN held flat with no
   trend** (survival ratio > 1 historically — mechanically impossible — and tiny counts).
6. **Reconstruct and invert** (`:362-365`):
   `prev_used = exp_prev·surv_ratio_used`; `true = prev_used/1e4·N`; `s = R/true`.
7. **Prior σ** (`_sigma_logit`, `_rel_prev`, `:204-220`): since `R` is real every year, the
   only tail uncertainty is imputed prevalence. `rel_prev` = 0.07 (observed) → 0.12
   (interp) → `0.10 + 0.035·(year−2018)` (extrap); combined with `1/√R` and pushed to the
   logit scale by delta method; floored at 0.05 (0.50 for AIAN). Result: `σ_logit ≈ 0.12`
   for well-observed big groups, ballooning to `≈ 0.5` by 2024.
8. **Emit** (`_write_anchor_module`, `:228-292`): a `[7 race, 9 year]` matrix; rows 0–4 from
   the above, rows 5 (Unknown) / 6 (Multi-race) get the weak fallback `s=0.40, σ=0.50` (no
   de Graaf category). Also emits `PREV_RACE_YEAR(_SIGMA)` for the full-margin anchor;
   rows 5/6 are `NaN` so the margin term skips them.

### 1.2 Two properties to state in any write-up

- **The anchor is not fully independent evidence.** `exp_prev` uses the *same* Morris θ_LB
  the model pins in Stage 1, and `s` is derived as `R/(prev·N)` from the *same* recorded
  counts the model fits. So pinning θ, anchoring `s` to this surface, and fitting the
  recorded Binomial are **mutually consistent by construction** — which is why convergence
  improved dramatically, but it means the anchor is de Graaf prevalence filtered through the
  project's own θ and recorded counts, not a clean external instrument.
- **The held-flat tail is blind to the post-2020 NIPT surge.** The backtest that chose
  "constant" pre-dates NIPT-for-all. If net survival actually fell in 2021–2024 (more
  detection → more termination), holding `surv_ratio` flat forces that decline to be
  reconstructed as *lower recording*, biasing recent-year `s` down and the recent-year total
  up. Most exposed part of the surface; `degraaf_tail.py` exists as the sensitivity bracket.

---

## 2. The identifiability diagnostic is stale relative to the anchor

### 2.1 What it computes

`_identifiability_correlations` (`selection/diagnostics.py:67-93`) takes, per race `i`, the
posterior draws of `eta_term_race[i]` and `s_race[i]` and computes their Pearson `r`.
`identifiability_table` / `identifiability_pairplot` (`:153-222`) flag `|r| > 0.7` as
"prior-driven — data cannot distinguish termination from recording" and `|r| < 0.7` as
"data-informed." Here `s_race` is a **Deterministic**: the logit-scale year-average of the
anchored `s_race_year` (`selection/model.py:284`).

### 2.2 Why it was sound before the anchor

Previously `s` had a free intercept and free race offsets. `eta_term_race` and `s_race`
were both genuinely free and sit on opposite sides of the ridge `recorded ∝ η·s`. If the
sampler could fix only the product, it traded one against the other draw-by-draw →
strong **negative** posterior correlation. So `|r|→1` was a real signal of
non-identifiability; small `|r|` meant the data pinned each separately.

### 2.3 Why it is now compromised (the finding)

`s_race_year` is no longer free — it is `Normal(anchor_logit, anchor_σ)` with
`anchor_σ ≈ 0.12` for well-observed groups. So for races 0–4 the posterior of `s_race` is
**prior-dominated and nearly constant**. A nearly-constant variable has almost no
covariance with anything, so:

> `corr(eta_term_race, s_race) → 0` **mechanically**, because `s_race` barely moves — not
> because the data separated termination from recording.

The diagnostic therefore reports `|r| < 0.7` ("data-informed") for exactly the groups where
`s` is most tightly **assumed**. It now conflates the two situations it was built to
distinguish:

- **identified** — data pinned η and s separately (what low `r` used to mean), vs
- **assumed** — s was fixed externally, so it cannot co-vary with η (what low `r` mostly
  means now).

Aggravating factors: (a) `s_race` is a *year-average*, shrinking its variance further; (b)
the real η↔s trade-off is at cell level across the full covariate interaction, so a
race-marginal correlation is coarse regardless. The only place a high `|r|` could still
appear is the **unanchored** groups (Unknown idx 5, Multi-race idx 6, σ=0.50) — where it is
expected and equally uninformative.

**Consequence:** wherever the identifiability table appears in a report, it can give false
reassurance that the demographic decomposition is data-identified when, for the
surveillance-anchored races, the recording side is essentially pinned.

### 2.4 What would be honest instead

1. **Prior→posterior shrinkage readout.** For each anchored `s_race_year[i,t]`, report
   `posterior_SD / prior_SD` and `(posterior_mean − prior_mean)/prior_SD`. If posterior SD ≈
   prior SD and the mean barely moved, label it "assumed here, not estimated" — the opposite
   of the current "data-informed."
2. **Lean on variant B as the real test.** B loosens `s` (σ×2) and tightens `η_term`; the
   material movement in the total and decomposition under B (~40k → ~48k) *is* the honest
   measure of how much `s` is doing by assumption. The A↔B spread is a better
   identifiability statement than the correlation table.
3. **Correlate the parameters that actually trade off.** If a correlation diagnostic is
   kept, compute it between the ridge quantities — e.g. aggregate recording level vs
   `eta_term_int` — not the race-marginal `s_race` the anchor has frozen.

### 2.5 Suggested action

Not wrong code — stale interpretation. Options, smallest first: (a) add a caveat wherever
the table renders (`docs/models/selection/index.qmd`) noting that low `|r|` under an
anchored `s` reflects the anchor, not identification; (b) add the §2.4.1 shrinkage readout
as the primary identifiability diagnostic and demote the correlation to secondary; (c)
formalise the §2.4.2 variant-B comparison as a standing prior-sensitivity panel. (b)+(c) are
the substantive fix; (a) is the minimum before the table is cited.

---

## 3. Parameter recovery validates the machinery, not the science

`tests/test_selection_parameter_recovery.py` draws one truth from variant C's priors,
simulates 1,620 cells (15/month × 12 × 9yr, mean size 1,500), fits the full spec (600
draws, 2 chains, `target_accept=0.95`), and asserts posterior-95%-CI coverage ≥70% for
`theta_lb_age`, `eta_term_{race,age,year}`, `s_race`, `eta_detect_year_age`, plus
`eta_term_int` mean within 0.5 logit and max R̂ < 1.05. The docstring calls it "the single
most load-bearing test." It is a necessary guard — a broken sampler or mis-wired
parameterisation trips it — but its scope is narrow:

### 3.1 It is a perfectly-specified, in-prior recovery

Truth is drawn from the **same** priors the fit uses (`TrueParams.from_priors(
variant_C_default())`), and `selection/simulate.py` mirrors `model.py` line-for-line
(identical `η = 1 − detect·term`, `s = invlogit(s_race_year + s_edu)`, `p_recorded`; the
zero-sum interaction double-centred to match `ZeroSumNormal`). So it tests whether NUTS can
invert the generative map **when the model is exactly correct** — by construction there is
no misspecification to detect, which is the failure mode that actually threatens the real
fit.

### 3.2 Two of the six recoveries are near-vacuous

- `theta_lb_age` is pinned at σ=0.001, so truth ≈ Morris and posterior ≈ Morris — coverage
  is guaranteed; it proves the pin works, not identification.
- `s_race` recovers for the same reason: truth is drawn from the tight anchor prior and the
  fit uses that anchor, so `s` is effectively handed to the model. This is prior
  propagation, not estimation — the same fact behind §2.3.

The only two genuinely informative recoveries are `eta_term_year` (the zero-mean drift,
identified by the year signal) and `eta_detect_year_age` (the zero-sum interaction, the
best-sampled part). Those carry weight; the rest are prior propagation. The test wisely does
**not** assert recovery of `eta_detect`'s main effects, since the detect-vs-term split is
not identified — so its scope is implicitly "the margins we claim are identifiable."

### 3.3 It does not exercise the η×s ridge, and the design is easier than reality

Because the simulated `s` truth *equals* the anchor prior mean (± its tight σ), the ridge is
dissolved before sampling — the fit recovers η only because a consistent `s` was pre-supplied.
The test would give **no warning** in the case that matters: an anchor wrong relative to the
true recording rate. A real stress test would draw the `s` truth *away* from the anchor and
check whether the posterior chases the data or stays glued to the (wrong) prior — a
prior-data-conflict test, which this is not. Separately, `simulate_cells` draws covariates
**independently and uniformly** with ~1,500-size cells, so it lacks the collinearity
(education↔payer, age↔race) and sparse/unequal cells that drive the real fit's
prior-dependence. Recovery is therefore easier in simulation than on real data.

**Bottom line:** parameter recovery confirms the fit inverts its own generative model and
that the two zero-sum time terms are identified. It does **not** validate the model against
reality and cannot detect anchor misspecification — the concrete form of the
"'fits the data' ≠ 'correct'" limitation. External validation (a linked registry) does not
exist.

## 4. Sampling settings are partly computational remedies for the ridge

Two presets (`config.py:41-60`): `dev` (1000/1000, 2 chains, `target_accept=0.9`) and
`reporting` (1500/1500, 4 chains, `target_accept=0.95`). `sampling.py` runs prior-predictive
→ posterior → posterior-predictive via nutpie (fallback to PyMC NUTS), seed 47. The
revealing detail is *why*: the config docstring says the model "has a known η/s
identification challenge and needs tighter stepping than the usual 0.9," and `eta_detect_age`
σ was tightened 0.5→0.1 because otherwise "the sampler wandered the ridge and variant A
failed to converge (r-hat 1.73, ESS 6 at the 25–29 band)." So the high `target_accept` and
the tight age-σ are partly **computational fixes for a badly-conditioned posterior**, not
purely modelling choices. Consequence for interpretation: the tight `eta_detect_age`
posterior should not be read as data-driven confidence — it is a sampler accommodation. State
this wherever that parameter is reported.

## 5. The trend-reconstruction layer (and a scoped `year_trends.py` bug)

### 5.1 Architecture correction — one Bayesian model, not two

The separate HSGP outcomes regression was **removed** on 2026-04-21 (commit 733161e,
"Remove Bayesian regression model") as redundant once the selection model covered the same
goal "with a more identifiable decomposition." Its shared fit infrastructure (`sample`,
`io`, `RunConfig`, `FitContext`) moved into the `selection` package. So the current
architecture is: **one Bayesian inference model** (selection) + **three deterministic
reconstruction scripts** that read its posterior (`year_trends.py`, `year_standardised.py`,
`year_age_interaction.py`) + **one non-Bayesian prevalence lookup** (`previous_model_yearly.py`,
1989–2024 context). The year/trend results are post-hoc reconstructions, not a second model.

### 5.2 `year_trends.py` reconstruction bug (= issue #67 P1), scoped

On `main`, `load_variant` reconstructs `det` (`year_trends.py:79-86`) from
`eta_detect_int + year + age + race + edu + payer` — **omitting `eta_detect_year_age`**,
which the fitted `eta_detect` includes (`model.py:189-194`). The docstring (`:19-21`) and
closing note (`:216-221`) assert the opposite of the model ("eta_detect has NO year-by-age
interaction"), contradicting the sibling `year_age_interaction.py` (which exists to analyse
that very term and errors if a fit lacks it, `:63-65`) and `year_standardised.py` (which
reads `eta_detect_year_age`, `:67-69`). Classic staleness after the interaction was added.

**Impact, scoped** (narrower than a code-only read suggests):

- **The headline `reduction`/`true_per10k` are correct** — computed from the *saved*
  `p_ds_lb` posterior (`:115-116, 121-124`), which was fitted *with* the interaction.
- **Only the displayed `eta_detect`/`eta_term` split lines are affected** (`:117-118`),
  and since the interaction is **flat** (June finding: +0.015 log-odds/band, CI crosses
  zero) even those are barely off — approximately right by luck, not correctness.
- **The guardrail exists but doesn't fire:** `recon_max_err` (`:130`) is printed (`:195`)
  but never asserted, so a future non-flat interaction would silently corrupt the split.
- **Secondary:** the split takes posterior means of each component then applies `inv_logit`
  (`cmean`/`cscalar`), so by Jensen the reported `eta_detect`/`eta_term` *levels* carry a
  small bias independent of the interaction. The reduction avoids this (uses `p_ds_lb` draws).

### 5.3 Overlap with the issue-67 branch — do not duplicate

`dev/codex/issue-67-model-review-fixes` **fully fixes this**, and goes further:
- adds `eta_detect_year_age` to the reconstruction;
- switches to **per-draw** reconstruction (`draw_vector`/`draw_scalar` → mean), which also
  removes the §5.2 Jensen bias as a bonus;
- corrects the docstring and closing note ("eta_detect includes a year-by-age interaction");
- adds `RECON_TOL = 1e-8` and a fail-fast `RuntimeError` in `main()` for either variant.

**Residual gaps** (vs issue #67's acceptance criteria, if we want to close them): no
dedicated **regression test** constructing a posterior with non-zero `eta_detect_year_age`
and checking the reconstruction; and no **shared reconstruction helper** — `year_trends.py`
and `year_standardised.py` still duplicate the (now-correct) reconstruction. Recommendation:
land the issue-67 branch rather than re-fixing; add the regression test + shared helper as an
optional follow-up.

## 6. Related internal notes

- `notes/20260623-degraaf-recording-anchor.md` — anchor motivation, full-margin term, results.
- `notes/20260628-degraaf-corrected-prevalence-extraction.md` — `--degraaf-tail` sensitivity.
- `notes/20260622-predictors-bayesian-model.md` — the current model + PU framing.
- `notes/20260707-modelling-review-and-writeup-options.md` — the review this expands (§5.1).
