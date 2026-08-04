> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).

# Making the post-window prevalence/recording allocation explicit (`DSP009`)

**Date:** 2026-08-04

**Status:** Implemented and fitted. `DSP009` is registered, tested and documented
in `docs/models/README.md`. The reported fits are extended runs over 2004-2024 —
4 chains × 4,000 tune + 4,000 draws at `target_accept` `0.99`, all clearing the
convergence gate with zero divergences — produced in a local environment that is
**not release-conformant**, so regenerate before citation. Companion to the
[surveillance workbook extraction note](20260803-degraaf-surveillance-workbook-extraction.md),
which implemented the anchor and identified this gap, and to the
[model-family review](20260803-dsp-core-model-family-review.md), whose Finding 5
is the general form of the problem.

**Two findings, and the second was not the objective.** `DSP009` makes the
post-window prevalence/recording allocation explicit, and shows that forbidding it
biases the 2016-2024 total down by about `1.8%` while narrowing its interval by a
factor of `1.4`. Separately, and more consequentially, fitting it properly exposed
a **degenerate mode in the anchored likelihood** — one chain in four escaped to a
region where the surveillance anchor effectively switches off. A per-chain audit of
all 19 anchored fits in the repository confirms **no existing `DSP007` or `DSP008`
result is contaminated**, and identifies the three conditions the mode jointly
requires.

## Question

`DSP008` fixed the level: latent Down-syndrome livebirth prevalence is observed
through the surveillance programmes' overlapping five-year window means, so the
2016-2024 total no longer rests on the reduction-rate CSV. What it did not fix is
the *trajectory* after the windows stop.

The workbook note recorded the consequence plainly: under `DSP008` latent
prevalence rises to `13.8` per 10,000 around 2018 and then falls to about `12.9`
by 2024, and the model "is not discovering that prevalence fell; it is inheriting
the constant-`s` assumption and reporting the consequence." A specification
allowing `s` to drift after the last window would attribute the same decline to
recording and fit equally well.

That is an unattributed modelling default sitting underneath a headline number.
`DSP009` converts it into a stated prior with reportable corners.

## What `DSP008` leaves unidentified, precisely

Surveillance windows are centred. With mid-years running `2004`-`2018` and a
half-width of two, the last window constrains the mean over `2016`-`2020`, so the
anchored span ends at 2020 and **`2021`-`2024` carry no surveillance observation
at all**. Across those four years the cell likelihood sees only the product

```text
theta_age * eta_year * (s - f)
```

with nothing external on either factor. `eta` and `s` trade off exactly. Holding
`s` constant does not resolve that; it picks one end of it.

The size of the thing being allocated is not small. The review measured the
identified quantity `(R/N - f) / (sum(N * theta) / N)` falling `-19.9%` across
2016-2024, and every substantive claim about change over time is a choice about
how to divide that decline.

## Design

`DSP009` is `DSP008` plus a random walk on `logit s` over exactly the unanchored
years, and nothing else. Age model, reduction model, revision split and cell
definition are all unchanged, so the comparison isolates one thing.

**A new orthogonal axis, not a new recording model.** `recording_drift` is a
separate field on `CoreModelDefinition` taking `none` or `post_anchor`, rather
than a fourth value of `recording_model`. The drift is conceptually independent
of how recording varies cross-sectionally, and folding it into the existing enum
would multiply values combinatorially as either axis grows. Two combinations are
rejected at validation: without `reduction_model='anchor'` there is no last
covered year to drift from, and with `recording_model='year'` the centred year
offsets and the post-window walk are two parameterisations of the same
year-varying sensitivity.

**Where the drift starts is derived, not configured.** The last anchored model
year is `max(mid_year_idx) + half_width`, so it follows from the anchor CSV. For
2004-2024 that is 2020, giving four drifting years. A range whose windows reach
the final modelled year raises rather than silently producing a no-op.

**`recording_s` keeps its meaning.** It remains the anchored-era
revised-certificate sensitivity in a drifted fit, so it stays directly comparable
with `DSP006` and `DSP008` and with fits confined to 2016 onward. The drift is
carried separately as `recording_s_drift_logit`, identically zero for every year a
window still reaches, with `recording_s_drift_ratio` reporting the final year's
sensitivity relative to that anchored level.

**The drift shifts both certificate versions together.** It models the
certificate's recording behaviour over time, not a change in the gap between
revised and unrevised forms. Unrevised records only exist before 2016, always
inside the anchored span, so in practice the drift term is zero wherever a cell
is unrevised.

**The drift SD is fixed, not estimated.** This is the same discipline
`--anchor-obs-sigma-fixed` already applies to surveillance accuracy, and for the
same reason: the quantity is not identified, so a free parameter would let the
posterior geometry pick a value that reads as evidence. The default is `0.06` per
year, calibrated so the cumulative prior width over a four-year unanchored tail
spans this repository's own bracketing allocation. The de Graaf-derived recording
anchor in `notes/figures/recording_rates_anchor.csv` has `s` for Non-Hispanic
White falling `17%` over 2016-2024, about `0.12` logit units across four years, or
one cumulative SD at that value. That is a calibration to bracket, **not** an
estimate, and the anchor cannot arbitrate the question anyway — it is built from
the same recorded counts and the same Morris curve, as the
[anchor diagnostic](20260707-s-anchor-and-identifiability-diagnostic.md) records.

### The two corners

| Configuration | Post-window decline attributed to |
| --- | --- |
| `--recording-s-drift-sigma 0` | Prevalence (identical to `DSP008`) |
| default drift `0.06` | Both, in the ratio of the drift SD to the anchor's state variances |
| `--anchor-forecast-flat --recording-s-drift-sigma 0.20` | Recording |

`--anchor-forecast-flat` holds latent prevalence at its last anchored value
instead of forecasting it, by masking the latent increments from that year
onward. It applies to `DSP007` and `DSP008` as well, so the all-recording corner
can be run against any anchored specification.

The zero-drift corner is *exactly* `DSP008`, not approximately: the graph carries
identical free random variables, identical named variables and identical
log-probability at a shared point. That is asserted in
`tests/test_core_reduction_model.py`, so the corner cannot drift apart from its
parent through later edits. It is also why the all-prevalence row below is a
`DSP008` fit rather than a redundant `DSP009` run.

## The drift exposed a degenerate mode in the anchored likelihood

This was not the expected finding and it matters more than the headline, because
it applies to `DSP007` and `DSP008` as well.

Fitting `DSP009` at the reporting profile gave max R-hat `1.0111` — marginal, and
easy to read as needing a longer run. Extending to 4,000 tune plus 4,000 draws at
`target_accept` `0.99` made it **far worse**, not better: max R-hat `1.5299`,
minimum ESS `7`. Degradation under better sampling is the signature of a
pathology that improved exploration *finds* rather than fixes.

Per-chain means locate it exactly. Three of four chains agree to four significant
figures; one escaped:

| Quantity | Chains 0, 2, 3 | Chain 1 |
| --- | ---: | ---: |
| `anchor_obs_sigma` | `0.0125` | `0.8417` |
| `anchor_level_sigma` | `0.0119` | `0.1310` |
| `recording_s` | `0.3350` | `0.0060` |
| `prevalence_year` 2004, per 10,000 | `12.69` | `720.9` |
| `eta_year` 2004 | `0.702` | `39.88` |

### What the escaped mode actually is

Decomposing the log-probability at each chain's mean settles the mechanism, and it
is not the one an earlier draft of this note asserted.

| Component | Chains 0, 2, 3 | Chain 1 | Delta |
| --- | ---: | ---: | ---: |
| Cell Binomial log-likelihood | `-3,596.8` | **`-3,513.1`** | `+83.7` |
| Surveillance observation log-likelihood | `+41.5` | `-157.2` | `-198.7` |
| `anchor_obs_sigma` prior | `+2.74` | `-138.92` | `-141.66` |
| `anchor_level_sigma` prior | `+3.51` | `-17.77` | `-21.27` |
| `recording_s_logit` prior | `-1.15` | `-13.99` | `-12.84` |
| **Total** | | | **`≈ -291`** |

The escaped mode **fits the recorded counts better**, by `84` log units. It pays
for that by discarding the surveillance anchor, which an inflated
`anchor_obs_sigma` makes affordable: at `0.84` the observation equation contributes
almost nothing, and the prior on the SD is the single largest cost in the table.

So this is the `eta * s` ridge the whole project documents. The cell likelihood on
its own prefers a lower `s` and a higher total — consistent with Boulet's
record-linkage sensitivity and with the bounds in the group note, both of which
suggest the fitted `s` sits at the low end. The anchor is what holds `s` at `0.335`.
Weaken it and the likelihood slides down the ridge.

It is a **genuine local mode, about `291` log units down**, not a competitive one
and not a flat plateau. A chain reaches it during tuning, when step sizes are still
large, and then cannot get back: returning requires `anchor_obs_sigma` to shrink
while `eta` is still wrong, which is expensive in both terms at once. Tighter
stepping at `target_accept` `0.99` made return *less* likely, which is why extending
the run made the diagnostics worse rather than better.

**An earlier draft of this note attributed the mode to the `theta * eta` clip.
That was wrong.** The clip binds in `16%` of cells there, so the mode is not
sustained by a gradient-free plateau — `84%` of cells still supply gradient, and the
mode is a real basin with a real barrier around it. What the clip *does* do is
weaken the escape: measured on the same graph, the restoring gradient in the latent
level drops roughly fourteenfold as `eta` crosses `26` and the clip starts to bind.
So the clip helps hold a chain there without having put it there. That distinction
matters because it changes the remedy — see below.

**Fixing the observation SD closes it.** Re-running the same extended
configuration with `--anchor-obs-sigma-fixed 0.05` gives max R-hat `1.0015` and
minimum ESS `1,722`. This is also what the workbook note already argues should be
done on separate grounds: an estimated observation SD measures only whether the
windows are *mutually consistent*, never whether the surveillance prevalences are
*accurate*, so fixing it is the honest sensitivity axis. The methodological
preference and the numerical requirement coincide, so every fit reported below
fixes it.

**A fixed observation SD is now the default for anchored models**, at `0.05`, in
`build_core_reduction_model` as well as the fit CLI, so the escape route is closed
by default rather than by remembering a flag. `--anchor-obs-sigma-estimated` opts
back out and the fit warns when it does. Note what this changes: anchored fits run
without arguments now report an interval roughly twice as wide as before, because
they have stopped asserting that surveillance prevalence is measured to about one
percent. The mean barely moves — `44,589` against `44,544` for `DSP008` — so this
is a correction to reported precision, not to the level.

### The clip is now a barrier with a gradient

Fixing the observation SD removes today's route to the mode. It does not remove the
flat region, and a flat region is a hazard for any future specification that
reaches it. The clip is therefore retained only for what it is needed for — keeping
`theta * eta` a valid Binomial probability — with a smooth barrier alongside it that
supplies the gradient the clip destroys:

```text
overshoot = sum( softplus(k * (theta * eta - 1)) / k )
Potential  = -w * overshoot            # k = 200, w = 1000
```

`softplus(k x) / k` is `~0` for `x` well below zero and `~x` well above, so the term
is inert while `theta * eta` is a valid probability and grows with a non-zero
gradient once it is not. The calibration matters and is generous: `theta` peaks at
`0.038` at age 50, so `theta * eta` reaches one only near `eta = 26`, against a
posterior `eta` of about `0.6`. Even at `eta = 1` — the anchored prevalence equalling
the Morris no-reduction expectation, which the model deliberately permits as a
`rho < 0` diagnostic — the largest `theta * eta` is `0.038` and the barrier
evaluates to `exp(-194)`, which is zero in double precision. It cannot perturb any
fit that could be reported, and two tests pin that: the barrier term is below
`1e-30` at plausible values and the log-probability matches the clip-only graph to a
relative `1e-12`, while past the clip it costs hundreds of log units and strengthens
the restoring gradient.

One side effect worth knowing: PyMC warns that potentials are ignored during
prior-predictive sampling, so anchored fits now emit that warning. It is harmless
here — prior draws of the latent level stay near the anchored prevalence and never
approach `eta = 26` — and the review already records that prior-predictive draws are
saved but never consumed.

**The barrier alone rescues the configuration that failed.** Re-running the exact
fit that went degenerate — `DSP009` at 4,000 draws with
`--anchor-obs-sigma-estimated`, so the observation SD is free and the old escape
route is open — now gives max R-hat `1.0019` and minimum ESS `1,797`, against
`1.5299` and `7` before. All four chains agree: `anchor_obs_sigma` `0.0123`-`0.0125`,
`recording_s` `0.335`, `eta` `0.702`. The per-chain audit returns CLEAN.

The mechanism is confirmed by what the barrier did *not* do. The largest overshoot
term across all 16,000 retained draws is `1.8e-85`, so the barrier never engaged in
the posterior at all. It engaged only during tuning, and expelled the chain that
would otherwise have been trapped. So the basin is created by the anchor being
discardable, but a chain *stays* there because the clip flattens the escape
gradient. Repairing either is sufficient, and both are now in place.

Fitted results are unchanged, as they must be. At matched seeds and settings the
2016-2024 total moves `44,544` to `44,555` for `DSP008` and `45,370` to `45,359` for
`DSP009` — about two Monte Carlo standard errors, in opposite directions, with `s`
agreeing to four decimal places and interval widths to `0.02` percentage points.

An earlier draft of this note reported `DSP009` results from the standard
reporting profile. Those are withdrawn: that run had not yet found the second
mode, so its interval was an artefact of incomplete exploration.

### No existing anchored run is contaminated

`scripts/audit_anchored_chain_health.py` audits every anchored fit under
`output/` per chain rather than through pooled statistics, which is the only way
this failure is visible — three healthy chains out of four produced a max R-hat of
`1.0111`, indistinguishable from a run that merely needs lengthening. It flags a
chain by the share of its draws with `eta > 1.5` and by between-chain dispersion in
`recording_s`, and exits non-zero under `--strict`.

**All 18 anchored runs other than the one above are CLEAN.** That includes both
frozen `DSP007` runs, all five frozen `DSP008` runs, and the extended fits reported
here. In every clean run the four chains agree on `recording_s` to better than
`0.3%`, `anchor_obs_sigma` sits at `0.012`-`0.014` where it is free, and `eta`
never exceeds `0.88` — against `50.9` in the escaped chain. **No number recorded
anywhere in this repository from an anchored fit is affected.**

The escaped chain sat in the mode for **100% of its draws**, so it entered during
tuning and never left. That is what a deep basin behind a real barrier predicts,
and it also means the failure is not a transient excursion that longer sampling
would average away.

### What the mode actually requires

Auditing across specifications turns the mechanism into a factorial result. All
rows are 4 chains × 4,000 draws at `target_accept` `0.99`, the settings at which the
mode appears:

| Model | Drift | Observation SD | Prevalence tail | Verdict |
| --- | --- | --- | --- | --- |
| `DSP007` | none | free | forecast | CLEAN |
| `DSP008` | none | free | forecast | CLEAN |
| `DSP009` | `post_anchor` | free | forecast | **DEGENERATE** |
| `DSP009` | `post_anchor` | free | flat | CLEAN |
| `DSP009` | `post_anchor` | fixed `0.05` | forecast | CLEAN |
| `DSP009` | `post_anchor` | fixed `0.10` | forecast | CLEAN |

Three ingredients are jointly necessary at these run lengths: a free observation
SD, a drifting `s`, and a forecasting rather than pinned prevalence tail. Remove any
one and the mode does not appear. So the earlier framing — that the drift merely
shortens a path `DSP007` and `DSP008` were already on — is **too pessimistic about
those two**. At matched settings neither finds it, and the drift is doing real work
in opening the route.

What cannot be concluded is that the mode is *absent* from the `DSP007`/`DSP008`
posteriors. A clean verdict is evidence about what four chains explored, not a proof
of unimodality, and ten of the clean runs have fewer than 4,000 draws per chain —
precisely the regime in which this mode was invisible in `DSP009`. The audit prints
that caveat rather than letting a clean row read as a guarantee.

One incidental finding: the frozen `output/refit2004/DSP008-obs0.05` run carries
max R-hat `1.0147`, above the `<1.01` gate, and nothing in the repository records
it. It is clean on the per-chain audit, so this is a convergence shortfall rather
than the mode.

## Results

All fits are 2004-2024, 4 chains × 4,000 tune + 4,000 draws at `target_accept`
`0.99`, with the surveillance observation SD fixed for the reason above.
Distinct seeds per row. The 2016-2024 total is computed on the posterior draws, so
the interval is on the sum rather than a sum of per-year intervals. Every row has
max R-hat below `1.002`, minimum ESS above `1,800`, and **zero divergences**.

Because the width is largely a function of the assumed surveillance accuracy, the
comparison is run at two values of that assumption. The `0.05` panel is the
primary; the `0.10` panel exists to show the finding does not depend on it.

### Surveillance observation SD fixed at `0.05`

| Fit | 2016-2024 total | 89% ETI | Width | vs. corner | MCSE |
| --- | ---: | --- | ---: | ---: | ---: |
| `DSP008` — all prevalence | `44,544` | `43,295`-`45,819` | `5.66%` | — | `5.6` |
| `DSP009` — drift `0.06` | `45,370` | `43,612`-`47,201` | `7.91%` | `+1.85%` | `7.6` |
| `DSP009` — flat + `0.20`, all recording | `45,795` | `44,316`-`47,303` | `6.52%` | `+2.81%` | `6.2` |

| Fit | `s` revised | `s₂₀₂₄` / anchored | Prevalence 2024 vs 2018 | 2021-2024 subtotal |
| --- | ---: | ---: | ---: | ---: |
| `DSP008` | `0.3363` | `1.0000` | `-6.25%` | `18,858` |
| `DSP009` drift `0.06` | `0.3364` | `0.9629` | `-1.68%` | `19,616` |
| `DSP009` flat corner | `0.3364` | `0.9437` | `+0.18%` | `19,981` |

### Surveillance observation SD fixed at `0.10`

| Fit | 2016-2024 total | 89% ETI | Width | vs. corner | Prevalence 2024 vs 2018 |
| --- | ---: | --- | ---: | ---: | ---: |
| `DSP008` — all prevalence | `44,405` | `42,318`-`46,582` | `9.60%` | — | `-6.34%` |
| `DSP009` — drift `0.06` | `45,184` | `42,744`-`47,763` | `11.11%` | `+1.75%` | `-1.95%` |

### What the rows say

**Forbidding drift biases the total down, by about `1.8%`.** The shift is
`+826` births at `0.05` and `+779` at `0.10`, against two-combined-MCSE bands of
`26` and `47` — far above simulation noise, and stable across the surveillance
assumption. The mechanism is direct: allowing `s` to fall in the unanchored years
means the same recorded counts imply more true cases.

**The total is insensitive to *how* the decline is divided, once division is
permitted at all.** The drifted row and the all-recording corner sit `0.94%`
apart, against `1.85%` between the drifted row and the no-drift corner. The
consequential choice is whether `s` may move, not how far.

**The effect lands where it should.** The 2021-2024 subtotal rises `4.0%` from
`18,858` to `19,616` while the anchored 2016-2020 span moves `0.3%`. A drift
bleeding into the anchored years would mean it was absorbing signal the windows
already constrain, so this is a check on the implementation as much as a result.

**`DSP008`'s post-2018 prevalence decline is largely an artefact of its own
constant-`s` assumption.** `DSP008` reports latent prevalence falling `6.25%` from
2018 to 2024. Permitting drift cuts that to `1.68%`, and the all-recording corner
turns it to `+0.18%`. The apparent decline is mostly the model booking a falling
recorded rate to the only free factor it has.

**The level the anchor identifies is untouched.** `s` revised is `0.3363` or
`0.3364` in every row of the primary panel — agreement to four decimal places.
The drift changes the trajectory without disturbing the level, which is what keeps
the comparison with `DSP006` and the modern-window fits valid.

**The drift matters most if you believe surveillance is accurate.** The interval
widens `1.40×` at an observation SD of `0.05` but only `1.16×` at `0.10`, because
by `0.10` surveillance uncertainty already dominates the total. So the two open
assumptions are not additive: resolving surveillance accuracy would *increase* how
much the allocation question matters, not reduce it.

**A side observation that corroborates the workbook note.** Freeing the
observation SD instead of fixing it at `0.05` moves `DSP008`'s mean by `-0.1%`
(`44,589` against `44,544`) while halving its width (`2.87%` against `5.66%`).
That is the note's own finding — the mean is robust, the width is almost entirely
a function of an unknown — reproduced here at a longer run.

## What this buys, and what it does not

**It buys candour, not identification.** Nothing after 2020 distinguishes falling
prevalence from falling recording, and `DSP009` does not pretend otherwise. What
changes is that the choice is now a named parameter with reportable corners
instead of a property of the default specification. The three rows above are a
**sensitivity envelope, not a credible interval** — the same distinction the
family already draws for the `lambda`/`delta` calibration grid, and the draws must
not be pooled or model-averaged across them.

**It shows the constant-`s` assumption is not free.** Forbidding drift, as every
other anchored model does, biases the 2016-2024 total down by about `1.8%` and
narrows its interval by a factor of `1.4` at the primary surveillance assumption.
Neither is a rounding error, and both were previously invisible.

**It cost more than it was expected to.** The exercise was scoped as a
presentational improvement — same evidence, honest interval. It also surfaced a
degenerate mode that had been latent in the anchored likelihood since `DSP007`,
and that finding is worth more than the interval. Specifications that open new
directions in a posterior are a way of stress-testing the ones already there.

### One independent corroboration, not yet written up

A separate diagnostic run during this session gives the first evidence on this
question that is not circular. The 2003 certificate records Down syndrome as one
checkbox in a shared congenital-anomaly item, and several conditions on that same
item have no prenatal-reduction channel worth speaking of — hypospadias is
male-only and not prenatally diagnosable, cleft lip and palate are essentially
never terminated in the US, limb reduction likewise. Their recorded rate is
therefore close to a direct reading of the item's recording sensitivity.

Age-standardised and pooled, that control panel falls `-9.3%` from 2016-2018 to
2022-2024 with roughly `5,300` flags a year, against `-17.4%` for the Down
syndrome identified ratio over the same span. Taken at face value it splits the
decline about evenly between recording and reduction. That is the same direction
`DSP009` reports, from a completely independent route.

On rate it points slightly further than the drift prior admits. The panel's
`-9.3%` spans about six years between period midpoints, roughly `-1.6%` a year,
while the fitted `s₂₀₂₄`/anchored ratio of `0.9629` is `-3.7%` across four years,
roughly `-0.9%` a year. If the panel is right, the default drift SD of `0.06` is
conservative and the all-recording corner is the better guide of the two — which
is an argument for reporting the envelope rather than the middle row.

It is **not** used to set the drift prior and must not be read as validating it:
the figures are aggregate ratios computed in a non-conformant environment, a
single shared factor across all anomaly checkboxes is refuted by the panel itself
(cyanotic congenital heart disease *rose* `+15.9%` over the same window, plausibly
from pulse-oximetry screening), and the between-condition spread is wide enough
that trusting one control rather than another moves the implied split from `17%` to
`86%` recording. Turning it into an identification strategy needs each control's
true prevalence trend pinned from published NBDPN surveillance, which is the
proposed `DSP010`. The wider exploration of options this came from has not been
written up.

## Caveats

- **The drift SD has no evidential basis.** It is calibrated to bracket a range,
  not estimated. Any reported figure is conditional on it, and the corners are
  the honest presentation.
- **The reporting profile is marginal for this model family, and marginal in a
  misleading direction.** At the standard profile neither `DSP008` (`1.0100`) nor
  drifted `DSP009` (`1.0111`) clears the `<1.01` R-hat gate, and the `DSP009`
  value understates the problem by a wide margin — extending the run took it to
  `1.5299`. A marginal R-hat in this family should be treated as a reason to
  extend the run, never as a near-pass.
- **`anchor_obs_sigma` should not be estimated**, on the reporting argument
  alone. A free observation SD also admits the degenerate mode above, which is why
  `--anchor-obs-sigma-fixed` is now the default and why the `theta * eta` clip now
  carries a smooth barrier. Both are in place, and either is independently
  sufficient against the mode; the reporting argument is what keeps the SD fixed
  even so.
- **All standard-profile fits carried divergences**, and
  `scripts/fit_core_reduction_model.py` never reads `sample_stats.diverging` — the
  review's Finding 7 item 2. The counts here were extracted by hand. Notably the
  *passing* corner carried the most divergences at the standard profile, so the
  gate and the geometry were not telling the same story. The extended
  fixed-SD fits carry none.
- **`DSP009` needs more tuning than `DSP008`.** The drift deliberately opens a
  ridge, and short chains wander along it rather than exploring it. At 150 tune
  plus 150 draws `DSP008` converges to max R-hat `1.024` while `DSP009` reaches
  `2.3` with an effective sample size near `3` and posterior means far outside any
  plausible range (`eta` of `414`). Do not shorten the profiles for it, and read
  R-hat on `recording_s_drift_innovation_raw` rather than only on the cumulated
  `recording_s_drift_logit`, whose anchored-era coordinates are identically zero
  and so return NaN diagnostics by construction.
- **The model comparison script cannot compare these fits.**
  `scripts/compare_core_reduction_models.py` fails with a duplicate-index error on
  any pair of revision-split runs, because such cells carry two rows per
  `(year, age)` and the common-grid posterior-predictive plot pivots without
  aggregating over the revision dimension. This predates `DSP009` — two `DSP008`
  runs fail identically — and affects `DSP006` too. The comparison in this note
  was computed directly from the saved InferenceData instead.
- **This does not touch the cross-sectional problem.** Group, payer and education
  splits still identify only the product `eta(g) * s(g)`; see the
  [group-identification note](20260803-group-reduction-recording-identification.md).
- **Everything upstream still applies.** The false-positive rate is still the
  mis-derived `7.8e-5` default, ART is still uncorrected in the cell definition,
  and the surveillance prevalences still carry no uncertainty — which the workbook
  note identifies as the binding constraint on the whole model.

## Reproducing

The anchored fits read `output/degraaf_surveillance/expected_births_anchor.csv`,
so the extraction script must run first:

```bash
python scripts/extract_degraaf_surveillance.py
```

All three rows use the same extended settings, which are **beyond** the reporting
profile. The profile alone does not clear the convergence gate for this family, so
`--tune 4000 --draws 4000 --target-accept 0.99` is the minimum for a citable fit.

The all-prevalence corner is a `DSP008` fit rather than
`DSP009 --recording-s-drift-sigma 0`, because those two are provably identical:

```bash
python scripts/fit_core_reduction_model.py DSP008 --profile reporting --years 2004-2024 --random-seed 47 --tune 4000 --draws 4000 --target-accept 0.99 --anchor-obs-sigma-fixed 0.05
```

The divided-by-prior allocation:

```bash
python scripts/fit_core_reduction_model.py DSP009 --profile reporting --years 2004-2024 --random-seed 101 --tune 4000 --draws 4000 --target-accept 0.99 --anchor-obs-sigma-fixed 0.05
```

The all-recording corner:

```bash
python scripts/fit_core_reduction_model.py DSP009 --profile reporting --years 2004-2024 --random-seed 202 --anchor-forecast-flat --recording-s-drift-sigma 0.20 --tune 4000 --draws 4000 --target-accept 0.99 --anchor-obs-sigma-fixed 0.05
```

Repeat with `--anchor-obs-sigma-fixed 0.10` for the second panel. Omitting
`--anchor-obs-sigma-fixed` reproduces the degenerate mode; run it that way only to
confirm the diagnosis.

The per-chain audit covers every anchored fit under `output/` and needs no
arguments. Pass directories to narrow it, and `--strict` to make a contaminated
run fail a pipeline:

```bash
python scripts/audit_anchored_chain_health.py --strict
```

Seeds are deliberately distinct across the three rows. The review recorded that a
scenario fit sharing seed 47 with its baseline had its MCSE-based materiality
classification withdrawn, so shared seeds are unsafe for corner comparisons.

The comparison tables were computed directly from the saved InferenceData, because
`scripts/compare_core_reduction_models.py` cannot compare revision-split runs (see
caveats). The 2016-2024 total is summed over model-year indices 12-20 on the draws
before summarising.

## Recommended next steps

1. **Do not report a single row.** Until the trajectory question is settled,
   report the envelope and say which corner is preferred and why. `DSP008` alone
   understates both the total and its uncertainty.
2. **This is a question for Gert, and it is already on the list.** Ask 7 in the
   workbook note asks whether the `7.3%` prevalence decline his column H implies
   for 2016-2024 is an intended estimate or a by-product of extrapolating the
   recording fraction. That is precisely the choice `DSP009` parameterises: his
   column H direction is the all-prevalence corner. His answer selects a row.
   Ask 3 — whether windows after 2016-2020 exist or are expected — would retire
   the question outright for part of the span.
3. **Build the anomaly-panel design as `DSP010`.** It is the only route
   identified so far that can actually *divide* the post-window decline rather
   than parameterise the division, and it works in exactly the years the anchor
   does not reach. Curate the control set against published NBDPN prevalence, and
   test the shared-factor restriction against the anchor over 2016-2018, where the
   two overlap.
4. **The anchor-off mode is closed; keep auditing anyway.** All three remedies
   are **done**: the frozen runs are audited and clean, the observation SD is fixed
   by default, and the clip now carries a smooth barrier that restores the escape
   gradient. Either of the last two is independently sufficient. What remains is
   habit rather than work — run `scripts/audit_anchored_chain_health.py --strict`
   after any anchored refit, since the audit costs seconds and this failure was
   invisible in every pooled statistic. If a future specification wants a free
   observation SD back, it is now safe to try, but check it per chain first.
5. **Read divergences at fit time** and fail the gate on them, alongside the
   other verification items in the review's recommendation 7. Add a per-chain
   agreement check too: this failure was invisible in the pooled summary but
   obvious in one line of per-chain means.
6. **Fix the comparison script** for revision-split models, with a regression
   test — `tests/test_compare_core_reduction_models.py` currently exercises only
   unsplit cells.
7. **Revisit the reporting profile for anchored models.** Two of three fits
   miss the R-hat gate at the profile intended for publication, which is a
   profile problem rather than three separate model problems.
