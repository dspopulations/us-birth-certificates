> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-5).

# Bayesian Model Inventory

The aggregate Down syndrome birth-certificate accounting models use stable
`DSPnnn` identifiers. The numbers index the historical order in which models
entered the reproducible fitting workflow; they are not a hierarchy and they do
not imply that the highest-numbered model is preferred.

| Model | Status | Age resolution | Recording structure | Combined reduction | Purpose |
| --- | --- | --- | --- | --- | --- |
| `DSP001` | Discretisation sensitivity | Seven bands | Constant `s` | One value per year | Original core accounting model; retained to measure the effect of evaluating the Morris curve in broad bands. |
| `DSP002` | Band-resolution sensitivity | Seven bands | Partially pooled `s_year` | One value per year | Tests year-varying recording under the original broad-band age approximation. |
| `DSP003` | Age-structure diagnostic | NCHS single-age codes | Constant `s` | Smooth age pattern within year | Tests how much residual age structure can be absorbed by combined reduction while preserving each year's natural-DS-weighted surveillance margin. |
| `DSP004` | Preferred structure; conditional reference | NCHS single-age codes | Constant `s` | One value per year | Removes the broad-band Morris approximation while retaining the simplest transparent reduction-recording structure; totals remain conditional on calibration scenarios. |
| `DSP005` | Year-recording sensitivity | NCHS single-age codes | Partially pooled `s_year` | One value per year | Tests whether year-specific recording materially changes the preferred exact-age baseline. |
| `DSP006` | Measurement-era control | NCHS single-age codes | Separate revised / unrevised `s` | One value per year | Splits recording sensitivity by 2003-certificate revision so a window spanning the 2004-2015 phase-in can identify that measurement shift instead of absorbing it into a time trend. Needs a year range crossing the phase-in. |
| `DSP007` | Level identification | NCHS single-age codes | Constant `s` | Consequence of an anchored prevalence | Replaces the reduction-rate prior with a latent annual prevalence observed through the surveillance programmes' overlapping five-year window means, so the level is set by data rather than imported. |
| `DSP008` | Level identification with era control | NCHS single-age codes | Separate revised / unrevised `s` | Consequence of an anchored prevalence | Combines the `DSP007` anchor with the `DSP006` revision split. Both fixes matter independently, so this is the specification that carries them together. |
| `DSP009` | Post-window allocation candour | NCHS single-age codes | Revised / unrevised `s`, drifting past the last window | Consequence of an anchored prevalence | Adds a random walk on `logit s` over the years no surveillance window covers. `DSP008` holds `s` constant there, so a falling recorded rate can only be read as falling prevalence; `DSP009` makes that allocation an explicit prior. It does not identify the split — see below. |

All models use the same Quarto template at
`docs/models/selection_core_reduction/index.qmd`. The fit CLI copies that
template into each run directory and records the selected model in `config.json`.

“NCHS single-age codes” is not exact at both endpoints: code 12 represents ages
10-12 and code 50 represents ages 50 and over. The Morris curve is evaluated at
representative ages 12 and 50 for those pooled cells.

Typical commands:

```bash
python scripts/fit_core_reduction_model.py DSP001 --profile reporting --render
python scripts/fit_core_reduction_model.py DSP002 --profile reporting --render
python scripts/fit_core_reduction_model.py DSP003 --profile reporting --render
python scripts/fit_core_reduction_model.py DSP004 --profile reporting --render
python scripts/fit_core_reduction_model.py DSP005 --profile reporting --render
python scripts/fit_core_reduction_model.py DSP006 --profile reporting --years 2004-2024
python scripts/fit_core_reduction_model.py DSP008 --profile reporting --years 2004-2024
python scripts/fit_core_reduction_model.py DSP009 --profile reporting --years 2004-2024
python scripts/compare_core_reduction_models.py \
  output/selection_core_reduction/DSP001/<timestamp> \
  output/selection_core_reduction/DSP004/<timestamp>
python scripts/compare_core_reduction_models.py \
  output/selection_core_reduction/DSP004/<timestamp> \
  output/selection_core_reduction/DSP005/<timestamp>
python scripts/compare_core_reduction_sensitivities.py \
  output/selection_core_reduction/DSP004/<reference-run> \
  output/selection_core_reduction/DSP004/<sensitivity-run> [...] \
  --output-dir output/selection_core_reduction/comparisons/<comparison-name>
```

## DSP004 coherent surveillance-calibration controls

`DSP004` also exposes two sensitivity controls without assigning a new model
identifier:

- `--reduction-error-correlation`, stored in `config.json` as
  `reduction_error_correlation`, sets the common correlation $\lambda$ among
  yearly logit-scale reduction-prior errors while preserving every year's
  marginal prior variance; and
- `--reduction-calibration-shift-logit`, stored as
  `reduction_calibration_shift_logit`, applies a fixed common logit shift
  $\delta$ to the complete reduction trajectory.

The default `lambda=0`, `delta=0` configuration reproduces the independent,
unshifted prior. The pre-specified primary grid has seven unique scenarios:

| Role | $\lambda$ | $\delta$ |
| --- | ---: | ---: |
| Baseline | `0` | `0` |
| Correlation sensitivity | `0.5` | `0` |
| Correlation sensitivity | `0.9` | `0` |
| Common-level stress | `0` | `-0.4` |
| Common-level stress | `0` | `-0.2` |
| Common-level stress | `0` | `+0.2` |
| Common-level stress | `0` | `+0.4` |

All seven scenarios hold `f=7.8e-5` and the observed/extrapolated reduction-
prior logit SDs at `0.20 / 0.45`. The shift values are stress values, not
validated bounds. Aggregate materiality is pre-specified as either an absolute
change of at least 5% in the total posterior mean or an increase of at least
25% in the 89% ETI width. The decomposition checks are an absolute change of at
least `0.05` in mean $s$ and at least 10% in the posterior mean of draw-by-draw
model-implied missed true cases, $T(1-s)$ for constant-$s$ `DSP004`.

If an aggregate rule triggers, the protocol adds joint corners
`(lambda, delta)=(0.9, -0.4)` and `(0.9, +0.4)` and checks them against the
separate-axis scenario envelope. That envelope is descriptive, not a posterior
interval; the scenario draws are not pooled or model-averaged. These controls
test common error correlation and a common level shift. They do not test the
extrapolated-tail slope or establish trend robustness, and all missed-case
quantities remain population aggregates rather than individual classifications.

The completed primary grid triggers the aggregate rule. Its posterior means
span 37,610-50,190, with outer 89% ETI endpoints of 35,023-52,177. The required
`(lambda, delta)=(0.9, +0.4)` corner gives 34,806 (32,727-36,924), outside that
separate-axis total envelope. Primary decomposition means span `0.300-0.401`
for $s$ and `22,572-35,151` missed true cases. The original negative-corner fit
shared seed 47 with the baseline, so its borderline MCSE classification was not
retained. An independent-seed refined run gives 46,336.589
(44,331.066-48,337.931), a `+4.704%` mean change. Its distance from the 5%
threshold is 131.025 births, exceeding its two-combined-MCSE band of 89.274
births, so the aggregate mean change is classified as immaterial. Its refined
`s=0.325` and missed-count mean of 31,298 (29,276-33,312), `+7.123%` from
baseline, are inside both primary decomposition envelopes and trigger neither
decomposition rule. The positive-shift corner remains outside both primary
decomposition envelopes at `s=0.433` and 19,769 missed cases, so its interaction
conclusion is unaffected. `DSP004` is therefore retained as the preferred
accounting structure and its independent, unshifted fit as a conditional
reference, not as a posterior that incorporates shared surveillance-source
uncertainty. Results must be reported by scenario; their envelope is not a
credible interval.

Broad-age posterior-predictive coverage remains `1/7`: coherent surveillance
calibration does not resolve the residual maternal-age allocation. The next
model-adequacy gate is the mirrored age-on-recording diagnostic; neither age
allocation should be treated as identified from certificates alone.

## Anchored models: the surveillance observation SD is fixed

Anchored models (`DSP007` onward) hold the surveillance observation SD **fixed**,
at `0.05` by default, rather than estimating it. Two independent reasons point the
same way.

**Reporting.** An estimated SD measures only whether the overlapping windows are
mutually *consistent* with a smooth latent path. It cannot measure whether the
surveillance prevalences are *accurate*, because the source workbook supplies no
uncertainty at all. Estimating it returns about `0.012` and an interval on the
2016-2024 total of `2.87%`, which amounts to asserting that surveillance
prevalence is measured to about one percent. Nothing supports that, and the
[workbook note](../../notes/20260803-degraaf-surveillance-workbook-extraction.md)
says plainly not to report it.

**Numerical.** A free SD admits a degenerate mode. Its half-normal prior does not
prevent it reaching `0.84`, and at that value the observation equation contributes
almost nothing to the log-probability, so the anchor effectively switches off.
Latent prevalence then runs up until $\theta\eta$ exceeds one for every maternal
age, where `p_ds_lb` is clipped — a flat region with no gradient, and therefore an
absorbing state. Recording sensitivity collapses towards zero to keep the product
near the observed recorded rate. One chain in four escaped there in a `DSP009` fit
at 4,000 draws per chain, and pooled convergence statistics did not make it
obvious: three healthy chains still gave a max R-hat of `1.0111`.

The fixed value is an **assumption about surveillance accuracy, not an estimate**.
Report across the sensitivity axis and say which value was chosen:

```bash
python scripts/fit_core_reduction_model.py DSP008 --years 2004-2024 --anchor-obs-sigma-fixed 0.10
```

`--anchor-obs-sigma-estimated` opts back out. It is not recommended, the fit warns
when it is used, and any such run should be checked per chain:

```bash
python scripts/audit_anchored_chain_health.py --strict
```

That audit walks every anchored fit under `output/`, flags a chain by the share of
its draws with $\eta > 1.5$ and by between-chain dispersion in `recording_s`, and
exits non-zero when a run is not clean. All anchored fits predating this default
have been audited and none is contaminated.

## DSP009 post-window allocation controls

Surveillance windows are centred, so with mid-years running to 2018 the anchored
span ends at 2020 and the years after it carry no external observation of
prevalence at all. Every anchored model except `DSP009` holds `s` constant
across those years, which means a falling recorded rate has only one place to go:
the fit reports falling prevalence. That is a consequence of the constant-`s`
default rather than a finding, and a specification letting `s` drift instead fits
the same data equally well.

`DSP009` makes the choice explicit. Two controls set where the post-window
decline is booked, and **neither is identified by the data** — nothing after the
last window distinguishes falling prevalence from falling recording, so the split
is decided by the prior. Report the corners with any drifted fit.

| Configuration | Post-window decline attributed to | Command |
| --- | --- | --- |
| All prevalence | Prevalence | `DSP009 --recording-s-drift-sigma 0` (identical to `DSP008`) |
| Divided by prior | Both, in the ratio of the drift SD to the anchor's state variances | `DSP009` (drift SD `0.06`) |
| All recording | Recording | `DSP009 --anchor-forecast-flat --recording-s-drift-sigma 0.20` |

The default drift SD of `0.06` per year is calibrated so its cumulative width
over a four-year unanchored tail spans this repository's own bracketing
allocation: the de Graaf-derived recording anchor in
`notes/figures/recording_rates_anchor.csv` has `s` for Non-Hispanic White falling
17% over 2016-2024, about `0.12` logit units across four years, or one cumulative
SD at that value. It is a stated assumption, not evidence.

`recording_s` remains the anchored-era revised sensitivity in a drifted fit, so
it stays directly comparable with `DSP006` and `DSP008`. The drift is carried
separately as `recording_s_drift_logit`, exactly zero for every year a window
still reaches, with `recording_s_drift_ratio` reporting the final modelled year's
sensitivity relative to its anchored-era level. `--anchor-forecast-flat` holds
latent prevalence at its last anchored value instead of forecasting it, and
applies to `DSP007` and `DSP008` as well.

Two properties are worth stating because they bound what the model can be asked
to do. The drift shifts revised and unrevised certificates together: it models
recording behaviour over time, not a change in the gap between certificate
versions. And a drifted fit should be expected to **widen** the interval on the
2016-2024 total rather than narrow it, because it stops asserting an allocation
the data cannot supply.

`DSP009` needs more tuning than `DSP008` does. The drift deliberately opens a
ridge — prevalence and recording trade off exactly along it after the last window
— and short chains wander along that ridge instead of exploring it. At 150 tune
plus 150 draws `DSP008` converges to max R-hat `1.024` while `DSP009` reaches
`2.3` with an effective sample size near `3` and posterior means far outside any
plausible range. Both profiles are healthy; do not shorten them for `DSP009`, and
read the R-hat on `recording_s_drift_innovation_raw` rather than only on the
cumulated `recording_s_drift_logit`.

## DSP004 race-surveillance audit

A no-refit audit reconstructs the independent, unshifted reporting `DSP004`
fit by exact maternal age, year, and the current seven race/Hispanic-origin
groups. The project lead confirmed on 2026-08-03 that the de Graaf source points
are centred five-year estimates based on maternal race and that prevalence is
the ratio of numerator and denominator counts pooled across each window. The
label 2016 therefore represents 2014-2018, of which the frozen fit supports only
2016-2018; it is excluded from the aligned comparison. The label 2018
represents 2016-2020 and is fully supported.

The earlier annual-label 2016/2018 comparison and cross-year transport result
are superseded and provide no evidence of repeatability. For the sole complete
window, the pooled count ratio is the sole source-aligned estimand. Its existing
comparison gives a material 2018 composition discrepancy (`TV` about 0.0803;
`WRMS` about 0.2184), including Asian/Pacific Islander and Hispanic
relative-rate contrasts. The earlier equal-year-rate calculation was
numerically close but is now a superseded sensitivity, not decision evidence.
Summing the native source birth denominators implies 25,128 named-group true
births, compared with a model posterior mean of 24,865; the source denominator
contains about 1.50% more named-group births than the model cohort. Applying
the same source rates to model births gives 24,781. The pooled data therefore
contain some absolute-scale information, but it remains inseparable from the
unresolved denominator mapping and cannot yet serve as a second national-scale
anchor.

The incomplete 2016 window means there is no temporal replication.
Hispanic-origin precedence, multi-race bridging, material source/model
denominator differences, source covariance, and overlap with the national
reduction evidence remain unresolved. The mirrored age-on-recording gate also
remains outstanding. The audit therefore records `calibration_eligible=false`
and authorises no race layer. Its local output must also be regenerated in a
release-conformant environment before it can be sealed. Resolve those source
and denominator definitions, obtain a second complete window, establish source
covariance and evidence dependence, and complete the mirrored age-on-recording
gate before reconsidering a time-invariant, composition-preserving race
extension.

The comparisons are descriptive and in-sample. `DSP004` is preferred over
`DSP001` because it removes an avoidable age-discretisation approximation, not
because it resolves the remaining age misfit. `DSP005` checks sensitivity to
year-varying recording. `DSP003` assigns the residual maternal-age pattern to
combined reduction while holding recording constant by age; its better
in-sample fit is therefore not evidence for that mechanism. None of the models
shows that birth-certificate counts alone identify recording separately from
pre-livebirth reduction. The headline estimates remain conditional on external
Morris and surveillance information and on the false-positive scenario. The
working false-positive range has little effect on the `DSP004` total under the
current reduction-prior widths, but it materially changes recording sensitivity;
widening the independent annual reduction priors approximately doubles the
headline interval width.

The [exact-age ablation note](../../notes/20260802-dsp004-dsp005-exact-age-ablations.md)
records the matched results and decision. The
[DSP003 note](../../notes/20260802-dsp003-age-reduction-extension.md) records the
age-structure and measurement sensitivities. The
[DSP004 measurement sensitivity note](../../notes/20260802-dsp004-false-positive-surveillance-sensitivity.md)
records the false-positive and reduction-prior-width grid and its conditional
interpretation. The
[coherent surveillance-calibration analysis](../../notes/20260803-dsp004-coherent-surveillance-calibration.md)
records the pre-specified correlated-error and common-shift protocol, fitted
results and conditional reporting decision. The
[race-surveillance audit](../../notes/20260803-dsp004-race-surveillance-audit.md)
records the no-refit protocol, reconstruction checks, descriptive findings and
fail-closed model decision.
