> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-5).

# DSP004 no-refit race-surveillance audit

**Date:** 2026-08-03

**Status:** Centred-window pooled-ratio protocol revision. The project lead
confirmed on 2026-08-03 that each labelled de Graaf estimate is centred on a
five-year window, uses maternal race, and calculates prevalence as a pooled
count ratio. Those clarifications supersede the initial
annual-label comparison: its 2016/2018 repeatability and cross-year transport
results are not aligned evidence and must not be cited as such. The frozen
`DSP004` fit supports all five years only for the estimate labelled 2018. The
remaining category-crosswalk, denominator, covariance, replication and
model-adequacy gaps keep calibration fail-closed. No race model is authorised
by this audit.

The numerical materiality thresholds and fail-closed fit permissions were
written before the first scripted audit run in this working session.
Preliminary source inspection had already established the available years and
approximate scale of several margins, so the rules were prospective but not a
blind pre-registration. Window and aggregation alignment were subsequently
corrected from the project lead's source-definition clarifications; choosing
the pooled operator is therefore definition-driven, not selected from the two
earlier calculations by whichever result was more favourable.

## Question

Can the centred five-year de Graaf surveillance prevalence estimates by
maternal race and Hispanic origin add calibration information to `DSP004`
beyond its national annual reduction priors?

The intended division of labour is:

- the annual reduction prior sets the national true-livebirth scale;
- race surveillance may inform the allocation of that total across the five
  supported named groups; and
- race-specific certificate counts may subsequently inform relative recording
  sensitivity, conditional on an externally supported true-birth allocation.

This phase does not fit race effects. It first asks whether the one fully
supported external window has a material pattern under the confirmed pooled
count-ratio operator, and whether the source definitions are compatible enough
to justify another model. With only one complete window, this phase cannot
establish temporal repeatability or transport.

## Source distinction

The corrected workbook contains two different types of quantity:

1. column R/L is surveillance-programme prevalence per 10,000, reported as a
   centred five-year pooled count ratio labelled 2016 and 2018; and
2. columns H/I are all-year estimated true counts and prevalence calculated
   from certificate counts divided by a regression-filled recording fraction.

Only the first source family is eligible for the primary audit, and only its
fully supported 2018-centred point enters the aligned comparison. Treating
columns H/I as an independent likelihood while also fitting the same
certificate counts would reuse the certificate evidence. The filled 2019-2024
values may be considered later as an explicitly circular sensitivity, not as
new surveillance observations.

The reproducible audit reads the corrected full extraction at
`data/us-births-degraaf-prevalence-recording-2000-2024.csv`. This retains the
workbook birth and recorded-count denominators beside the raw surveillance
field, allowing category alignment to be checked rather than assumed. The
shorter tracked ethnicity CSV contains the same raw surveillance values but
cannot support that denominator audit.

## Confirmed and unresolved source-alignment gate

The project lead confirmed on 2026-08-03 that:

- a point labelled year $c$ represents the centred window
  $\{c-2,c-1,c,c+1,c+2\}$; and
- the source groups are based on maternal race; and
- five-year prevalence is calculated by pooling the numerator and denominator
  counts across that window before taking their ratio.

This confirmation records the working source definition used by the project;
it is not being presented as a quotation from source documentation. The
repository still does not establish:

- the Hispanic-origin precedence and race-bridging rules, especially for
  multi-race births;
- whether the surveillance catchment denominator is intended to represent the
  national natality denominator used here; or
- the sampling covariance within or between source windows.

The audit therefore compares the source point only with model-implied counts
and births pooled over the same five years. The earlier equal-weight mean of
annual prevalence rates is not source-aligned and is excluded from the
decision. It may remain in provenance for the superseded formula-sensitivity
run, but it is not a second estimand or observation. The pooled comparison may
not be promoted to a fitted calibration likelihood until the remaining
category, denominator and covariance issues are resolved.

## Frozen analytic cohort and support

- Fit: the unshifted, independent reporting `DSP004` reference from the
  coherent-calibration phase.
- Birth cohort: the exact `DSP004` inclusion rule—non-missing maternal age and
  non-missing confirmed-or-pending DS indicator.
- Maternal-age representation: exact NCHS age codes with ages at or below 12
  represented by 12 and ages at or above 50 represented by 50, matching the
  fitted Morris curve.
- Surveillance labels and windows:

  - 2016 represents 2014-2018. The frozen fit covers only 2016-2018, or three
    of the five required years. It is left-truncated and is excluded from the
    aligned comparison; it must not be compared with a truncated three-year
    model quantity as though that represented the centred source estimate.
  - 2018 represents 2016-2020. All five years are present in the frozen fit, so
    this is the sole fully aligned comparison window.
- Named race support: NH White, NH Black, NH AIAN, NH Asian/Pacific Islander,
  and Hispanic.
- Unsupported groups: Unknown and NH Multi-race are reported separately for
  cohort coverage but excluded from both named-group composition denominators.
  They are neither discarded silently nor assigned a surveillance target.

The audit must verify that age-by-year birth and recorded-case totals rebuilt
from the race-stratified DuckDB query exactly reproduce the saved `DSP004`
cells before calculating any comparison.

It must sum the workbook's annual race-specific birth and recorded counts over
each centred window and report them beside the corresponding model-cohort
pooled totals. This is a category and denominator mapping audit. The summed
source births supply the native denominator for the confirmed pooled ratio,
but unexplained differences from the model cohort remain evidence that the
category mapping is not yet established. In particular, neither the model's NH
Multi-race group nor a source-bridged allocation of those births may be
silently folded into the five named-group comparison.

For the machine-readable gate, a denominator discrepancy is material when the
absolute named-group total difference exceeds 1% in any complete window or
any one named group's absolute relative difference exceeds 5%. This is a
mapping audit, not a sampling test. The source-alignment gate remains unresolved
even if these numerical thresholds are not crossed, because matching totals
alone would not establish matching definitions.

## No-refit reconstruction

For race group $r$, exact maternal age $a$, and year $y$, define

$$
A_{ray}=N_{ray}\theta_a,
$$

where $\theta_a$ is the exact-age Morris/de Graaf expected DS livebirth
probability absent prenatal reduction. For every saved posterior draw $d$,

$$
T_{ry}^{(d)}=\left(\sum_a A_{ray}\right)\eta_y^{(d)},
\qquad
P_{ry}^{(d)}=10{,}000\frac{T_{ry}^{(d)}}{N_{ry}}.
$$

`DSP004` has one national $\eta_y=1-\rho_y$ per year. The common $\eta_y$
cancels from named-group shares within a single year. It does not generally
cancel after combining five years because race groups have different birth and
maternal-age compositions across those years. The centred-window comparison
therefore propagates the saved annual $\eta_y$ draws into both absolute
prevalence and named-group composition.

For a complete centred window $W_c$, the audit calculates the source-aligned
pooled prevalence for every posterior draw:

$$
P^{(d)}_{rc}
=10{,}000\frac{\sum_{y\in W_c}T_{ry}^{(d)}}
{\sum_{y\in W_c}N_{ry}},
$$

which pools model-implied true counts and births before taking their ratio.
This matches the confirmed source operator. An equal-weight mean of the five
annual prevalence rates is not used in the aligned audit.

For absolute accounting, the model uses
$\sum_{y\in W_c}T_{ry}^{(d)}$. Let $N^{source}_{ry}$ be the tracked
race-specific source birth denominator. The source-native implied count is

$$
\widehat T^{native}_{rc}
=\left(\sum_{y\in W_c}N^{source}_{ry}\right)
\widehat P_{rc}/10{,}000.
$$

The audit also reports a model-birth-standardised analogue,

$$
\widehat T_{rc}
=\left(\sum_{y\in W_c}N_{ry}\right)
\widehat P_{rc}/10{,}000.
$$

The native count uses the summed source denominator; the standardised count
isolates the prevalence contrast from differences between source and model
birth totals. Reporting both makes the denominator contribution visible but
does not prove that the underlying race definitions match.

## Primary descriptive estimands

For the five named groups in the complete 2018-centred window under the pooled
count-ratio construction:

- model and surveillance prevalence per 10,000;
- model, source-native, and model-birth-standardised implied true-birth counts;
- named-group case shares $q^{model}_{rc}$ and $q^{surv}_{rc}$;
- the share difference in percentage points;
- the prevalence and case-share ratios, surveillance divided by model;
- total-variation distance

  $$
  TV_c=\frac{1}{2}\sum_r
  \left|q^{surv}_{rc}-q^{model}_{rc}\right|,
  $$

  interpreted as the fraction of named-group true births that would need to
  be reallocated to reproduce the surveillance composition; and
- surveillance-share-weighted root-mean-square log discrepancy

  $$
  WRMS_c=\sqrt{\sum_r q^{surv}_{rc}
  \left[\log(q^{surv}_{rc}/q^{model}_{rc})\right]^2}.
  $$

Here $c=2018$. There is one source-aligned construction and one surveillance
observation.

The material group rule uses the race-versus-NH-White relative-rate residual,
rather than treating the race source as another independent national-scale
anchor:

$$
D^{(d)}_{rc}
=\log\frac{\widehat P_{rc}}{\widehat P_{White,c}}
-\log\frac{P^{(d)}_{rc}}{P^{(d)}_{White,c}}.
$$

The reported relative-rate ratio is $\exp(E[D^{(d)}_{rc}])$. This removes a
common multiplicative scale from the group contrast, although it does not
resolve dependence on the national reduction evidence or the category
crosswalk.

AIAN is reported but cannot trigger a race extension by itself because its
surveillance series is known to be unstable.

## Recording sensitivity is not identified by this comparison

The centred source prevalence cannot be combined with a single annual recorded
count to estimate recording sensitivity. Under the confirmed pooled-count
operator, the audit reports the descriptive window analogue

$$
\widehat s_{rW}=
\frac{\sum_{y\in W}R_{ry}
-\left(\sum_{y\in W}N_{ry}-\widehat T_{rW}\right)f}
{\widehat T_{rW}},
\qquad
\widehat T_{rW}=\widehat P_{rW}\sum_{y\in W}N_{ry}/10{,}000.
$$

The audit evaluates this once using the summed source births and source
recorded counts, and once using the model-cohort births and recorded counts
after standardising the source rate to that denominator. Their agreement or
disagreement is a mapping diagnostic, not identification of race-specific
recording. Both remain invalid for calibration until the category crosswalk,
denominator and source uncertainty are reconciled. In particular, the earlier
year-label approximation and the shortcut $R/T$ are both ineligible.

## Conditional surveillance-uncertainty scenarios

The source supplies no sampling covariance in the tracked files. The audit
therefore treats uncertainty only as labelled assumptions, not as confidence or
credible intervals. On log prevalence, each scenario contains a common source
component plus a race-specific component:

| Scenario | Common CV | Non-AIAN race-specific CV | AIAN race-specific CV |
| --- | ---: | ---: | ---: |
| Narrow | 5% | 5% | 25% |
| Moderate decision scenario | 10% | 10% | 35% |
| Wide | 20% | 15% | 50% |

The common component cancels from within-window log-ratios. For the pooled
construction in the complete 2018-centred window, the audit reports the
Mahalanobis distance of the three non-AIAN log-ratios against NH White under
the scenario's race-specific covariance and compares it with an 89%
chi-squared reference threshold. This is a conditional discrepancy scale, not
a p-value, because the assumed CVs are not source-provided standard errors. The
source point remains one observation and must not be multiplied as independent
evidence.

## No cross-window transport or repeatability claim

The revised audit performs no 2016-to-2018 or 2018-to-2016 transport check.
The 2016-centred source estimate requires 2014-2018, but the frozen `DSP004`
reference begins in 2016. A transport calculation based on the available
2016-2018 subset would compare a three-year model construction with a five-year
source construction and would not be a valid pseudo-holdout. Consequently only
the 2018-centred window can contribute aligned descriptive evidence, and
temporal repeatability is not assessable. Even with a future 2014 extension,
the two source windows share 2016-2018, or three of their five years, so they
would remain overlapping rather than independent validation targets.

## Frozen single-window descriptive materiality rules

A **single-complete-window pooled descriptive signal** requires all of:

1. `TV >= 0.05` or `WRMS >= log(1.10)` under the pooled construction;
2. relative to NH White, at least two non-AIAN groups have a
   surveillance-to-model **relative-rate ratio** of at least 1.15 or at most
   0.85; and
3. the non-AIAN discrepancy exceeds the conditional 89% reference under the
   moderate uncertainty scenario.

Passing these rules would mean only that the confirmed pooled comparison has a
material 2018-centred discrepancy. It would not establish repeatability,
transportability or calibration eligibility. Category mapping remains
unresolved, source covariance is unavailable, and there is only one complete
window.

If those gates are later satisfied, any first fitted extension will be limited
to time-invariant, partially pooled race offsets and must preserve the national
annual reduction margin exactly. A race-by-year reduction or recording surface
is ruled out: the available source support cannot identify one.

## Superseded annual-label run and revised audit

The initial audit was run without refitting `DSP004` using the following
invocation:

```bash
python scripts/audit_core_race_surveillance.py \
  --fit-dir \
    output/selection_core_reduction/DSP004/20260803-calibration-base-reporting \
  --duckdb-path data/us_births.db \
  --surveillance-csv \
    data/us-births-degraaf-prevalence-recording-2000-2024.csv \
  --years 2016,2018 \
  --output-dir \
    output/selection_core_reduction/audits/DSP004-race-surveillance/20260803-initial-audit
```

The comparison tables, plots and decision first produced by this invocation
treated the labelled years as annual comparison points. They are superseded by
the centred-window clarification and provide no aligned evidence. The current
pooled-only bundle records window support, omits 2016 from the aligned
comparison, uses the confirmed pooled count ratio for 2018, and makes no
transport or repeatability decision. An intermediate centred-window bundle
that paired pooled and equal-year constructions is also superseded in part:
its equal-year rows and formula-sensitivity decision are not source-aligned.

The machine-readable decision, provenance hashes, tables, and plots remain
local and gitignored. The bundle records hashes of the protocol, audit script,
saved fit and surveillance source; the Git revision and dirty state; the Python
and package versions; and the effective invocation and DuckDB table. A clean
committed rerun would be required before treating it as a sealed release
artefact.

The current local run is also not release-environment conformant. Its recorded
runtime has `dse-research-utils 0.5.0` and `pandas 3.0.3`, whereas the repository
declares `dse-research-utils 0.9.0` and `pandas >=3.0.5`.
`dse-check-env environment.yml` additionally reports that the checked-in
environment has drifted from the canonical shared scientific core. These are
release-reproducibility blockers, not evidence against the arithmetic result;
the audit must be rerun after the repository and local environment are
reconciled.

### Fit health and reconstruction identity

The saved reference fit passed the audit's scientific-health gate:

- zero divergences;
- maximum unrounded Rhat `1.00250`;
- minimum bulk ESS `1,221.64`; and
- minimum tail ESS `1,559.74`.

The cached summary's means and unrounded health columns were recomputed from
and matched against `idata.nc`, so a stale or mixed summary file would fail the
audit rather than supply the health decision.

The implementation also refuses a non-reporting fit, a shifted or correlated
reduction-prior sensitivity, a changed false-positive scenario, a different
age/end-point contract, a source race-label/code mismatch, or substitution of
the regression-filled `est_true_*` columns for missing raw surveillance. The
DuckDB connection is opened read-only. Focused tests cover these contracts,
reconstruction identities, centred-window support and pooled-ratio
calculations, permanently blocked fit permissions, provenance output, and an
end-to-end check that the source database is unchanged.

Collapsing the seven reconstructed race groups reproduced every saved
age-by-year birth and recorded-count cell exactly. Recomputing the saved
posterior quantities gave maximum absolute discrepancies of `0` for
`p_ds_lb`, `3.64e-12` for annual true counts, and `4.37e-11` for the total true
count. The race audit therefore describes the same cohort and posterior as the
saved `DSP004` fit; its substantive discrepancies are not reconstruction
errors.

### Superseded annual-label findings

The initial results that compared the source points labelled 2016 and 2018 with
the corresponding single model years are withdrawn from the aligned evidence.
This includes their total-variation and `WRMS` discrepancies, group ratios,
conditional Mahalanobis distances, implied counts, and the bidirectional
transport calculation. Those calculations answered an annual-label question,
whereas the source points represent 2014-2018 and 2016-2020 respectively.

The earlier apparent agreement across two labels therefore cannot be described
as repeatability. Nor can its large transport improvement be retained as a
pseudo-holdout result. Only regenerated results for the complete 2016-2020
window under the pooled count-ratio construction are eligible as the current
descriptive audit. The audit does not classify the source point as
falling inside or outside the model posterior interval: that interval contains
only `DSP004` uncertainty, while source uncertainty and covariance remain
unavailable.

### Centred-window pooled findings

The pooled-only no-refit bundle was regenerated at
`output/selection_core_reduction/audits/DSP004-race-surveillance/20260803-centred-window-audit`.
It confirms five-of-five model-year support for the 2018-labelled 2016-2020
window and only three-of-five support for the 2016-labelled 2014-2018 window.
All model comparison fields for the partial 2016 window are therefore null;
the audit does not truncate the window, extrapolate the missing reduction
draws, or substitute the annual 2016 value.

For the complete 2018 window, the pooled source rate applied to model births
gave total variation `0.08030`, `WRMS` `0.21836`, and an absolute
coherence difference of `-0.327%` after retaining the model estimate for
Unknown and NH Multi-race. Using the summed source birth denominators instead
gave total variation `0.08059`, `WRMS` `0.21684`, and an absolute
coherence difference of `+1.026%`. Both composition distances cross their
pre-specified materiality thresholds of 0.05 and
$\log(1.10)=0.0953$.

The native source denominator implies `25,128.34` true births among the five
named groups, compared with the model posterior mean `24,864.68`. The
model-birth-standardised source analogue is `24,780.66`. This separation is
important: the pooled surveillance data contain some absolute-scale
information, but about 1.5% more named-group births are represented in the
source denominator than in the model cohort. Native counts therefore cannot be
entered as an additional likelihood until that mapping difference and the
source covariance are represented.

Using NH White as the relative-rate reference, the source-to-model ratios are
approximately 1.135 for NH Black, 1.348 for NH AIAN, 0.509 for NH
Asian/Pacific Islander, and 1.379 for Hispanic. Under the frozen pooled rule,
NH Asian/Pacific Islander and Hispanic are the two non-AIAN groups with at
least a 15% discrepancy. AIAN is reported but excluded from this rule because
its small and definition-sensitive denominator was a pre-specified
limitation.

The moderate conditional uncertainty scenario gives squared Mahalanobis
distance 56.44 against the conditional reference value 6.03. This calculation
includes model posterior covariance, but the source component remains an
assumed coefficient-of-variation scenario rather than source-provided
uncertainty. It is a robustness scale, not a test statistic or posterior
probability.

The current decision records
`single_complete_window_pooled_descriptive_signal=true`. The superseded
two-window field remains null, temporal replication and transport are not
evaluable, `calibration_eligible=false`, and every race-layer fit
authorisation remains false. The earlier equal-year-rate calculation was
numerically close, but it is retained only as provenance for a superseded
sensitivity and is not present in the pooled-only decision.

### Denominator reconciliation

The 2018-centred source and model denominators are not interchangeable. Across
the five named groups, the summed source births are `18,661,084`, versus
`18,385,935` in the model cohort: a source-minus-model difference of
`+1.50%`, crossing the 1% named-total threshold. By group, the differences
are:

- NH White: `+1.31%`;
- NH Black: `+3.19%`;
- NH AIAN: `+22.36%`;
- NH Asian/Pacific Islander: `+1.76%`; and
- Hispanic: `+0.11%`.

Only AIAN crosses the 5% group threshold. The corresponding source-minus-model
recorded-count differences are +72, +32, +17, +6 and 0. The 2016 source pooled
births and recorded counts are retained, but no full model-window denominator
comparison is calculated because 2014-2015 are outside the frozen fit.

These differences may reflect cohort filters, Hispanic precedence, race
bridging, or another construction difference; the audit cannot attribute
them. They neither fill the missing 2014-2015 model support nor establish that
the source categories are identical to the model categories. The material flag
is therefore a mapping stop rule, not evidence that either denominator is
wrong.

### Fail-closed decision and next gate

`calibration_eligible` remains false even though the one complete window passes
the pooled descriptive rules. The audit authorises neither a
time-invariant race layer nor a race-by-year or absolute-scale layer. The
blockers are:

1. the 2016-centred window is incomplete and only one centred five-year source
   window is fully covered by the frozen `DSP004` reference, so replication
   and transport cannot be assessed;
2. Hispanic-origin precedence, multi-race bridging and the full category
   crosswalk remain unresolved, although maternal race is now confirmed;
3. the material source-versus-model denominator differences above remain
   unexplained;
4. source-provided sampling covariance is absent;
5. overlap with, and resulting dependence on, the national reduction evidence
   is unresolved; and
6. the mirrored age-on-recording model-adequacy gate remains outstanding.

The next gate is documentary and methodological: obtain the Hispanic-origin
and multi-race definitions; reconcile the denominators; establish source
covariance and overlap with the national reduction evidence; support a second
complete centred window, either with compatible 2014-2015 model inputs or
another source point; and complete the mirrored age-on-recording
model-adequacy check. Only then should the project reconsider one
time-invariant, partially pooled **relative-composition** race layer that
preserves the national annual reduction margin. The present audit remains a
conditional descriptive national association and does not support a race
mechanism or a race-specific correction.

## Interpretation limits

- This is aggregate population accounting. It does not identify individual
  unrecorded DS births.
- A race offset would combine any race-associated difference in prenatal
  reduction, transportability of the Morris curve, surveillance ascertainment,
  denominator construction, and category mapping. It would not identify a
  screening, termination, access, biological, or geographical mechanism.
- Unknown and NH Multi-race remain externally unanchored.
- Race calibration cannot repair or explain the outstanding `1/7` broad-age
  posterior-predictive coverage. The mirrored age-on-recording diagnostic
  remains the next model-adequacy gate before adopting a fitted race model.
- Absolute race prevalence must not be added as a second independent national
  anchor unless its covariance with the national reduction evidence is
  represented. The initial fitted candidate, if justified, will use relative
  race composition while preserving the national total.
