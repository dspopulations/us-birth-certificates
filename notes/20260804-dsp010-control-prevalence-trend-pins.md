> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).

# Pinning the DSP010 control conditions' own prevalence trends

This closes the first recommended next step of the
[DSP010 note](20260804-dsp010-anomaly-panel-recording-factor.md): every
`true_trend_log_per_year` in the anomaly-panel curation table was `0.0`, meaning
"believed stable, not verified". The panel reads the controls' common movement as the
certificate item's recording sensitivity, so a real prevalence trend shared by the
controls is perfectly confounded with a recording trend and no comparison inside the
panel can see it. That made the zeros the single most consequential unmeasured input
in the design.

Three of the four controls now carry a measured trend. The fourth was **refused a
pin**, which is the more interesting result.

## Why the Texas Birth Defects Registry, and not the NBDPN pooled reports

The DSP010 note named NBDPN annual reports as the source to read. Two facts moved the
work to a single state instead.

**The national reports give pooled cohorts, not annual series.** The 2016-2020 national
estimates and their 1999-2001, 2004-2006 and 2010-2014 predecessors are five-year
pooled prevalences. Two pooled numbers give one slope across their midpoints, and the
contributing-programme roster changes between cohorts, so part of any such slope is
composition rather than prevalence. That is the same artefact the panel already guards
against for maternal age.

**Hypospadias is absent from the national tables entirely** — and it is the control
that matters most, being male-only, undetectable prenatally, and the condition showing
the largest decline in the certificate data.

The Texas Birth Defects Registry publishes case counts and prevalence per 10,000 live
births for each monitored defect by single delivery year, 1999-2022, covering the whole
state in every year. It qualifies on the criterion that decides everything here — it
does not ascertain cases from the birth-certificate anomaly item, so its series cannot
re-import the recording decline the panel is trying to measure. From the report's own
methods:

> The Texas Birth Defects Registry uses active surveillance. This means it does not
> require reporting by hospitals or medical professionals.

> Regardless of the source of demographic information for this report, all diagnostic
> information was abstracted from medical records.

Case-finding is facility record review, and **birth** certificates enter only to supply
demographics for cases already found. That is the circularity trap avoided: had the
source used the live-birth anomaly checkbox for case-finding, subtracting its trend
would have cancelled our own signal while appearing to strengthen the design.

The registry does use one certificate for case-finding, and it needs stating precisely
rather than glossed. From 2009 it also screens **fetal death** certificates, including
those "with a congenital anomaly reported on the certificate", to identify potential
cases. That is a different form for a different population from the one DSP010 measures,
and every one of these controls was chosen because it has essentially no fetal loss or
termination, so the fetal-death channel contributes a negligible share of their cases
against a live-birth denominator. It is not negligible in principle for gastroschisis,
which does carry some fetal loss: if fetal-death anomaly reporting decayed the way the
live-birth item did, part of the gastroschisis pin would be recording rather than
prevalence. The four controls are unaffected either way.

Two properties of the registry are worth carrying forward as caveats rather than
hiding. Cases include all pregnancy outcomes, so the tabulated rate is all-outcome
cases over live births; for controls with no termination channel — which is why they
were chosen — that is the same quantity to within the termination rate. And diagnoses
count up to one year after delivery, far beyond a certificate completed at discharge,
so TBDR *levels* sit above certificate levels. Neither affects a trend unless the
termination rate or the diagnostic timing moved, and the second is a genuine open
question for hypospadias.

## Method

Quasi-Poisson log-linear regression of annual counts on delivery year with a log
live-births offset. The dispersion scaling is load-bearing: annual counts vary far more
than Poisson because ascertainment itself moves, and the unscaled standard error would
claim precision the series does not have. Scaling is never applied downwards — an
under-dispersed short series is luck, not extra information.

Denominators are recovered as the median of `cases / rate × 10,000` across all
whole-population blocks, which beats any single block whose implied value is limited by
the rate being printed to two decimals. Hypospadias is reported per 10,000 male live
births and gets its own denominator; the recovered male share is `0.5111`, as it should
be. Because a wrong denominator would rescale every slope silently, the recovered values
are checked against Texas resident live births from CDC WONDER — a source already tracked
in this repository and entirely unrelated to the registry report — and agree to within
`0.55%` in the worst year and `0.05%` in five of seven.

**The primary window is 2010-2022, not the panel's own 2016-2024.** A birth-prevalence
trend is a slow quantity and the short window is demonstrably unreliable here: over
2016-2022 the limb-reduction slope comes out **positive** and disagrees in sign with the
national pooled series, while over 2010-2022 it agrees to within `0.001` log per year.
Seven annual points on a few hundred cases cannot separate a trend from noise. The
short window is fitted and shipped anyway, as a documented sensitivity.

**Two independent gates decide whether a trend may be pinned**, because each alone rests
on a parametric choice the other does not.

The *breakpoint scan* asks whether the series contains a level shift. A trend and a step
look alike to a straight line and mean opposite things — a step is an ascertainment or
coding change, and pinning it as biology would inject a registry artefact into the model.
Every interior year is tried as a breakpoint and a shift beyond `3` standard errors
refuses the pin. Its weakness is that it measures the shift against a fitted trend, so a
sceptic can dispute the trend and shrink the shift.

The *dispersion gate* asks the model-free question of whether a straight line describes
the series at all. A line fitting more than `3` times worse than Poisson has no slope
worth pinning, whatever the cause. The four pinned controls sit at `0.73`–`1.81`.

Both gates fire on exactly one condition, and neither fires on any other, so the refusal
below does not depend on which one you trust. A refused condition keeps `0.0` with the
reason recorded in the table.

## Results

| Condition | Slope, log/yr | SE | Dispersion | Largest level shift | Pinned |
| --- | ---: | ---: | ---: | --- | :---: |
| `ca_hypo` hypospadias | `+0.00524` | `0.00453` | `5.01` | **`-16.3%` at 2019, z = `-6.0`** | **no** |
| `ca_clpal` cleft palate alone | `-0.00118` | `0.00657` | `1.78` | `+12.0%` at 2017, z = `+1.2` | yes |
| `ca_cleft` cleft lip ± palate | `-0.00281` | `0.00495` | `1.81` | `+10.0%` at 2018, z = `+1.4` | yes |
| `ca_limb` limb reduction | `-0.00648` | `0.00537` | `1.24` | `+17.2%` at 2019, z = `+2.4` | yes |
| `ca_gast` gastroschisis | `-0.03798` | `0.00533` | `0.73` | `-12.5%` at 2016, z = `-1.7` | yes |
| `reference_ds` Down syndrome | `+0.00130` | `0.00392` | `1.56` | `+13.1%` at 2018, z = `+2.7` | never |

`ca_limb` sums the registry's upper- and lower-limb categories because the certificate
carries a single limb-reduction checkbox. A child with both is counted in both rows, so
the summed *level* is slightly overstated; the summed trend is unaffected while that
overlap share is stable.

`reference_ds` is carried through as an independent series and never pinned — Down
syndrome is the model's own outcome. Note that TBDR counts terminations, so this is
total rather than live-birth prevalence and is **not** directly comparable to the
model's target; it is recorded because it comes from medical records rather than
certificates, not because it settles anything.

### Hypospadias is a step, not a trend

The series is well described as **a rising line, one level break at 2019, then a rising
line again** — and badly described by anything simpler:

| Fit | Slope, log/yr | Dispersion |
| --- | ---: | ---: |
| 2010-2018 | `+0.02347 ± 0.00426` | `1.47` — a line fits |
| 2019-2022 | `+0.01611 ± 0.01234` | `0.65` — a line fits |
| 2010-2022 | `+0.00524 ± 0.00453` | **`5.01` — a line does not fit** |

The step between the segments is `-16.3%` (z = `-6.0`), and the registry's published
methods document no change of definition, coverage or coding at 2019. Two further
symptoms point the same way: the whole-span slope's *sign flips with the window*
(`+0.005` from 2010, `-0.005` from 2013, `-0.019` from 2016), and the dispersion stays
above `3` on every window that spans 2019.

One caveat on the `z = -6.0` specifically: it is the shift measured *against a fitted
trend*, and because that trend is strongly positive the step has to be large to explain
the data. Ignore the trend and pool 2010-2018 against 2019-2022 and the raw gap is only
`-3.0%` (z = `-1.8`). The refusal therefore does not rest on the step's size — it rests
on the model-free observation that a straight line fits each segment and fails on the
span, which is why the script gates on dispersion as well as on the step.

**No reading of this series supports a sustained decline.** If the step is a registry
artefact, hypospadias' true prevalence *rose* through the panel window at `+1.6` to
`+2.3%`/yr, and the certificate's `-15.5%` decline is *more* than fully recording — a
positive pin would have made the recording finding stronger, not weaker. If the step is
real prevalence — a one-year `16%` fall in hypospadias, which is not biologically
credible — the window average is roughly flat. Zero sits between those and is the
conservative choice; `true_trend_log_per_year` carries no uncertainty once pinned, so a
contested value would be worse than zero. It is now a recorded decision with a reason
attached rather than a silent default.

That hypospadias has been *rising* in Texas is not an artefact of the recent window: the
1999-2022 slope is `+0.0150` log per year. It matches the long-documented US rise.

### Gastroschisis: the curation judgement was right

Gastroschisis was excluded from the panel on the grounds that its recorded decline is
genuine prevalence, not recording. Active surveillance confirms it: `-0.038` log per
year, the largest and cleanest trend in the table, and the long series shows the classic
US rise to a 2010 peak followed by a sustained fall. The national pooled series shows
the same shape (`3.73 → 4.49 → 5.12 → 4.10`). With a pinned trend it can now be
readmitted as a fifth control, which is the falsification test the DSP010 note asked
for.

### The national cross-check agrees on every testable condition

Over the two most recent pooled cohorts — adjacent and recent, because a
1999-to-2020 comparison reports a *rise* for gastroschisis, which has been falling
throughout the panel's span:

| Condition | National, log/yr | Texas, log/yr | |
| --- | ---: | ---: | :--- |
| `ca_clpal` | `+0.01062` | `+0.00231` | agree |
| `ca_cleft` | `-0.00525` | `-0.00298` | agree |
| `ca_limb` | `-0.01035` | `-0.00929` | agree |
| `ca_gast` | `-0.03703` | `-0.04514` | agree |

Texas tracks the national pooled series in sign everywhere and closely in magnitude for
the two conditions that actually move. That is the evidence for treating one state as a
proxy for a national panel — it is not proof, and it is the largest remaining assumption
in this extraction.

## What this implies for DSP010

**The shared prevalence trend prior was well chosen.** The pinned controls imply a
shared trend of `-0.00262` log per year against a prior of `0 ± 0.00400`. That prior
was pure assertion when DSP010 was built, and its two-standard-deviation width was
nearly as large as the entire measured item decline. It now has external support: the
measured value sits at two thirds of one prior standard deviation from zero.

**The pins do not reduce the controls' disagreement — they increase it.** Netting each
pinned trend out of the panel's own span comparison leaves a *larger* spread than it
started with, because the pins remove the most from the condition that was already
closest to zero and nothing at all from the biggest outlier:

| Condition | Certificate change | Pin removes | Residual |
| --- | ---: | ---: | ---: |
| `ca_hypo` | `-16.78%` | `+0.00%` | `-16.78%` |
| `ca_clpal` | `-8.02%` | `-0.71%` | `-7.32%` |
| `ca_cleft` | `-2.25%` | `-1.69%` | `-0.56%` |
| `ca_limb` | `-13.79%` | `-3.89%` | `-9.91%` |

Cochran `Q` rises from `33.42` to `40.11`, `I²` from `91.0%` to `92.5%`, and the
between-condition SD from `7.45%` to `8.23%`. What the pins do move is the *level*: the
random-effects common change goes from `-10.11%` to `-8.62%`, a shrink of about `1.5`
percentage points that anticipates almost exactly what the refit below does to the fitted
factor. The controls' surveillance trends are
essentially uncorrelated with their certificate declines, which is itself a finding: the
disagreement the panel has to absorb is **not** explained by their true prevalence. After
removing measured prevalence, hypospadias still reads `-16.8%` where cleft lip reads
`-0.6%`, and those cannot both be one item-wide recording factor. The shared-factor
restriction comes out of this under more strain, not less.

Note that `panel_heterogeneity` is computed on observed rates at load time, *before* the
model subtracts `known_log_trend`, so the fit prints the same `I² = 91%` either way. The
diagnostic that exists to warn about disagreement cannot currently see the pins.

### Refit result

`DSP010` refitted against the pinned table, with the unpinned baseline refitted at the
same seed and settings so those two differ only in the curation file, and a third fit at
the γ corner described below. 2004-2024, 4 chains × 4,000 draws, **zero divergences
throughout**, all three CLEAN on the per-chain audit.

| | Unpinned | Pinned | Pinned, γ = 0 |
| --- | ---: | ---: | ---: |
| 2016-2024 total | `45,828` | `45,653` | `45,666` |
| 89% ETI | `43,887`-`48,097` | `43,799`-`47,782` | `43,879`-`47,665` |
| Width | `9.19%` | `8.72%` | `8.29%` |
| Item recording factor | `-7.63%` | `-6.38%` | `-6.34%` |
| its 89% ETI | `-15.71%` to `+1.31%` | `-14.39%` to `+2.67%` | `-13.07%` to `+1.48%` |
| its width | `17.02` pp | `17.06` pp | `14.55` pp |
| `s` 2024 vs 2016 | `0.9516` | `0.9595` | `0.9593` |
| Prevalence 2024 vs 2018 | `-2.37%` | `-3.06%` | `-3.01%` |
| `recording_s` | `0.3370` | `0.3369` | `0.3368` |
| Loading | `0.935 ± 0.454` | `0.936 ± 0.453` | `0.941 ± 0.456` |
| Shared trend γ | `+0.00009 ± 0.00397` | `+0.00019 ± 0.00387` | fixed at `0` |
| `panel_condition_trend_scale` | `0.01431 ± 0.00662` | `0.01471 ± 0.00646` | `0.01469 ± 0.00647` |

**Two quantities moved and the rest did not.** The item recording factor moved `+1.25`
percentage points (`18.6` combined MCSE) and the total fell `174` births (`12.5`
combined MCSE); the total's interval narrowed by about half a percentage point. Against
that, the loading, γ, the idiosyncratic scale and `recording_s` are unchanged to three
decimals, and the factor's own interval is flat at `17` pp.

### The γ corner: switching off the shared trend buys width, not a different answer

With surveillance carrying the shared component, `--panel-prevalence-trend-sigma 0`
stops being an assertion and becomes the reading that says "the measurement got it all".
It moves the estimate not at all — total `+13` births (`0.8` MCSE), factor `+0.04`
percentage points (`0.6` MCSE) — and takes `2.5` percentage points out of the factor's
interval, from `17.06` to `14.55` pp. That is what it looks like when a free parameter
was contributing prior noise rather than signal, which is exactly what the pins
established: γ's own posterior had returned its prior untouched in both earlier fits.
So the division of labour is clean. The pins moved the level; γ = 0 removes the width
that the level no longer needs.

It is a slightly stronger claim than the evidence supports, and the reason is
hypospadias. γ = 0 asserts that surveillance measured the whole shared trend, but only
three of the four controls are pinned; hypospadias' own trend sits in neither the offsets
nor γ. The corner belongs in the envelope, not at its centre.

**This fit needed a second attempt, and the first one must not be quoted.** The initial
γ = 0 run reported max R-hat `1.5272` and min ESS `4` with zero divergences — the
signature of a chain that stopped moving rather than of bad curvature. Chain 3 had
collapsed during tuning: step size `0.0`, `recording_s` variance exactly zero,
`panel_condition_trend_scale` stuck at `0`, the funnel boundary of its own HalfNormal.
Chains 0-2 agreed with each other and with the pinned fit to four decimals. A reseed with
6,000 tuning iterations and `target_accept` 0.995 converged cleanly, and
`audit_anchored_chain_health.py` independently marks the failed run SUSPECT and the retry
CLEAN. Removing γ tightens the geometry around a hierarchical scale that can now be
pulled harder toward zero, so this corner is the one in the family most likely to need
the longer tuning — worth knowing before anyone reruns it.

**The `scale/√n` mechanism did not fire.** `panel_condition_trend_scale` rose only
`2.8%`, well inside its own SD, despite `Q` rising a fifth. The heterogeneity statistic
uses Poisson within-condition variance alone, while the model has a free
`panel_idiosyncratic_scale` absorbing annual noise — so what looks like sharpened
disagreement to DerSimonian-Laird is largely invisible to the fit. An earlier draft of
this note predicted a narrowing through this channel; that prediction was wrong, and the
narrowing that did occur came from the factor's centre moving toward zero, leaving the
loading's freedom less room to move the total.

**The headline survives, weakened.** The item's recording sensitivity still falls, still
contradicting `DSP008`'s constant-`s` reading, and now with three of four controls' true
prevalence measured externally rather than assumed. It should be said plainly that the
factor's 89% interval includes zero in **both** fits: this is a point estimate with a
direction, not an interval that excludes no change.

## What is still missing

- **The uncertainty channel does not exist.** `true_trend_log_per_year` enters the
  likelihood as a fixed offset, so pinning asserts a point estimate and discards the
  standard errors in the table above — which are not small relative to the slopes they
  accompany. Carrying them needs a per-condition trend *prior* centred on surveillance,
  or at minimum a settable centre for `panel_prevalence_trend`, whose `mu` is currently
  hardcoded to zero. Until then the pinned fit is a sensitivity corner, not the answer,
  and this repository has twice been bitten by exactly this kind of unsupported
  precision.
- **2023 and 2024 are extrapolated.** TBDR data ends at 2022; the panel runs to 2024.
  The pin covers seven of the panel's nine years and the model applies the fitted slope
  to the remaining two.
- **One state, about a tenth of US births.** Corroborated in sign against the national
  pooled series, unverified against any second annual series.
- **Hypospadias remains unpinned**, and it is the control carrying the largest
  certificate decline. Resolving the 2019 step — by correspondence with the registry, or
  by finding a second active-surveillance annual series — is now the highest-value open
  input, inheriting that title from the zeros this note removed.
- **`panel_heterogeneity` cannot see the pins.** It reads observed rates before
  `known_log_trend` is subtracted, so a pinned run prints the same `I² = 91%` as an
  unpinned one and the `I² > 50%` warning fires identically. It should net the pinned
  trends out and report the residual, which is what the fit actually has to explain.
- **Nothing propagates the pins into the DSP010 note's own six-fit envelope.** That
  table is still entirely unpinned. Deciding which of these three fits is the headline
  `DSP010` — and whether the γ corner joins the envelope — is a separate call.

## Reproducing

```bash
curl -sSL -o data/1999-2022-tbdr-2-prevbyyear.xlsx \
  "https://www.dshs.texas.gov/sites/default/files/birthdefects/annualreport/1999-2022-tbdr-2-prevbyyear.xlsx"
```

```bash
python scripts/extract_surveillance_anomaly_trends.py --strict
```

`--strict` exits non-zero while any pin is refused, which it currently is, so the
hypospadias step cannot be forgotten. Outputs land in
`output/surveillance_anomaly_trends/`, and the two model-facing files are installed to
`data/us-births-anomaly-surveillance-trends.csv` and
`data/us-births-anomaly-panel-conditions-pinned.csv`. Pass `--no-install` to leave the
tracked copies alone.

To fit `DSP010` against the pinned table:

```bash
python scripts/fit_core_reduction_model.py DSP010 --years 2004-2024 \
    --panel-conditions-csv data/us-births-anomaly-panel-conditions-pinned.csv \
    --profile reporting --draws 4000 --tune 4000 --chains 4 --target-accept 0.99
```

The unpinned baseline in the comparison above is the same command without
`--panel-conditions-csv`, run at the same seed so the two fits differ only in the
curation file. Do not compare against a differently-seeded run: the 89% interval width
moves by around `0.2` percentage points between seeds, which is a large fraction of the
difference being measured.

The γ corner needs the longer tuning; at the default it lost a chain to the scale funnel:

```bash
python scripts/fit_core_reduction_model.py DSP010 --years 2004-2024 \
    --panel-conditions-csv data/us-births-anomaly-panel-conditions-pinned.csv \
    --panel-prevalence-trend-sigma 0 --random-seed 20260804 \
    --profile reporting --draws 4000 --tune 6000 --chains 4 --target-accept 0.995
```

```bash
python scripts/audit_anchored_chain_health.py --strict
```

Run the audit on anything before quoting it. The fit script prints its own R-hat verdict
and it is the first thing to read: the failed γ run above wrote a complete `idata.nc` that
looks perfectly ordinary until you check.

## Sources

- Texas Birth Defects Registry Annual Report, Table 2A, deliveries 1999-2022, and its
  Methods section (Texas Department of State Health Services, September 2025).
  <https://www.dshs.texas.gov/texas-birth-defects-epidemiology-surveillance/birth-defects-data-publications/tbdr-annual-report>
- Stallings et al. (2024) National population-based estimates for major birth defects,
  2016-2020. *Birth Defects Research* 116(1):e2301.
  <https://doi.org/10.1002/bdr2.2301>
