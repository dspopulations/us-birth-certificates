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

Birth and fetal-death certificates enter only to supply demographics for cases already
found. That is the circularity trap avoided: had the source used the anomaly checkbox
for case-finding, subtracting its trend would have cancelled our own signal while
appearing to strengthen the design.

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
be.

**The primary window is 2010-2022, not the panel's own 2016-2024.** A birth-prevalence
trend is a slow quantity and the short window is demonstrably unreliable here: over
2016-2022 the limb-reduction slope comes out **positive** and disagrees in sign with the
national pooled series, while over 2010-2022 it agrees to within `0.001` log per year.
Seven annual points on a few hundred cases cannot separate a trend from noise. The
short window is fitted and shipped anyway, as a documented sensitivity.

**A breakpoint scan gates the pin.** A trend and a level shift look alike to a straight
line and mean opposite things — a step is an ascertainment or coding change, and pinning
it as biology would inject a registry artefact into the model. Every interior year is
tried as a breakpoint; a condition whose largest standardised shift exceeds `3` is
refused a pin and left at zero with the reason recorded in the table.

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

The refusal is decisive and survives both windows: z = `-6.0` over 2010-2022 and
z = `-3.6` over 2016-2022, with dispersion `5.01` — a straight line simply does not
describe this series. The rate runs `74.55, 72.97, 76.00` for 2016-2018 and then
`65.64, 65.76, 69.51, 68.04` for 2019-2022, and it *rises* within each segment
(`+0.95%`/yr before, `+1.61%`/yr after). Over 2010-2022 the fitted slope is positive.
Whatever happened between 2018 and 2019 was a one-off level change of about `-10%`, and
the registry's published methods document no change of definition, coverage or coding
at that point.

This matters because the two readings imply opposite corrections. If the step is a
registry artefact, hypospadias' true prevalence was flat to rising, and the
certificate's `-15.5%` decline is *entirely or more than* recording. If the step is real
prevalence, a large part of that `-15.5%` is not recording at all. The extraction cannot
tell, and `true_trend_log_per_year` carries no uncertainty once pinned, so a contested
value would be worse than zero. Zero stays — but it is now a recorded decision with a
reason attached rather than a silent default.

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

**The spread across controls is much smaller than the panel's residual heterogeneity.**
The pinned trends span `-0.0065` to `0.0`, where the panel's own fitted per-condition
deviations had to explain a spread from `-1.85%`/yr to `+1.26%`/yr at `I² = 91%`.
Moving the real part of that divergence into a fixed offset should shrink
`panel_condition_trend_scale` and with it the `scale/√n` uncertainty the common factor
inherits — so this should *narrow* DSP010's interval, the first thing in this line of
work that would.

**The item recording decline should survive.** Arithmetic, not a fit: eight years of the
pinned shared trend is about `-2.1%`, against a measured item factor of `-7.51%`, which
leaves roughly `-5.4%` as recording. The finding that the item's recording sensitivity
fell — contradicting `DSP008`'s constant-`s` reading — is weakened but not overturned.
**No model has been refitted. Every number in this paragraph is arithmetic on the old
posterior and must be replaced by a fit before it is quoted.**

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
python scripts/fit_core_reduction_model.py --model dsp010 --panel-conditions-csv data/us-births-anomaly-panel-conditions-pinned.csv
```

## Sources

- Texas Birth Defects Registry Annual Report, Table 2A, deliveries 1999-2022, and its
  Methods section (Texas Department of State Health Services, September 2025).
  <https://www.dshs.texas.gov/texas-birth-defects-epidemiology-surveillance/birth-defects-data-publications/tbdr-annual-report>
- Stallings et al. (2024) National population-based estimates for major birth defects,
  2016-2020. *Birth Defects Research* 116(1):e2301.
  <https://doi.org/10.1002/bdr2.2301>
