> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).

# Anchoring the DSP series on surveillance-based expected live births

**Date:** 2026-08-03

**Status:** Data-extraction audit, era audit, and three fitted model extensions
(`DSP006` revision-split recording; `DSP007`/`DSP008` surveillance-anchored
level). All figures come from
`data/cor verwissel jaren overzicht prevalencties races usa birth cert vanaf 2000 ALT3.xlsx`
and `data/us_births.db`, reproduced by
[`scripts/extract_degraaf_surveillance.py`](../scripts/extract_degraaf_surveillance.py)
and
[`scripts/audit_certificate_revision_era.py`](../scripts/audit_certificate_revision_era.py),
in a local environment that is **not release-conformant**; regenerate before
citation. The refit figures are reporting-profile runs but were produced in the
same non-conformant environment, so they are indicative rather than citable.

**Amended twice.** First to add the era audit and the 2004-start refit, in
response to the question of whether the model window should start earlier; then
on 2026-08-04 to implement the surveillance observation equation as `DSP007` and
`DSP008`. The design section below is retained as written, and the "anchored
model" section records what the implementation actually produced — which differs
from what the design section anticipated in one important respect: the interval
turned out to be governed by an unknown rather than narrowed by the anchor.

**This note builds on
[the 28 June extraction note](20260628-degraaf-corrected-prevalence-extraction.md),
which already decoded the same workbook** — the column semantics, the five
per-race regression coefficients, the 2000–2014/2016/2018 coverage, the race-code
mapping, and the fact that `cor verwissel jaren` refers to Gert's fix of the
swapped 2002/2003 birth-certificate figures. Those are re-verified here, not
newly discovered. Also a companion to the
[model-family review](20260803-dsp-core-model-family-review.md), the
[false-positive note](20260803-false-positive-channel-identification.md) and the
[group-identification note](20260803-group-reduction-recording-identification.md).

## Summary

The workbook extracts cleanly and reconstructs exactly: every internal identity
holds to zero relative error, and the six chart trendlines refit from their
observed points to within `2e-10`. It does compare against birth-certificate
totals, and those totals are ours — columns C and D match `us_births.db`
cell-for-cell in 14 of 25 years.

What this pass adds beyond the June decoding:

1. **The race denominators are inconsistent across years**, in two distinct and
   separable ways. This is the material new finding. It distorts group-level
   results while leaving the pooled total intact to four decimal places.
2. **Three of the five per-race trendlines are weak** (R² `0.14`–`0.32`) yet are
   extrapolated six years past the last observation. The **pooled row's
   reporting fraction is fitted in every year**, including years where all five
   race-specific values were observed.
3. **A pooled surveillance anchor**, recomputed on our own counts, lands within
   `-0.3%` to `+2.6%` of the `DSP004` posterior total.
4. **Gert's reporting fraction and the model's `s` are different estimands** and
   reconcile exactly through the false-positive allowance.
5. **The reset's real decision is about trajectory, not level.** Gert's column H
   and a surveillance-anchored prevalence path agree on the 2016–2024 total to
   about 2% but disagree on its shape: his implied prevalence *falls* `7.3%`
   across the tail, a flat anchor by construction does not.
6. **The reporting fraction steps rather than trends**, and the step is the 2003
   certificate revision. Extending the model window to 2004 and splitting on
   certificate version (`DSP006`) **nearly halves the interval on the 2016–2024
   total** while leaving the headline within `0.6%`. Extending *without* the
   split biases it up `3.6%`.
7. **The surveillance anchor is implemented** (`DSP007`/`DSP008`) and does fix
   the identification problem: the level no longer comes from the reduction CSV.
   But it **does not deliver a precision gain so much as relocate an
   assumption** — the interval runs from `2.87%` to `16.19%` depending on an
   assumed surveillance accuracy the workbook does not supply, while the mean
   moves only `1.1%`. That single unknown is now the binding constraint on the
   whole model.

## What the workbook contains

One sheet, 150 rows (25 years × 5 race groups + a birth-weighted pooled row),
plus a side block holding the surveillance matrix and six charts. The column
semantics are in the [June note](20260628-degraaf-corrected-prevalence-extraction.md);
three points are worth restating because the reset depends on them.

**Only one quantity in the file is external.** The surveillance prevalence
(`L`/`R`) is the new information. The birth-certificate counts, the five-year
running prevalences and the "percentage reported" are arithmetic on data we
already hold, so they should be recomputed rather than imported. The extraction
script does exactly that.

**The five-year windows are centred on the row's year.** Row 14 is 2002 and its
sums draw on 2000–2004, so `L` and `R` carry each window's value against its
mid-year.

**The pooled row's surveillance value is not an external datum.** It is
`Σ R_race · P_race / Σ P_race`, a birth-composition-weighted mean of the five
race-specific figures, reproduced to `1.4e-16`.

Two definitional details: `down_ind = 1` reproduces column `C` exactly, so the
workbook counts **confirmed and pending flags together**; and column `D` includes
births whose anomaly status is unknown, which the model's cohort excludes — a
`0.18%` difference over 2016–2024, small enough to ignore but worth stating.

### Everything reconstructs

| Check | Max relative error |
| --- | ---: |
| `U == Q/R` | `0.000e+00` |
| `G == U` where surveillance observed (75 cells) | `0.000e+00` |
| `G ==` fitted line where surveillance missing (50 cells) | `0.000e+00` |
| `O ==` centred five-year sum of `C` | exact |
| `R == L` | `0.000e+00` |

### The regressions are recoverable, and three of them are weak

Excel stores a trendline's *specification* but not its fitted coefficients, so
the literals in column `G` can only have arrived by being read off a chart and
typed in. Refitting least squares of the observed `U` on the year index recovers
them, which means the fill is fully reproducible without the workbook:

| Race | n | Slope refit | Rel. error | Intercept refit | Rel. error | R² |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| nhw | 15 | `0.001692095` | `2.2e-13` | `0.421962462` | `1.3e-15` | `0.704` |
| nhb | 15 | `0.005150633` | `6.5e-14` | `0.220699387` | `8.8e-16` | `0.906` |
| his | 15 | `0.001980572` | `1.7e-10` | `0.316338543` | `5.0e-13` | `0.235` |
| as/pi | 15 | `0.001836718` | `2.2e-14` | `0.320411014` | `1.7e-16` | `0.137` |
| ai/an | 15 | `0.009189841` | `2.7e-14` | `0.335633908` | `3.3e-16` | `0.318` |
| pooled | 15 | `0.001714238` | `2.3e-13` | `0.366315772` | `1.5e-16` | `0.762` |

The R² column is the finding, not the reproduction. The trend is well determined
for Non-Hispanic White and Black; for Hispanic, Asian/Pacific Islander and AIAN
the line explains `14%` to `32%` of the variance, and those three are
nonetheless extrapolated six years forward.

**The pooled row has no observed `G` at all.** Its column `G` is empty for all 25
years; the pooled line lives in the side block and is fitted throughout. Any use
of a pooled reporting fraction is therefore using a fitted value even in years
where all five race-specific values were observed.

### Where the reporting fraction is observed

| Years | Reporting fraction `G` |
| --- | --- |
| 2000–2001 | fitted line, extrapolated **backwards** |
| 2002–2014 | observed |
| 2015 | fitted line, interpolated |
| 2016 | observed |
| 2017 | fitted line, interpolated |
| 2018 | observed |
| 2019–2024 | fitted line, extrapolated **forwards** |

Six of the 25 years — a quarter of the series, and the six most recent — rest on
a linear extrapolation, structurally the same problem the review recorded for the
reduction-rate CSV tail.

## The race denominators are inconsistent

Comparing column `D` against `us_births.db` by year and `mracehisp_c`, 38
race-year cells differ across 11 years, in two mechanisms.

**The multi-race code is treated inconsistently.** `mracehisp_c = 6`
(non-Hispanic, more than one race) appears from 2014 and reaches 91,634 births by
2024. In 2014, 2015, 2017, 2018 and 2019 the workbook distributes it across nhw,
nhb, as/pi and ai/an — the 2014 excesses are `+35,481`, `+22,683`, `+12,650`,
`+4,491`, summing to exactly the 75,305 multi-race births. In 2016 and 2020–2024
it is dropped entirely. The denominator definition therefore changes from year to
year within a single series.

**Pacific Islander births are grouped with AIAN in four years.** In 2016, 2020,
2021 and 2022 as/pi is short by exactly the amount ai/an is long: `9,350`,
`9,630`, `9,536`, `10,128` births. These are precisely the NHOPI counts —
`mrace6 = 5` for 2020–2022, and `mrace15` codes 11–14 (Hawaiian, Guamanian,
Samoan, other Pacific Islander) summing to `9,350` births and 4 flags for 2016.
NHOPI belongs with Asian/Pacific Islander, not with American Indian/Alaska
Native.

The consequence concentrates in the smallest group. AIAN has 24,000–40,000 births
a year, so adding ~9,500 NHOPI births moves its denominator by about 30% — and it
is AIAN that carries the implausible `0.66` reporting fraction in 2016 and `0.51`
in 2018 against a series otherwise near `0.40`.

Two smaller discrepancies: 2005 differs by `+5` births across three groups, and
2023 by `-9,123` spread across all five, consistent with a
provisional-versus-final release.

**The pooled total is unaffected.** Recomputing on our own counts reproduces
Gert's pooled fraction to four decimal places in every observed year, because the
errors mostly move births between two small groups. This asymmetry is the
practical conclusion: **the workbook is sound for a pooled anchor and unsafe for
race-specific work.** That bears directly on the
[race-surveillance audit](20260803-dsp004-race-surveillance-audit.md), whose
fail-closed decision looks better founded in light of it.

## The anchor

Taking only the surveillance prevalences from the workbook and recomputing on our
own counts, with multi-race and unknown-origin births carried at the
composition-weighted mean of the five observed groups:

| Mid-year | Window | Births (5yr) | Prev./10⁴ | Expected/yr | Recorded/yr | Reported | Residual |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2000 | 1998–2002 | `20,031,387` | `11.6821` | `4,680.2` | `1,791.2` | `0.3827` | `1.0%` |
| 2002 | 2000–2004 | `20,337,729` | `12.5183` | `5,091.9` | `1,869.0` | `0.3671` | `0.8%` |
| 2004 | 2002–2006 | `20,661,214` | `12.4901` | `5,161.2` | `1,956.2` | `0.3790` | `0.7%` |
| 2006 | 2004–2008 | `21,116,910` | `12.5924` | `5,318.2` | `2,022.2` | `0.3802` | `0.7%` |
| 2008 | 2006–2010 | `20,997,330` | `12.5272` | `5,260.8` | `2,016.0` | `0.3832` | `0.7%` |
| 2010 | 2008–2012 | `20,322,113` | `12.5292` | `5,092.4` | `1,949.0` | `0.3827` | `0.7%` |
| 2012 | 2010–2014 | `19,868,060` | `12.7820` | `5,079.1` | `1,941.6` | `0.3823` | `1.1%` |
| 2014 | 2012–2016 | `19,844,580` | `13.3721` | `5,307.3` | `2,026.0` | `0.3817` | `2.0%` |
| 2016 | 2014–2018 | `19,609,281` | `13.1618` | `5,161.9` | `2,072.4` | `0.4015` | `2.9%` |
| 2018 | 2016–2020 | `18,999,754` | `13.4780` | `5,121.6` | `2,060.8` | `0.4024` | `3.1%` |

(Odd mid-years omitted for brevity; the full 17 rows are in
`expected_births_anchor.csv`.)

Expected Down-syndrome live births sit between `4,680` and `5,346` a year across
the span. Pooled prevalence rises from `11.68` to `13.48` per 10,000, a
log-linear trend of `0.573%` a year (SE `0.079`pp) with residual SD `1.68%`.

The "residual" column is the share of births carried at the imputed mean rather
than an observed group prevalence. It grows from `0.7%` to `3.1%` as the
multi-race code fills out, so the imputation matters most in exactly the recent
years the model cares about.

### Cross-check against `DSP004`

`DSP004`'s cohort is 33,527,704 births over 2016–2024 with 17,809 flags, and its
posterior total is `44,254.9`.

| Anchor basis | Prevalence/10⁴ | Expected total | vs `DSP004` |
| --- | ---: | ---: | ---: |
| 2016 window held flat | `13.1618` | `44,128` | `-0.3%` |
| 2018 window held flat | `13.4780` | `45,189` | `+2.1%` |
| Log-linear trend extrapolated | `13.539` | `45,393` | `+2.6%` |

Three bases bracket the posterior within about two percent. Given that the review
found the total's level to be prior-only and its interval roughly 2.7× too
narrow, this is genuine corroboration — but note what it corroborates. It
corroborates the *level*, which is what the prior already asserted, and it does
not vindicate the interval.

### Gert's fraction and the model's `s` are different estimands

Gert's pooled reporting fraction reaches `0.4024` for the 2018 window; the
`DSP004` posterior sensitivity is `0.3402`. These differ by exactly the
false-positive allowance:

```text
f · N = 7.8e-5 × 33,527,704 = 2,615 of 17,809 flags = 14.7%
0.4024 × (1 − 0.147) = 0.3433   vs   posterior s = 0.3402
```

Gert's fraction is recorded-over-expected with every flag treated as real; `s` is
true-positive sensitivity after 14.7% of flags are assigned to false positives.
They must not be compared directly, and a reset must state which of the two the
anchor pins. The `14.7%` rests on the mis-derived `f = 7.8e-5` default that the
[false-positive note](20260803-false-positive-channel-identification.md)
recommends replacing, so the reconciliation is exact only under the current
default.

## The real decision is trajectory, not level

This is the substantive point for the reset, and it is where the two available
approaches genuinely disagree.

Gert's column H estimates true counts as `recorded / G`. Because `G` rises
linearly while recorded flags fall after 2020, his **implied prevalence falls
through the tail**:

| Year | Implied true prevalence /10⁴ (col H) | Mean `G` |
| --- | ---: | ---: |
| 2016 | `13.664` | `0.4259` |
| 2018 | `13.782` | `0.3976` |
| 2020 | `13.442` | `0.4024` |
| 2022 | `12.579` | `0.4103` |
| 2024 | `12.665` | `0.4183` |

His five-race total for 2016–2024 is `43,194` at an overall fraction of `0.4020`,
matching the `~43,200` recorded in the June note. Holding the 2018-window
prevalence flat over the same five-race births gives `44,133`, `2.2%` higher.

So the two approaches agree on the **total** to about two percent while
disagreeing on its **shape**: Gert's prevalence declines `7.3%` from 2016 to
2024, an imposed flat or rising anchor does not. The June note found the same
divergence from the other direction — our backtested imputation runs 10–40% above
Gert's column I by 2024 — and found the selection posterior robust to it
(`-1.5%`).

The causal directions are opposite, and that is the choice to make explicitly:

- **Gert's direction.** Recorded counts are informative about true prevalence,
  given an extrapolated recording fraction. Falling recorded counts imply falling
  prevalence.
- **Anchor direction.** Surveillance prevalence is informative about true
  prevalence, and the recording fraction absorbs the residual. Falling recorded
  counts imply falling *recording*.

Neither is testable from these data after 2018, because nothing external observes
the tail. The honest response is to carry both as scenarios and to let the
interval reflect the disagreement, rather than to pick one and report a narrow
posterior.

## What anchoring on expected live births would change

The current construction sets `T_y = Σ_a N_{y,a} · θ_a · (1 − ρ_y)`, with `ρ` read
from `us-births-reduction-rates-1989-2024.csv`
([`selection/core_reduction.py:50`](../src/dspopulations_us_birth_certificates/selection/core_reduction.py)).
Because that series was itself built as `1 − S/M`, the Morris curve cancels and
`T_y = N_y · S_y` — conditional on a consistency nothing in the pipeline
enforces. Anchoring directly on `T_y = π_y · N_y` removes both the Morris curve
and the reduction ratio from the level, and makes that cancellation an identity
rather than a coincidence.

**What it fixes.** The review's Finding 1 was that the total's level is
prior-only: the likelihood identifies the product `θ_a · (1 − ρ_y) · (s − f)`, and
the common level is not. An observation equation on `π` supplies the missing
constraint from data. That is the substantive argument for the reset and it is a
good one.

**What it does not fix.**

- *Coverage.* Surveillance ends at the 2016–2020 window. Of the nine years in the
  model window only two mid-years are anchored; 2021–2024 stays unanchored. The
  extrapolation moves rather than disappears — with the advantage that in a
  state-space form it can carry honest uncertainty.
- *Resolution.* The 17 windows overlap by four years in five. They span 23
  birth-years, so they carry roughly `4.6` independent observations, not 17.
  Treating them as independent would overstate precision by about `1.9×`, and
  single-year variation is not identifiable from this series at all.
- *Age structure.* The anchor is a total, so it cannot replace `θ(age)`. The age
  gradient is what identifies `f` and the age pattern of recording; the anchor
  pins the level. They are complementary and the reset should keep both.
- *Uncertainty on `π`.* The workbook attaches none — every prevalence is a bare
  point value with no interval, no contributing-programme count and no covered
  birth denominator. Anchoring on a point value with an arbitrary observation
  variance would repeat, in a new place, the mistake of pinning the level by
  assumption.

### Suggested form

The selection model already implements this pattern: `PREV_RACE_YEAR` in
`selection/recording_anchor.py` is a de Graaf true-prevalence target applied as a
soft full-margin potential, generated by `scripts/derive_recording_rates.py` with
a backtested imputation for 2019–2024 and sigmas that widen across the tail. That
imputation is more defensible than Gert's linear extrapolation and is already
unit-tested. **The first move is therefore a port, not a build.**

Where a new construction is warranted is the observation equation, because the
overlapping windows need handling. Treat log prevalence as a latent annual series
with a local linear trend, observed through the windows:

```text
log π_y = log π_{y-1} + δ_y ;    δ_y = δ_{y-1} + ε_y
obs_w   = log( mean( π_y : y in window w ) ) + ν_w
```

Averaging over the window in the observation equation is what makes the overlap
harmless: each window constrains a five-year mean of the latent path, so
overlapping windows share latent years rather than double-counting evidence.
Forecasts for 2021–2024 then fall out of the same state equation with intervals
that widen as they should — the review measured the process variance needed at
`1.5%`–`1.8%` year-to-year in log prevalence, against the current implied
`7.8%`/`17.6%`.

Fit the pooled anchor first. Race-specific anchors need the denominator errors
resolved, and the AIAN series is too noisy — 12 to 33 flags a year — to carry a
group-specific trend.

## Should the model window start earlier?

Surveillance ends in 2018 while the model window starts in 2016, so only two
mid-years are anchored. Extending the window backwards is therefore attractive,
and the birth-certificate data is not the constraint: the database holds
1989–2024 (142.9M records), and
`us-births-surveillance-prevalence-1989-2024.csv` carries a *national*
surveillance prevalence with genuine annual variation from 1989 to 2018.

The diagnostics changed the reason for doing it. Reproduce with
[`scripts/audit_certificate_revision_era.py`](../scripts/audit_certificate_revision_era.py).

### The reporting fraction does not trend — it steps

| Era | Pooled reporting fraction |
| --- | --- |
| 15 windows, mid-2000 to 2014 | **`0.3793`**, sd `0.0047` |
| 2 windows, 2016 and 2018 | **`0.4019`** |

| Model of the pooled reporting fraction | Coefficient | R² |
| --- | ---: | ---: |
| Linear trend in year | `+0.001219` | `0.560` |
| Revised-certificate coverage | `+0.015561` | `0.439` |
| **Step at 2015** | `+0.022668` | **`0.746`** |
| Linear trend, 2000–2014 only | `+0.000563` | — |

The step fits far better than a line, and the trend fitted *only* on the flat era
is `+0.00056`/yr against Gert's pooled `+0.00171`/yr. **His slope is three times
steeper because two high windows tip a line through fifteen flat ones.** That is
the concrete mechanism by which the six-year forward extrapolation goes wrong,
and it is invisible from inside a 2016–2024 window.

### The step is the 2003 certificate revision

The revision is identifiable per record — `ca_down`/`ca_downs` populated, versus
the unrevised `uca_downs` only — and coverage runs 0% (2003) → 18.4% (2004) →
77.4% (2010) → 96.4% (2014) → 100% (2016).

Pooling the ten years in which both versions are in use, revised certificates
record `4.99` flags per 10,000 against `4.54` unrevised, a ratio of `1.098`.
Conditioning properly on maternal age and the year's reduction, the model puts it
higher (below).

The raw contrast is confounded by age composition and, more seriously, by state
composition: states adopted the revised certificate at different times, so
comparing versions within a year compares early adopters with late adopters. Our
imported database carries **no state identifier in any year** — only
`mbstate_rec` (mother's nativity) and `restatus` (residence status). Per the
[CDC WONDER feasibility note](20260803-cdc-wonder-state-level-feasibility.md) the
1989–2004 public-use files do carry state, so 2004 could in principle be resolved
by re-importing those columns, but geographic detail is absent from 2005 onward
and WONDER's coverage starts at 2016. The 2005–2015 phase-in — eleven of the
twelve relevant years — therefore cannot be resolved by state from any available
source. The phase-in bounds the size of the measurement shift; it is not a clean
natural experiment, and it will not become one.

### What breaks across eras

| Start | Anomaly status unknown | Race unknown | Source field | Confirmed/pending |
| --- | ---: | ---: | --- | --- |
| 1989 | **`17.57%`** | `5.12%` | `downs` | none |
| 1993 | `6.52%` | `1.26%` | `downs` | none |
| 1996 | `2.12%` | `1.46%` | `downs` | none |
| 2003 | `1.34%` | `0.70%` | `uca_downs` | none |
| **2004** | **`1.17%`** | `0.79%` | `uca_downs` + `ca_down(s)` | within revised only |
| 2016 | `0.17%` | `0.92%` | `ca_down(s)` | complete |

`mage_c` is **0% null in every year from 1989**, so the Morris `θ(age)` curve
carries across the whole span unchanged — the one thing that might have blocked
the extension does not. `mracehisp_c` is harmonised 1989–2024 by construction;
only the multi-race category is 2014+.

**Confirmed/pending is contaminated before 2016.** Every unrevised flag is coded
`ca_down_c = 'C'` although the unrevised certificate has no confirmation step —
all 1,465 unrevised 2004 flags are 'C', contributing zero pending. So pre-2016
"confirmed" mixes genuinely-confirmed with never-assessed cases, and the
two-channel design in the
[false-positive note](20260803-false-positive-channel-identification.md) must be
restricted to revised records. On that basis the pending share is a consistent
`60.9%` (sd `6.1`pp), running `69%` in 2004 down to `54%` in 2024.

**2004 is the recommended start.** It is where the revised indicator becomes
usable, it brings 13 anchored surveillance windows instead of 2, and anomaly
status is 98.8% complete. 1996 is available for a national-anchor-only extension;
1989 is not advisable at 17.6% unknown status.

## The 2004-start refit

`DSP006` (`recording_s_revision`) extends `DSP004` by splitting every cell on
certificate version and giving each its own recording sensitivity — one extra
parameter. `recording_s` remains the *revised* sensitivity so it stays directly
comparable with fits confined to 2016 onward, and the unrevised level is a logit
offset from it. Centring is deliberately absent: the two levels are
distinguishable measurement regimes, not exchangeable groups.

Reporting profile, 4 chains × 1,500 draws, max R-hat `1.0033`, min ESS `1,146`.

| Fit | 2016–2024 total | ETI width | `s` |
| --- | ---: | ---: | --- |
| `DSP004`, 2016–2024 (frozen) | `44,255` [`41,934`–`46,565`] | `10.46%` | `0.3402` |
| `DSP004`, 2004–2024 | `45,866` [`44,566`–`47,155`] | `5.64%` | `0.3271` |
| **`DSP006`, 2004–2024** | **`44,505`** [`43,219`–`45,814`] | **`5.83%`** | `0.3379` revised / `0.2954` unrevised |

Three findings.

**Extending the window nearly halves the interval**, from `10.46%` to under `6%`
of the mean. The 2016–2024 total is constrained by 21 years of data rather than
9, and the reduction priors for 2004–2019 sit in the observed rather than the
extrapolated regime.

**Extending without the revision split biases the modern total upward by
`3.6%`** (`45,866` against `44,255`). A single constant `s` is dragged down by
the unrevised years to `0.3271`, and a lower sensitivity implies a *higher* true
count for the same recorded flags. Pooling incomparable measurement eras is not
a free lunch.

**The revision split removes that bias while keeping the precision.** `DSP006`
gives `44,505`, within `0.6%` of the frozen 2016–2024 run, at an interval barely
wider than the pooled long-window fit. Its revised sensitivity `0.3379` sits
within `0.7%` of the `0.3402` estimated from the modern window alone — a
model fitted over 21 years recovers the same modern recording rate, which is
evidence that the split separates the eras correctly rather than absorbing
signal.

The revision effect is large and unambiguous:

```text
s revised    0.3379  [0.3286, 0.3475]
s unrevised  0.2954  [0.2852, 0.3063]
ratio        1.1441  [1.1098, 1.1792]     P(revised > unrevised) = 1.000
```

### What the revision refit does *not* do

The level still comes from the reduction CSV via `from_reduction_csv`. The
surveillance observation equation is implemented separately, in `DSP007`/`DSP008`
below; `DSP006` on its own is the window extension plus the revision covariate,
nothing more, and under it the Morris cancellation and the prior-only level both
still stand.

Two caveats. The revised/unrevised contrast is confounded by unobserved state
composition, and for 2005–2015 no available source can resolve it, so `1.144`
should be read as the size of the measurement shift and not as a causal estimate
of the revision's effect. And within fully revised
records the flag rate still fell from `5.438` per 10,000 in 2016–2018 to `5.131`
in 2022–2024, **`-5.7%` on a constant instrument** — so something real happened
after 2018 that the revision does not explain, and no amount of pre-2016 data
resolves whether it was prevalence or recording. Extending the window sharpens
that question; only post-2018 surveillance can settle it.

## The anchored model

`DSP007` replaces the reduction-rate prior with the observation equation
proposed above; `DSP008` adds the revision split. Latent annual log prevalence
follows a local linear trend, is observed through the surveillance programmes'
centred five-year window means, and sets the reduction by the accounting
identity

```text
eta_year = prevalence_year / Morris_expected_prevalence_year
```

so the reduction is a *consequence* of an anchored prevalence rather than an
imported prior. The latent series is padded two years either side of the model
window, so a window centred on the first or last modelled year remains usable;
2004–2024 admits 13 windows, mid-2004 to mid-2018.

Three properties are worth stating because they were the point of the exercise.

**The level is now identified by data.** `rho_logit_year` does not exist in an
anchored fit. The reduction CSV is still loaded so the report can contrast the
superseded prior with the anchored posterior, and the config records
`reduction_prior_enters_likelihood: false` so no reader can mistake the retained
comparison series for the prior the model used.

**The overlap is handled rather than ignored.** Each window constrains the mean
of its own five latent years, so overlapping windows share latent years instead
of double-counting evidence. The config also records
`effective_independent_windows` — `3.8` for 2004–2024 — so the 13 windows are
never presented as 13 observations.

**The forecast widens.** 2019–2024 have no window, and the ETI on latent
prevalence grows from `0.63` per 10,000 in 2018 to `0.96` in 2024, a 52%
widening across six unanchored years.

### Results, 2004–2024

| Fit | Level from | `s` | 2016–2024 total | ETI width |
| --- | --- | --- | ---: | ---: |
| `DSP004`, 2016–2024 (frozen) | reduction CSV | `0.3402` | `44,255` [`41,934`–`46,565`] | `10.46%` |
| `DSP004` | reduction CSV | `0.3271` | `45,866` [`44,566`–`47,155`] | `5.64%` |
| `DSP006` | reduction CSV | `0.3379` / `0.2954` | `44,505` [`43,219`–`45,814`] | `5.83%` |
| `DSP007` | **surveillance** | `0.3254` | `45,487` [`44,804`–`46,240`] | `3.16%` |
| **`DSP008`** | **surveillance** | `0.3347` / `0.2970` | **`44,589`** [`43,952`–`45,231`] | **`2.87%`** |

All five rows are reporting profile. `DSP008` reaches max R-hat `1.0100`, exactly
at the `<1.01` gate, with min ESS `589`; `DSP007` is cleaner at `1.0042` and
`762`. The `DSP008` value needs a longer run before it is cited, though the dev
and reporting fits agree to `0.02%` on the total, which argues the borderline
statistic is sampler noise rather than a specification problem.

The era-pooling bias recurs in the anchored model exactly as it does in the
prior-based one: `DSP007` with one constant `s` gives `45,487`, about `2%` above
`DSP008`'s `44,589`. So the revision split matters independently of how the level
is set, and `DSP008` is the specification that carries both fixes.

`DSP008`'s `44,589` sits within `0.7%` of both `DSP006` and the frozen
2016–2024 run. Four routes to the modern total — a short window with a prior
level, a long window with a prior level and the revision split, and the anchored
versions of each — agree to about a percent.

### The interval is not a precision gain. It is a relocated assumption

This is the finding that matters most, and it cuts against the headline.

The estimated observation SD is `0.0122` — the 13 windows reconcile with a
smooth latent path to about `1.2%`. But that quantity measures
only whether the windows are *mutually consistent*. It cannot measure whether the
surveillance programmes' prevalences are *accurate*, because the workbook
supplies no uncertainty at all. Fixing the observation SD at larger values is
therefore the honest sensitivity axis, and `--anchor-obs-sigma-fixed` exists for
exactly that:

Dev profile, so the levels are indicative; the relationship is the point.

| Surveillance observation SD | 2016–2024 total | ETI width |
| --- | ---: | ---: |
| estimated (`0.012`) | `44,580` | `2.87%` |
| fixed `0.05` | `44,522` | `5.86%` |
| fixed `0.10` | `44,441` | `9.69%` |
| fixed `0.20` | `44,088` | `16.19%` |

The **mean is robust** — it moves by `1.1%` across the whole range, which is the
real gain, because the level is now determined by data under any of these
assumptions. The **width is almost entirely a function of an unknown.** At a
plausible 10% surveillance error the interval is `9.69%`, essentially the
`10.46%` the original 2016–2024 fit reported.

So anchoring does not narrow the interval so much as **move where the narrowness
comes from**: previously from tight reduction priors that the review found
roughly 2.7× too confident, now from an assumption that surveillance prevalence
is measured almost exactly. The improvement is real but it is one of *candour*,
not precision — the new assumption is single, explicit, and answerable by one
question to Gert, where the old one was diffuse and buried in a CSV.

**Do not report `2.87%`** as the interval on the 2016–2024 total. Until Gert
supplies uncertainty, report the sensitivity row that matches a defensible view
of surveillance accuracy, and say which was chosen.

### It also resolves the trajectory question — by assumption, not by evidence

Under `DSP008` the latent prevalence rises to `13.8` per 10,000 around 2018 and
then falls to about `12.9` by 2024. That is Gert's direction, and it is worth
being clear about why the model says it.

2019–2024 have no surveillance window, so latent prevalence there is constrained
by the state equation *and by the recorded flag counts*. With `s` held constant
across those years, a falling flag rate can only be explained by falling
prevalence. The model is not discovering that prevalence fell; it is inheriting
the constant-`s` assumption and reporting the consequence. A specification
allowing `s` to drift after 2018 would attribute the same decline to recording
and would fit equally well. Nothing in the data after 2018 distinguishes them —
which is the same conclusion the era audit reached from the other direction.

## Asks of Gert

1. **Uncertainty on the surveillance prevalences.** Confidence intervals, or the
   number of contributing programmes and the birth denominator each window
   covers, so an observation variance can be derived rather than assumed. This is
   now demonstrably **the binding constraint on the entire model**: with the
   anchor implemented, the interval on the 2016–2024 total runs from `2.87%` to
   `16.19%` purely as a function of this one unknown, while the mean barely
   moves. Nothing else we can do ourselves narrows it.
2. **2015 and 2017.** Why are these two windows absent when 2014, 2016 and 2018
   are present? Are the underlying data recoverable?
3. **Windows after 2016–2020.** Are 2019 or 2020 mid-years available or expected?
   Six of the nine years in our model window are currently unanchored.
4. **Race harmonisation.** Confirm the intended treatment of the multi-race code
   from 2014 — it is included in five years and dropped in six — and confirm that
   grouping Pacific Islander births with AIAN in 2016 and 2020–2022 is an error
   rather than a definition.
5. **2023 vintage.** Our counts exceed the workbook's by `9,123` births spread
   across all five groups. Which release does the workbook use?
6. **Adjustments already applied.** Are the surveillance prevalences adjusted for
   ascertainment, and do they cover live births only?
7. **Trajectory after 2020.** His column H implies prevalence fell `7.3%` from
   2016 to 2024. Is that intended as an estimate, or is it a by-product of
   extrapolating the recording fraction? His view on which direction is more
   plausible would settle the main open modelling choice.
8. **Which file is canonical.** `ALT3` implies alternatives 1 and 2.

## Recommended next steps

1. **Adopt 2004–2024 with `DSP006` as the identification window**, reporting
   2016–2024 from it. This is done and reproducible; it is the cheapest
   available improvement to the interval, and it does not depend on anything
   further from Gert.
2. ~~Port the pooled anchor into the core DSP family.~~ **Done** as
   `DSP007`/`DSP008`. What remains is to decide, with Gert's input on
   surveillance accuracy, which sensitivity row to report — and to reconcile the
   core family's anchor with the selection model's `PREV_RACE_YEAR`, which
   applies the same idea by race as a soft potential and whose backtested tail
   imputation is still the better-tested treatment of 2019-2024.
3. **Keep `θ(age)`.** The anchor pins the level; the age gradient identifies `f`
   and the recording shape. The reset should add the anchor, not remove the age
   model.
4. **Restrict the confirmed/pending channel model to revised records**, so
   `s_C`/`s_P` and `f_C`/`f_P` are estimated on a consistent instrument. That
   makes the two-channel design estimable from 2004 rather than 2016.
5. **Carry the trajectory disagreement as an explicit scenario axis**
   (Gert's column I tail versus a flat or trended anchor), the way
   `--degraaf-tail` already does for the selection model, and report the spread
   rather than one narrow posterior.
6. **Fix the race denominators before any group-level anchoring**, using our own
   `mracehisp_c` throughout, and decide explicitly how multi-race births are
   handled — they are `2.5%` of births by 2024 and have no surveillance
   prevalence of their own.
7. **Do not attach an invented observation variance to `π`.** Until Gert supplies
   uncertainty, prefer the widening sigmas already backtested in
   `derive_recording_rates.py` over a fixed guess, and say in any write-up that
   the interval is conditional on it.
8. **Decide how to track the workbook.** The June note settled that its content
   is publishable aggregate data rather than DUA-restricted microdata, subject to
   Frank's confirmation before external publication. The open question is only
   whether the `.xlsx` binary itself belongs in a public repository, or whether
   the extracted CSVs suffice. Until that is decided the file is untracked and
   **not** covered by `.gitignore`, so a bulk `git add` would sweep it in.

Note that the two surveillance CSVs already in `data/` are *not* in conflict:
`us-births-surveillance-prevalence-1989-2024.csv` is a separate national series
spanning 1989 onward and is not a births-weighted aggregate of the by-ethnicity
values, which is why it differs from the workbook-derived pooled prevalence by
`-0.4%` to `+2.9%`. The June note records this. Its flat tail from 2018 and the
reduction CSV's rising tail are still worth reconciling as *assumptions*, but they
are not inconsistent measurements of the same thing.

## Reproducing

```bash
python scripts/extract_degraaf_surveillance.py
```

Read-only throughout: the workbook is parsed with the standard library, so the
canonical conda environment needs no Excel dependency, and `data/us_births.db` is
opened read-only. Outputs land in `output/degraaf_surveillance/`. `--strict` exits
non-zero while the workbook-versus-database discrepancies stand.

The era audit consumes that anchor and must run second:

```bash
python scripts/audit_certificate_revision_era.py
```

The anchored fits, and the surveillance-accuracy sensitivity:

```bash
python scripts/fit_core_reduction_model.py DSP008 --profile reporting --years 2004-2024 --output-dir output/refit2004/DSP008-reporting
```

```bash
python scripts/fit_core_reduction_model.py DSP008 --profile dev --years 2004-2024 --anchor-obs-sigma-fixed 0.10 --output-dir output/refit2004/DSP008-obs0.10
```

`DSP007`/`DSP008` read the anchor from `output/degraaf_surveillance/`, so the
extraction script must run first. The prior-based refits, reporting profile:

```bash
python scripts/fit_core_reduction_model.py DSP004 --profile reporting --years 2004-2024 --output-dir output/refit2004/DSP004-baseline-reporting
```

```bash
python scripts/fit_core_reduction_model.py DSP006 --profile reporting --years 2004-2024 --output-dir output/refit2004/DSP006-revision-reporting
```

The 2016–2024 comparison is the frozen `DSP004` calibration-base reporting run.
`DSP006` requires a year range spanning the phase-in: run on 2016–2024 alone every
record is revised, the offset is unidentified and reverts to its prior.
