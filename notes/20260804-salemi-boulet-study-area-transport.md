> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Fable 5).

# What Salemi and Boulet actually say about `s`, once you correct for where they were measured

**Date:** 2026-08-04

**Status:** Measured result, no model refit. Every figure below is reproduced by
`scripts/compare_study_area_recording.py` against the frozen extract and the
tracked CDC WONDER aggregates; the state table is written to
`notes/figures/study-area-recording-transport.csv`. Answers recommendation 5 of
the [core model family review](20260803-dsp-core-model-family-review.md) and the
Salemi route named as unexploited in the
[group identification note](20260803-group-reduction-recording-identification.md).

**Two findings, pulling in opposite directions.** First, the external anchor this
project has been citing does not exist: **Boulet reports `18%` birth-certificate
sensitivity for Down syndrome, not `40%`**, and the `40%` figure appears nowhere
in the paper. Three notes carry it, and two of them —
[the status review](20260514-status-review.md) and
[the reporting sweep](202604202200-selection-reporting-sweep-findings.md) — grade
fits against it. Second — and this is why
the correction does not cost the project its anchor — **both validation studies
were conducted in low-recording areas, and Florida is the third-lowest-recording
state in the country.** Transported to national recording level, Boulet gives
`0.374` and Salemi `0.319`, which bracket the `DSP` posterior `s = 0.344`. The
number the project has been using was approximately right for entirely the wrong
reason, and it is now right for a defensible one.

## Question

Boulet (2011) and Salemi (2017) are the only external evidence on
`P(recorded | DS livebirth)` — the one direction the `DSPnnn` family cannot
identify from certificate data alone. The family review asked what they actually
measure and whether their denominators are confirmed-or-pending or confirmed-only.
Both questions turn out to be answerable, and a third one — whether a
single-locality sensitivity can be used nationally at all — turns out to matter
more than either.

## What the two papers report for Down syndrome

Both papers give Down syndrome its own row. Neither headline figure is the DS
figure, which is how the misattribution arose: Boulet's abstract leads with `23%`
across six defects, Salemi's with `19.1%` across seven.

| | Boulet 2011 | Salemi 2017 |
|---|---|---|
| setting | metro Atlanta, MACDP | Florida, FBDR + enhanced surveillance |
| births | 1995–2005 | 2007–2011 |
| certificate | 1989 revision | 2003 revision |
| DS sensitivity | `18.1%` (113/625) | `24.6%` (364/1478), 95% CI 22.4–26.8 |
| DS PPV | `97.4%` (113/116) | `87.3%` (84.1–90.5) |

Salemi additionally splits on the karyotype sub-field, which maps directly onto
the NVSS `ca_down_c` `C`/`P` states: karyotype-confirmed sensitivity `7.0%`
(PPV `89.6%`), pending `17.7%` (PPV `86.4%`).

**This resolves the family review's definitional question.** Boulet's `18.1%` is
unambiguously the confirmed-or-pending analogue: the 1989-revision Georgia
certificate carried a flat list of 22 anomaly checkboxes with no karyotype
sub-field, so the distinction did not exist to be made. The confirmed-only
comparator is Salemi's `7.0%`, not Boulet's figure.

The nearest `40%`-shaped numbers in Boulet are the composite sensitivity in
hospitals under 1,000 annual births (`37.5%`) and the threefold
prevalence gap for non-Hispanic White infants (implying `~33%`). Neither is a
Down syndrome sensitivity. The provenance should be corrected wherever it appears.

## Why the check could not be run by state

Recorded DS prevalence varies about ninefold across states, so a locally-measured
sensitivity means little until you know how that locality records. The direct
version of that check is unavailable:

- The natality extract carries **no geography at all** — only `mbstate_rec`
  (nativity) and `restatus`. No ingest script has ever referenced a state column.
- NCHS withdrew geographic detail from the public-use files **with the 2005 data
  year**, so Salemi's window is unreachable in principle, not just unbuilt. The
  [WONDER feasibility note](20260803-cdc-wonder-state-level-feasibility.md)
  records the same boundary.
- WONDER carried anomaly data for 2007–2013 but **withdrew it on 27 August 2015**
  from every database except `Natality 2016-2024 (expanded)`. Salemi's window was
  covered and no longer is.
- Boulet's window is partly recoverable — the 1989–2004 public-use microdata do
  carry state — but the raw source files are no longer present under `data/`.

So the comparison was run on the margin that survives. It does not need state at
all, because each paper reports its own study-area recorded count; what was
missing was the national comparator, and the extract has that.

## The transport

    factor     = national recorded DS prevalence / study-area recorded prevalence
    s_national = s_study x factor

National recorded DS prevalence per 10,000 covered births, using the core models'
own definition of a recorded case (`down_ind`, confirmed-or-pending):

| window | recorded | covered births | per 10,000 |
|---|---|---|---|
| 1995–2005 | 19,772 | 43,057,164 | `4.59` |
| 2007–2011, 2003-revision area | 6,984 | 14,465,860 | `4.83` |
| 2016–2024 | 17,809 | 33,527,704 | `5.31` |

The 2016–2024 row reproduces the frozen extract's `17,809` flags and its
`7,984`/`9,825` confirmed/pending split exactly, so the query matches the model's
cell construction rather than approximating it.

Florida is compared against the **2003-revision reporting area** rather than all
births, since Florida adopted the 2003 revision in 2004 and the unrevised
remainder is not the right comparator for it.

| | study area | national | factor | `s` measured | `s` national |
|---|---|---|---|---|---|
| Boulet / metro Atlanta | `2.22` | `4.59` | `2.068` | `0.181` | **`0.374`** |
| Salemi / Florida | `3.72` | `4.83` | `1.297` | `0.246` | **`0.319`** |

Salemi's paper does not print its birth denominator; `1,120,000` is the NCHS final
natality figure for Florida resident livebirths 2007–2011 and is the only external
number in the calculation. Across a generous `1.05M`–`1.20M` range the transported
sensitivity moves `0.299`–`0.342`, so the conclusion does not rest on it.

### The transport is legitimate, and that is checkable

Scaling a sensitivity by a recorded-prevalence ratio is only valid if the study
area is **ordinary in true prevalence and unusual only in recording**. Otherwise
the ratio absorbs a real prevalence difference and the transport is meaningless.
Each study reports its verified-registry prevalence, so this can be tested
directly against the project's own surveillance series:

| | study-area true prevalence | project surveillance, same years | difference |
|---|---|---|---|
| Boulet / MACDP | `11.97` per 10,000 | `11.68` | `+2.4%` |
| Salemi / FBDR | `13.20` per 10,000 | `12.59` | `+4.8%` |

Both agree within `5%`. The study areas are unremarkable in how many babies with
Down syndrome are born there and remarkable only in how many get recorded, which
is exactly the condition the transport requires. This is the strongest part of the
result: the correction is not an assumption, it is measured on both margins.

## State dispersion, and where the two study areas sit

From the tracked WONDER extract, 2016–2024 pooled, 50 states with unsuppressed
counts (Vermont suppressed):

- national `5.31` per 10,000
- range `1.44` (Hawaii) to `13.43` (Utah) — **`9.3`-fold**
- `p10 = 3.64`, `p25 = 4.56`, `p50 = 6.17`, `p75 = 7.80`, `p90 = 9.62`
- **Florida `2.57` = `0.48x` national, 6th percentile**
- Georgia `4.45` = `0.84x` national, 24th percentile

Florida is the third-lowest-recording state in the country. Choosing Florida and
metro Atlanta to validate birth-certificate recording was, for this purpose, close
to a worst case — entirely reasonable for the authors, whose question was about
their own surveillance programmes, and badly misleading for anyone reading their
sensitivities as national.

The births-weighted SD of log recorded prevalence across states is **`0.375`**,
which at `s ~ 0.25` is about `0.281` on the logit scale. The random-effects
between-study SD implied by the two papers is `tau = 0.264` logits
(`Q = 10.7`, `df 1`, `p = 0.001`). **Between-study heterogeneity in the validation
literature is about the same size as between-state heterogeneity in the US** — two
independent routes to the same dispersion, which is reassuring for both.

## Two side results worth keeping

**The 2003 revision did not improve recording.** Salemi's central claim can be
tested directly here, because revision adoption was staggered across states from
2004 to 2016 and the extract distinguishes the layouts. Restricted to 2006–2010,
where the split is closest to balanced (`63%` revised), recorded prevalence is
`4.84` on revised records against `4.88` on unrevised — **ratio `0.992`**. Pooling
the whole 2004–2015 span gives `1.078`, but that number is confounded: early
adopters record high and late holdouts record low, so the endpoints reflect
adoption order rather than form design. The balanced window is the reading, and it
corroborates Salemi. This supports keeping `recording_s_unrevised_offset` centred
at zero, which it already is.

**The false-positive rate may not need changing after all.** Salemi's DS PPV of
`87.3%` sits close to the `85.3%` the funnel implies
(`recorded_raw 17,776 -> recorded_corrected 15,166`), so the documented units
error in `f = 7.8e-5` is far less consequential than the review feared. Salemi's
directly measured rate is `53` false positives over `~1.12M` Florida non-cases,
or `4.7e-5` — but that is a Florida rate, and transporting it by the same `1.297`
factor gives `6.1e-5`, within striking distance of the shipped value. Whether
false positives scale with sensitivity is not established, so this is a reason for
caution about replacing `f`, not a licence to.

## What to change

1. **Fix the Boulet provenance — applied.** Dated correction callouts are now in
   [the family review](20260803-dsp-core-model-family-review.md) (Finding 6 and
   recommendation 5), [the status review](20260514-status-review.md), and
   [the reporting sweep](202604202200-selection-reporting-sweep-findings.md).
   The withdrawn figure is quoted in each so the record stays legible rather than
   being silently rewritten. Both notes that graded fits against `40%` reach the
   same conclusion against the transported `~37%`, so only the attribution
   changed. `UNKNOWN_S = 0.40` in `scripts/derive_recording_rates.py` is
   documented as a weak neutral fallback rather than as Boulet's figure, and
   needs no change — it happens to sit close to the transported value.
2. **Report the transported sensitivities as the independent check on `s`.** This
   is what recommendation 5 asked for. `0.319` and `0.374` against a posterior of
   `0.344` (89% ETI `0.326`–`0.363`) is corroboration, and it is currently unreported.
3. **Do not tighten the prior on `s` to match.** `recording_s_logit = logit(0.5)`,
   `sigma = 1.0` should stay weak. The transported literature agrees with the de
   Graaf anchor, and both ultimately reference surveillance prevalence, so folding
   it into the prior would double-count evidence that is already in the anchor.
   Its value is as a check, not as an input.
4. **Stop treating the confirmed-only definition as an option worth carrying.**
   Salemi settles it: karyotype-confirmed costs `72%` of cases and buys `3` points
   of PPV (`89.6%` vs `86.4%`). A pending flag is very nearly as trustworthy as a
   confirmed one.
5. **Take Boulet's Table 3 as priors for any future differential-recording layer.**
   These are Down-syndrome-specific adjusted odds ratios, not composite, and being
   within-setting ratios they transport far better than levels do. Suggested prior
   SDs inflate the reported standard errors by `1.5x`:

   | covariate | log OR | SE | prior SD |
   |---|---|---|---|
   | NH-Black vs NH-White | `-0.94` | 0.273 | `0.41` |
   | NH-other vs NH-White | `-1.90` | 0.789 | `1.18` |
   | Hispanic vs NH-White | `-0.62` | 0.417 | `0.63` |
   | nulliparous vs multiparous | `-0.64` | 0.278 | `0.42` |
   | maternal age <35 vs >=35 | `-0.29` | 0.247 | `0.37` |
   | education <high school | `-0.49` | 0.401 | `0.60` |
   | hospital >=2500 births vs <1000 | `-0.99` | 0.429 | `0.64` |

   Hospital birth volume is **not on the public-use file**, and it is one of the
   two strongest and most replicated predictors in both papers. That part of the
   recording variation is structurally unmeasurable here, which argues for a state
   random effect to absorb it — with `0.375` on the log scale as its prior width.

6. **Prefer a false-positive rate to a PPV wherever the model is stratified.** PPV
   is prevalence-dependent, and DS prevalence varies sevenfold across maternal-age
   strata. Holding `f` and `s` at Salemi's values, implied PPV runs from `75.5%`
   for mothers under 20 to `98.1%` for mothers 40+. Salemi's own Table 2 confirms
   the gradient: the adjusted risk ratio for a false-positive flag at age >=35 is
   `0.6` (0.4–0.9). A constant PPV applied across age strata would be wrong in the
   direction that matters most.

## Limitations

- **The Florida birth denominator is external and unverifiable here.** Reported
  range `0.299`–`0.342` for the transported sensitivity.
- **The transport assumes recording ratios are stable within a study window.** For
  each study the comparator is contemporaneous, so no temporal extrapolation is
  needed; the 2016–2024 WONDER data only establishes that Florida records
  persistently low. It has in fact got relatively worse — `0.77x` national in
  2007–2011, `0.48x` in 2016–2024.
- **A residual gap survives on the confirmed-only definition.** Salemi's `7.0%`
  transported (and adjusted for the confirmed-share difference, `27.6%` in Florida
  against `33.1%` nationally for the same years) reaches about `0.109`, against the
  model's confirmed-only fit of `0.186`. Still a factor of `1.7`, where
  confirmed-or-pending lands inside the posterior interval. Carried as an open
  roadmap item at
  [family review recommendation 5b](20260803-dsp-core-model-family-review.md),
  with the leading lead being that the `0.186` run set `f = 0` — worth about a
  factor of `0.8` on the C/P rows of the same sensitivity table — and that Salemi
  supplies the right rate for confirmed flags directly at roughly `1.1e-5`
  (`12` false positives among `115` confirmed). Until that refit is done the
  confirmed-only sensitivity should be treated as externally unvalidated.
- **Counting details.** `births_revised` uses a slightly wider non-null condition
  than `births`, so the revised share reads `100.2%`–`100.3%` from 2016 —
  immaterial at `0.3%` but visible in the by-year output. The `17,809` here versus
  `17,776` in the funnel is the model's additional `mage_c` filter, `0.19%`.
- **Neither study covers the 2016–2024 target window.** Salemi ends in 2011,
  Boulet in 2005. Both found sensitivity flat over their spans — 16 combined years
  across two states — which supports a small drift prior but is not proof. Neither
  can see the cfDNA screening era, where more prenatally-confirmed diagnoses
  should push sensitivity up.
- **Both reference standards are incomplete**, biasing the measured sensitivities
  *upward* (MACDP is `95%` complete at two years; Salemi's active case-finding
  covered `70%` of Florida livebirths). Correcting for that lowers the transported
  figures slightly, toward rather than away from the posterior.

## Reproduction

```
python scripts/compare_study_area_recording.py
```

Reads `data/us_births.db`,
`data/us-births-surveillance-prevalence-1989-2024.csv` and
`data/us-births-wonder-state-pooled-2016-2024.csv`; writes
`notes/figures/study-area-recording-transport.csv`. No refit, no microdata
release — every output is an aggregate.
