> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).

# CDC WONDER as a route to state-level Down syndrome birth-certificate counts

**Date:** 2026-08-03

**Status:** Feasibility assessment **with measured results**. The CDC WONDER
data-use agreement was accepted on the project lead's explicit instruction and
thirteen queries were run against the `D149` database on 2026-08-03. Five
aggregate extracts were written to `data/` (listed below). Every suppression
figure in this note is observed in a returned result set, not predicted. Nothing
was refitted, and no model change is authorised by this note.

An earlier draft of this note estimated suppression from national prevalence
before any query was run. Those estimates were **too pessimistic for the fine
cross-tabulations** — the recorded mass is concentrated, not spread, so most
cells of a deep cross-tab are structural zeros rather than censored — and
**slightly too optimistic for the pooled state table**, where Vermont is
suppressed. The measured figures below supersede them.

## Question

The public-use natality microdata carry no geographic detail from the 2005 data
year onward, so the recorded Down syndrome counts feeding the `DSPnnn` models
cannot be resolved by state. Can CDC WONDER supply sufficient state-level cells
instead?

Answer: **yes for 2016-2024**, more usefully than expected, at 5-year maternal
age resolution rather than exact age, and only through the interactive interface.
No for 1989-2015 via WONDER — but the 1989-2004 public-use microdata already
carry state, so the real gap is 2005-2015.

## What CDC WONDER holds

### Down syndrome is present in exactly one database

The [Natality 2016-2024 (expanded)](https://wonder.cdc.gov/natality-expanded-current.html)
database (`D149`) carries the full 2003-revision congenital anomaly block. Down
syndrome is variable `D149.V132`, coded **Confirmed (1); Pending (2); No (3);
Unknown or Not Stated (9)** — the same four states `ca_down_c` harmonises to, and
confirmed selectable as a group-by variable, not merely as a filter. Both the
`confirmed_or_pending` default and the `--confirmed-only` sensitivity are
reproducible without redefinition.

Also present, and relevant to the false-positive work: **Suspected Chromosomal
Disorder** (`D149.V133`, same coding) and **Congenital Anomalies Checked**
(`D149.V135`).

Anomaly data were **withdrawn from every other natality database on 27 August
2015**. The [natality documentation](https://wonder.cdc.gov/wonder/help/natality.html)
records that they were previously available for births in 2007-2013 only. The
2007-2024, 2003-2006 and 1995-2002 databases carry no Down syndrome field.

### Stratification limits

- **State of residence** is `D149.V21-level1`; region, division, HHS region and
  county are also available. Territories are excluded.
- **Five group-by variables maximum** per query.
- **Single-year maternal age was removed as a group-by in May 2022.** The finest
  maternal age stratification offered is 5-year bands (`D149.V1`, 9 groups;
  `D149.V38`, 10; `D149.V39`, 13 — the 13-group set only splits the teenage
  years). Single-year age survives solely behind the *Average Age of Mother*
  measure. WONDER cannot reproduce `DSP004`'s exact-age cell structure even
  nationally.
- There is **no combined race/Hispanic-origin variable**. Reconstructing
  `mracehisp_c` requires crossing `D149.V42` (Single Race 6) with `D149.V43`
  (Hispanic Origin), which costs two of the five group-by slots.

### Suppression, as documented and as observed

Documented: all statistics of 1-9 births are suppressed (tightened in May 2022
from sub-national only); totals and sub-totals are suppressed when they include a
suppressed component; and the data-use restrictions separately forbid
*publishing* counts of 9 or fewer or rates based on them. Since May 2023 **zero
values are reported**, so a suppressed cell means the count lies in `[1, 9]`, not
`[0, 9]`.

Observed, and not anticipated from the documentation: **WONDER disables the
totals row entirely whenever suppression is possible in the result set** — it
returns "Totals are not available for these results due to suppression
constraints" rather than a suppressed total. Margins must therefore be obtained
from separate, shallower queries. This is exactly the workaround the aggregated
likelihood needs, so it costs nothing but it must be planned for.

## Measured results

National denominator across the window is **33,511,275** live births, matching
the published NCHS totals for 2016-2024 exactly. National recorded Down syndrome
(Confirmed + Pending) is **17,783**, or **5.31 per 10,000**.

### Cross-check against the pipeline

WONDER's 17,783 sits **0.15% below** the 17,809 recorded flags the frozen
`DSP004` run reports for the same window. The gap is in the expected direction
and of the expected size: WONDER counts births to residents of the 50 states and
DC only, excluding territories and non-residents. This is a useful independent
validation of the certificate extract.

By year, WONDER gives 2164, 2039, 2106, 2024, 1954, 1901, 1857, 1855, 1883 — each
slightly above the corresponding
`data/us-births-degraaf-prevalence-recording-2000-2024.csv` figure, consistent
with that file's denominator excluding records of unknown race.

### Suppression by design

Numerator is Down syndrome Confirmed + Pending. "Captured" is the share of the
national 17,783 falling in cells reported exactly.

| Design | Cells | Observed | Suppressed | Zero | Captured |
| --- | --- | --- | --- | --- | --- |
| Year (national) | 9 | 9 | 0 | 0 | 100% |
| State, pooled | 51 | 50 | 1 | 0 | 99.97% |
| State x year | 459 | 367 | 86 | 6 | 97.3% |
| State x race6 x Hispanic, pooled | 1,377 | 176 | 299 | 902 | 95.0% |
| State x year x race6 | 4,131 | 436 | 1,064 | 2,631 | 83.5% |
| State x year x race6 x Hispanic | 12,393 | 516 | 1,697 | 10,180 | 74.5% |

Denominators are effectively unaffected: zero suppressed cells at one and two
group-by variables, and even the four-way cross-tab reports 99.985% of all births
exactly.

### Depth limits: where a multi-predictor cell model fails

A second round of queries measured what happens as predictors are added. Age is
`D149.V1` (Age of Mother 9), education `D149.V5`.

| Design | By-vars | Enumerated | Observed | Suppressed | Zero | Captured |
| --- | --- | --- | --- | --- | --- | --- |
| State x age band | 2 | 459 | 252 | 100 | 107 | 97.3% |
| State x education | 2 | 561 | 314 | 122 | 125 | 97.0% |
| State x age band x education | 3 | 5,049 | 593 | 1,479 | 2,977 | 71.3% |
| State x year x age band | 3 | 4,131 | 606 | 1,893 | 1,632 | 61.0% |
| State x age band x race6 x Hispanic | 4 | 12,393 | 404 | 1,311 | 10,678 | 80.6% |
| State x age band x race6 x Hispanic x education | 5 | **truncated** | 417 | not listed | not listed | 40.6% |
| State x year x age band x race6 x Hispanic | 5 | **truncated** | 321 | not listed | not listed | 26.3% |

At five group-by variables WONDER stops enumerating the table. It returns only
the exactly-observed cells with the message *"The full results are too long to be
displayed. Due to suppression constraints rows that are zero, suppressed or a
total will not be available"*, and silently sets Show Zero Values to Disabled.

This is materially worse than censoring. Interval censoring requires knowing
**which** cells are censored; at this depth WONDER will not tell you, so the
missing 59-74% of recorded mass sits in a set of cells that cannot be
enumerated from the numerator query at all. The cliff is driven by result length
rather than by the variable count as such — the 12,393-row four-way tables
enumerate fully, the ~100,000-row five-way tables do not.

Three separate limits therefore bind before censoring becomes the problem:

1. **Maternal age in single years does not exist as a group-by** (removed May
   2022). Only 5-year bands. This is unavailability, not censoring, and it alone
   prevents reproducing `DSP004`'s exact-age cell structure at any geography.
2. **Five group-by variables maximum.** State + year + age band + race +
   Hispanic origin consumes all five and leaves no slot for SES or anything
   else. Reconstructing `mracehisp_c` costs two slots by itself.
3. **Enumeration truncation** at large result sizes, which destroys the
   censoring pattern rather than merely censoring values.

Note also that WONDER's SES content is proxy only — mother's education
(`D149.V5`), source of payment for delivery (`D149.V109` / `V110`), WIC
(`D149.V66`) and marital status (`D149.V27`). There is no income or occupation
item.

**Consequence.** A state-resolved *joint cell array* over maternal age,
ethnicity, SES and further predictors is not obtainable from WONDER. What is
obtainable is a set of **shallow overlapping margins** — state x year, state x
age band, state x race x Hispanic, state x education — each 95-100% exact at two
by-variables and tractably censored at three. Main effects and selected two-way
interactions are identifiable from a margin set; the full joint array is not. So
WONDER supports state-level recording-rate estimation, and cannot support a
state-resolved analogue of the predictors classifier (aim 2) or a state-resolved
`DSP004` cell structure. For those, restricted-use microdata (option D) is the
only route.

**Vermont is suppressed even pooled over nine years** — at most 9 recorded flags
against 48,125 births, below 1.9 per 10,000. Under the confirmed-only definition
Hawaii and Wyoming are suppressed as well.

### Confirmed / Pending split

National confirmed-only is **7,974** against 17,783 Confirmed + Pending — so
**Pending is 55% of all recorded flags**, the majority. This bears directly on
the [false-positive channel note](20260803-false-positive-channel-identification.md),
which uses the confirmed/pending split as an identifying signal, and it means the
`--confirmed-only` sensitivity is a much larger perturbation than a
robustness-check framing would suggest.

### The substantive finding: recorded prevalence varies about ninefold by state

Among the 50 jurisdictions reported exactly, pooled recorded prevalence spans
**1.44 to 13.43 per 10,000 — a 9.3-fold range**:

| Lowest | per 10,000 | | Highest | per 10,000 |
| --- | --- | --- | --- | --- |
| Hawaii | 1.44 | | Utah | 13.43 |
| Mississippi | 2.14 | | South Dakota | 13.28 |
| Florida | 2.57 | | Alaska | 10.37 |
| Tennessee | 3.55 | | Idaho | 10.24 |
| Texas | 3.60 | | Nebraska | 9.74 |
| California | 3.64 | | Iowa | 9.61 |

True livebirth prevalence cannot plausibly vary this much between states. Maternal
age structure and termination rates differ, but not by ninefold. Almost all of
this spread is **recording completeness**, which means state resolution supplies
a large and previously unused source of variation for identifying the recording
process — the quantity the whole selection model is organised around. It also
means a single national `s` is averaging over jurisdictions that differ by close
to an order of magnitude.

Two cautions. Some of the highest values (Utah, South Dakota, Idaho, Alaska,
Nebraska) exceed most surveillance estimates of *true* livebirth prevalence,
which cannot be explained by recording completeness alone and points at either
false-positive recording or a genuine age/termination gradient; this interacts
with the fixed `f = 7.8e-5` assumption. And the censoring is informative in the
adverse direction — the worst-recording jurisdictions are exactly those most
likely to be suppressed, so dropping suppressed cells would bias estimated
recording upward.

## How to use these data in the models

The age structure `DSP004` needs sits on the **expected** side, and denominators
are essentially never suppression-limited. WONDER is needed only for the
numerator. So:

1. **Aggregated likelihood.** Retain full age x race resolution on the expected
   side and observe recorded Down syndrome at whatever aggregation WONDER
   reports, with the model summing expected counts over the constituent cells.
2. **Interval censoring.** A suppressed cell contributes `P(1 <= R <= 9)` —
   directly expressible in PyMC. Combined with the separately-queried pooled
   margins, an exact nine-year state total plus the observed year-cells
   constrains the censored remainder tightly. At state x year this recovers the
   remaining 2.7% of mass across 86 cells rather than discarding it.

The residual approximation is band-averaging the Morris curve over 5-year
maternal age bands, weighted by the national within-band single-year
distribution from our own microdata. That is the quantity the `DSP001` /
`DSP004` comparison in [`docs/models/README.md`](../docs/models/README.md)
already measures, so its size is known rather than assumed.

## Extracts written

All are aggregate, non-record-level, and contain no count between 1 and 9 —
WONDER never returns one. Columns: key fields, `births` / `births_status`
(denominator, all Down syndrome values), `ds_cp` / `ds_cp_status` (Confirmed +
Pending). Status is `observed`, `suppressed` or `zero`.

| File | Rows |
| --- | --- |
| `data/us-births-wonder-national-year-2016-2024.csv` | 9 |
| `data/us-births-wonder-state-pooled-2016-2024.csv` (adds `ds_confirmed`) | 51 |
| `data/us-births-wonder-state-year-2016-2024.csv` | 459 |
| `data/us-births-wonder-state-race-pooled-2016-2024.csv` | 1,377 |
| `data/us-births-wonder-state-year-race6-2016-2024.csv` | 2,753 |

Rows whose denominator and numerator are both zero are dropped from the two
largest tables as structurally empty. The `State x year x race6 x Hispanic`
four-way table was pulled and verified but **not committed** — at 766 KiB it is
twice the size of everything else in `data/` combined, and it is the least
informative table (74.5% captured, 1,697 censored cells, 5,446 zeros). Regenerate
it from the query specification below if a full state x year x `mracehisp_c`
model is ever wanted.

### Query specification

Database `D149`. Group-by codes: `D149.V21-level1` state, `D149.V20` year,
`D149.V42` single race 6, `D149.V43` Hispanic origin. Numerator sets
`V_D149.V132` to `1` and `2` (or `1` alone for confirmed-only); denominator
leaves it `*All*`. All queries set `V_D149.V20=*All*`, `O_show_zeros=true`,
`O_show_suppressed=true`, and export format `tsv`.

## Coverage and alternative routes

| Years | State-resolved Down syndrome from | Notes |
| --- | --- | --- |
| 1989-2004 | Public-use microdata directly | State of residence present; the 1989-certificate `DOWNS` checkbox is a different and more poorly recorded item, and the 2003-revision transition years are mixed-layout |
| 2005-2015 | Nothing public | Geographic detail dropped from public files [from the 2005 data year](https://www.psc.isr.umich.edu/dis/data/kb/answer/1047.html); WONDER holds no anomaly data for these years |
| 2016-2024 | CDC WONDER `D149` | Extracted, see above |

A [restricted-use vital statistics data request](https://www.cdc.gov/nchs/nvss/nvss-restricted-data.htm)
under a signed NCHS data-use agreement supplies geographic detail in the
microdata from 2005 onward. That is the only route to exact-age state cells and
the only way to close 2005-2015.

## Open question: automation

The WONDER API **cannot return sub-national data** — location fields can be
neither grouped nor limited through it, enforced server-side. Since CDC provides
an API for automated access and deliberately excludes geography from it,
scripting the HTML form to obtain state data works around a deliberate design
decision, even though nothing in the data-use restrictions addresses automation.

The queries behind this note were driven programmatically over a browser session
whose agreement had been accepted interactively — appropriate for a one-off pull
of a fixed extract, and the session identifier expires. **A committed, reusable
scraper is a different proposition and has not been written.** If the extracts
need periodic refresh, the choices are to re-run the pull by hand, or to raise
scripted access with CDC WONDER support (`cwus@cdc.gov`) rather than assume it.
This is a decision for the project lead, not a technical gap.

## Options

**A. Pooled state extract.** Already in hand. 50 of 51 jurisdictions exact,
Vermont censored. Supports state-level recording-completeness comparison against
external surveillance with maternal age structure carried entirely by the
denominator. No new model structure needed.

**B. State x year with interval censoring.** Already in hand. 97.3% of mass
exact, 86 censored cells anchored by the option-A margins. Recovers temporal
variation in state recording. Requires a censored observation term only.

**C. State x race.** Already in hand in two forms — pooled with the Hispanic
split (95.0% captured), and by year without it (83.5%). Connects to the open
questions in the [race-surveillance audit](20260803-dsp004-race-surveillance-audit.md);
that audit's fail-closed conclusion is unaffected by this note.

**D. Restricted-use file request.** Bypasses WONDER from 2005 onward, gives
exact-age state cells, closes 2005-2015. Long lead time and an institutional
agreement, but the only route to a state-resolved model at `DSP004`'s actual cell
structure.

## Recommendation

The ninefold state spread in recorded prevalence is a stronger result than the
feasibility question that prompted this work, and it is the reason to proceed:
it is direct evidence that recording completeness is the dominant source of
variation in the certificate data, and it is measured rather than assumed.

Take **B** — it is already extracted and needs only a censored observation term
on top of the existing expected-count machinery. Treat any state-resolved fit as
conditional on the same calibration scenarios that gate the national totals;
state resolution adds structure to the recording process, not new evidence about
the national true-birth scale. Pursue **D** in parallel if state resolution is
intended to become a reported component rather than a diagnostic.

Before any of that, two things in this note deserve checking on their own terms:
the 55% Pending share against the assumptions in the false-positive work, and the
handful of states whose recorded prevalence exceeds plausible true prevalence.

## Verification status

Measured from returned result sets: all cell counts, suppression counts, captured
shares, the national and per-year totals, the confirmed/pending split, the
per-state prevalences, and the automatic disabling of totals under possible
suppression.

Read from documentation, not independently tested: the 27 August 2015 withdrawal
of anomalies from the other databases; the pre-May-2022 and pre-May-2023
suppression history; the availability of state in the 1989-2004 public-use
microdata.

Not attempted: any county-level query, and any query using payment source, WIC or
marital status as a group-by (their availability is read from the group-by list,
their suppression behaviour is untested).
