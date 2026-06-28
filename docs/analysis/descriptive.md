> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 4.8).

# Descriptive review of recorded Down syndrome live births, 1989–2024

> [!WARNING]
> **Preliminary template.** This is the scaffold for the descriptive report, not the
> rendered report. Section prose is settled; the figures, tables, and headline numbers are
> placeholders to be produced by a companion script (see [§10](#10-how-this-report-is-generated)).
> All data and estimates in this study are provisional.

This report documents **study aim #1** — the numbers and characteristics of babies recorded
with Down syndrome (DS) in US NCHS/NVSS natality data from 1989 to 2024 — and the
co-occurring-conditions portion of **aim #4**. It describes the *recorded* population only.
The predicted-missing extension (the ~60% of cases absent from birth certificates) and its
demographic caveats live in [`predicted.qmd`](predicted.qmd) (with the model-variant comparison in
[`compare_predicted.qmd`](compare_predicted.qmd)); the formal estimation of missed
cases lives in the selection model ([`../models/selection/index.qmd`](../models/selection/index.qmd)).

---

## 1. Purpose and scope

- **Population.** Every US live birth 1989–2024 in the combined natality file (~142.9 M records;
  `data/us_births.db`, table `us_births`). A **recorded DS birth** is `down_ind = 1`, i.e.
  `ca_down_c ∈ {C confirmed, P pending}`.
- **Questions.**
  1. How many DS live births were recorded each year, and how does that compare with the number
     expected from surveillance? (the **recording rate / ascertainment**, headline ≈ 40 %.)
  2. What are the maternal, pregnancy, and infant characteristics of recorded DS births, and how
     do they shift over the period?
  3. Among recorded DS cases, how often are co-occurring conditions noted on the certificate, and
     how does that compare with published clinical estimates? (aim #4.)
- **Out of scope here** (handled elsewhere, cross-reference rather than duplicate): the
  predicted-missing population and the recorded-vs-predicted comparison ([`predicted.qmd`](predicted.qmd),
  aims #3/#4, 2016–2024 only); the Bayesian selection model ([`../models/selection/`](../models/selection/index.qmd)).

---

## 2. Data, definitions, and method

### 2.1 The Down syndrome item and the recorded population

`ca_down_c` harmonises the certificate DS item across eras to a single C/P/N/U domain:

| `ca_down_c` | Meaning | In `down_ind`? |
| --- | --- | --- |
| `C` | Confirmed (1989-cert "anomaly reported"; 2003-cert "confirmed") | 1 (recorded) |
| `P` | Pending (2003-cert only, **from 2004**) | 1 (recorded) |
| `N` | No / not reported | 0 |
| `U` | Unknown / not classifiable | 0 (treated as not recorded) |
| `NULL` | Item not on the certificate (non-reporting area) | 0 (treated as not recorded) |

### 2.2 The three eras (the dominant structural fact)

Almost every variable's availability and coding is bounded by the certificate revision. Treat
these three eras consistently throughout and annotate the transition band on every time series:

| Era | Years | Notes |
| --- | --- | --- |
| 1989 certificate | 1989–2002 | DS item is `downs`/`uca_downs`; **no "pending"**; many 2003-cert items absent. |
| 2003-cert transition | 2003–2013 | States adopt the revised certificate gradually → 2003-cert items are **sparse / partial**; do not read partial-rollout sparsity as a real trend. |
| Full 2003 certificate | 2014–2024 | 2003-cert items effectively complete; multi-race identifiable. |

### 2.3 Denominators and surveillance benchmarks

- **Denominator** for rates: all live births in the year (`COUNT(*)`), always complete.
- **Expected DS live births** (for the recording rate) come from external estimates already
  materialised per record. Use these:

| Series (column) | Coverage | Meaning | Caveat to surface |
| --- | --- | --- | --- |
| `p_ds_lb_wt` | 1989–2024 | Per-year surveillance prevalence (de Graaf-based). Expected count = `SUM(p_ds_lb_wt)`. | **2018 value carried forward through 2024**; precision break at 2014/2015 (spliced vintages). |
| `p_ds_lb_nt` | 1989–2024 | Morris maternal-age risk, **no terminations** (upper bound). Expected = `SUM(p_ds_lb_nt)`. | Upper bound; gap to `p_ds_lb_wt` ≈ terminations. |
| `p_ds_lb_nt_reduc` | 1989–2024 | `p_ds_lb_nt × (1 − reduction[year])` — age-risk **after** terminations; the like-for-like expected-live-births series. | Reduction rate **extrapolated 2020–2024**. |
| `p_ds_lb_wt_mage` | 1989–**2018** | Expected prevalence split at maternal age 35 → recording rate **by age band**. | Stops at 2018 (NULL after). |
| ethnicity prevalence CSV | 2000–**2018** | de Graaf expected DS per 10 k by `mracehisp_c` 1–5 → recording rate **by group**. | Stops at 2018; **2015 and 2017 blank**; **no multi-race (code 6)**; loaded as table `us_births_est_prevalence_ethnicity`, **not joined** in the pipeline — compute the comparison in the report. |

> [!IMPORTANT]
> **Do not** use `ds_case_weight` as an "expected" series in any recording-rate figure — it is a
> modelling weight *derived from* recording rates, so dividing recorded by it is circular. The
> placeholder columns `p_ds_lb_nt_mage`, `p_ds_lb_wt_ethn`, `p_ds_lb_nt_ethn` are declared but
> **NULL** — do not use them.

### 2.4 Reproducibility

All numbers come from read-only DuckDB queries against `data/us_births.db` (59 residual records
carry no `year` and are dropped from every time series via `year IS NOT NULL`). Each section below
carries the query that produces its figure/table. The **target state** ([§10](#10-how-this-report-is-generated))
is a `docs/analysis/descriptive.qmd` template plus a companion `scripts/analyse_descriptive.py`,
mirroring `predicted.qmd` / `scripts/analyse_predicted.py`.

---

## 3. How to read this report (cross-cutting caveats)

1. **Recorded ≈ 40 % of expected.** Birth certificates under-report DS by roughly 60 %; every
   recorded count and rate here is an *ascertainment-depressed* observation, not true prevalence.
2. **Confirmed-only is the era-robust series.** "Pending" (`P`) exists only from 2004, so a
   confirmed-vs-pending split is structurally all-confirmed before 2004; a confirmed + pending
   total has a step at 2004. Show `P` separately and keep a confirmed-only line for long trends.
3. **Unknown / NULL is a data-quality signal, not epidemiology.** The `U`/NULL share spikes around
   the 2003 cert changeover. Rising unknowns change the recordable denominator; never read them as
   incidence change.
4. **Era-bounded availability.** Flag the 2003–2013 transition band on every series; give every
   2003-cert dimension an explicit coverage warning.
5. **Multi-race only from 2014.** `mracehisp_c = 6` (NH multi) is identifiable only 2014+. Use the
   5-category collapse for 1989–2024 trends and a 6-category panel for 2014+; never silently drop
   code 6.
6. **Some characteristics cannot be pooled across eras.** Maternal education is years-of-schooling
   pre-2003 (`dmeduc`) vs attainment level 2003+ (`meduc`) — two-panel, no single trend. Marital
   status lives in two columns (`dmar` 1989–2002 & 2014+, `mar` 2003–2013) and must be coalesced.

---

## 4. Section A — Recorded numbers and trends, 1989–2024  *(aim #1)*

### A1. Master table: recorded DS by year × confirmation status

*Table `recorded_by_status.csv`.* Per-year counts of `C`, `P`, `N`, `U`, NULL, total births, and
recorded DS rate per 10 000.

```sql
SELECT year,
       COUNT(*) FILTER (WHERE ca_down_c = 'C')      AS confirmed,
       COUNT(*) FILTER (WHERE ca_down_c = 'P')      AS pending,
       COUNT(*) FILTER (WHERE ca_down_c = 'N')      AS no,
       COUNT(*) FILTER (WHERE ca_down_c = 'U')      AS unknown,
       COUNT(*) FILTER (WHERE ca_down_c IS NULL)    AS not_on_cert,
       COUNT(*)                                     AS births,
       SUM(down_ind)                                AS recorded_ds,
       1e4 * SUM(down_ind) / COUNT(*)               AS recorded_per_10k
FROM us_births WHERE year IS NOT NULL
GROUP BY year ORDER BY year;
```

### A2. Recorded DS births by year, confirmed vs pending  *(headline figure)*

*Figure `recorded_confirmed_pending.png`.* Stacked bar (confirmed bottom, pending top), 1989–2024,
with the 2003–2013 transition band shaded; overlay a confirmed-only line as the era-robust trend.
Recorded DS runs roughly flat-to-slightly-rising (~1.6 k–2.2 k/yr) against a *falling* total-birth
denominator — so the rate per 10 000 rises faster than the count.

### A3. Total births and recorded rate per 10 000

*Figure `recorded_rate_per_10k.png`.* Line of `recorded_per_10k` by year, annotated that the level
reflects **recording completeness, not prevalence** (the surveillance benchmark `p_ds_lb_wt` is
≈ 13–14 per 10 000, against a recorded rate of ≈ 5/10 000).

### A4. The "unknown / not-on-certificate" trend  *(data quality)*

*Figure `ds_status_unknown.png`.* Line of `unknown` (and NULL) by year, framed as a cert-changeover
artefact concentrated around 2003. State explicitly that the project treats `U` and NULL as
"not recorded".

---

## 5. Section B — Recorded vs surveillance-expected / recording rate  *(aim #1 headline)*

### B1. Recorded vs expected counts by year

*Figure `recorded_vs_expected.png`.* Per year: recorded bars vs expected lines from
`SUM(p_ds_lb_wt)` (surveillance) and `SUM(p_ds_lb_nt_reduc)` (age-risk after terminations); the
visible gap is the under-reporting.

```sql
SELECT year,
       SUM(down_ind)           AS recorded,
       SUM(p_ds_lb_nt)         AS expected_no_term,
       SUM(p_ds_lb_nt_reduc)   AS expected_after_term,
       SUM(p_ds_lb_wt)         AS expected_surveillance
FROM us_births WHERE year IS NOT NULL
GROUP BY year ORDER BY year;
```

### B2. Recording rate over time  *(the ≈ 40 % headline)*

*Figure `recording_rate_by_year.png`.* `recorded / expected` by year against both denominators,
with a ~36–43 % reference band. Compute the ratio per year rather than asserting a flat 40 %.

### B3. Recording rate by maternal-age band  *(≤ 2018)*

*Figure `recording_rate_by_age.png`.* Recording rate for `mage_c < 35` vs `≥ 35`, using
`p_ds_lb_wt_mage`. **Truncate at 2018** (the age-band CSV stops there).

### B4. Recording rate by race/Hispanic origin  *(de Graaf, 2000–2018)*

*Figure `recording_rate_by_race.png`.* Recording rate per `mracehisp_c` group: recorded ÷ expected,
where the group expected is either the
age-structure-adjusted `SUM(p_ds_lb_nt_reduc)` within group (in-DB, full span) or the de Graaf
ethnicity prevalence (external CSV, **2000–2018, no 2015/2017, codes 1–5 only**). Tests whether
under-reporting is uniform or **differential** across groups. Footnote multi-race (code 6) as
having no de Graaf anchor.

---

## 6. Section C — Maternal characteristics of recorded cases

For each cut, show the distribution **among recorded DS cases** and, as a reference, **among all
births** (recorded-case counts are confounded by the shifting characteristics of the whole cohort).

### C1. Maternal age  *(1989–2024)*

*Figures `recorded_ds_by_mage.png`, `mean_mage_by_year.png`.* Single-year `mage_c` distribution of
recorded DS mothers; mean maternal age of recorded DS vs all births by year (the age gap and its
drift). Also `AVG(down_ind)` by `mage_c` — recorded DS *prevalence* rising steeply with age — and the
recorded-vs-expected-count overlay by age (the gap widens at older ages). Maternal age is the
dominant DS risk axis: present age-standardised counts alongside crude.

### C2. Race / Hispanic origin  *(`mracehisp_c`)*

*Figure `recorded_ds_by_race.png`.* Stacked-share area over time: 5-category (1 NH White, 2 NH Black,
3 NH AIAN, 4 NH Asian/PI, 5 Hispanic) for 1989–2024; a 6-category panel adding NH multi-race for
2014+. Pair with the by-group recording rate from [B4](#b4-recording-rate-by-racehispanic-origin-de-graaf-20002018).

### C3. Maternal education  *(two panels, non-poolable)*

*Figure `recorded_ds_by_education.png`.* **Panel 1** 1989–2002 years-of-schooling (`meduc6`, the
banded recode, or `dmeduc`, raw years — two encodings of the same pre-2003 scheme); **Panel 2**
2014–2024 attainment level (`meduc`). The 2003–2013 transition is shown as a coverage
caveat, not a trend (each record carries only one of the two schemes).

### C4. Marital status  *(coalesced, 1989–2024)*

*Figure `recorded_ds_by_marital.png`.* Married vs unmarried share by year, coalescing
`dmar` (1989–2002, 2014+) and `mar` (2003–2013); fold Puerto Rico code 3 into unmarried; exclude
`9` unknown from the base.

### C5. Maternal nativity  *(2014+)*

*Figure `recorded_ds_by_nativity.png`.* US-born vs foreign-born (`mbstate_rec`), **2014–2024 only**
(sparse in the transition era; the pre-2003 analogue `mplbir` is a different coding).

### C6. Payment source and WIC  *(2009+; near-complete 2014+)*

*Figure `recorded_ds_by_payer.png`.* Payer mix (`pay_rec`: Medicaid / Private / Self-pay / Other,
drop `9`) and WIC-receipt share. Both items first appear ~2009 but are only near-complete from 2014,
so report the **2014–2024** window for comparability and flag 2009–2013 as partial. Socioeconomic
context / recording-completeness covariates.

---

## 7. Section D — Pregnancy and infant characteristics

### D1. Plurality and infant sex

*Table `recorded_ds_plurality.csv`, figure `recorded_ds_sex.png`.* Singleton vs multiple
(`dplural`, 1989–2024; small cells — a table, not a trend line). Infant sex (`sex`, **2003+ only** —
`CSEX` is not imported pre-2003); expect a slight male excess.

### D2. Birthweight and gestational age  *(2003+)*

*Figures `recorded_ds_birthweight.png`, `recorded_ds_gestation.png`.* Birthweight distribution and
% low-birthweight (< 2500 g) of recorded DS vs all (`dbwt`, drop 9999); preterm share from
`gestrec10` (drop 99). **2003–2024 only** (`DBIRWT`/`GESTAT10` pre-2003 use different names/boundaries
and are not loaded). DS skews to lower birthweight / earlier gestation.

### D3. Delivery, attendant, prenatal care

*Figure `recorded_ds_delivery.png`.* Cesarean rate (`dmeth_rec`, **2006+** collapsed form — note its
role changes pre-2006); attendant-type mix (`attend`, one of the few **1989–2024** care dimensions);
prenatal-care initiation (`precare`, 2014+).

---

## 8. Section E — Co-occurring conditions and newborn morbidity  *(aim #4, 2014+)*

> [!NOTE]
> New analysis — not prototyped in the notebooks. Compute rates **among recorded DS cases**
> (`down_ind = 1`) over the **Y/N base only** (exclude `U`/blank), **2014–2024** for comparability.
> Certificate checkbox reporting under-ascertains, so every rate here is a **lower bound** on
> clinical co-occurrence.

### E1. Co-occurring congenital anomalies

*Figure `recorded_ds_cooccurring.png`, table `recorded_ds_cooccurring.csv`.* Percentage of recorded
DS cases flagged `Y` for each item: suspected chromosomal disorder (`ca_disor`), cyanotic congenital
heart disease (`ca_cchd`), and the other structural-anomaly checkboxes (`ca_cdh`, `ca_omph`,
`ca_gast`, `ca_limb`, `ca_cleft`, `ca_clpal`, `ca_anen`, `ca_mnsb`, `ca_hypo`).

```sql
SELECT
  100.0 * COUNT(*) FILTER (WHERE ca_cchd = 'Y')
        / NULLIF(COUNT(*) FILTER (WHERE ca_cchd IN ('Y','N')), 0) AS pct_cchd,
  100.0 * COUNT(*) FILTER (WHERE ca_disor = 'C')
        / NULLIF(COUNT(*) FILTER (WHERE ca_disor IN ('C','P','N')), 0) AS pct_chromosomal_disorder
  -- ... repeat per ca_* item
FROM us_births WHERE down_ind = 1 AND year >= 2014;
```

> [!IMPORTANT]
> **Aim #4 comparison.** Set each rate against published population-level DS co-occurrence estimates
> (e.g. congenital heart defects in DS ≈ 40–50 %). Birth-certificate `ca_cchd` (a single cyanotic-CHD
> checkbox) will read far lower — the point of the comparison is the *direction and magnitude of
> under-ascertainment*, not exact agreement.

### E2. Newborn morbidity / intervention

*Figure `recorded_ds_morbidity.png`.* Share of recorded DS newborns with low 5-minute APGAR (< 7,
`apgar5`/`apgar5r`, available 2003+), NICU admission (`ab_nicu`), and assisted ventilation
(`ab_aven1`/`ab_aven6`), **2014–2024**. Note these partly reflect that sicker DS newborns (e.g. with
CHD) are the ones flagged.

---

## 9. Limitations — what the raw data cannot show

- **No geography.** The public-use file carries no state, county, or sub-national identifier
  (`restatus` is only resident/non-resident status) — no maps, no state-level recording-rate
  comparisons, no urban/rural split.
- **Live births only.** The file cannot observe pregnancies ending in termination after a prenatal
  DS diagnosis, so it cannot estimate true antenatal prevalence; the surveillance benchmarks inject
  that externally.
- **No record linkage.** No person identifiers — a birth cannot be linked to infant death, to the
  mother across years, or to any later outcome.
- **No DS subtype or severity.** `ca_down_c` is a single checkbox — no trisomy-21 vs translocation
  vs mosaic, no phenotype, no ascertainment pathway (prenatal vs postnatal), and "pending" is never
  resolved within the file.
- **Limited pre-2003 clinical detail.** Infant sex, birthweight, gestation boundaries, APGAR, and
  *all* 2003-cert checkboxes (NICU, ventilation, co-occurring anomalies beyond the DS box, risk
  factors, payment, WIC, BMI) are absent or non-comparable for 1989–2002.
- **Under-reporting and selection.** Recorded cases are ~40 % of expected and are likely
  *severity-selected* (sicker babies more often flagged) — characteristic distributions of recorded
  cases need not match all DS births.

---

## 10. How this report is generated

**Current state.** This `.md` is the scoping/template document.

**Target state** (recommended, mirroring the sibling analysis docs): convert to
`docs/analysis/descriptive.qmd` rendered by a companion `scripts/analyse_descriptive.py` that writes
`config.json` + per-figure CSVs/PNGs into an output directory and renders the template there —
the same `script → CSV → .qmd` pattern as `predicted.qmd` / `scripts/analyse_predicted.py`. The
analysis logic to port is `notebooks/0001-characteristics-recorded-births.py` (recorded-by-year,
confirmed/pending bar, recording-rate line, unknown trend), extended with Sections C–E above. On
conversion, the YAML front-matter, the `code-fold` / `# | echo: false` chunk idiom, the
`_render_summary_table` helper, `{#fig-… .lightbox}` figures, and the Quarto callout form of the
AI-assisted disclosure all apply (see the conventions in `predicted.qmd`).

**Cross-references.** This descriptive baseline (recorded cases, aim #1) hands off to
[`predicted.qmd`](predicted.qmd) for the predicted-missing extension (aims #3/#4) and to
[`../models/selection/index.qmd`](../models/selection/index.qmd) for the formal missed-case
estimation. The recording-rate evidence in [§5](#5-section-b--recorded-vs-surveillance-expected--recording-rate-aim-1-headline)
is the empirical basis for the ~60 % under-reporting figure those documents rely on.

### Figure / table inventory

| ID | Section | Artefact |
| --- | --- | --- |
| `recorded_by_status.csv` | A1 | Per-year status counts + rate |
| `recorded_confirmed_pending.png` | A2 | Confirmed/pending stacked bar |
| `recorded_rate_per_10k.png` | A3 | Recorded DS per 10 000 |
| `ds_status_unknown.png` | A4 | Unknown/NULL trend |
| `recorded_vs_expected.png` | B1 | Recorded vs expected counts |
| `recording_rate_by_year.png` | B2 | Recording rate ≈ 40 % |
| `recording_rate_by_age.png` | B3 | By age band (≤ 2018) |
| `recording_rate_by_race.png` | B4 | By race/Hispanic (2000–2018) |
| `recorded_ds_by_mage.png`, `mean_mage_by_year.png` | C1 | Maternal age |
| `recorded_ds_by_race.png` | C2 | Race/Hispanic shares |
| `recorded_ds_by_education.png` | C3 | Education (two panels) |
| `recorded_ds_by_marital.png` | C4 | Marital status |
| `recorded_ds_by_nativity.png` | C5 | Nativity (2014+) |
| `recorded_ds_by_payer.png` | C6 | Payer / WIC (2014+) |
| `recorded_ds_plurality.csv`, `recorded_ds_sex.png` | D1 | Plurality, sex |
| `recorded_ds_birthweight.png`, `recorded_ds_gestation.png` | D2 | Birthweight, gestation |
| `recorded_ds_delivery.png` | D3 | Delivery / attendant / prenatal |
| `recorded_ds_cooccurring.{png,csv}` | E1 | Co-occurring anomalies |
| `recorded_ds_morbidity.png` | E2 | Newborn morbidity |
