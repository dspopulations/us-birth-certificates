# C-only vs C+P: predicted-missing cohort comparison

**Date:** 2026-04-22
**Reports:** `output/compare_predicted/{c_effect_full, c_effect_clinical, nonclinical_effect}/index.html`
**Commit:** `e7d1bb1` (C-only model variants)

## Setup

Four reporting-profile LightGBM models were fit on 2016–2024 NVSS natality:

| Model | Features | Label | n_leaves (tuned) | best_iter | AP (valid) | ROC-AUC |
|---|---|---|---|---|---|---|
| `usbc10_m1`   | full, post-prune (28) | C+P → 1 | 180 | 158 | 0.0324 | 0.889 |
| `usbc10_m1_cn`| full, re-pruned under C-only (26) | C → 1, P dropped | 116 | 282 | 0.0245 | **0.908** |
| `usbc11_m0`   | clinical + mage (24) | C+P → 1 | 180 | 570 | 0.0310 | 0.883 |
| `usbc11_m1_cn`| clinical + mage, re-pruned under C-only (18) | C → 1, P dropped | 41  | 1157 | 0.0262 | **0.902** |

AP drops under C-only as expected (positive class shrinks ~46%; baseline AP
scales with prevalence). ROC-AUC **improves** by 1.7–1.9 pp under C-only,
consistent with a cleaner label producing a more separable ranking problem.

Predictions and a year×month `ceil(1.5 × recorded)` missing-flag quota were
written back to DuckDB for each variant:

| Variant | `predictions` column | `missing` flag column | `missing` TRUE |
|---|---|---|---|
| usbc10_m1    | `p_ds_lb_pred_01` | `ds_pred_missing`    | 26,742 |
| usbc10_m1_cn | `p_ds_lb_pred_13` | `ds_pred_missing_13` | 12,002 |
| usbc11_m0    | `p_ds_lb_pred_02` | `ds_pred_missing_02` | 26,742 |
| usbc11_m1_cn | `p_ds_lb_pred_14` | `ds_pred_missing_14` | 12,002 |

The C-only cohorts are roughly half the size of C+P. This is a mechanical
consequence of the quota: `WHERE down_ind = 1 AND predictions IS NOT NULL`
counts only rows that were in the training set, so for the C-only variants
P rows contribute neither to the quota base nor to the candidate pool. The
flag densities per year×month are therefore comparable across variants, but
totals differ.

Recorded-cohort row counts at 2016–2024: 17,809 births coded C+P; 7,984
coded C only.

## Headline findings

### 1. Label change (C/P → C-only) collapses a label-feature tautology

The largest shift across any comparison is in `ca_disor` (suspected
chromosomal disorder):

| Comparison | `ca_disor = Pending` in predicted-missing | Δ vs left |
|---|---|---|
| `usbc10_m1` (C+P) | 29.4% | — |
| `usbc10_m1_cn` (C-only) | 2.3% | **−27.2 pp** |
| `usbc11_m0` (C+P, clinical) | 25.6% | — |
| `usbc11_m1_cn` (C-only, clinical) | 2.7% | **−22.9 pp** |

Baseline in the recorded cohort is 4.5%. Both C+P models were flagging
`ca_disor = Pending` cases at 6× their recorded rate. This is a label-
feature tautology: `ca_disor` takes values {Confirmed, Pending, No,
Unknown}, and "Pending chromosomal disorder" is by construction almost
co-extensive with "Pending Down syndrome" at certificate filing time.
Under the C+P label, the model learns "`ca_disor = P` → positive" as a
near-deterministic rule, and then flags any unrecorded row with
`ca_disor = P` as a missed case.

Under the C-only label this pseudo-signal collapses: pending chromosomal
disorder is not the same as *confirmed* Down syndrome, so `ca_disor = P`
becomes a weak positive indicator like any other. The C-only predicted-
missing pool reverts close to the recorded base rate. This is the
cleanest evidence from the comparison that the C+P label was partly
riding a recording-process artefact rather than DS biology.

### 2. Predicted-missing shifts older and more clinically severe under C-only

Both label-effect comparisons show the same direction:

| Variable | Δ (C-only − C+P), usbc10 | Δ, usbc11 | Recorded % |
|---|---|---|---|
| mage 35–39 years     | +4.9 pp | +3.4 pp | 30.0% |
| mage 40–44 years     | +0.8 pp | +0.9 pp | 20.9% |
| mage 45–49 years     | +0.1 pp | +1.9 pp | 2.2%  |
| mage 20–29 years     | **−6.1 pp** | **−6.3 pp** | 23.9% |
| `ab_nicu = Yes`      | +8.2 pp | +5.3 pp | 58.3% |
| `ca_cchd = Yes`      | +18.1 pp | +17.5 pp | 5.6% |
| `ab_aven1 = Yes`     | +7.3 pp | +3.8 pp | 28.9% |
| `ab_aven6 = Yes`     | +5.0 pp | +6.7 pp | 15.5% |
| `dbwt < 2500 g`      | +7.5 pp | +4.5 pp | 25.4% |
| `gestrec10 28–33 wk` | +2.0 pp | +1.5 pp | 6.9%  |
| `ca_disor = Confirmed` | +16.8 pp | +11.1 pp | 1.8% |

The C-only model is *less* distributed across mid-age mothers (20-29) and
*more* concentrated in advanced maternal age plus clinically severe
neonatal outcomes (NICU, ventilation, low birth weight, preterm,
confirmed chromosomal disorder, cyanotic CHD). This is consistent with
the hypothesis that pending codes are over-generated in younger / less-
severe presentations where post-discharge cytogenetic confirmation is
less likely to complete before the certificate is filed — and that
"pending → DS" is a noisier label in those subpopulations.

### 3. SES shifts: C-only flags fewer Medicaid, more private-insurance mothers

| Variable | Δ (C-only − C+P), usbc10 | Δ, usbc11 | Recorded % |
|---|---|---|---|
| `pay_rec = Medicaid` | −3.3 pp | −2.1 pp | 41.8% |
| `pay_rec = Private`  | +4.5 pp | +2.6 pp | 47.8% |
| `meduc ≤ HS`         | −4.6 pp | −2.6 pp | 39.0% |
| `meduc ≥ Bachelor`   | **+4.1 pp** | +2.4 pp | 32.4% |

C-only's predicted-missing pool is more privately-insured and more
college-educated than C+P's. Mechanistically this can be read either way:

- Under C+P, *pending* cases concentrate in Medicaid / lower-education
  populations (perhaps because follow-up paperwork is less complete), so
  the C+P model learns "this demographic + birth certificate features →
  likely positive" via that pending pathway. Removing `P` positives
  removes that gradient from the training signal.
- Alternatively, advanced maternal age and private-insurance coverage
  may be more causally linked to *confirmed* DS (older mothers having
  longer, more tightly-monitored pregnancies with earlier cytogenetic
  confirmation). Then C-only is closer to DS biology and C+P was polluted.

Both readings predict the observed shift. The test profile permutation
importance for `usbc10_m0_cn` (see fit artefacts) showed `bfacil3` going
*negative* (permuting it improved AP) — i.e. under C-only, birth facility
type encodes noise rather than DS signal, supporting reading (1).

### 4. Race/ethnicity: feature-set effect, not label effect

Race and Hispanic-origin shifts are small in the label-effect comparisons
(≤ 2 pp in absolute terms), but much larger when the feature set changes
under a fixed (C-only) label:

| Variable | Δ (C-only − C+P), usbc10 | Δ, usbc11 | Δ (usbc11 − usbc10), C-only | Recorded % |
|---|---|---|---|---|
| NH White   | +2.0 pp | +0.9 pp | −2.3 pp | 52.4% |
| NH Black   | −0.2 pp | −0.2 pp | **+4.1 pp** | 11.9% |
| NH Asian   | −0.5 pp | +0.0 pp | **+3.3 pp** | 3.5%  |
| Hispanic   | −1.2 pp | −0.8 pp | **−5.7 pp** | 27.9% |

The clinical-only C-only model over-represents NH Black and NH Asian
relative to the recorded baseline (14.2% / 6.6% vs recorded 11.9% / 3.5%)
and under-represents Hispanic (22.6% vs 27.9%). This matches the
direction predicted by Boulet et al. (2011) — that NH Black and NH Asian
infants are under-ascertained on birth certificates more heavily than
NH White or Hispanic infants — and is **not** an artefact of the C-only
label change (which shifted race shares by <2 pp on its own). The
sociodemographic-feature removal is what's doing the work.

This is an important qualification on the earlier selection-bias probe:
it's the feature-set choice that recovers the Boulet signal, not the
label cleanup. The C-only label cleanup is orthogonal and addresses a
separate confound (pending-code recording noise).

### 5. Gestational hypertension surfaces under C-only clinical

`rf_ghype = Yes` was ~noise in both C+P models (≤ 1 pp delta from
baseline). Under C-only clinical-only (`usbc11_m1_cn`), it lands at
13.4% in the predicted-missing pool vs 9.4% recorded — the single
largest *new* signal to emerge. The permutation-importance file for
`usbc11_m0_cn` shows it at 0.0005 (borderline retained). Worth
flagging as a candidate for closer look in the Bayesian outcomes model,
but treat as preliminary given the narrow margin above the retention
threshold.

## Per-comparison summary

### A. `c_effect_full` — `usbc10_m1` vs `usbc10_m1_cn`

Full feature set, C/P label only. Dominant shifts are clinical
intensification (NICU +8, CCHD +18, VLBW +3, ventilation +7) plus the
`ca_disor` tautology collapse (−27). SES drift toward private /
higher-education is moderate (~4 pp). Race shifts ≤ 2 pp.

### B. `c_effect_clinical` — `usbc11_m0` vs `usbc11_m1_cn`

Clinical-only feature set, C/P label only. Same direction as (A) with
smaller magnitudes on most clinical variables (CCHD and `ca_disor`
collapse survive at similar size because those features are in both
models); the clinical-only model has less headroom to redistribute
across the missing non-clinical features. Notable new signal:
`rf_ghype = Yes` +5.8 pp.

### C. `nonclinical_effect` — `usbc10_m1_cn` vs `usbc11_m1_cn`

Both C-only; differ only in whether non-clinical features (race,
education, payer, paternal age) are in the model. Stripping non-clinical
features shifts the predicted-missing pool:

- Towards advanced maternal age (40-44 +2.6, 45-49 +2.5)
- Towards NH Black (+4.1) and NH Asian (+3.3), away from Hispanic (−5.7)
- Towards the higher-severity clinical tail (`rf_ghype = Yes` +6.0,
  `ab_aven1 = Yes` +3.8, `dbwt < 2500 g` +2.5)
- `ca_disor = Confirmed` drops by 5.6 pp (clinical-only model loses the
  disorder-confirmation signal, since confirmed chromosomal disorder is
  a near-tautology with DS)

This is the "cleanest" comparison for the Boulet-style
under-ascertainment hypothesis: same label, same size of missing-flag
pool, different feature-set ideology.

## Caveats

- **Sample size of C-only positives.** Valid-set positives: 1,597
  (same for both C-only variants, same split). Train-set positives:
  6,387. That's above the flagged 1,500-2,000 lower bound but tight.
  Tail calibration at the 99.9th-percentile score bin has wider CIs than
  the C+P runs.

- **Quota denominator differs.** The year×month `1.5 × recorded` quota
  uses recorded-within-training, which for C-only excludes P rows.
  Cross-variant comparisons of the absolute *count* of predicted-missing
  cases are not meaningful; the demographic/clinical *shares* are.

- **`apgar5` data-preparation bug.** The load_predictors_data filter is
  `apgar5 >= 10 AND apgar5 <= 10` (data_utils.py:185-187), which drops
  all values except exactly 10 and effectively zeroes the feature's
  variance. apgar5 appears below the importance threshold in every fit
  for this reason, not because the 5-minute APGAR score is uninformative
  for DS. Out of scope for this work but worth a separate bug.

- **Hyperparameter tune quality.** The C+P baselines (`usbc10_m1`,
  `usbc11_m0`) were fit with `--no-optimize` using `DEFAULT_PRIOR_BEST_PARAMS`
  (which happens to equal each variant's declared `params`). The C-only
  variants ran fresh Optuna searches under the test profile on their
  M0-level feature sets, then the M1-level variants inherited those
  tuned values without re-tuning. A fairer apples-to-apples comparison
  would re-run Optuna on all four under reporting; the current runs are
  a reasonable first cut but the AP differences should be read as a
  lower bound on potential performance.

- **`ca_disor` tautology is structural, not a tuning issue.** Even if
  the C+P models were re-tuned with `ca_disor` dropped from the feature
  set, the point of (1) stands: the C+P label is partly defined by
  "pending chromosomal disorder" (through `ca_down = 'P'`), so there is
  no feature-engineering fix that fully separates them. The label
  change is the correct remedy.

- **`ds_case_weight` and downstream Bayesian outcomes.** The selection
  model (`dspopulations_us_birth_certificates/selection`) and the
  Bayesian outcomes model use `down_ind = 1` (C+P) as the recorded
  indicator and `ds_pred_missing*` as the missing flag. None of that
  has been rerun with the C-only variants; the DuckDB columns are
  available (`ds_pred_missing_13`, `_14`) if and when we want to redo
  those analyses on the cleaner label.

## Next steps

- Decide whether downstream analyses (selection model, outcomes model,
  predicted analyses report) should switch to the C-only variants as
  the default, run both for comparison, or keep C+P and use C-only only
  as a diagnostic. The tautology finding (1) is a strong argument for
  C-only becoming the primary for any analysis that uses
  `ds_pred_missing*` as the missed-cases cohort.
- Re-tune the C+P baselines under the reporting profile to make the
  label-effect comparisons (A, B) fully apples-to-apples.
- Fix `apgar5` filter bug in `data_utils.py` and rerun feature
  selection. APGAR5 may enter the retained set under C-only once it
  has variance.
- Consider a further-pruned `usbc11_m2_cn` that re-examines features
  still near threshold under C-only (e.g. `rf_fedrg` at 1.1e-4, barely
  retained) to see if feature-selection noise is inflating any
  comparison.
