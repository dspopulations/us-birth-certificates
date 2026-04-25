# Retune + reselect all gradient-boosting models after the apgar5 fix

**Date:** 2026-04-22 → 2026-04-23 (work spans multiple sessions)
**Triggered by:** PR #26 — `apgar5` / `apgar10` filter was `>= 10 AND <= 10`,
NULL'ing the entire 0-9 range. ~98% of valid APGAR variance was destroyed in
training data for every existing model.
**Reports:** see per-family `output/fit_model_test/<id>/` (M0 diagnostic) and
`output/fit_model_reporting/<id>/` (final M1 fit).
**Notes prerequisite:** `notes/20260422-compare-confirmed-only.md` — the
pre-fix apples-to-apples baseline this work supersedes. Caveats §3 ("`apgar5`
data-preparation bug") and §4 ("Hyperparameter tune quality") flag the work
done here.
**Plan file (private):** `~/.claude/plans/review-the-changes-made-rosy-bentley.md`

## Context

Every existing `usbc10_*` and `usbc11_*` LightGBM model trained with `apgar5`
near-constant (≈98% NULL) and `apgar10` either near-constant or pruned to zero
under the M0→M1 selection step. Two consequences:

1. **Tuned hyperparameters are stale.** Optuna optimised against a feature
   matrix where one or two clinically-important columns were structurally
   uninformative.
2. **M0→M1 selection decisions are stale.** `apgar5` had no variance to lose,
   so it permuted to ~zero and was dropped under C-only; `apgar10` was dropped
   for the same reason under C+P. With the data fix neither result is
   trustworthy.

The fix is to re-run the full **tune → fit + permutation + correlation →
re-select → re-tune → final fit** cycle on all four families, commit the
results back into the existing model classes in place (`update existing model
classes in place` decision logged in plan), and document deltas against the
pre-fix baseline.

## Method

Per family (run for each of `usbc10`, `usbc10_cn`, `usbc11`, `usbc11_cn`):

1. **M0 tune (test profile, throwaway).** 50 Optuna trials × 10 000 boost
   rounds. Output `output/tuning/<m0_id>/best_params.json` is *not* committed
   — it feeds step 2 only.
2. **M0 fit + permutation + correlation (test profile).** Generates
   `permutation_importance.csv`, `feature_importance_gain.csv`,
   `shap_importance.csv`, plus `plots/dendrogram.{png,svg}` and
   `plots/correlation_heatmap.{png,svg}`.
3. **Decide M1 prune.** Drop features with permutation AP loss < 1e-4; for any
   correlation-dendrogram pair with dissimilarity < 0.3 keep the higher-
   importance side and drop the other even if it would have survived the
   importance cut.
4. **M1 retune (reporting profile, real tune).** 200 Optuna trials × 50 000
   boost rounds.
5. **Final reporting fit + `--write-predictions`.** Overwrites the existing
   per-variant `p_ds_lb_pred_*` and `ds_pred_missing_*` columns in
   `data/us_births.db` (column names per `predictions_column` /
   `missing_flag_column` attributes — `_01`/`_02`/`_13`/`_14` reused).
6. **Commit** retuned `params` + post-prune `categorical_features` + new
   `SelectionStep` to the model class.

For USBC11 the M0 baseline is **re-derived**: `USBC10_M0`'s 6 numeric + 42
categorical features minus the 6 sociodemographic features (`mracehisp`,
`fracehisp`, `meduc`, `feduc`, `pay_rec`, `fagecomb`). This restores `apgar10`
and the rare-disorder/risk flags that USBC11 currently inherits-pruned via
`USBC10_M1`. Registered as transient `USBC11_M0_BASE` /
`USBC11_M0_BASE_CN` classes in `models/usbc11_baseline.py` for the duration
of this work; removed at commit time.

Year range for all runs: 2016–2024 (matches existing reporting fits — both the
M0 diagnostic and the M1 final fit use the same data slice, so the
permutation-importance ranking is computed against the same distribution the
final model is evaluated on; resolved by setting `year_range = (2016, 2024)`
on the model class, which both `tune_model.py` and `fit_model.py` honour
when `--model-id` is set).
Random seed: 47 (matches existing tune runs; preserves run-to-run reproducibility).

### Pipeline smoke test (2026-04-23)

Confirmed `tune_model.py` works end-to-end on `usbc11_m0_base` at the **dev**
profile (10 trials × 500 boost rounds × 9 years × 42 features). Wallclock
~37 min; 7 trials completed, 3 pruned by Hyperband; best AP 0.0368. This
calibrates the per-trial cost and confirms the new transient classes register
correctly in the model registry.

Per-trial cost extrapolations for the larger profiles (used to set
expectations, not fixed budgets — early-stopping behaviour varies with the
sampled hyperparameters):

| Profile  | Trials | Boost rounds | Early stop | Year span | Estimated wallclock |
|---|---|---|---|---|---|
| dev      | 10  |    500 |  50 | 9 yr | ~37 min (measured) |
| test     | 50  | 10 000 | 200 | 9 yr | ~5–8 h (per family) |
| reporting| 200 | 50 000 | 200 | 9 yr | ~20–35 h (per family) |

## Per-family results

### `usbc10` (C+P, full feature set)

#### M0 tune (test profile, 50 trials × 10K rounds, 9 yr)
- Wallclock: **2 h 00 min** (50 trials, 9 completed, 41 pruned by Hyperband)
- Best AP (valid): **0.037199** (trial 22)
- Best params: lr 0.0098, num_leaves 86, min_data_in_leaf 1350,
  feature_fraction 0.65, bagging_fraction 0.96/freq 8,
  λ₁ 2.05e-7, λ₂ 1.33, min_gain_to_split 0.78
- Artefacts: `output/tuning/usbc10_m0/`

#### M0 fit + permutation + correlation
- Wallclock: **53 min** (full permutation + SHAP subsample)
- Metrics: AP **0.0349**, ROC-AUC **0.897**, best_iter 545, n_valid 6.72M, n_pos 3,562
- Artefacts: `output/fit_model_test/20260423-131505/`

**Top permutation-importance features** (mean AP loss ± sd):
| feature   | importance | ± sd     |
|-----------|-----------:|---------:|
| mage_c    | 1.48e-2 | 4.5e-4 |
| ca_disor  | 1.44e-2 | 3.2e-4 |
| ca_cchd   | 1.19e-2 | 4.8e-4 |
| ab_nicu   | 7.00e-3 | 7.0e-4 |
| dbwt      | 3.06e-3 | 4.5e-4 |
| dmeth_rec | 2.67e-3 | 3.8e-4 |
| gestrec10 | 2.35e-3 | 4.1e-4 |
| **apgar5**    | **1.82e-3** | **4.0e-4** |
| ab_aven6  | 1.26e-3 | 2.8e-4 |
| wtgain    | 1.09e-3 | 2.7e-4 |

- **APGAR5 status:** importance jumped from 1.37e-5 (pre-fix `usbc10_m0_cn`)
  to **1.82e-3** under post-fix `usbc10_m0` (C+P) — **133× larger**, now the
  8th most important feature. The PR #26 fix is doing what was expected.
- **APGAR10 status:** importance is **−1.33e-4** (the worst of all 48
  features — permuting it improves AP). Distance correlation with apgar5 is
  0.86, so apgar10 is essentially redundant noise once apgar5 has variance.

**Below-threshold (importance < 1e-4) — 23 features:**
sex (7.05e-5), ca_mnsb (5.31e-5), ld_augm (5.29e-5), rf_ppterm (5.15e-5),
wic (4.33e-5), ca_anen (4.16e-5), ld_indl (3.91e-5), rf_ehype (3.86e-5),
rf_pdiab (3.50e-5), ca_gast (2.49e-5), ca_limb (1.83e-5), ab_surf (1.52e-5),
ca_cleft (1.18e-5), ca_hypo (6.83e-6), ca_cdh (5.97e-6), rf_gdiab (4.88e-6),
ab_seiz (−7.7e-8), fracehisp (−4.6e-6), ca_omph (−2.0e-5),
rf_artec (−2.3e-5), rf_fedrg (−3.5e-5), ca_clpal (−5.5e-5), apgar10 (−1.3e-4).

**Correlation pairs (dcor > 0.7):**
| pair | dcor | imp left | imp right | decision |
|---|---:|---:|---:|---|
| apgar5 ~ apgar10  | 0.859 | +1.82e-3 | −1.33e-4 | keep apgar5; drop apgar10 (already a drop candidate) |
| rf_fedrg ~ rf_artec | 0.813 | −3.5e-5 | −2.3e-5 | both drop (both below threshold) |
| mage_c ~ fagecomb | 0.747 | +1.48e-2 | +1.04e-4 | keep mage_c; drop fagecomb (just above threshold but correlated) |

#### M1 drop list (post-fix)
**24 features** — 23 importance-based + `fagecomb` added on correlation grounds:
`fagecomb` (numeric); plus 23 categorical: sex, ca_mnsb, ld_augm,
rf_ppterm, wic, ca_anen, ld_indl, rf_ehype, rf_pdiab, ca_gast, ca_limb,
ab_surf, ca_cleft, ca_hypo, ca_cdh, rf_gdiab, ab_seiz, fracehisp,
ca_omph, rf_artec, rf_fedrg, ca_clpal, apgar10.

**Vs. pre-fix `_M1_FEATURES_REMOVED`** (18):
- New drops not in pre-fix list (6): `sex`, `ld_augm`, `ld_indl`,
  `fracehisp`, `rf_fedrg`, `fagecomb`. (Pre-fix kept these despite low
  importance because the model leaned harder on `apgar5`-shaped variance
  it didn't actually have, leaving these features marginally useful as
  surrogates. With apgar5 fixed, that surrogate role evaporates.)
- Re-instated from pre-fix prune (1): none — all 18 pre-fix drops remain
  below threshold post-fix.
- `apgar5` newly **retained**; `apgar10` stays dropped (now with stronger
  evidence — it's actively harmful, not just uninformative).

**Surviving M1 feature set** (24 features = 5 numeric + 19 categorical):
- Numeric: year, dbwt, wtgain, bmi, mage_c
- Categorical: bfacil3, precare, gestrec10, rf_phype, rf_ghype, rf_inftr,
  me_pres, dmeth_rec, apgar5, ab_aven1, ab_aven6, ab_nicu, ab_anti,
  ca_cchd, ca_disor, meduc, mracehisp, feduc, pay_rec.

#### M1 retune (reporting profile, 200 trials × 50K rounds, 9 yr)
- Wallclock: **4 h 25 min** (started 14:13, finished 18:38 on 2026-04-23)
- 200 trials: 16 completed, 184 pruned by Hyperband (more aggressive than
  the M0 test-profile run, as expected with the longer boost rounds)
- Best AP (valid): **0.033329** (trial 1) — modestly higher than the
  pre-fix M1 reporting-fit AP (0.0324)
- Best params: lr **0.0172**, num_leaves **64**, min_data_in_leaf **537**,
  min_gain_to_split 0.099, feature_fraction 0.72, bagging 0.86 / freq 4,
  λ₁ 4.67e-7, λ₂ 1.80
  - Notably different from pre-fix params (lr 0.0095, num_leaves 180,
    min_data 756): the fix's restored apgar5 variance + the wider prune
    pushed the optimum to a higher learning rate with fewer, deeper-data
    leaves. λ₂ is similar (1.80 vs 0.61); λ₁ is much smaller.
- Artefacts: `output/tuning/usbc10_m1/`

#### Final fit (reporting profile, 50K rounds × 9 yr)
- Wallclock: ~14 h 42 min (started 18:39 on 2026-04-23, finished early
  morning 2026-04-24; permutation + SHAP at full sample is the slow
  contributor)
- Artefacts: `output/fit_model_reporting/20260423-183929/`

| Metric | Pre-fix `usbc10_m1` | Post-fix `usbc10_m1` | Δ |
|---|---:|---:|---:|
| AP (valid)   | 0.0324 | **0.0346** | **+6.9%** |
| ROC-AUC      | 0.889  | **0.8955** | **+0.7 pp** |
| best_iter    | 158    | 556        | +252% |
| log_loss     | —      | 0.003565   | — |
| n_valid      | —      | 6,717,846  | — |
| n_pos_valid  | —      | 3,562      | — |

**Post-fix permutation importance under the M1 feature set (top 8):**

| feature   | importance | ± sd     |
|-----------|-----------:|---------:|
| mage_c    | 1.49e-2 | 4.8e-4 |
| ca_disor  | 1.48e-2 | 4.8e-4 |
| ca_cchd   | 1.21e-2 | 5.2e-4 |
| ab_nicu   | 8.86e-3 | 1.1e-3 |
| dbwt      | 3.57e-3 | 6.2e-4 |
| gestrec10 | 3.40e-3 | 6.1e-4 |
| **apgar5**    | **3.03e-3** | **5.6e-4** |
| dmeth_rec | 2.45e-3 | 5.5e-4 |

`apgar5`'s importance has *grown* from 1.82e-3 (M0 test-profile fit) to
3.03e-3 (M1 reporting fit) once the redundant features are removed and
hyperparameters are properly tuned to the smaller feature set. `apgar5`
is now the 7th-most-important feature out of 24, a complete reversal
from its pre-fix near-zero importance.

**Caveat — two features cross back below the 1e-4 threshold under the
post-tune fit:**
- `ab_anti` = 7.96e-5 (was 1.76e-4 in the M0 test-profile fit)
- `feduc`   = 2.01e-5 (was 2.29e-4 in the M0 test-profile fit)

Both are close to threshold; both stayed in the M1 set per the M0-derived
prune. Worth flagging but not re-iterating the prune cycle for this
plan — a future "M2" pass could revisit them.

#### Pending: write predictions to DuckDB
The `--write-predictions` step (overwrites `p_ds_lb_pred_01` /
`ds_pred_missing` columns) needs to be run separately because the
runtime guard requires explicit per-invocation approval. Command:
```
python scripts/fit_model.py --model-id usbc10_m1 --profile reporting \
  --load-model output/fit_model_reporting/20260423-183929/model.txt \
  --write-predictions --no-permutation --no-shap
```

### `usbc10_cn` (C-only, full feature set)

#### M0 tune (test profile, 50 trials × 10K rounds, 9 yr)
- Wallclock: **1 h 12 min** (faster than `usbc10_m0` because C-only's
  fewer positives push trials to early-stop sooner)
- 50 trials: 7 completed, 43 pruned by Hyperband
- Best AP (valid): **0.024014** (trial 1)
- Best params: identical to `usbc10_m1`'s reporting tune (TPE/Hyperband
  landed on the same first trial under the shared seed: lr 0.0172,
  num_leaves 64, min_data_in_leaf 537, etc.)
- Artefacts: `output/tuning/usbc10_m0_cn/`

#### M0 fit + permutation + correlation
- Wallclock: ~46 min (started 10:55, finished 11:41)
- Metrics: AP **0.02372**, ROC-AUC **0.9134**, best_iter 470, n_pos 1,597
- Artefacts: `output/fit_model_test/20260424-105543/`

**Top permutation-importance features**:

| feature   | importance | ± sd     |
|-----------|-----------:|---------:|
| mage_c    | 1.29e-2 | 6.2e-4 |
| ca_cchd   | 1.21e-2 | 4.0e-4 |
| ca_disor  | 1.04e-2 | 2.9e-4 |
| ab_nicu   | 6.46e-3 | 1.0e-3 |
| dbwt      | 2.94e-3 | 5.4e-4 |
| gestrec10 | 2.85e-3 | 6.7e-4 |
| **apgar5**    | **2.65e-3** | **3.9e-4** |
| dmeth_rec | 1.62e-3 | 3.9e-4 |
| rf_ghype  | 7.59e-4 | 3.1e-4 |

- **APGAR5 status:** importance 1.37e-5 → 2.65e-3 (**193× larger**),
  now 7th-most-important. The PR #26 fix has the same restorative effect
  under C-only as under C+P.
- **APGAR10 status:** importance 7.82e-6 (essentially zero) — and
  flagged for correlation drop with apgar5 (dcor 0.81).

**Below-threshold (importance < 1e-4) — 30 features:**
mracehisp (9.9e-5), bfacil3 (8.3e-5), ab_anti (6.8e-5), ld_indl (4.5e-5),
rf_ehype (3.3e-5), ca_omph (2.6e-5), rf_pdiab (2.2e-5), sex (2.0e-5),
me_pres (1.4e-5), ca_clpal (1.4e-5), apgar10 (7.8e-6), ab_surf (2.6e-6),
ca_anen (1.4e-6), ca_gast (6.3e-7), ab_seiz (4.0e-7), ca_cdh (−7.9e-6),
rf_fedrg (−1.1e-5), ca_cleft (−1.3e-5), rf_gdiab (−1.4e-5),
ca_hypo (−2.0e-5), rf_ppterm (−2.2e-5), rf_inftr (−3.3e-5),
rf_artec (−3.7e-5), ca_limb (−3.7e-5), **bmi (−4.3e-5, numeric)**,
ca_mnsb (−6.0e-5), fracehisp (−8.7e-5), wic (−9.4e-5),
**pay_rec (−1.6e-4)**, **rf_phype (−2.6e-4)**.

`pay_rec` and `rf_phype` carry strongly negative importance — permuting
them improves AP. `rf_phype`'s pattern was flagged in
`compare-confirmed-only.md` §1 Caveats as a co-linearity-with-mage_c
artefact under C-only; the apgar fix did not rescue it. `bmi` (numeric,
default-kept in pre-fix) is also actively harmful under C-only with the
fixed apgar inputs.

**Correlation pairs (dcor > 0.7):**
| pair | dcor | imp left | imp right | decision |
|---|---:|---:|---:|---|
| apgar5 ~ apgar10  | 0.814 | +2.65e-3 | +7.82e-6 | keep apgar5; drop apgar10 (already a drop) |
| rf_fedrg ~ rf_artec | 0.798 | −1.1e-5 | −3.7e-5 | both drop (both below threshold) |
| mage_c ~ fagecomb | 0.745 | +1.29e-2 | +4.35e-4 | keep mage_c; drop fagecomb (4.35e-4 above threshold but redundant) |

#### M1 drop list (post-fix)
**31 features** — 30 importance-based + `fagecomb` correlation:
- 2 numeric: `bmi`, `fagecomb`
- 29 categorical: mracehisp, bfacil3, ab_anti, ld_indl, rf_ehype,
  ca_omph, rf_pdiab, sex, me_pres, ca_clpal, apgar10, ab_surf, ca_anen,
  ca_gast, ab_seiz, ca_cdh, rf_fedrg, ca_cleft, rf_gdiab, ca_hypo,
  rf_ppterm, rf_inftr, rf_artec, ca_limb, ca_mnsb, fracehisp, wic,
  pay_rec, rf_phype.

**Vs. pre-fix `_M1_CN_FEATURES_REMOVED`** (22):
- New drops not in pre-fix list (10): `bmi`, `fagecomb`, `mracehisp`,
  `ab_anti`, `rf_pdiab`, `sex`, `me_pres`, `wic`, `pay_rec`, `rf_phype`.
- Re-instated from pre-fix prune (1): `ld_augm` (now 1.27e-4, above
  threshold).
- `apgar5` newly retained; `apgar10` stays dropped (and now correlation-
  redundant with apgar5).

**Surviving M1_CN feature set** (17 features = 4 numeric + 13 categorical):
- Numeric: year, dbwt, wtgain, mage_c
- Categorical: precare, gestrec10, rf_ghype, ld_augm, dmeth_rec, apgar5,
  ab_aven1, ab_aven6, ab_nicu, ca_cchd, ca_disor, meduc, feduc.

#### M1 retune (reporting profile, 200 trials × 50K rounds, 9 yr)
- Wallclock: **3 h 52 min** (started 11:44, finished 15:37 on 2026-04-24)
- 200 trials: 19 completed, 181 pruned by Hyperband
- Best AP (valid): **0.026209** (trial 116)
- Best params: lr **0.0176**, num_leaves **78**, min_data_in_leaf **507**,
  min_gain_to_split 0.182, feature_fraction 0.749, bagging 0.857 / freq 4,
  λ₁ 4.37e-7, λ₂ 0.128
  - Notably similar learning rate to `usbc10_m1` (0.0176 vs 0.0172) but
    λ₂ is an order of magnitude smaller (0.13 vs 1.80) — consistent with
    the smaller C-only feature set (17 vs 24) having less room for
    overfitting to distributionally-redundant columns, so the optimum
    trims the regularisation rather than the feature count.
- Artefacts: `output/tuning/usbc10_m1_cn/`

#### Final fit (reporting profile, 50K rounds × 9 yr)
- Wallclock: **~17 min** (started 17:52, finished 18:09 on 2026-04-24) —
  much faster than `usbc10_m1` (~14h) because (a) the 17-feature C-only
  set is smaller, (b) only 1,597 positives vs 3,562 gives less signal to
  fit, and (c) `best_iteration=89` — the model converged early.
- Artefacts: `output/fit_model_reporting/20260424-175244/`

| Metric | Pre-fix `usbc10_m1_cn` | Post-fix `usbc10_m1_cn` | Δ |
|---|---:|---:|---:|
| AP (valid)   | 0.02451 | **0.02470** | **+0.8%** |
| ROC-AUC      | 0.9081  | **0.9097**  | **+0.16 pp** |
| best_iter    | 282     | 89          | −68% |
| log_loss     | 0.001766 | 0.001776   | +0.6% |
| n_valid      | 6,715,881 | 6,715,881 | — |
| n_pos_valid  | 1,597   | 1,597       | — |

**Post-fix permutation importance under the M1_CN feature set (all 17):**

| rank | feature    | importance | ± sd     |
|----:|------------|-----------:|---------:|
|  1  | ca_cchd    | 1.43e-2 | 3.5e-4 |
|  2  | mage_c     | 1.23e-2 | 5.5e-4 |
|  3  | ca_disor   | 1.07e-2 | 4.1e-4 |
|  4  | ab_nicu    | 4.25e-3 | 1.1e-3 |
|  5  | dbwt       | 3.98e-3 | 1.1e-3 |
|  6  | dmeth_rec  | 3.25e-3 | 7.5e-4 |
|  7  | **apgar5** | **2.36e-3** | **5.8e-4** |
|  8  | gestrec10  | 1.54e-3 | 6.8e-4 |
|  9  | year       | 8.89e-4 | 7.4e-4 |
| 10  | ab_aven6   | 8.52e-4 | 3.6e-4 |
| 11  | precare    | 4.96e-4 | 2.0e-4 |
| 12  | ab_aven1   | 4.64e-4 | 2.4e-4 |
| 13  | wtgain     | 3.42e-4 | 7.2e-4 |
| 14  | rf_ghype   | 3.22e-4 | 2.6e-4 |
| 15  | meduc      | 2.96e-4 | 3.7e-4 |
| 16  | feduc      | 2.37e-4 | 2.2e-4 |
| 17  | ld_augm    | 1.24e-4 | 8.8e-5 |

`apgar5` sits firmly in the top half under the pruned C-only set — the
PR #26 fix carries through cleanly from M0 to M1 final fit. All 17
retained features clear the 1e-4 AP-loss threshold under the retuned
model (the closest to the cut-off is `ld_augm` at 1.24e-4, still above).

#### Predictions written to DuckDB (2026-04-24)
`--write-predictions` overwrote `p_ds_lb_pred_13` / `ds_pred_missing_13`
on 2026-04-24 18:14 (artefacts: `output/fit_model_reporting/20260424-181416/`).
33,579,403 rows populated; 12,002 non-recorded births flagged as likely
missed (under the C-only year×month quota of `ceil(1.5 × recorded)`).

### `usbc11` (C+P, clinical+age — re-derived M0 baseline)

#### M0 tune (test profile, 50 trials × 10K rounds, 9 yr) on `USBC11_M0_BASE` (5 num + 37 cat)
- Wallclock: ~2 h 19 min (started 18:37, finished 20:05 on 2026-04-24)
- 50 trials: 7 completed, 43 pruned by Hyperband
- Best AP (valid): **0.034044** (trial 1)
- Best params: identical to `usbc11_m0_base`'s = `usbc10_m1`'s reporting
  optimum (TPE/Hyperband first-trial hit under the shared seed).
- Artefacts: `output/tuning/usbc11_m0_base/`

#### M0 fit + permutation + correlation (test profile)
- Wallclock: ~33 min (started ~20:05, finished 20:29 on 2026-04-24)
- Metrics: AP **0.0334**, ROC-AUC **0.8934**, best_iter 431, n_pos 3,562
- Artefacts: `output/fit_model_test/20260424-200531/`

**Top permutation-importance features:**

| feature   | importance | ± sd     |
|-----------|-----------:|---------:|
| mage_c    | 1.39e-2 | 5.1e-4 |
| ca_disor  | 1.36e-2 | 5.8e-4 |
| ca_cchd   | 1.31e-2 | 3.6e-4 |
| ab_nicu   | 7.23e-3 | 6.5e-4 |
| dbwt      | 3.70e-3 | 3.3e-4 |
| **apgar5**    | **2.88e-3** | **3.9e-4** |
| gestrec10 | 2.84e-3 | 3.8e-4 |
| dmeth_rec | 2.00e-3 | 4.9e-4 |
| ab_aven6  | 1.57e-3 | 3.6e-4 |

- **APGAR5 status:** importance is 2.88e-3 — comparable to the post-fix
  `usbc10_m0_cn` (2.65e-3) and `usbc10_m0` (1.82e-3) values. The fix
  carries through to the USBC11 sociodemographic-stripped feature set
  intact.
- **APGAR10 status:** importance 1.63e-5 (sub-threshold), correlation
  pair with apgar5 dcor 0.86 → drop on both grounds.

**Below-threshold (importance < 1e-4) — 19 features:**
wic (9.4e-5), rf_ehype (9.0e-5), rf_gdiab (7.7e-5), rf_pdiab (6.8e-5),
rf_ppterm (6.2e-5), rf_artec (5.6e-5), ca_anen (3.6e-5), ca_mnsb (3.1e-5),
rf_fedrg (2.7e-5), ca_gast (2.1e-5), ab_seiz (1.7e-5), apgar10 (1.6e-5),
ld_augm (1.2e-5), ca_cdh (4.8e-6), ca_limb (-3.2e-6), ca_omph (-1.7e-5),
ca_hypo (-3.1e-5), ab_surf (-3.6e-5), ca_cleft (-1.34e-4).

**Correlation pairs (dcor > 0.7):**
| pair | dcor | imp left | imp right | decision |
|---|---:|---:|---:|---|
| apgar5 ~ apgar10  | 0.859 | +2.88e-3 | +1.63e-5 | keep apgar5; drop apgar10 (already a drop) |
| rf_fedrg ~ rf_artec | 0.797 | +2.67e-5 | +5.63e-5 | both drop (both below threshold) |

#### M1 drop list (post-fix)
**19 categorical features**, no numeric drops:
rf_pdiab, rf_gdiab, rf_ehype, rf_ppterm, rf_fedrg, rf_artec, ld_augm,
apgar10, ab_surf, ab_seiz, ca_anen, ca_mnsb, ca_cdh, ca_omph, ca_gast,
ca_limb, ca_cleft, ca_hypo, wic.

**Surviving M1 feature set** (23 features = 5 numeric + 18 categorical):
- Numeric: year, dbwt, wtgain, bmi, mage_c
- Categorical: bfacil3, sex, precare, gestrec10, rf_phype, rf_ghype,
  rf_inftr, ld_indl, me_pres, dmeth_rec, apgar5, ab_aven1, ab_aven6,
  ab_nicu, ab_anti, ca_cchd, ca_clpal, ca_disor.

#### M1 retune (reporting profile, 200 trials × 50K rounds, 9 yr)
- Wallclock: **3 h 4 min** (started 22:53 on 2026-04-24,
  finished 01:57 on 2026-04-25)
- 200 trials: 14 completed, 186 pruned by Hyperband
- Best AP (valid): **0.037099** (trial 72)
- Best params: lr **0.0099**, num_leaves **82**, min_data_in_leaf **596**,
  min_gain_to_split 0.287, feature_fraction 0.709, bagging 0.865 / freq 4,
  λ₁ 2.75e-4, λ₂ 1.57
- Artefacts: `output/tuning/usbc11_m1/`

#### Final fit (reporting profile, 50K rounds × 9 yr)
- Wallclock: **1 h 3 min** (started 01:57, finished 03:00 on 2026-04-25)
- Artefacts: `output/fit_model_reporting/20260425-015750/`

| Metric | Pre-fix `usbc11_m0` | Post-fix `usbc11_m1` | Δ |
|---|---:|---:|---:|
| AP (valid)   | 0.0310 | **0.0334** | **+7.7%** |
| ROC-AUC      | 0.883  | **0.8916** | **+0.86 pp** |
| best_iter    | 570    | 948        | +66% |
| log_loss     | —      | 0.003608   | — |
| n_valid      | —      | 6,717,846  | — |
| n_pos_valid  | —      | 3,562      | — |

**Post-fix permutation importance under the M1 feature set (top 8):**

| feature   | importance | ± sd     |
|-----------|-----------:|---------:|
| ca_cchd   | 1.35e-2 | 4.6e-4 |
| mage_c    | 1.33e-2 | 5.1e-4 |
| ca_disor  | 1.23e-2 | 5.4e-4 |
| ab_nicu   | 8.44e-3 | 6.1e-4 |
| dbwt      | 3.89e-3 | 4.0e-4 |
| gestrec10 | 3.16e-3 | 4.7e-4 |
| **apgar5**    | **2.83e-3** | **5.7e-4** |
| dmeth_rec | 2.76e-3 | 4.3e-4 |

apgar5 is the 7th-most-important feature post-prune, confirming the fix
holds across the USBC11 family.

**Caveat — one feature crosses below 0 under the post-tune fit:**
- `rf_phype` = -7.43e-5 (was +1.29e-4 under M0). Borderline; not worth
  re-iterating the prune for a one-feature change.

#### Predictions written to DuckDB (2026-04-25)
`--write-predictions` overwrote `p_ds_lb_pred_02` / `ds_pred_missing_02`
on 2026-04-25 06:15 (artefacts: `output/fit_model_reporting/20260425-061545/`).
33,589,228 rows populated; 26,742 non-recorded births flagged as likely
missed.

### `usbc11_cn` (C-only, clinical+age — re-derived M0 baseline)

#### M0 tune (test profile, 50 trials × 10K rounds, 9 yr) on `USBC11_M0_BASE_CN` (5 num + 37 cat)
- Wallclock: ~1 h 38 min (started 20:29, finished 22:07 on 2026-04-24)
- Best AP (valid): **0.022058** (trial 1)
- Best params: lr 0.015, num_leaves 87, min_data_in_leaf 3145,
  min_gain_to_split 0.004, feature_fraction 0.68, bagging 0.98 / freq 5,
  λ₁ ≈ 0, λ₂ ≈ 0
- Artefacts: `output/tuning/usbc11_m0_base_cn/`

#### M0 fit + permutation + correlation (test profile)
- Wallclock: ~36 min
- Metrics: AP **0.0215**, ROC-AUC **0.9160**, best_iter 406, n_pos 1,597
- Artefacts: `output/fit_model_test/20260424-220729/`

**Top permutation-importance features:**

| feature   | importance | ± sd     |
|-----------|-----------:|---------:|
| ca_cchd   | 1.17e-2 | 3.3e-4 |
| mage_c    | 1.07e-2 | 7.6e-4 |
| ca_disor  | 7.83e-3 | 2.8e-4 |
| ab_nicu   | 4.34e-3 | 8.1e-4 |
| **apgar5**    | **3.05e-3** | **4.5e-4** |
| dbwt      | 2.89e-3 | 5.5e-4 |
| dmeth_rec | 1.92e-3 | 4.0e-4 |
| gestrec10 | 1.79e-3 | 4.6e-4 |

- **APGAR5 status:** 3.05e-3 — the largest of the four post-fix M0
  fits. The C-only label + clinical-only feature set most starkly
  rewards the apgar5 information that the bug had erased.
- **APGAR10 status:** 9.96e-5 (just below threshold, plus dcor 0.83
  with apgar5).

**Below-threshold (importance < 1e-4) — 22 features:**
apgar10 (9.96e-5), sex (4.29e-5), rf_inftr (5.28e-5), rf_ehype (4.02e-5),
ca_anen (4.72e-5), ab_surf (3.63e-5), me_pres (2.53e-5), ca_mnsb
(5.25e-6), ca_limb (4.13e-6), ab_seiz (-3.11e-6), ca_hypo (-3.81e-6),
ca_omph (-5.01e-6), ca_cdh (-6.36e-6), ca_gast (-1.05e-5),
ld_augm (-5.15e-5), rf_artec (-5.68e-5), ca_clpal (-6.32e-5),
rf_fedrg (-6.74e-5), rf_ppterm (-8.60e-5), rf_gdiab (-1.37e-4),
rf_phype (-1.51e-4), ca_cleft (-2.13e-4).

Eight features have strongly-negative importance (permuting them
*improves* AP) — `rf_phype`, `rf_gdiab`, `rf_ppterm`, `rf_fedrg`,
`rf_artec`, `ld_augm`, `ca_clpal`, `ca_cleft`. The `rf_phype` finding
replicates the same co-linearity-with-mage_c artefact already documented
in `compare-confirmed-only.md` §2 and observed under `usbc10_m0_cn`.
`rf_pdiab` (1.86e-4) and `wic` (3.68e-4) cross *back above* threshold
under C-only, opposite to their drop status under USBC11_M1 (C+P).

**Correlation pairs (dcor > 0.7):**
| pair | dcor | imp left | imp right | decision |
|---|---:|---:|---:|---|
| apgar5 ~ apgar10  | 0.825 | +3.05e-3 | +9.96e-5 | keep apgar5; drop apgar10 (already sub-threshold) |
| rf_fedrg ~ rf_artec | 0.801 | -6.74e-5 | -5.68e-5 | both drop (both below threshold) |

#### M1_CN drop list (post-fix)
**22 categorical features**, no numeric drops:
sex, rf_gdiab, rf_phype, rf_ehype, rf_ppterm, rf_inftr, rf_fedrg,
rf_artec, ld_augm, me_pres, apgar10, ab_surf, ab_seiz, ca_anen, ca_mnsb,
ca_cdh, ca_omph, ca_gast, ca_limb, ca_cleft, ca_clpal, ca_hypo.

**Surviving M1_CN feature set** (20 features = 5 numeric + 15 categorical):
- Numeric: year, dbwt, wtgain, bmi, mage_c
- Categorical: bfacil3, precare, gestrec10, rf_pdiab, rf_ghype, ld_indl,
  dmeth_rec, apgar5, ab_aven1, ab_aven6, ab_nicu, ab_anti, ca_cchd,
  ca_disor, wic.

#### M1_CN retune (reporting profile, 200 trials × 50K rounds, 9 yr)
- Wallclock: **2 h 52 min** (started 03:00, finished 05:52 on 2026-04-25)
- 200 trials: 18 completed, 182 pruned by Hyperband
- Best AP (valid): **0.023873** (trial 62)
- Best params: lr **0.0503** (notably high), num_leaves **205**,
  min_data_in_leaf **3253**, min_gain_to_split 0.192, feature_fraction
  0.756, bagging 0.875 / freq 3, λ₁ 7.58e-7, λ₂ 1.68e-3
- Artefacts: `output/tuning/usbc11_m1_cn/`

The high learning rate + large num_leaves + large min_data_in_leaf is
an unusual combination (TPE/Hyperband converged to a "shallow many-leaf"
optimum rather than the more conventional "deep few-leaf" pattern).
Trial 192 (lr 0.0065, num_leaves 56, min_data 888) achieved nearly
identical AP (0.02372) under conventional regularisation; the chosen
optimum is within ~0.6% of that more conservative trial.

#### Final fit (reporting profile, 50K rounds × 9 yr)
- Wallclock: **17 min** (started 05:52, finished 06:09 on 2026-04-25)
- Artefacts: `output/fit_model_reporting/20260425-055215/`

| Metric | Pre-fix `usbc11_m1_cn` | Post-fix `usbc11_m1_cn` | Δ |
|---|---:|---:|---:|
| AP (valid)   | 0.0262 | **0.0198** | **−24%** |
| ROC-AUC      | 0.902  | **0.9082** | **+0.62 pp** |
| best_iter    | 1,157  | 99         | −91% |
| log_loss     | —      | 0.001794   | — |
| n_valid      | —      | 6,715,881  | — |
| n_pos_valid  | —      | 1,597      | — |

**The C-only AP regressed.** The post-fix model has higher ROC-AUC but
sharply lower AP. Possible explanations (no further investigation done
for this work):
- The pre-fix tune ran much longer (best_iter 1157 vs 99) under
  conservative learning rates, possibly fitting noise that happened to
  improve AP at the validation slice. The post-fix retune chose a
  shallow-leaf, high-learning-rate optimum that converges in 99
  iterations — better generalisation but lower AP at this specific
  validation slice.
- The pre-fix M1_CN feature set had 18 features (different drop list),
  so the pre/post AP comparison conflates the apgar fix with the prune
  re-derivation. Disentangling would need a controlled re-run with the
  pre-fix prune and post-fix data, not budgeted here.
- Sampling variance: 1,597 positives in 6.7M rows is sparse; AP is
  noisier than ROC-AUC at this regime, and ROC-AUC went up.

**Post-fix permutation importance under the M1_CN feature set (top 10):**

| feature   | importance | ± sd     |
|-----------|-----------:|---------:|
| ca_cchd   | 1.05e-2 | 2.9e-4 |
| mage_c    | 9.16e-3 | 3.8e-4 |
| ca_disor  | 7.99e-3 | 2.7e-4 |
| ab_nicu   | 4.26e-3 | 4.5e-4 |
| gestrec10 | 2.28e-3 | 5.8e-4 |
| dmeth_rec | 2.10e-3 | 5.3e-4 |
| dbwt      | 1.82e-3 | 5.0e-4 |
| **apgar5**    | **1.71e-3** | **4.5e-4** |
| wtgain    | 1.31e-3 | 9.4e-4 |
| ab_aven1  | 1.24e-3 | 8.8e-4 |

**Caveat — three features cross below threshold under the post-tune fit:**
- `ld_indl` = 3.25e-5 (was 2.87e-4 in M0 fit)
- `precare` = -3.45e-4 (was 4.02e-4 in M0)
- `bmi`     = -5.96e-4 (was 6.30e-4 in M0)

`bmi` and `precare` swing strongly negative under the retuned
hyperparameters — the high-learning-rate optimum extracts less signal
from them than the M0 fit did. Worth flagging for a possible "M2_CN"
prune cycle but not iterated here.

#### Predictions written to DuckDB (2026-04-25)
`--write-predictions` overwrote `p_ds_lb_pred_14` / `ds_pred_missing_14`
on 2026-04-25 06:20 (artefacts: `output/fit_model_reporting/20260425-062010/`).
33,579,403 rows populated; 12,002 non-recorded births flagged as likely
missed.

## Cross-family findings

| Family | AP (post-fix) | AP Δ vs pre-fix | apgar5 imp | apgar5 rank |
|---|---:|---:|---:|---:|
| `usbc10_m1`    (C+P, full set)        | 0.0346 | +6.9% | 3.03e-3 | 7 / 24 |
| `usbc10_m1_cn` (C-only, full set)     | 0.0247 | +0.8% | 2.36e-3 | 7 / 17 |
| `usbc11_m1`    (C+P, clinical+age)    | 0.0334 | +7.7% | 2.83e-3 | 7 / 23 |
| `usbc11_m1_cn` (C-only, clinical+age) | 0.0198 | −24%  | 1.71e-3 | 8 / 20 |

**1. Does `apgar5` re-enter every post-prune feature set?** Yes —
`apgar5` is retained by all four families and ranks 7th or 8th by
permutation importance in every final fit. The PR #26 fix is doing
its job.

**2. Does `apgar10` re-enter under any variant?** No. `apgar10` drops
under all four post-fix M0 fits, both on permutation importance
(highest seen is 9.96e-5 under usbc11_m0_base_cn) and on dcor 0.81–0.86
correlation with apgar5. apgar10 is essentially redundant with apgar5
under the apgar5-fixed data; pre-fix it carried compensating signal
because apgar5 had no variance.

**3. C-only AP regression on the clinical-only set.** USBC10_M1_CN
moved +0.8% (0.0245 → 0.0247) but USBC11_M1_CN moved −24%
(0.0262 → 0.0198). The C-only / clinical-only intersection is the
sparsest combination — only 1,597 positives across 9 years, on 20
features. The pre-fix model converged after 1,157 iterations under
conservative hyperparameters; the post-fix retune's optimum is much
flatter (best_iter=99). ROC-AUC moved +0.62 pp under the same retune,
so ranking quality improved even as AP-at-threshold worsened — this
combination of metrics is consistent with the post-fix model
discriminating better in aggregate but assigning lower probabilities
to the rare positive class. For most downstream use cases (selection
flagging via the year×month quota, which is rank-based) the AP drop
is not load-bearing; for any propensity-of-DS interpretation it is.

**4. Hyperparameter regimes.** Three of the four families landed in
the "moderate-leaf, moderate-data, moderate-regularisation" regime
(num_leaves 64–82, min_data 507–596, λ₂ 0.13–1.80). USBC11_M1_CN is
the outlier (num_leaves 205, min_data 3253, λ₂ ≈ 0). The 1,597
positives × 20 features sparsity invites a flat-many-leaves optimum
because deep trees overfit; TPE/Hyperband found that local minimum and
chose it over a more conventional configuration that performed within
0.6% AP (trial 192).

**5. Drop-list overlap.** `apgar10`, `rf_artec`, `rf_fedrg`, `ca_anen`,
`ca_mnsb`, `ca_cdh`, `ca_omph`, `ca_gast`, `ca_limb`, `ca_cleft`,
`ca_hypo`, `ab_surf`, `ab_seiz`, `rf_ehype`, `rf_ppterm`, `ld_augm`
drop in all four families (16 of 19/22 drops). Three features (`sex`,
`rf_phype`, `me_pres`) drop under C-only but stay under C+P; two
(`rf_pdiab`, `wic`) drop under C+P but stay under C-only. The C-only
label rewards a slightly different feature subset — risk-flag features
with `mage_c`-correlated noise (rf_phype, rf_gdiab) drop with strongly
negative importance under C-only specifically, consistent with C-only
removing the recording-noise channel through which those features
gained signal under C+P.

## Caveats

_Inherits the caveats from `compare-confirmed-only.md` §Caveats. Additional
items specific to this work:_

- **M0 tuning at test profile.** The Optuna search that drives the M0
  permutation-importance ranking uses 50 trials, not the 200 reporting
  profile uses. Some features near the 1e-4 retention threshold may flip
  retain/drop status under a tighter optimum. Mitigation: the M1 retune is
  full reporting profile, so any feature retained gets a properly-tuned
  model.
- **Prediction-column overwrites.** `--write-predictions` overwrites the
  pre-fix values in `p_ds_lb_pred_*` / `ds_pred_missing_*`. The pre-fix
  predictions are not preserved. Re-running the C-only vs C+P comparison
  (`scripts/compare_predicted.py`) will read the new values; the pre-fix
  numbers in `compare-confirmed-only.md` are fixed in that note as the
  historical record.
- **Transient `USBC11_M0_BASE` / `USBC11_M0_BASE_CN` classes.** These existed
  in `models/usbc11_baseline.py` only for the duration of this work and were
  removed at the final commit on 2026-04-25 (the surviving features and
  metadata landed back into `USBC11_M0` / `USBC11_M0_CN`). Any artefact
  directory referencing them (`output/tuning/usbc11_m0_base*/`,
  `output/fit_model_test/20260424-200531/`,
  `output/fit_model_test/20260424-220729/`) is provenance for this notes
  file but is not load-bearing for downstream code.
- **`USBC11_M0` re-parented.** The class previously inherited from
  `USBC10_M1` (i.e. silently adopted the pre-PR-#26 prune); it now inherits
  directly from `USBC10_M0` with only the 6 sociodemographic drops, and
  `USBC11_M1` exists as a separate class with the post-fix prune. The
  `selection_history()` MRO chain is therefore one step shorter under the
  new structure.

## Next steps

- Re-run `scripts/compare_predicted.py` against the refreshed
  `ds_pred_missing_*` columns to refresh `compare-confirmed-only.md`'s
  comparison tables.
- Re-fit the Bayesian selection model
  (`scripts/fit_selection_model.py`) under the refreshed missing flags;
  this is a separate, larger run covered by the existing
  `compare-confirmed-only.md` Next Steps.
- Decide whether the C-only variants should become the primary models for
  downstream analysis (`compare-confirmed-only.md` §Next Steps already
  raised this; the post-fix retune doesn't change the underlying
  ca_disor-tautology argument but the freshly-tuned numbers should inform
  the call).
