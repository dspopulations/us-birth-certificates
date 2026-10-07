> [!NOTE]
> AI-assisted revision by Codex (GPT-6), 7 October 2026.

# Retuning classifiers after the Apgar-score fix

**Original date:** 22 to 25 April 2026. Historical research note, revised for clarity in October 2026.

PR [#26](https://github.com/dspopulations/us-birth-certificates/pull/26) corrected an Apgar filter that retained only score 10 and set scores 0 to 9 to NULL. Classifiers trained on the earlier data needed new tuning and feature selection.

The work repeated tuning, fitting, permutation checks, feature selection and a final fit for the four classifier families. The exploratory pass used 50 trials; the final pass used 200. All used 2016 to 2024 and seed 47. These are historical settings, not current profile defaults.

## Main comparisons

| Metric | Pre-fix `usbc10_m1` | Post-fix `usbc10_m1` | Δ |
|---|---:|---:|---:|
| AP (valid)   | 0.0324 | **0.0346** | **+6.9%** |
| ROC-AUC      | 0.889  | **0.8955** | **+0.7 pp** |
| best_iter    | 158    | 556        | +252% |
| log_loss     |,      | 0.003565   |, |
| n_valid      |,      | 6,717,846  |, |
| n_pos_valid  |,      | 3,562      |, |

| Metric | Pre-fix `usbc10_m1_cn` | Post-fix `usbc10_m1_cn` | Δ |
|---|---:|---:|---:|
| AP (valid)   | 0.02451 | **0.02470** | **+0.8%** |
| ROC-AUC      | 0.9081  | **0.9097**  | **+0.16 pp** |
| best_iter    | 282     | 89          | −68% |
| log_loss     | 0.001766 | 0.001776   | +0.6% |
| n_valid      | 6,715,881 | 6,715,881 |, |
| n_pos_valid  | 1,597   | 1,597       |, |

| Metric | Pre-fix `usbc11_m0` | Post-fix `usbc11_m1` | Δ |
|---|---:|---:|---:|
| AP (valid)   | 0.0310 | **0.0334** | **+7.7%** |
| ROC-AUC      | 0.883  | **0.8916** | **+0.86 pp** |
| best_iter    | 570    | 948        | +66% |
| log_loss     |,      | 0.003576   |, |
| n_valid      |,      | 6,717,846  |, |
| n_pos_valid  |,      | 3,562      |, |

| Metric | Pre-fix `usbc11_m1_cn` | Post-fix `usbc11_m1_cn` | Δ |
|---|---:|---:|---:|
| AP (valid)   | 0.0262 | **0.0198** | **−24%** |
| ROC-AUC      | 0.902  | **0.9082** | **+0.62 pp** |
| best_iter    | 1,157  | 99         | −91% |
| log_loss     |,      | 0.001762   |, |
| n_valid      |,      | 6,715,881  |, |
| n_pos_valid  |,      | 1,597      |, |

| Family | AP (post-fix) | AP Δ vs pre-fix | apgar5 imp | apgar5 rank |
|---|---:|---:|---:|---:|
| `usbc10_m1`    (C+P, full set)        | 0.0346 | +6.9% | 3.03e-3 | 7 / 24 |
| `usbc10_m1_cn` (C-only, full set)     | 0.0247 | +0.8% | 2.36e-3 | 7 / 17 |
| `usbc11_m1`    (C+P, clinical+age)    | 0.0334 | +7.7% | 2.83e-3 | 7 / 23 |
| `usbc11_m1_cn` (C-only, clinical+age) | 0.0198 | −24%  | 1.71e-3 | 8 / 20 |

AP is average precision against the certificate label. It measures ranking of recorded cases. It does not measure sensitivity for true Down syndrome among births with no record of the condition.

Five-minute Apgar became a retained predictor in all four final fits. Ten-minute Apgar remained excluded. The comparisons support refitting after a preparation change. They do not establish that Apgar caused the changes in other predictors' importance, since feature sets and tuned parameters also changed.

## Database writes

The note recorded new classifier scores and quota flags in April. The C+P classifiers each selected 26,742 unrecorded births; `usbc10_cn` selected 12,002 and `usbc11_cn` selected 12,002. These counts come from quotas and are not counts of verified missed cases. Database columns can have been replaced since then.

The four score columns were `p_ds_lb_pred_01`, `_02`, `_13` and `_14`. The companion flags were `ds_pred_missing`, `ds_pred_missing_02`, `ds_pred_missing_13` and `ds_pred_missing_14`. Use each model class to check current column names.

The [confirmed-only comparison](20260422-compare-confirmed-only.md) preserves the pre-fix comparison. The [workflow guide](../docs/modelling-workflow.md) gives current commands and paths. The detailed trial logs and scratch plans have been removed from this note because they are not needed to repeat the current workflow.
