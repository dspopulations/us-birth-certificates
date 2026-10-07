> [!NOTE]
> AI-assisted revision by Codex (GPT-6), 7 October 2026.

# Confirmed-only and confirmed-or-pending classifier labels

**Original date:** 22 April 2026. Historical research note, revised for clarity in October 2026.

This compares four classifiers before the Apgar fix. The [retuning note](20260422-1132-retune-afterapgar-fix.md) supersedes its performance estimates. The label comparison remains relevant.

Confirmed-only models count `C` as a positive label. Confirmed-or-pending models count `C` and `P`. A pending karyotype is not a verified false positive. A confirmed entry is not an independent clinical validation of the certificate.

## Original comparison

| Model | Features | Label | n_leaves (tuned) | best_iter | AP (valid) | ROC-AUC |
|---|---|---|---|---|---|---|
| `usbc10_m1`   | full, post-prune (28) | C+P → 1 | 180 | 158 | 0.0324 | 0.889 |
| `usbc10_m1_cn`| full, re-pruned under C-only (26) | C → 1, P dropped | 116 | 282 | 0.0245 | **0.908** |
| `usbc11_m0`   | clinical + mage (24) | C+P → 1 | 180 | 570 | 0.0310 | 0.883 |
| `usbc11_m1_cn`| clinical + mage, re-pruned under C-only (18) | C → 1, P dropped | 41  | 1157 | 0.0262 | **0.902** |

| Variant | `predictions` column | `missing` flag column | `missing` TRUE |
|---|---|---|---|
| usbc10_m1    | `p_ds_lb_pred_01` | `ds_pred_missing`    | 26,742 |
| usbc10_m1_cn | `p_ds_lb_pred_13` | `ds_pred_missing_13` | 12,002 |
| usbc11_m0    | `p_ds_lb_pred_02` | `ds_pred_missing_02` | 26,742 |
| usbc11_m1_cn | `p_ds_lb_pred_14` | `ds_pred_missing_14` | 12,002 |

| Comparison | `ca_disor = Pending` in predicted-missing | Δ vs left |
|---|---|---|
| `usbc10_m1` (C+P) | 29.4% |, |
| `usbc10_m1_cn` (C-only) | 2.3% | **−27.2 pp** |
| `usbc11_m0` (C+P, clinical) | 25.6% |, |
| `usbc11_m1_cn` (C-only, clinical) | 2.7% | **−22.9 pp** |

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

| Variable | Δ (C-only − C+P), usbc10 | Δ, usbc11 | Recorded % |
|---|---|---|---|
| `pay_rec = Medicaid` | −3.3 pp | −2.1 pp | 41.8% |
| `pay_rec = Private`  | +4.5 pp | +2.6 pp | 47.8% |
| `meduc ≤ HS`         | −4.6 pp | −2.6 pp | 39.0% |
| `meduc ≥ Bachelor`   | **+4.1 pp** | +2.4 pp | 32.4% |

| Variable | Δ (C-only − C+P), usbc10 | Δ, usbc11 | Δ (usbc11 − usbc10), C-only | Recorded % |
|---|---|---|---|---|
| NH White   | +2.0 pp | +0.9 pp | −2.3 pp | 52.4% |
| NH Black   | −0.2 pp | −0.2 pp | **+4.1 pp** | 11.9% |
| NH Asian   | −0.5 pp | +0.0 pp | **+3.3 pp** | 3.5%  |
| Hispanic   | −1.2 pp | −0.8 pp | **−5.7 pp** | 27.9% |

Here, "predicted-missing" is the historical name for a classifier-selected cohort. The quota sets its size. Its composition therefore cannot validate the number of missed cases.

The label changes both the positive class and the ranking target. Scores from these models need not be comparable on the same scale. Cohort differences also reflect separate feature selection and tuning. Interpret them as sensitivity to the analysis definition, rather than as evidence about which cohort contains true missed cases.
