> [!NOTE]
> AI-assisted revision by Codex (GPT-6), 7 October 2026.

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 4.8).

# Correcting the de Graaf prevalence extraction

**Original date:** 28 June 2026. Historical research note, revised for clarity in October 2026.

The corrected extraction distinguished observed surveillance prevalence from a workbook prediction and its correction factors. Earlier anchor estimates used the wrong series. This note preserves the source-column mapping and the numerical comparison.

| Sheet col | Meaning | Coverage |
| --- | --- | --- |
| C, D | recorded DS count, total births (birth certificates) | 2000–2024 |
| **E** | birth-certificate DS prevalence /10k (`= C/D × 10⁴`) | 2000–2024 |
| Q | column E as a 5-year running average | 2000–2024 |
| **R** (= L) | **surveillance-programme** prevalence /10k, 5-year running | **2000–2014, 2016, 2018 only** |
| Q/R (col U) | "percentage reported", birth-cert ÷ surveillance (the recording fraction) | observed years |
| **G** | recording fraction with gaps filled by a per-race **linear regression** | 2000–2024 |
| H, **I** | estimated **true** count, true prevalence /10k (`= C/G`, then `/D × 10⁴`) | 2000–2024 |

| group | intercept (2000) | slope / yr |
| --- | --- | --- |
| nhw | 0.421962 | 0.001692 |
| nhb | 0.220699 | 0.005151 |
| his | 0.316339 | 0.001981 |
| as/pi | 0.320411 | 0.001837 |
| ai/an | 0.335634 | 0.009190 |

| column | source col | notes |
| --- | --- | --- |
| `year`, `race`, `mracehisp_c` | A, B | keys; `mracehisp_c` = our code (1 nhw, 2 nhb, 3 ai/an, 4 as/pi, 5 his) |
| `recorded_bc`, `births_bc` | C, D | birth-certificate recorded DS, total births (corrected) |
| `bc_prev_per10k` | E | birth-certificate DS prevalence /10k |
| `recording_frac_g` | G | recording fraction, gaps regression-filled (all years) |
| `est_true_count`, `est_true_prev_per10k` | H, I | Gert's estimated **true** count / prevalence /10k (all years) |
| `surveillance_prev_per10k` | R/L | surveillance prevalence /10k; **blank** for 2015, 2017, 2019–2024 |

| group | 2024 ours | 2024 Gert (col I) |
| --- | --- | --- |
| NH White | 14.13 | 11.67 |
| NH Black | 16.27 | 12.69 |
| NH Asian/PI | 10.38 | 7.39 |
| Hispanic | 17.44 | 15.74 |

| anchor | total true DS 2016–2024 | 95% CI |
| --- | --- | --- |
| production (`--anchor-margin`) | 40,637 | 39,138–42,205 |
| Gert col-I tail (`--degraaf-tail`) | 40,041 | 38,437–41,607 |

The file's year labels are not sufficient to define an annual observation. The [August workbook audit](20260803-degraaf-surveillance-workbook-extraction.md) established that the surveillance values represent centred five-year windows. That later audit supersedes annual interpretations of these rows.

Keep source prevalence, projected prevalence and correction factors separate in any extraction. Check reconstruction against the workbook before fitting. A matched projection does not make the source an independent annual estimate.

The original before/after model totals describe development fits. They are not a validation of national prevalence or a reason to prefer one degree of prior tightness.
