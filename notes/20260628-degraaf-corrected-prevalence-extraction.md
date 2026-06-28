# De Graaf corrected surveillance prevalence & recording fractions by ethnicity (2000–2024)

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 4.8).

> [!WARNING]
> Work in progress. All data and models are preliminary.

**Date:** 2026-06-28

## Source

Gert de Graaf supplied a corrected workbook of surveillance-based Down syndrome
prevalence and birth-certificate recording, by ethnic group, 2000–2024:

`data/new/cor verwissel jaren overzicht prevalencties races usa birth cert vanaf 2000 ALT3.xlsx`

His key fix in this version: **the 2002 and 2003 birth-certificate figures, previously
swapped, are now correct.** The swap was in the birth-certificate columns only.

### Decoding the workbook (one sheet, per ethnic group × year)

Five groups — nhw, nhb, his, as/pi, ai/an (`mracehisp_c` 1, 2, 5, 4, 3). Per Gert's note:

| Sheet col | Meaning | Coverage |
| --- | --- | --- |
| C, D | recorded DS count, total births (birth certificates) | 2000–2024 |
| **E** | birth-certificate DS prevalence /10k (`= C/D × 10⁴`) | 2000–2024 |
| Q | column E as a 5-year running average | 2000–2024 |
| **R** (= L) | **surveillance-programme** prevalence /10k, 5-year running | **2000–2014, 2016, 2018 only** |
| Q/R (col U) | "percentage reported" — birth-cert ÷ surveillance (the recording fraction) | observed years |
| **G** | recording fraction with gaps filled by a per-race **linear regression** | 2000–2024 |
| H, **I** | estimated **true** count, true prevalence /10k (`= C/G`, then `/D × 10⁴`) | 2000–2024 |

The 2002/2003 correction flows from C/D/E into Q, U, G, H and I. The surveillance input
R/L is independent of the birth-certificate data and is **unchanged** by the fix.

Per-race recording-fraction regression lines (`G = intercept + slope · yr_idx`, `yr_idx`
= year − 2000), read from the workbook's chart formulas:

| group | intercept (2000) | slope / yr |
| --- | --- | --- |
| nhw | 0.421962 | 0.001692 |
| nhb | 0.220699 | 0.005151 |
| his | 0.316339 | 0.001981 |
| as/pi | 0.320411 | 0.001837 |
| ai/an | 0.335634 | 0.009190 |

## Reconciliation with our existing CSVs — no correction required

`data/us-births-estimated-prevalence-ethnicity-2000-2018.csv` (`year, mracehisp_c,
prevalence`) holds the **surveillance** prevalence (workbook column R/L). Cell-by-cell diff
against the corrected workbook: **exact match for all 85 overlapping cells (max abs diff
5 × 10⁻⁹)**, including 2002 and 2003; 2015/2017 blank in both. Because the swap did not
touch the surveillance column, our values were already correct and remain so.

The other lookup CSVs are not in this workbook and cannot be updated from it:

- `us-births-surveillance-prevalence-1989-2024.csv` (overall `p_ds_lb_wt`) is a separate
  national series spanning 1989+; it is **not** a births-weighted aggregate of the
  by-ethnicity surveillance values (differs ~2–4 %).
- `us-births-reduction-rates-1989-2024.csv` is overall elective-termination reduction.
- `us-births-estimated-prevalence-maternal-age-1989-2018.csv` is the Morris age model.

The recording-rate pipeline (`scripts/derive_recording_rates.py`, on the eta-reanchor
branch) consumes **only** surveillance prevalence and re-derives recording rates from our
own microdata with its own backtested imputation. It is therefore unaffected by the fix,
and it deliberately does **not** use Gert's regression-filled estimates (columns G / I).

## What we captured — `data/us-births-degraaf-prevalence-recording-2000-2024.csv`

A faithful, full-precision extraction of the corrected workbook (125 rows = 25 years ×
5 groups), so Gert's corrected work is captured and reproducible without the xlsx. Columns:

| column | source col | notes |
| --- | --- | --- |
| `year`, `race`, `mracehisp_c` | A, B | keys; `mracehisp_c` = our code (1 nhw, 2 nhb, 3 ai/an, 4 as/pi, 5 his) |
| `recorded_bc`, `births_bc` | C, D | birth-certificate recorded DS, total births (corrected) |
| `bc_prev_per10k` | E | birth-certificate DS prevalence /10k |
| `recording_frac_g` | G | recording fraction, gaps regression-filled (all years) |
| `est_true_count`, `est_true_prev_per10k` | H, I | Gert's estimated **true** count / prevalence /10k (all years) |
| `surveillance_prev_per10k` | R/L | surveillance prevalence /10k; **blank** for 2015, 2017, 2019–2024 |

This is **reference data, not a Stage-5 pipeline input** — it is not read by
`scripts/duckdb_prepare.py`. The duplicated `surveillance_prev_per10k` is the same series
already committed (and verified above) in `us-births-estimated-prevalence-ethnicity-2000-2018.csv`.

### Licensing / DUA

This is de Graaf's **published** surveillance plus highly-aggregated counts (5 groups ×
year), not NCHS restricted microdata, so it is not DUA-restricted — the same provenance and
reasoning as the existing committed `us-births-estimated-prevalence-ethnicity-2000-2018.csv`
and the [recording-anchor note](20260623-degraaf-recording-anchor.md). Confirm with Frank
before any external publication.

## Not done here (open modelling decisions)

- **Adopting Gert's estimated-true prevalence (column I) for 2015/2017/2019–2024.** His
  regression fill is an alternative to our backtested survival-ratio imputation in
  `derive_recording_rates.py`. Whether to switch is the open "how hard to push de Graaf"
  decision in the [recording-anchor note](20260623-degraaf-recording-anchor.md), not a data fix.
- **Extending the surveillance comparison past 2018.** Raw surveillance still stops at 2018
  (2015/2017 absent); only the regression-filled estimate covers 2019–2024.
