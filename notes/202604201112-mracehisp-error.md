# `mracehisp` artefact in the predictor analysis — scope mismatch, fix, and follow-up

> [!WARNING]
> This note was drafted by an AI coding assistant (Claude, Opus 4.7).
> Numbers were pulled from the live DuckDB on 2026-04-20; SQL is
> reproducible from the snippets in the appendix. Treat the analysis
> and the proposed plan as a draft for human review.

## Summary

The `analyse_predicted` report at
`output/analyse_predicted/20260420-095418/` showed `mracehisp = 6`
("Non-Hispanic more than one race") accounting for **28.93 % of recorded
DS births and 31.04 % of predicted-missing DS births** within the
report's coverage window. The NCHS published share for mothers
reporting more than one race in 2023 is ≈ 3 %, so the figure was
implausible by an order of magnitude.

**The figure was an artefact of a scope mismatch, not a feature bug.**
Three independent things had to be true for it to appear:

1. The gradient-boosting models (`usbc09`/`usbc10`/`usbc11`) train on
   the post-2014-revision window only — `ModelDefinition.year_range =
   (2016, 2024)` (`src/dspopulations_us_birth_certificates/models/base_model.py:61`).
   Within that window the raw NCHS `mracehisp` and `fracehisp` columns
   carry a single, consistent coding scheme.
2. The raw NCHS `mracehisp` column reuses the same integer codes with
   different meanings on either side of the 2014 revision —
   pre-2014 `6 = Non-Hispanic White`, from 2014 on
   `6 = Non-Hispanic more than one race`. Same break for `fracehisp`.
3. `scripts/analyse_predicted.py` defaulted `--years` to `None`,
   restricting only to `p_ds_lb_pred_01 IS NOT NULL`. That column had
   stale predictions populated for 2005–2024 (left over from an earlier
   model variant whose `year_range` extended below 2016 — the current
   `write_predictions_to_duckdb` updates only the rows it scored, so
   the older predictions were never cleared). The report therefore
   pulled in 2005–2015 rows where `mracehisp = 6` means *Non-Hispanic
   White*, and labelled them under the 2014+ scheme as *Non-Hispanic
   more than one race*.

So the model was always trained on a clean schema; the **report's row
scope was wider than the model's training scope**, and the resulting
regime collision produced the headline artefact.

## Evidence

### `mracehisp = 6` jump across the 2014 boundary (all births)

A coding break, not a demographic shift:

| Year | rows with `mracehisp = 6` |
|-----:|--------------------------:|
| 2010 | 2,163,096 |
| 2012 | 2,134,726 |
| 2013 | 2,130,029 |
| 2014 |    75,305 |
| 2015 |    77,914 |

### Coverage of the prediction columns in the live DuckDB (pre-fix)

| Column | Years populated | Source |
|---|---|---|
| `p_ds_lb_pred_01` | **2005–2024** (≈99 % per year) | usbc10 + ghosts from a previous broader-window run |
| `p_ds_lb_pred_02` | 2016–2024 only | usbc11 (training window respected) |
| `ds_pred_missing` | derived from `p_ds_lb_pred_01` quota → flagged across **2005–2024** | inherits the wider scope |

### DS-recorded births by `mracehisp` × year (illustrative)

Within the prediction-coverage rows the report was reading:

|   Year |   `mracehisp = 1` recorded |   `mracehisp = 6` recorded |
|-------:|---------------------------:|---------------------------:|
| 2005   |                        330 |                      1,258 |
| 2010   |                        316 |                      1,167 |
| 2013   |                        318 |                      1,176 |
| 2014   |                      1,182 |                         36 |
| 2020   |                      1,045 |                         43 |
| 2024   |                        963 |                         51 |

Restricting the same query to 2016–2024 collapses the artefact and
returns plausible shares: NH White 58.4 %, Hispanic 25.8 %,
NH Black 10.6 %, NH Asian 4.1 %, NH AIAN/NHOPI/multi-race together
≈ 1 %.

## Fix applied

Five-step change, applied on the `dev/model-usbc11` branch:

1. **`scripts/analyse_predicted.py`** — `--years` default changed from
   `None` to `"2016-2024"` so the report's row scope matches the model
   training window. The flag remains available for one-off overrides
   (e.g. when the longitudinal Bayesian work plotted in
   `plans/readme.md` pulls in the full 1989–2024 series).
2. **`src/dspopulations_us_birth_certificates/predicted_analyses.py`**
   — reverted the earlier exploratory change that had swapped the raw
   `mracehisp` for a harmonised `mracehisp_c`. With the year scope
   pinned to 2016+, the raw column is the more granular axis (the
   2014+ scheme keeps NH Asian only / NH NHOPI only / NH more than one
   race as separate buckets that `mracehisp_c` would collapse). Same
   for `fracehisp`. The `CategoryGrouping.min_year` field added in the
   exploratory pass was dropped.
3. **One-shot DuckDB cleanup** — NULLed stale `p_ds_lb_pred_01` and
   `ds_pred_missing` for `year < 2016` so the existing `usbc10`
   prediction column reflects only that model's actual training window.
   Cleared 51,558,420 stale predictions and 35,925 stale missing-flags
   (recorded counts logged inline in the run; reproducible via the SQL
   in the appendix). 2016+ counts unchanged (verified row-by-row
   before/after).
4. **`docs/analysis/predicted.qmd`** — `mracehisp` and `fracehisp`
   sections restored to use the raw columns with a brief preamble
   explaining the 2014-revision coding caveat and pointing to this
   note. The "key concern" callout reframed in cautious language —
   the actual direction of the recorded vs predicted shift needs to
   be re-read off the regenerated chart, not asserted in advance.
5. **Re-render** — `python scripts/analyse_predicted.py --render` to
   regenerate the artefacts under the corrected scope.

## What this fix does *not* do

- **It does not retrain the gradient-boosting models.** `usbc10` and
  `usbc11` were already training on a single-schema window, so they
  are unchanged. The new `p_ds_lb_pred_01` coverage matches their
  training year-range; old ghost rows are gone.
- **It does not build a `mracehisp_c` / `fracehisp_c` recode.** Those
  remain useful future work for the longitudinal aim 1 in
  `plans/readme.md` ("characterise DS births from 1989 to 2024"),
  where any analysis that crosses the 2014 boundary will hit the same
  schema collision. Sketch retained below.
- **It does not change the LightGBM feature set** — `mracehisp` and
  `fracehisp` remain in `CATEGORICAL_BASE`. They are safe to keep
  there as long as the model's `year_range` is post-2014.

## Future work — harmonised race/ethnicity recodes

Strictly optional for the current 2016–2024 work, but needed before the
long-horizon analyses can use parental race/ethnicity coherently. The
project already builds a maternal recode (`mracehisp_c`) in
`scripts/duckdb_prepare.py:359-435`; a paired paternal recode is
straightforward.

### Build `fracehisp_c`

Mirror the existing maternal pipeline (`scripts/duckdb_prepare.py:359-435`):

1. Add `FRACE_C`, `FHISP_C`, `FRACEHISP_C` enum entries to `Variables`
   (`src/dspopulations_us_birth_certificates/variables.py`) and dtypes
   to the `COMPUTED` dict.
2. Three `UPDATE us_births SET ... = CASE ... END` statements in
   `duckdb_prepare.py`:
    - **`frace_c`** — bridge `frace15` → `fracerec` → `fbrace` → `frace`
      to 1=White / 2=Black / 3=AIAN / 4=Asian-PI, parallel to the
      `mrace_c` build. The collapsed Asian/PI bucket lines up with
      `mrace_c` so maternal/paternal axes are comparable.
    - **`fhisp_c`** — bridge `fhisp_r` → `fhispx` → `ufhisp` → `orracef`
      with the same Mexican / Puerto Rican / Cuban / Other-Hispanic /
      Unknown / NH mapping as `mhisp_c`. ORRACEF needs the ORRACEM-style
      special-casing (codes 6–8 are non-Hispanic).
    - **`fracehisp_c`** — combine via the `mracehisp_c` template:
      `WHEN fhisp_c BETWEEN 1 AND 4 THEN 5 / WHEN fhisp_c = 5 THEN NULL
      / ELSE frace_c`. Decide explicitly whether to keep an extra
      "father absent / unknown ethnicity" bucket — the raw `fracehisp`
      has a code-9 bucket for that case (≈ 12.9 % of 2014+ births in
      this dataset) with no maternal analogue. Recommendation: emit
      `NULL`, on the same grounds as `mracehisp_c`, and let downstream
      models handle missingness explicitly.
3. Add a smoke test (`tests/` does not yet exist — create it as part
   of this work) that asserts post-2014 `fracehisp_c` shares are within
   ~1 pp of post-2014 raw `fracehisp` shares (after collapsing 4+5 →
   4), and that pre-2014 `fracehisp_c` shares look like a reasonable
   demographic distribution rather than the regime-mixed artefact.
4. Re-run `scripts/duckdb_prepare.py` to populate the new columns.

### Audit of other raw NCHS columns with regime breaks

Quick sweep template — large step changes at certificate-revision
boundaries (1989, 2003/2004, 2013/2014) are the symptom:

```sql
SELECT year, COUNT(*) AS n
FROM   us_births
WHERE  <col> = <code-of-interest>
GROUP  BY year
ORDER  BY year;
```

Columns to spot-check: `mhispx`, `mhisp_r`, `umhisp`, `orracem` and
their paternal equivalents (already feeding into `*_c` recodes — but
worth confirming no analysis uses them raw); `meduc` (covers two cert
revisions but the codes are reasonably consistent); `mracerec` /
analogues (already correctly referenced only inside `mrace_c`
construction).

### When the longitudinal work begins

Two retrofit steps:

- Update `CATEGORICAL_BASE` in `variables.py:614-657`:
  `Variables.MRACEHISP` → `Variables.MRACEHISP_C`, ditto paternal.
- Audit notebook references: `notebooks/00013-predictions-2.py`,
  `notebooks/0001-variable-selection.py`,
  `notebooks/0011-model-predictions-1.py`,
  `notebooks/notes-race-ethnicity.py`.

Models can then be refit on a wider `year_range` without inheriting
the regime collision through their feature set. None of this is
required while the modelling stays inside the 2016+ window.

## Appendix — reproduction queries

Run against `data/us_births.db`. The first three are evidence; the
last two are the cleanup applied in step 3 of the fix.

```sql
-- code-6 jump across the 2014 boundary (all births)
SELECT year, COUNT(*) AS n
FROM   us_births
WHERE  mracehisp = 6
GROUP  BY year
ORDER  BY year;

-- prediction-column coverage by year (pre-fix shape: stale 2005-2015
-- entries in p_ds_lb_pred_01)
SELECT year,
       SUM(CASE WHEN p_ds_lb_pred_01 IS NOT NULL THEN 1 ELSE 0 END)
           AS n_pred_01,
       SUM(CASE WHEN p_ds_lb_pred_02 IS NOT NULL THEN 1 ELSE 0 END)
           AS n_pred_02,
       SUM(CASE WHEN ds_pred_missing THEN 1 ELSE 0 END) AS n_flagged
FROM   us_births
GROUP  BY year
ORDER  BY year;

-- recorded DS births within the corrected 2016-2024 scope
SELECT mracehisp, COUNT(*) AS n,
       ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS pct
FROM   us_births
WHERE  year BETWEEN 2016 AND 2024
   AND mracehisp IS NOT NULL
   AND down_ind = 1
GROUP  BY mracehisp
ORDER  BY mracehisp;

-- the cleanup applied in step 3 (run once, not idempotent in spirit)
UPDATE us_births SET p_ds_lb_pred_01 = NULL  WHERE year < 2016;
UPDATE us_births SET ds_pred_missing = FALSE WHERE year < 2016;
```
