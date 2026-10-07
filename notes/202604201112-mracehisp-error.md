> [!NOTE]
> AI-assisted revision by Codex (GPT-6), 7 October 2026.

> [!WARNING]
> This note was drafted by an AI coding assistant (Claude, Opus 4.7).
> Numbers were pulled from the live DuckDB on 2026-04-20; SQL is
> reproducible from the snippets in the appendix. Treat the analysis
> and the proposed plan as a draft for human review.

# Audit of the combined race and Hispanic-origin code

**Original date:** 20 April 2026. Historical research note, revised for clarity in October 2026.

The April audit found that `mracehisp = 6` had been treated as a residual or unknown group. In the source coding it represents non-Hispanic people of more than one race. The original pipeline also used race and Hispanic-origin fields whose names and code meanings changed across years.

## Audit evidence

| Year | rows with `mracehisp = 6` |
|-----:|--------------------------:|
| 2010 | 2,163,096 |
| 2012 | 2,134,726 |
| 2013 | 2,130,029 |
| 2014 |    75,305 |
| 2015 |    77,914 |

| Column | Years populated | Source |
|---|---|---|
| `p_ds_lb_pred_01` | **2005–2024** (≈99 % per year) | usbc10 + ghosts from a previous broader-window run |
| `p_ds_lb_pred_02` | 2016–2024 only | usbc11 (training window respected) |
| `ds_pred_missing` | derived from `p_ds_lb_pred_01` quota → flagged across **2005–2024** | inherits the wider scope |

|   Year |   `mracehisp = 1` recorded |   `mracehisp = 6` recorded |
|-------:|---------------------------:|---------------------------:|
| 2005   |                        330 |                      1,258 |
| 2010   |                        316 |                      1,167 |
| 2013   |                        318 |                      1,176 |
| 2014   |                      1,182 |                         36 |
| 2020   |                      1,045 |                         43 |
| 2024   |                        963 |                         51 |

These counts describe the April database. They are not a current validation of the prepared data.

## Current resolution

`scripts/duckdb_prepare.py` derives `mrace_c`, `mhisp_c` and `mracehisp_c`. The last variable keeps multi-race as its own category. Unknown values remain missing in the database; the selection cell builder gives them a separate category.

The selection model uses Unknown at index 5 and multi-race at index 6. See the [group-scoping note](20260628-multi-race-selection-group-scoping.md) and [preparation guide](../docs/data-preparation.md). Old six-category model artefacts cannot be read with the current seven-category labels.

Check each source field against the guide for its year. A field name alone does not show that its categories have the same meaning across years. The [historical coding reference](../previous/us-birth-certificates/data-preparation.md) preserves the source-code lists.
