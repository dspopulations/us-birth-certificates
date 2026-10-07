> [!NOTE]
> AI-assisted revision by Codex (GPT-6), 7 October 2026.

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 4.8).

# Keeping multi-race separate from Unknown

**Original date:** 28 June 2026. Historical research note, revised for clarity in October 2026.

Multi-race is a reported category, not a missing value. The selection cells now retain seven race/Hispanic-origin groups.

| Index | Group |
| ---: | --- |
| 0 | Non-Hispanic White |
| 1 | Non-Hispanic Black |
| 2 | Non-Hispanic American Indian or Alaska Native |
| 3 | Non-Hispanic Asian or Pacific Islander |
| 4 | Hispanic |
| 5 | Unknown |
| 6 | Non-Hispanic multi-race |

The original proposal placed the two final groups in a different order. The table above follows the implemented `RACE_LEVELS` order. Read old six-group artefacts using their own manifest, and refit if a report needs the current groups.

The de Graaf source used by the anchor has no matching multi-race or Unknown prevalence series. Both groups receive a weak fallback recording prior, with no prevalence-margin target. Adding a distinct model group preserves its births and counts; it does not provide external evidence for its recording rate.

The combined database code and the model index are separate encodings. See [the preparation guide](../docs/data-preparation.md) and [selection cells](../src/dspopulations_us_birth_certificates/selection/data.py). Do not use a model index as an NVSS code.
