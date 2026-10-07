> [!NOTE]
> AI-assisted revision by Codex (GPT-6).

# Study aims and limits

> [!WARNING]
> This exploratory study is in progress. Estimates remain conditional on the data sources and model assumptions.

The study has five aims:

1. Describe the numbers and characteristics of births recorded with Down syndrome from 1989 to 2024. Compare recorded counts with surveillance-based estimates.
2. Model the recorded Down syndrome checkbox for 2016–2024. Compare feature sets and label definitions, and describe associations with recorded status.
3. Estimate expected Down syndrome livebirth counts and expected missed counts for 2016–2024 with Bayesian population models. Report uncertainty and sensitivity to external calibration, false positives and recording assumptions.
4. Describe co-occurring conditions among recorded cases. Explore population estimates only where assumptions about recording within each condition stratum can be stated and tested through sensitivity analyses.
5. Explore age, year and demographic patterns in recorded births and modelled expected counts. Report associations and distinguish them from causal claims.

## What the data can establish

A classifier trained on the recorded checkbox estimates `P(recorded DS | characteristics)`. It does not directly estimate either `P(true DS | characteristics)` or recording sensitivity, `P(recorded DS | true DS, characteristics)`. The recorded probability reflects true prevalence, sensitivity and false positives.

The unrecorded population contains both non-cases and missed cases. Treating every unrecorded birth as a negative cannot resolve that mixture. The quota-based `ds_pred_missing*` flags select high-scoring births; their counts are set by a multiplier, not by validation against diagnoses. A chromosomal-disorder checkbox without a Down syndrome checkbox does not establish a missed Down syndrome diagnosis.

Aggregate estimation also needs assumptions. Birth-certificate counts alone cannot separate true prevalence, prenatal reduction, recording sensitivity and false positives. The Bayesian models use external information to constrain that separation. A posterior interval describes uncertainty conditional on the chosen model; it does not include every possible source bias.

The microdata used here has no state identifier. Demographic associations may therefore reflect geographic differences in access or recording. State-level aggregate comparisons can inform checks but cannot add a state effect to individual birth records.

## Current work

Use the [model inventory](../docs/models/README.md) for model roles and the [workflow](../docs/modelling-workflow.md) for commands. The [September code-review note](../notes/20260905-dsp-code-review-fixes.md) records changes that require refitting older DSP analyses. Review their surveillance weighting, observation-error assumptions, validation and provenance before reporting new estimates.

The [research-note index](../notes/readme.md) identifies the supporting source checks and historical fits. Earlier proposed aims and completed implementation tasks are not the current work plan.
