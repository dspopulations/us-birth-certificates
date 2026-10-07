> [!NOTE]
> AI-assisted revision by Codex (GPT-6).

> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-5).

# Bayesian accounting model inventory

DSP identifiers record the order of model development. A larger number does not mean a preferred model. These aggregate models are separate from the [three-stage selection model](../../src/dspopulations_us_birth_certificates/selection/README.md).

For true Down syndrome probability `q`, recording sensitivity `s` and false-positive probability `f`, the recorded probability is `q*s + (1-q)*f`. Certificate counts constrain this recorded probability. External information and structural assumptions constrain its components.

> [!IMPORTANT]
> September 2026 changes revised the DSP003 calibration solver, the surveillance observation model, prior sampling and validation. August fits record the earlier implementation. Refit them before presenting estimates from the current model. See the [code-review fixes](../../notes/20260905-dsp-code-review-fixes.md).

## Model roles

| Model | Age resolution | Recording | Prevalence or reduction | Role |
| --- | --- | --- | --- | --- |
| DSP001 | Seven bands | Constant | Annual reduction priors | Original baseline and age-discretisation sensitivity |
| DSP002 | Seven bands | Centred year offsets | Annual reduction priors | Year-recording sensitivity at band resolution |
| DSP003 | NCHS single-age codes | Constant | Smooth age reduction calibrated to annual margins | Age-allocation diagnostic |
| DSP004 | NCHS single-age codes | Constant | Annual reduction priors | Simple exact-age reference structure |
| DSP005 | NCHS single-age codes | Centred year offsets | Annual reduction priors | Year-recording sensitivity to DSP004 |
| DSP006 | NCHS single-age codes | Revised and unrevised levels | Annual reduction priors | Certificate-revision control |
| DSP007 | NCHS single-age codes | Constant | Surveillance-anchored annual prevalence | Direct prevalence anchor |
| DSP008 | NCHS single-age codes | Revised and unrevised levels | Surveillance-anchored annual prevalence | Anchor with revision control |
| DSP009 | NCHS single-age codes | Revision levels and post-anchor drift | Surveillance-anchored annual prevalence | Sensitivity to post-window allocation |
| DSP010 | NCHS single-age codes | Revision levels and anomaly-panel factor | Surveillance-anchored annual prevalence | Additional recording evidence under control-condition assumptions |

"Exact age" is shorthand. Code 12 pools ages 10–12 and code 50 pools ages 50 and over. The Morris curve uses representative ages 12 and 50 at those endpoints.

DSP004 replaces the broad-band approximation with a simpler exact-age reference. It is not a claim of adequate age fit or identified prenatal reduction. The historical fits still missed broad-age margins. DSP003 can absorb that pattern into reduction while holding recording constant; a better fit does not establish that allocation as the mechanism.

## Fitting and reporting

```bash
uv run python scripts/fit_core_reduction_model.py DSP004 --profile reporting --render
uv run python scripts/fit_core_reduction_model.py DSP008 --years 2004-2024 --profile reporting --render
uv run python scripts/fit_core_reduction_model.py --help
```

Use a range that crosses the 2004–2015 certificate phase-in to study revision-specific recording. All DSP models use `docs/models/selection_core_reduction/index.qmd`. The CLI copies it into `output/selection_core_reduction/<model_id>/<timestamp>/`. See the [workflow](../modelling-workflow.md) for inputs and checks.

Each posterior draw of the expected total is `sum(N_cell*q_cell)`. Its interval describes uncertainty in expected burden. It does not include the additional uncertainty in the unknown realised count. Older `true_count_*` names are compatibility aliases.

## Reduction-prior sensitivities

DSP001–DSP006 use the reduction-rate CSV. Its source derivation is incomplete, and its extrapolated tail remains an assumption. The working logit standard deviations are 0.20 before 2020 and 0.45 from 2020. The [family review](../../notes/20260803-dsp-core-model-family-review.md) questions whether 2019 should also receive the wider prior.

DSP004 exposes:

- `--reduction-error-correlation`, which correlates annual logit errors while preserving their marginal variances;
- `--reduction-calibration-shift-logit`, which shifts all annual prior centres;
- the false-positive and reduction-width controls shared by the fitting CLI.

The defaults are independent errors and zero shift. These values are working assumptions. The [calibration analysis](../../notes/20260803-dsp004-coherent-surveillance-calibration.md) found material changes in total and recording under alternative scenarios. Report scenarios separately. Their outer range is not a credible interval and their draws must not be pooled without justified probabilities.

## Surveillance-anchored models

DSP007–DSP010 model annual log prevalence. The CLI weights each surveillance window by annual birth counts, including years outside the fitted range. Missing weights stop the run. The observation model uses a joint Normal likelihood on log prevalence.

The default error correlation is derived from overlapping window weights. It represents shared annual errors of equal variance; it is not a measured surveillance covariance and does not account for every common bias. Check independent errors, partial overlap and non-overlapping windows as sensitivities.

The starting prevalence prior has median 0.0013 and log-scale standard deviation 0.25. The observation standard deviation is fixed at 0.05 by default. Neither scale is a measured source uncertainty. Vary them before reporting estimates. Estimating the observation scale can weaken the anchor and permit a high-prevalence, low-recording solution.

```bash
uv run python scripts/fit_core_reduction_model.py DSP008 --years 2004-2024 --anchor-obs-sigma-fixed 0.10
uv run python scripts/audit_anchored_chain_health.py --strict
```

Inspect `validation.json` as well as the per-chain audit. A numerical pass does not establish source accuracy or separate prevalence from recording.

## DSP009 post-window drift

A surveillance point centred on 2018 covers 2016–2020. It supplies no direct prevalence observation for 2021–2024. DSP008 holds recording constant in those years. DSP009 allows recording to drift, so its split between prevalence and recording depends on prior scales.

| Setting | Assumption |
| --- | --- |
| `--recording-s-drift-sigma 0` | No post-window recording drift; reproduces DSP008's recording structure |
| Default drift scale 0.06 | Both prevalence and recording can change |
| `--anchor-forecast-flat --recording-s-drift-sigma 0.20` | Prevalence stays at its last anchored level; recording can change |

Report these allocation checks with a drifted fit. `recording_s` is the reference revised-certificate sensitivity. `recording_s_drift_ratio` compares the final year with that anchored-era level. See the [drift note](../../notes/20260804-dsp009-post-anchor-recording-drift.md).

## DSP010 anomaly panel

DSP010 uses recorded congenital-anomaly rates from 2016 onward as an additional observation channel. Its control conditions are hypospadias, cleft palate alone, cleft lip with or without cleft palate, and limb reduction. Their changes can inform recording only under assumptions about their own prevalence, prenatal reduction and shared recording behaviour.

The controls disagree. The model allows condition-specific trends and an uncertain Down syndrome loading. The descriptive heterogeneity calculation omits some within-condition uncertainty, so its Q and I-squared values should not be treated as exact evidence about a shared factor.

The pinned condition table supplies Texas surveillance trends for three controls. It supplies no hypospadias trend. These fixed offsets do not propagate source uncertainty or establish national transport. Setting `--panel-prevalence-trend-sigma 0` imposes zero remaining shared prevalence trend; it is still an assumption.

A loading of one means equal changes in **log odds**, not equal proportional changes in sensitivity. `recording_s_panel_ratio` compares the final year with the panel reference year. It spans a different period from DSP009's drift ratio; compare annualised changes or counts on the same period.

```bash
uv run python scripts/fit_core_reduction_model.py DSP010 --years 2004-2024 --panel-conditions-csv data/us-births-anomaly-panel-conditions-pinned.csv --panel-prevalence-trend-sigma 0 --tune 6000 --target-accept 0.995
```

This reproduces the historical specification's settings with the current code, not its historical results. Compare the default table, a non-zero shared prevalence trend and alternative loadings. More tuning may be needed even with no divergences. See the [panel design](../../notes/20260804-dsp010-anomaly-panel-recording-factor.md) and [trend-source check](../../notes/20260804-dsp010-control-prevalence-trend-pins.md).

## Race calibration remains unresolved

The [race-surveillance audit](../../notes/20260803-dsp004-race-surveillance-audit.md) has one complete aligned window, centred on 2018. The estimate centred on 2016 needs years outside its frozen fit. The source pools numerator and denominator counts across each window; an equal-year mean of rates is not the same quantity.

Category mapping, denominator differences, source covariance and overlap with national evidence remain unresolved. The audit records `calibration_eligible=false`. Its comparisons do not establish temporal replication or authorise a race-specific calibration.
