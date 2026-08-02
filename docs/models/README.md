> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-5).

# Bayesian Model Inventory

The aggregate Down syndrome birth-certificate accounting models use stable
`DSPnnn` identifiers. The numbers index the historical order in which models
entered the reproducible fitting workflow; they are not a hierarchy and they do
not imply that the highest-numbered model is preferred.

| Model | Status | Age resolution | Recording structure | Combined reduction | Purpose |
| --- | --- | --- | --- | --- | --- |
| `DSP001` | Discretisation sensitivity | Seven bands | Constant `s` | One value per year | Original core accounting model; retained to measure the effect of evaluating the Morris curve in broad bands. |
| `DSP002` | Band-resolution sensitivity | Seven bands | Partially pooled `s_year` | One value per year | Tests year-varying recording under the original broad-band age approximation. |
| `DSP003` | Age-structure diagnostic | NCHS single-age codes | Constant `s` | Smooth age pattern within year | Tests how much residual age structure can be absorbed by combined reduction while preserving each year's natural-DS-weighted surveillance margin. |
| `DSP004` | Preferred simple-resolution baseline | NCHS single-age codes | Constant `s` | One value per year | Removes the broad-band Morris approximation while retaining the simplest transparent reduction-recording structure. |
| `DSP005` | Year-recording sensitivity | NCHS single-age codes | Partially pooled `s_year` | One value per year | Tests whether year-specific recording materially changes the preferred exact-age baseline. |

All five models use the same Quarto template at
`docs/models/selection_core_reduction/index.qmd`. The fit CLI copies that
template into each run directory and records the selected model in `config.json`.

“NCHS single-age codes” is not exact at both endpoints: code 12 represents ages
10-12 and code 50 represents ages 50 and over. The Morris curve is evaluated at
representative ages 12 and 50 for those pooled cells.

Typical commands:

```bash
python scripts/fit_core_reduction_model.py DSP001 --profile reporting --render
python scripts/fit_core_reduction_model.py DSP002 --profile reporting --render
python scripts/fit_core_reduction_model.py DSP003 --profile reporting --render
python scripts/fit_core_reduction_model.py DSP004 --profile reporting --render
python scripts/fit_core_reduction_model.py DSP005 --profile reporting --render
python scripts/compare_core_reduction_models.py \
  output/selection_core_reduction/DSP001/<timestamp> \
  output/selection_core_reduction/DSP004/<timestamp>
python scripts/compare_core_reduction_models.py \
  output/selection_core_reduction/DSP004/<timestamp> \
  output/selection_core_reduction/DSP005/<timestamp>
python scripts/compare_core_reduction_sensitivities.py \
  output/selection_core_reduction/DSP004/<reference-run> \
  output/selection_core_reduction/DSP004/<sensitivity-run> [...] \
  --output-dir output/selection_core_reduction/comparisons/<comparison-name>
```

The comparisons are descriptive and in-sample. `DSP004` is preferred over
`DSP001` because it removes an avoidable age-discretisation approximation, not
because it resolves the remaining age misfit. `DSP005` checks sensitivity to
year-varying recording. `DSP003` assigns the residual maternal-age pattern to
combined reduction while holding recording constant by age; its better
in-sample fit is therefore not evidence for that mechanism. None of the models
shows that birth-certificate counts alone identify recording separately from
pre-livebirth reduction. The headline estimates remain conditional on external
Morris and surveillance information and on the false-positive scenario. The
working false-positive range has little effect on the `DSP004` total under the
current reduction-prior widths, but it materially changes recording sensitivity;
widening the independent annual reduction priors approximately doubles the
headline interval width.

The [exact-age ablation note](../../notes/20260802-dsp004-dsp005-exact-age-ablations.md)
records the matched results and decision. The
[DSP003 note](../../notes/20260802-dsp003-age-reduction-extension.md) records the
age-structure and measurement sensitivities. The
[DSP004 measurement sensitivity note](../../notes/20260802-dsp004-false-positive-surveillance-sensitivity.md)
records the false-positive and reduction-prior-width grid and its conditional
interpretation.
