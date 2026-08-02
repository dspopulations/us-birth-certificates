> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-5).

# Bayesian Model Inventory

The aggregate Down syndrome birth-certificate accounting models use stable
`DSPnnn` identifiers. The numbers index the historical order in which models
entered the reproducible fitting workflow; they are not a hierarchy and they do
not imply that the highest-numbered model is preferred.

| Model | Status | Recording structure | Purpose |
| --- | --- | --- | --- |
| `DSP001` | Baseline | Constant `s` | Core age-by-year accounting model with surveillance-informed combined reduction and one global certificate recording sensitivity. |
| `DSP002` | First extension | Partially pooled `s_year` | Direct comparison against `DSP001` to test whether year-specific recording sensitivity materially changes the central accounting story. |

Both models use the same Quarto template at
`docs/models/selection_core_reduction/index.qmd`. The fit CLI copies that
template into each run directory and records the selected model in `config.json`.

Typical commands:

```bash
python scripts/fit_core_reduction_model.py DSP001 --profile reporting --render
python scripts/fit_core_reduction_model.py DSP002 --profile reporting --render
python scripts/compare_core_reduction_models.py \
  output/selection_core_reduction/DSP001/<timestamp> \
  output/selection_core_reduction/DSP002/<timestamp>
```

The direct comparison is descriptive. It should be read as a model-stability and
reporting-sensitivity check, not as evidence that birth-certificate counts alone
identify year-specific recording changes separately from the surveillance-
informed reduction trend.
