> [!NOTE]
> AI-assisted revision by Codex (GPT-6).

> [!NOTE]
> Drafted by Codex (GPT-6).

# Research figures and historical inputs

`20261108-recorded-ds-trends/` contains the reviewed age-dependent analysis of
confirmed and pending certificate records for 2018–2025. The [research note](../20261108-recorded-ds-age-dependent-trends.md)
states the model, uncertainty and limits. Its reproducible command is
`scripts/analyse_recorded_ds_trends.py`.

## Historical inputs

The older CSVs here retain aggregate evidence from earlier exploratory analyses. They
are not current results, and this documentation review did not regenerate them.
Their originating scripts are in `scripts/`, and the [research notes](../readme.md)
describe the assumptions and corrections.

The old rendered plots were removed because their estimates, labels or model
interpretations were superseded. In particular, classifier totals, stage-specific
trends and co-occurrence curves are conditional on assumptions that the earlier
labels did not state. Keep these CSVs for comparison, not as validated inputs to a
new report.

`dsp004_dag_extensions.png` and `.svg` are a conceptual model diagram. They include
proposed extensions and do not establish that every depicted model is implemented.
`scripts/model_dag_figure.py` contains the diagram source.

`study-area-recording-transport.csv` contains conditional national rescalings of
local validation results. Read the [transport note](../20260804-salemi-boulet-study-area-transport.md)
before using them.
