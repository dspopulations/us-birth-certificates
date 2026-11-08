> [!NOTE]
> Drafted by Codex (GPT-6).

> [!NOTE]
> AI-assisted revision by Codex (GPT-6).

# Exploratory notebooks

These percent-format Python notebooks preserve earlier experiments. Their numbered
names identify iterations, not a current run order. Some use superseded fields,
hard-coded model settings or configurations that are no longer in the repository.
Use the [script workflow](../docs/modelling-workflow.md) for current fits and reports.

The files whose names contain `unbiased` compare feature sets. That name does not
establish unbiased classification. Their labels come from certificates, and selected
unrecorded births are not verified missed cases.

Historical counts and metrics describe their original database and run. Requery
current data and check case definitions before reuse. Notebook code that writes
predictions changes the local database.

Jupytext uses `ipynb,py:percent`. The `.py` source is tracked and `.ipynb` files are
gitignored. Keep a local pair in sync when editing. Use the locked uv environment
and the repository's `notebook.mplstyle`.

## Plot setup

Run notebooks with `notebooks/` as the working directory. Initialise the shared
environment first, report the project package versions, then apply the local style:

```python
import matplotlib.pyplot as plt
from dse_research_utils.environment.setup import init_workbook
from dse_research_utils.metadata.packages import report_package_versions

from dspopulations_us_birth_certificates import PACKAGE_LIST

init_workbook()
report_package_versions(list(PACKAGE_LIST))
plt.style.use("../notebook.mplstyle")
```

Older notebooks call `repl_utils.print_environment_info()` for the first two steps.
Apply the local style after that call too. This order keeps the shared mathematical
fonts and colour cycle, then sets the local 9-point text and narrow text fonts.
The local font-family list includes DejaVu Sans for symbols absent from the text
font. Applying the shared style again resets the local text size and font choices.

For PyMC graphs, import `model_to_graphviz` from
`dse_research_utils.statistics.models.pymc_utils`. It sets the shared graph font.
For hand-built Graphviz diagrams, use `FONT_FAMILY_DEFAULT` from
`dse_research_utils.plot.styles` with a `sans-serif` fallback for the graph, nodes
and edges. Graph fonts are separate from matplotlib's notebook text settings.
