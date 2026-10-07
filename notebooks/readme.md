> [!NOTE]
> Drafted by Codex (GPT-6).

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
