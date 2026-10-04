> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-6).

# Shared utilities 0.17.0

The project now selects `dse-research-utils` from the published `v0.17.0` tag. The tag resolves to release commit `935bd38bdd09da05cd9895fb3ee73d38c27b3e7c`. The [library upgrade guide](https://github.com/dseinternational/research/blob/v0.17.0/docs/migrating-to-0.17.md) describes public sampling diagnostics, reductions of existing diagnostic tables, optional file-permission controls and the fix for nullable missing diagnostics.

The library's Python requirement, dependency minimums and extras are unchanged from `v0.16.2`. This project retains its existing extras, model specifications, sampling thresholds and validation rules. The lock refresh selects the new library tag while retaining unrelated package versions.

Install the updated environment with `uv sync --locked`. Both the installed distribution and `dse_research_utils.__version__` must report `0.17.0`. Numerical validation remains separate from the study's scientific assumptions. Keep saved results, historical manifests and recorded environments intact when checking compatibility.

The file adapter now uses the shared mode probe in the destination directory. It retains the first mode for the process and applies it after the writer. Its no-argument mode helper remains available. Selection validation uses the public energy diagnostic and table reductions. The strict R-hat cutoff, inclusive effective-sample-size and energy cutoffs, constant exemptions and failed checks for unavailable diagnostics remain local.
