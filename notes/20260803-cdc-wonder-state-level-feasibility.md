> [!NOTE]
> AI-assisted revision by Codex (GPT-6), 7 October 2026.

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).

# State-level CDC WONDER feasibility audit

**Original date:** 3 August 2026. Historical research note, revised for clarity in October 2026.

This was an August audit of available aggregate queries, suppression and the repository's lack of geographic identifiers. It is a dated service snapshot, not a guarantee that the live interface or coverage is unchanged.

## Query results

| Design | Cells | Observed | Suppressed | Zero | Captured |
| --- | --- | --- | --- | --- | --- |
| Year (national) | 9 | 9 | 0 | 0 | 100% |
| State, pooled | 51 | 50 | 1 | 0 | 99.97% |
| State x year | 459 | 367 | 86 | 6 | 97.3% |
| State x race6 x Hispanic, pooled | 1,377 | 176 | 299 | 902 | 95.0% |
| State x year x race6 | 4,131 | 436 | 1,064 | 2,631 | 83.5% |
| State x year x race6 x Hispanic | 12,393 | 516 | 1,697 | 10,180 | 74.5% |

| Design | By-vars | Enumerated | Observed | Suppressed | Zero | Captured |
| --- | --- | --- | --- | --- | --- | --- |
| State x age band | 2 | 459 | 252 | 100 | 107 | 97.3% |
| State x education | 2 | 561 | 314 | 122 | 125 | 97.0% |
| State x age band x education | 3 | 5,049 | 593 | 1,479 | 2,977 | 71.3% |
| State x year x age band | 3 | 4,131 | 606 | 1,893 | 1,632 | 61.0% |
| State x age band x race6 x Hispanic | 4 | 12,393 | 404 | 1,311 | 10,678 | 80.6% |
| State x age band x race6 x Hispanic x education | 5 | **truncated** | 417 | not listed | not listed | 40.6% |
| State x year x age band x race6 x Hispanic | 5 | **truncated** | 321 | not listed | not listed | 26.3% |

"Captured" refers to the fraction of records represented by unsuppressed cells in the queried margin. Suppressed cells are not zero counts. A model needs a defined treatment of those missing cells; dropping them can change both the numerator and population represented.

## State variation and saved extracts

| Lowest | per 10,000 | | Highest | per 10,000 |
| --- | --- | --- | --- | --- |
| Hawaii | 1.44 | | Utah | 13.43 |
| Mississippi | 2.14 | | South Dakota | 13.28 |
| Florida | 2.57 | | Alaska | 10.37 |
| Tennessee | 3.55 | | Idaho | 10.24 |
| Texas | 3.60 | | Nebraska | 9.74 |
| California | 3.64 | | Iowa | 9.61 |

| File | Rows |
| --- | --- |
| `data/us-births-wonder-national-year-2016-2024.csv` | 9 |
| `data/us-births-wonder-state-pooled-2016-2024.csv` (adds `ds_confirmed`) | 51 |
| `data/us-births-wonder-state-year-2016-2024.csv` | 459 |
| `data/us-births-wonder-state-race-pooled-2016-2024.csv` | 1,377 |
| `data/us-births-wonder-state-year-race6-2016-2024.csv` | 2,753 |

| Years | State-resolved Down syndrome from | Notes |
| --- | --- | --- |
| 1989-2004 | Public-use microdata directly | State of residence present; the 1989-certificate `DOWNS` checkbox is a different and more poorly recorded item, and the 2003-revision transition years are mixed-layout |
| 2005-2015 | Nothing public | Geographic detail dropped from public files [from the 2005 data year](https://www.psc.isr.umich.edu/dis/data/kb/answer/1047.html); WONDER holds no anomaly data for these years |
| 2016-2024 | CDC WONDER `D149` | Extracted, see above |

The wide spread in recorded prevalence can reflect true prevalence, recording, source coverage and case mix. It does not establish that most state variation is recording. Marginal state counts also do not provide record linkage or a joint state-by-age-by-demographic table.

The prepared national extract has no state or hospital identifier. Adding one requires a suitable source and a new derivation, not a new column inferred from maternal nativity or residency status. Query availability described here should be checked against the [WONDER documentation](https://wonder.cdc.gov/natality.html) before new extraction work.

Overlapping marginal tables share births and cases. Treating their likelihoods as independent can count evidence more than once. State effects need an explicit model of shared margins, suppressed counts, prevalence and recording; adding state parameters alone does not establish identification.

The [study-area comparison](20260804-salemi-boulet-study-area-transport.md) uses the saved aggregates for context. Its national sensitivity rescaling remains conditional on comparable true prevalence and false-positive rates.
