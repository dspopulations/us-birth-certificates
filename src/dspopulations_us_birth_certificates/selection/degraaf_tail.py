"""De Graaf column-I tail sensitivity for the full-margin prevalence anchor.

The production anchor (``recording_anchor.PREV_RACE_YEAR``) imputes 2019-2024 true
prevalence by holding each race's net survival ratio flat at its 2018 level, so tail
prevalence drifts *up* with the ageing maternal-age distribution. Gert de Graaf's
workbook instead extends the birth-certificate *recording fraction* linearly and back-
computes true prevalence as ``recorded / G``, so his tail prevalence falls *down* as
recorded counts decline post-2020. The two bracket the post-2020 (NIPS-era)
uncertainty; see ``notes/20260628-degraaf-corrected-prevalence-extraction.md`` and
``notes/20260623-degraaf-recording-anchor.md``.

This module provides the **sensitivity scenario**: keep the production anchor for
2016-2019 and all sigmas, but replace the 2020-2024 margin *target* for the five named
de Graaf race groups with his column I (``est_true_prev_per10k``). Production stays the
default; run with ``fit_selection_model.py --degraaf-tail``.

``DEGRAAF_TAIL_PREV`` rows = race idx 0-4 (NH White, NH Black, NH AIAN, NH Asian/PI,
Hispanic); columns = 2020-2024. Sourced verbatim from column ``est_true_prev_per10k`` of
``data/us-births-degraaf-prevalence-recording-2000-2024.csv`` (his corrected workbook).
"""

from __future__ import annotations

import numpy as np

from dspopulations_us_birth_certificates.selection.recording_anchor import ANCHOR_YEARS

DEGRAAF_TAIL_FIRST_YEAR = 2020

# de Graaf column I (est_true_prev_per10k) per 10k, named races idx 0-4, years 2020-2024.
DEGRAAF_TAIL_PREV = np.array([
    [ 12.4319,  11.3905,  11.2943,  11.4340,  11.6691],  # NH White
    [ 13.6380,  12.4423,  13.5157,  13.4767,  12.6935],  # NH Black
    [ 11.6191,  13.2603,  17.1110,  16.4000,  11.2217],  # NH AIAN
    [  6.7667,   7.6820,   9.4830,   8.8002,   7.3908],  # NH Asian/Pacific Islander
    [ 17.2164,  17.2285,  15.1253,  15.7403,  15.7403],  # Hispanic
])

DEGRAAF_TAIL_YEARS = tuple(y for y in ANCHOR_YEARS if y >= DEGRAAF_TAIL_FIRST_YEAR)


def apply_degraaf_tail(prev_full: np.ndarray) -> np.ndarray:
    """Splice de Graaf's column-I tail onto a copy of the production prevalence target.

    Replaces the 2020-2024 columns of the five named de Graaf races (idx 0-4) with
    ``DEGRAAF_TAIL_PREV``; all other cells (2016-2019, Unknown/Multi-race rows) are left
    untouched. Operates on the full ``[N_RACE, len(ANCHOR_YEARS)]`` surface, before any
    ``[:, :n_year]`` slice.
    """
    prev_full = np.asarray(prev_full, dtype=float)
    tail_cols = [j for j, y in enumerate(ANCHOR_YEARS) if y >= DEGRAAF_TAIL_FIRST_YEAR]
    if len(tail_cols) != DEGRAAF_TAIL_PREV.shape[1]:
        raise ValueError(
            f"anchor has {len(tail_cols)} tail years >= {DEGRAAF_TAIL_FIRST_YEAR}, "
            f"but DEGRAAF_TAIL_PREV has {DEGRAAF_TAIL_PREV.shape[1]} columns"
        )
    out = prev_full.copy()
    n_named = DEGRAAF_TAIL_PREV.shape[0]
    out[:n_named, tail_cols] = DEGRAAF_TAIL_PREV
    return out
