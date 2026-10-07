# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Colours by role for this study's figures.

A quantity keeps one colour in every figure that shows it, so recorded births,
for example, look the same in the summary, yearly and diagnostic figures. All
colours come from the DSE design tokens in
:mod:`dse_research_utils.plot.styles`. The design language allows six
categorical colours (``CHART_COLOURS``), uses sequential or diverging steps for
ordered values and draws text in ``TEXT_COLOUR`` or ``MUTED_TEXT_COLOUR``.
Colour is never the only signal: keep labels, markers and line styles.

Figures that compare variants colour each variant with :data:`VARIANT_COLOURS`
instead of the quantity colour.
"""

from __future__ import annotations

import numpy as np
from dse_research_utils.plot.styles import (
    CHART_COLOURS,
    MUTED_TEXT_COLOUR,
    SEQUENTIAL_CMAP,
    diverging_palette,
    sequential_palette,
)
from matplotlib.colors import to_hex

RECORDED_COLOUR = CHART_COLOURS[5]
"""DS births recorded on the birth certificate, including the observed recorded
counts in posterior predictive checks."""

ESTIMATED_COLOUR = CHART_COLOURS[2]
"""Estimated true DS live births from a model."""

NATURAL_COLOUR = CHART_COLOURS[0]
"""Expected DS live births without prenatal screening or termination."""

DETECTION_COLOUR = CHART_COLOURS[0]
"""Prenatal screening detection."""

TERMINATION_COLOUR = CHART_COLOURS[1]
"""Termination if detected."""

REDUCTION_COLOUR = CHART_COLOURS[3]
"""Reduction: DS pregnancies not born alive, as a share or an implied count."""

POSTERIOR_COLOUR = CHART_COLOURS[0]
"""A posterior summary in a diagnostic figure."""

REFERENCE_COLOUR = CHART_COLOURS[2]
"""An external reference compared with a posterior, such as a surveillance
source or the Morris age curve."""

VARIANT_COLOURS: dict[str, str] = {
    "A": CHART_COLOURS[0],
    "B": CHART_COLOURS[2],
    "C": CHART_COLOURS[1],
    "D": CHART_COLOURS[3],
}
"""Selection-model reporting variants."""

RACE_COLOURS: tuple[str, ...] = (
    *CHART_COLOURS[:5],
    MUTED_TEXT_COLOUR,
    CHART_COLOURS[5],
)
"""Race and Hispanic-origin groups, in ``selection.RACE_LEVELS`` order.

The five named groups take the first five chart colours, which is also their
order in the default property cycle. Unknown is grey and NH Multi-race takes the
sixth chart colour, so all seven model groups stay distinct.
"""

_DIVERGING_ENDS = diverging_palette(3)

DECREASE_COLOUR = _DIVERGING_ENDS[0]
"""A negative change: the low end of the diverging palette."""

INCREASE_COLOUR = _DIVERGING_ENDS[-1]
"""A positive change: the high end of the diverging palette."""

# SEQUENTIAL_CMAP runs from near-white to dark blue. Its first sequential token
# step, the lightest colour that reaches 3:1 on white, sits at 0.2.
_SEQUENTIAL_START = 0.2


def ordered_palette(n: int) -> list[str]:
    """Return ``n`` colours for ordered categories, least first.

    Three to five categories take the design tokens' sequential steps. Other
    counts are spaced evenly along ``SEQUENTIAL_CMAP`` from its first step, so
    every colour reaches 3:1 on white rather than starting at near-white.

    Parameters
    ----------
    n : int
        Number of ordered categories, at least 1.

    Returns
    -------
    list[str]
        ``n`` hex colours, lightest first.
    """
    if n < 1:
        raise ValueError(f"ordered_palette needs at least one category, not {n}")
    if 3 <= n <= 5:
        return sequential_palette(n)
    return [to_hex(SEQUENTIAL_CMAP(x)) for x in np.linspace(_SEQUENTIAL_START, 1.0, n)]
