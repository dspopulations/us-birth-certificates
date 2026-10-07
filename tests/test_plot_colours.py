# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Role colours and category palettes on the dse-research-utils 0.18.0 tokens."""

from __future__ import annotations

import pytest
from dse_research_utils.plot import styles
from matplotlib.colors import to_hex, to_rgb

from dspopulations_us_birth_certificates import plot_colours
from dspopulations_us_birth_certificates.plot_utils import _annotation_colour
from dspopulations_us_birth_certificates.predicted_analyses import (
    CATEGORY_GROUPINGS,
    category_colours,
)
from dspopulations_us_birth_certificates.selection.priors import RACE_LEVELS


def _contrast(foreground: str, background: str) -> float:
    def luminance(colour: str) -> float:
        channels = [
            c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
            for c in to_rgb(colour)
        ]
        return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]

    high, low = sorted((luminance(foreground), luminance(background)), reverse=True)
    return (high + 0.05) / (low + 0.05)


@pytest.mark.parametrize("variable", sorted(CATEGORY_GROUPINGS))
def test_every_grouping_has_one_colour_per_category(variable: str) -> None:
    # categorical_palette() raises above six chart colours, so a grouping with
    # more categories must name a palette.
    grouping = CATEGORY_GROUPINGS[variable]
    n = len(grouping.labels)
    colours = category_colours(n, grouping.colormap)
    assert len({to_hex(colour) for colour in colours}) == n


@pytest.mark.parametrize("n", range(1, 12))
def test_ordered_palette_reaches_three_to_one_on_white(n: int) -> None:
    colours = plot_colours.ordered_palette(n)
    assert len(colours) == n
    for colour in colours:
        assert _contrast(colour, styles.BACKGROUND_COLOUR) >= 3.0


def test_ordered_palette_uses_token_steps_where_they_exist() -> None:
    for n in (3, 4, 5):
        assert plot_colours.ordered_palette(n) == styles.sequential_palette(n)


def test_race_colours_cover_every_model_group() -> None:
    colours = plot_colours.RACE_COLOURS
    assert len(colours) == len(set(colours)) == len(RACE_LEVELS)
    # The named groups follow the default property cycle, as in the
    # descriptive figures; Unknown is grey.
    assert colours[:5] == styles.CHART_COLOURS[:5]
    assert colours[RACE_LEVELS.index("Unknown")] == styles.MUTED_TEXT_COLOUR


def test_roles_that_share_a_figure_are_distinct() -> None:
    c = plot_colours
    # Ascertainment funnel; previous-model yearly figure; yearly screening figure.
    assert len({c.NATURAL_COLOUR, c.ESTIMATED_COLOUR, c.RECORDED_COLOUR}) == 3
    assert len({c.ESTIMATED_COLOUR, c.RECORDED_COLOUR, c.REDUCTION_COLOUR}) == 3
    assert (
        len(
            {
                c.DETECTION_COLOUR,
                c.TERMINATION_COLOUR,
                c.REDUCTION_COLOUR,
                c.VARIANT_COLOURS["B"],
            }
        )
        == 4
    )
    # Selection-model yearly figure: the variants and recorded births.
    assert len({*c.VARIANT_COLOURS.values(), c.RECORDED_COLOUR}) == 5


def test_heatmap_labels_contrast_with_their_cells() -> None:
    assert _annotation_colour(styles.SEQUENTIAL_CMAP(0.0)) == styles.TEXT_COLOUR
    assert _annotation_colour(styles.SEQUENTIAL_CMAP(1.0)) == styles.BACKGROUND_COLOUR
