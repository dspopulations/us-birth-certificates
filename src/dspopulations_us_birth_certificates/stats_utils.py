# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Shared numeric helpers with the existing descriptive-table schema."""

import numpy as np
import pandas as pd
from dse_research_utils.math.constants import EPSILON as EPSILON
from dse_research_utils.ml.feature_dependence import (
    distance_corr_dissimilarity as distance_corr_dissimilarity,
)
from dse_research_utils.ml.feature_dependence import (
    distance_corr_dissimilarity_linkage as distance_corr_dissimilarity_linkage,
)
from dse_research_utils.ml.feature_dependence import (
    distance_corr_matrix as distance_corr_matrix,
)
from dse_research_utils.ml.feature_dependence import (
    mutual_info_dissimilarity as mutual_info_dissimilarity,
)
from dse_research_utils.ml.feature_dependence import (
    spearman_distance_matrix as spearman_distance_matrix,
)
from dse_research_utils.statistics.descriptive import describe as _shared_describe
from dse_research_utils.statistics.descriptive import (
    differential_entropy_standardized as differential_entropy_standardized,
)
from dse_research_utils.statistics.transforms import (
    convert_to_categorical as convert_to_categorical,
)
from dse_research_utils.statistics.transforms import (
    invlogit as invlogit,
)
from dse_research_utils.statistics.transforms import (
    logit as logit,
)
from dse_research_utils.statistics.transforms import (
    standardize as standardize,
)


def describe(series: list[float] | np.ndarray | pd.Series, alpha: float) -> pd.Series:
    """Use shared calculations while retaining the existing rows and their order."""
    return _shared_describe(series, alpha).drop(
        labels=["n_non_na", "anderson_stat", "anderson_pvalue"]
    )


def describe_all(df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Summarise numeric columns with this project's descriptive-table schema."""
    numeric_cols = df.select_dtypes(include=["number"]).columns
    return df[numeric_cols].apply(lambda col: describe(col, alpha))


def describe_all_grouped(
    df: pd.core.groupby.DataFrameGroupBy, alpha: float
) -> pd.DataFrame:
    """Retain grouped table layout and exclude grouping columns."""
    return df.apply(lambda group: describe_all(group, alpha), include_groups=False)
