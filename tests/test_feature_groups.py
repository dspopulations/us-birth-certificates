from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

from dspopulations_us_birth_certificates import feature_groups, ml_utils


def test_feature_groups_from_linkage_clusters_correlated_features() -> None:
    features = ["mage_c", "fagecomb", "ca_cchd", "ab_nicu"]
    distance = np.array(
        [
            [0.0, 0.1, 0.9, 0.9],
            [0.1, 0.0, 0.9, 0.9],
            [0.9, 0.9, 0.0, 0.2],
            [0.9, 0.9, 0.2, 0.0],
        ]
    )
    linkage = hierarchy.linkage(squareform(distance), method="average")

    groups = feature_groups.feature_groups_from_linkage(
        features, linkage, distance_threshold=0.3
    )

    assert list(groups.values()) == [
        ["mage_c", "fagecomb"],
        ["ca_cchd", "ab_nicu"],
    ]


def test_annotate_feature_group_combines_stage_hints() -> None:
    annotation = feature_groups.annotate_feature_group(["ca_disor", "ca_cchd"])

    assert annotation["dimension_hint"] == "certificate_proxy + clinical_severity"
    assert "s / certificate workflow" in annotation["candidate_stage"]
    assert "near-proxy" in annotation["interpretation_flag"]


def test_grouped_importance_to_csv_frame_serialises_features() -> None:
    df = pd.DataFrame(
        {
            "rank": [1],
            "group": ["cluster_01"],
            "features": [["mage_c", "fagecomb"]],
            "importance_mean": [0.1],
            "importance_std": [0.01],
        }
    )
    annotated = feature_groups.annotate_grouped_importance(df)
    csv_df = feature_groups.grouped_importance_to_csv_frame(annotated)

    assert json.loads(csv_df.loc[0, "features"]) == ["mage_c", "fagecomb"]
    assert "dimension_hint" in csv_df.columns


def test_group_permutation_importance_empty_groups_returns_columns() -> None:
    class _Estimator:
        def predict_proba(self, X):
            p = np.full(len(X), 0.1)
            return np.column_stack([1.0 - p, p])

    X = pd.DataFrame({"x": [0, 1, 2]})
    y = pd.Series([0, 0, 1])

    out = ml_utils.group_permutation_importance(_Estimator(), X, y, {})

    assert out.empty
    assert {"rank", "group", "features", "importance_mean"}.issubset(out.columns)
