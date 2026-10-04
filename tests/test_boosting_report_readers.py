# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Missing fits remain visible, while corrupt required files stop rendering."""

import re
from pathlib import Path

import pandas as pd
import pytest

REPORT = Path(__file__).resolve().parents[1] / "docs/models/usbc10/index.qmd"


def setup_cell():
    return re.findall(r"```\{python\}[^\n]*\n(.*?)```", REPORT.read_text(), re.S)[0]


def test_missing_files_keep_pending_messages(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    namespace = {}
    exec(setup_cell(), namespace)
    assert namespace["config"] == {"model_id": "<not yet rendered from a run>"}
    assert namespace["metrics"] == {}
    assert namespace["load_table"](Path("missing.csv")) == "missing.csv not present."


@pytest.mark.parametrize("contents", ["", "{broken", '{"r_hat": NaN}', '{"r_hat": 1e999}'])
def test_corrupt_json_stops_the_report(tmp_path, monkeypatch, contents):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.json").write_text(contents)
    with pytest.raises(ValueError, match="required JSON artefact"):
        exec(setup_cell(), {})


def test_present_csv_is_parsed_and_empty_document_is_invalid(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    namespace = {}
    exec(setup_cell(), namespace)
    source = Path("table.csv")
    source.write_text("value\n1\n")
    pd.testing.assert_frame_equal(namespace["load_table"](source), pd.DataFrame({"value": [1]}))
    source.write_text("")
    with pytest.raises(ValueError, match="required CSV artefact"):
        namespace["load_table"](source)


def test_required_readers_preserve_absence_and_index_policy(tmp_path):
    from dspopulations_us_birth_certificates.report_readers import (
        read_csv_artefact,
        read_json_artefact,
    )

    absent = tmp_path / "absent.json"
    with pytest.raises(FileNotFoundError):
        read_json_artefact(absent)
    assert read_json_artefact(absent, missing=None) is None
    present = tmp_path / "present.json"
    present.write_text("null")
    assert read_json_artefact(present, missing={"pending": True}) is None
    table = tmp_path / "summary.csv"
    table.write_text("variable,mean\na,1.0\n")
    assert read_csv_artefact(table, index_col=0).index.tolist() == ["a"]
