"""Independent historical positions and reviewed release exceptions.

All records are synthetic. No source microdata appears in these tests.
"""

from __future__ import annotations

import copy
import io
import zipfile

import duckdb
import pyarrow as pa
import pytest

from dspopulations_us_birth_certificates.ingest.database import import_year
from dspopulations_us_birth_certificates.ingest.layouts import (
    ROOT,
    Layout,
    allowed_tokens,
    load_layout,
    sha256,
    validate_layout,
)
from dspopulations_us_birth_certificates.ingest.parser import iter_batches
from dspopulations_us_birth_certificates.ingest.sources import decompressor_command
from dspopulations_us_birth_certificates.preparation import derive_columns


def synthetic_record(layout, overrides):
    raw = bytearray(b" " * layout.data["record_length"])
    for field in layout.fields:
        if field["status"] == "present":
            start = field["start"] - 1
            raw[start : start + field["width"]] = allowed_tokens(field)[0].encode()
    for start, value in overrides.items():
        raw[start - 1 : start - 1 + len(value)] = value
    separator = {"CRLF": b"\r\n", "LF": b"\n", "NONE": b""}[layout.data["separator"]]
    return bytes(raw) + separator


@pytest.mark.parametrize("year", [1989, 1991, 1992, 1996, 1997, 2002])
def test_old_certificate_positions_and_year(year):
    layout = load_layout(year)
    raw = synthetic_record(
        layout,
        {
            1: str(year).encode(),
            6: b"1",
            70: b"35",
            72: b"22",
            77: b"0",
            78: b"6",
            80: b"01",
            87: b"1",
            176: str(year).encode(),
            201: b"2",
            250: b"00",
            304: b"1",
            322: b"1",
        },
    )
    batch = next(iter_batches(io.BytesIO(raw), layout, 2, {}))
    row = batch.to_pylist()[0]
    assert (
        row["datayear"],
        row["biryr"],
        row["dmage"],
        row["mage36"],
        row["downs"],
    ) == (year, year, 35, 22, 1)
    assert row["dob_yy"] is None and row["dob_mm"] is None and row["sex"] is None
    with duckdb.connect() as con:
        con.register("batch", pa.Table.from_batches([batch]))
        con.execute("CREATE TABLE us_births AS SELECT * FROM batch")
        derive_columns(con, ROOT / "data")
        assert con.execute(
            "SELECT year,mage_c,mracehisp_c,ca_down_c FROM us_births"
        ).fetchone() == (year, 35, 1, "C")


def test_1992_race_and_1997_age_boundaries():
    for year, position, value, accepted in [
        (1991, 80, b"18", False),
        (1992, 80, b"18", True),
        (1996, 70, b"54", False),
        (1997, 70, b"54", True),
    ]:
        layout = load_layout(year)
        raw = synthetic_record(layout, {position: value})
        if accepted:
            assert next(iter_batches(io.BytesIO(raw), layout, 2, {})).num_rows == 1
        else:
            with pytest.raises(ValueError, match="Invalid source code"):
                list(iter_batches(io.BytesIO(raw), layout, 2, {}))


def test_2003_transition_age_and_education_alias():
    layout = load_layout(2003)
    raw = synthetic_record(
        layout,
        {
            15: b"2003",
            19: b"06",
            89: b"22",
            138: b"1",
            143: b"1",
            148: b"0",
            149: b"9",
            156: b"12",
            523: b"1",
        },
    )
    batch = next(iter_batches(io.BytesIO(raw), layout, 2, {}))
    row = batch.to_pylist()[0]
    assert row["mager41"] == 22 and row["dmeduc"] == 12 and row["umeduc"] is None
    assert row["mracehisp"] == 9 and row["uca_downs"] == 1
    with duckdb.connect() as con:
        con.register("batch", pa.Table.from_batches([batch]))
        con.execute("CREATE TABLE us_births AS SELECT * FROM batch")
        derive_columns(con, ROOT / "data")
        assert con.execute(
            "SELECT year,mage_c,ca_down_c FROM us_births"
        ).fetchone() == (2003, 35, "C")


def test_2015_unlisted_race_codes_keep_bridged_fallback():
    layout = load_layout(2015)
    raw = synthetic_record(
        layout,
        {
            9: b"2015",
            75: b"35",
            104: b"1",
            105: b"99",
            108: b"99",
            110: b"2",
            111: b"2",
            115: b"0",
            552: b"N",
        },
    )
    frequencies = {}
    batch = next(iter_batches(io.BytesIO(raw), layout, 2, frequencies))
    assert batch.column("mrace15").to_pylist() == [99]
    assert frequencies["mrace15"] == {"99": 1}
    with duckdb.connect() as con:
        con.register("batch", pa.Table.from_batches([batch]))
        con.execute("CREATE TABLE us_births AS SELECT * FROM batch")
        derive_columns(con, ROOT / "data")
        assert con.execute("SELECT mrace_c,mracehisp_c FROM us_births").fetchone() == (
            2,
            2,
        )
    current = load_layout(2024)
    with pytest.raises(ValueError, match="mrace15"):
        list(
            iter_batches(
                io.BytesIO(synthetic_record(current, {108: b"99"})), current, 2, {}
            )
        )


def test_2004_control_byte_exception_is_limited_to_unselected_position():
    layout = load_layout(2004)
    frequencies = {}
    raw = synthetic_record(layout, {1422: b"\r"})
    assert next(iter_batches(io.BytesIO(raw), layout, 2, frequencies)).num_rows == 1
    assert frequencies["__record_bytes"] == {"1422:0d": 1}
    with pytest.raises(ValueError, match="Nonprinting"):
        list(
            iter_batches(
                io.BytesIO(synthetic_record(layout, {1423: b"\r"})), layout, 2, {}
            )
        )
    data = copy.deepcopy(layout.data)
    data["record_byte_exceptions"][0]["position"] = 138
    with pytest.raises(ValueError, match="outside selected fields"):
        validate_layout(data, layout.source)


def test_2017_exception_count_and_same_numeric_value(tmp_path):
    original = load_layout(2017)
    data, source = copy.deepcopy(original.data), copy.deepcopy(original.source)
    data.update(member="synthetic.txt", compression_method=8)
    cache = tmp_path / "cache"
    cache.mkdir()
    guide = cache / "guide.pdf"
    guide.write_bytes(b"Synthetic guide")
    source.update(
        guide_urls=["https://example.test/guide.pdf"],
        guide_sha256=sha256(guide),
        guide_sha256_by_url={},
    )
    source["expected_counts"] = {"occurrence": 3, "resident": 2, "foreign_resident": 1}
    data["guide_sha256"] = source["guide_sha256"]
    layout = Layout(data, source, "test-layout")
    archive = cache / "synthetic.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr(
            data["member"],
            b"".join(
                synthetic_record(
                    layout, {104: str(r).encode(), 147: b"1 " if r == 1 else b"30"}
                )
                for r in (1, 3, 4)
            ),
        )
    source.update(
        archive_url="https://example.test/synthetic.zip", archive_sha256=sha256(archive)
    )
    output = tmp_path / "annual.db"
    report = import_year(layout, cache=cache, output=output, with_derivations=False)
    assert report["source_exceptions"][0]["observed_count"] == 1
    with duckdb.connect(str(output), read_only=True) as con:
        assert con.execute(
            "SELECT fagecomb FROM us_births WHERE restatus=1"
        ).fetchone() == (1,)
    field = next(f for f in data["fields"] if f["name"] == "fagecomb")
    field["source_code_exceptions"]["1 "]["expected_count"] = 2
    changed = Layout(data, source, "changed-count")
    with pytest.raises(ValueError, match="exception count changed"):
        import_year(
            changed, cache=cache, output=output, with_derivations=False, restart=True
        )
    field["source_code_exceptions"]["1 "]["canonical"] = "10"
    with pytest.raises(ValueError, match="Invalid source code exception"):
        validate_layout(data, source)


def test_ppmd_selects_capable_backend_or_explains_dependency(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "shutil.which", lambda name: "/usr/bin/bsdtar" if name == "bsdtar" else None
    )
    assert decompressor_command(tmp_path / "source.zip", "synthetic.txt", 98)[:2] == [
        "/usr/bin/bsdtar",
        "-xOf",
    ]
    monkeypatch.setattr("shutil.which", lambda name: None)
    with pytest.raises(RuntimeError, match="PPMd"):
        decompressor_command(tmp_path / "source.zip", "synthetic.txt", 98)


def test_annual_layouts_pin_all_guides_and_exceptions():
    for year in range(1989, 2026):
        layout = load_layout(year)
        assert layout.source["archive_sha256"]
        assert all(
            f.get("guide_page") for f in layout.fields if f["status"] == "present"
        )
        validate_layout(layout.data, layout.source)
    for year in (2009, 2010):
        layout = load_layout(year)
        assert len(layout.source["guide_sha256_by_url"]) == 3
        assert any("Addendum" in url for url in layout.source["guide_urls"])
    source = load_layout(2015).source
    assert source["expected_counts"]["occurrence"] == 3988733
    assert load_layout(2015).data["compression_method"] == 98
