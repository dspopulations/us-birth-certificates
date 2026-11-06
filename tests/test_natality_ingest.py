"""Source layouts, strict parsing, integrity checks and non-destructive pilot writes.

All records here are synthetic. No natality microdata is committed in fixtures.
"""

from __future__ import annotations

import copy
import io
import json
import zipfile

import duckdb
import pytest

from dspopulations_us_birth_certificates.ingest import database, published_validation
from dspopulations_us_birth_certificates.ingest.database import (
    import_year,
    protect_output,
)
from dspopulations_us_birth_certificates.ingest.layouts import (
    ROOT,
    Layout,
    allowed_tokens,
    load_layout,
    sha256,
    validate_layout,
)
from dspopulations_us_birth_certificates.ingest.parser import iter_batches
from dspopulations_us_birth_certificates.ingest.published_validation import (
    load_controls,
    validate_published,
)
from dspopulations_us_birth_certificates.ingest.sources import (
    CheckedReader,
    download,
    inspect_archive,
    open_member,
)
from dspopulations_us_birth_certificates.ingest.validation import compare_database
from dspopulations_us_birth_certificates.preparation import derive_columns
from dspopulations_us_birth_certificates.reference_data import (
    reference_coverage,
    require_complete_estimates,
)
from dspopulations_us_birth_certificates.variables import IMPORTED_VARS


def record(layout: Layout, *, resident: int = 1) -> bytes:
    data = bytearray(b" " * 1345)
    # Supply legal values to unrelated fields. The critical values and expected
    # results below use independently stated positions from the printed guide.
    for field in layout.fields:
        if field["status"] == "present":
            token = allowed_tokens(field)[0].encode()
            start = field["start"] - 1
            data[start : start + field["width"]] = token
    for start, token in {
        9: str(layout.year).encode(),
        13: b"06",
        19: b"2359",
        75: b"35",
        104: str(resident).encode(),
        107: b"6",
        108: b"15",
        115: b"0",
        117: b"6",
        175: b"00",
        283: b"26.6",
        337: b"0",
        456: b"1",
        552: b"P",
        558: b"1",
        561: b"9",
    }.items():
        data[start - 1 : start - 1 + len(token)] = token
    return bytes(data) + b"\r\n"


def make_local_source(tmp_path, year):
    original = load_layout(year)
    data, source = copy.deepcopy(original.data), copy.deepcopy(original.source)
    data.update(member="synthetic.txt", compression_method=8)
    source["expected_counts"] = {"occurrence": 3, "resident": 2, "foreign_resident": 1}
    cache = tmp_path / "cache"
    cache.mkdir()
    guide = cache / "guide.pdf"
    guide.write_bytes(b"Synthetic guide hash fixture; not a PDF")
    source.update(
        guide_urls=["https://example.test/guide.pdf"], guide_sha256=sha256(guide)
    )
    data["guide_sha256"] = source["guide_sha256"]
    layout = Layout(data, source, "synthetic-layout-v1")
    archive = cache / "synthetic.zip"
    records = [record(layout, resident=r) for r in (1, 3, 4)]
    if year == 2025:
        first = bytearray(records[0])
        first[104:106] = b"1 "
        first[106:109] = b"11 "
        first[116] = ord("1")
        records[0] = bytes(first)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr(data["member"], b"".join(records))
    source.update(
        archive_url="https://example.test/synthetic.zip", archive_sha256=sha256(archive)
    )
    return layout, cache, tmp_path / "pilot.db"


@pytest.fixture
def local_source(tmp_path):
    return make_local_source(tmp_path, 2024)


def test_catalogue_covers_all_years_but_only_reviewed_year_imports():
    catalogue = json.loads((ROOT / "config/natality/sources.json").read_text())
    assert [s["year"] for s in catalogue["sources"]] == list(range(1989, 2026))
    assert all(
        s["guide_urls"] and s["archive_url"].startswith("https://ftp.cdc.gov/")
        for s in catalogue["sources"]
    )
    with pytest.raises(ValueError, match="No reviewed layout"):
        load_layout(2023)
    assert load_layout(2025).year == 2025


def test_2024_positions_and_types_are_explicit():
    layout = load_layout(2024)
    fields = {f["name"]: f for f in layout.fields}
    assert set(IMPORTED_VARS) <= fields.keys()
    assert (fields["ca_down"]["start"], fields["ca_down"]["width"]) == (552, 1)
    assert fields["f_ca_downs"]["start"] == 558
    assert fields["dob_tt"]["type"] == "uint16"
    assert fields["imp_plural"]["source_name"] == "IMP_PLUR"
    assert fields["datayear"]["status"] == "absent"
    assert "02" not in allowed_tokens(fields["mager14"])
    assert "99.9" in allowed_tokens(fields["bmi"])


@pytest.mark.parametrize("change", ["missing", "overlap", "range", "padding"])
def test_bad_layout_fails_before_data_read(change):
    layout = load_layout(2024)
    data = copy.deepcopy(layout.data)
    present = [f for f in data["fields"] if f["status"] == "present"]
    if change == "missing":
        data["fields"].pop(0)
    elif change == "overlap":
        present[1]["start"] = present[0]["start"]
    elif change == "range":
        present[0]["start"] = 1330
    else:
        data["padding"]["width"] = 14
    with pytest.raises(ValueError):
        validate_layout(data, layout.source)


def test_parser_preserves_source_codes_and_types():
    layout = load_layout(2024)
    frequencies = {}
    batch = list(iter_batches(io.BytesIO(record(layout)), layout, 2, frequencies))[0]
    row = batch.to_pylist()[0]
    assert row["dob_yy"] == 2024 and row["datayear"] is None
    assert row["dob_tt"] == 2359 and row["no_risks"] == 0 and row["no_congen"] == 9
    assert str(row["bmi"]) == "26.6"
    assert row["ca_down"] == "P" and row["f_ca_downs"] == 1
    assert row["imp_plural"] == 1 and row["source_record"] == 1
    assert frequencies["ca_down"] == {"P": 1}


class ShortReads(io.BytesIO):
    def read(self, size):
        return super().read(min(size, 23))


def test_short_reads_and_batch_boundaries():
    layout = load_layout(2024)
    batches = list(iter_batches(ShortReads(record(layout) * 5), layout, 2, {}))
    assert [b.num_rows for b in batches] == [2, 2, 1]
    assert [b.column("source_record").to_pylist() for b in batches] == [
        [1, 2],
        [3, 4],
        [5],
    ]


@pytest.mark.parametrize(
    ("position", "token", "message"),
    [
        (9, b"2025", "dob_yy"),
        (552, b"Z", "ca_down"),
        (1331, b"X", "padding"),
        (1346, b" \n", "separator"),
        (1, b"\0", "Nonprinting"),
    ],
)
def test_bad_records_fail_with_record_number(position, token, message):
    layout = load_layout(2024)
    bad = bytearray(record(layout))
    bad[position - 1 : position - 1 + len(token)] = token
    with pytest.raises(ValueError, match=message):
        list(iter_batches(io.BytesIO(record(layout) + bad), layout, 1, {}))


def test_truncated_record_and_optional_final_separator():
    layout = load_layout(2024)
    with pytest.raises(ValueError, match="Truncated"):
        list(iter_batches(io.BytesIO(record(layout)[:-2]), layout, 2, {}))
    data = {**layout.data, "allow_final_record_without_separator": True}
    assert (
        list(
            iter_batches(
                io.BytesIO(record(layout)[:-2]),
                Layout(data, layout.source, "test"),
                2,
                {},
            )
        )[0].num_rows
        == 1
    )


def test_zip_checks_require_full_consumption_and_crc(local_source):
    layout, cache, _ = local_source
    archive = cache / "synthetic.zip"
    with pytest.raises(ValueError, match="fully read"):
        with open_member(archive, layout) as (reader, _):
            reader.read(10)
    member = inspect_archive(archive, layout)
    reader = CheckedReader(io.BytesIO(b"wrong"), member)
    while reader.read(2):
        pass
    with pytest.raises(ValueError, match="CRC"):
        reader.verify()
    with zipfile.ZipFile(archive, "a") as z:
        z.writestr("extra.txt", "unexpected member")
    with pytest.raises(ValueError, match="member list"):
        inspect_archive(archive, layout)


def test_changed_cached_download_is_not_skipped(tmp_path):
    path = tmp_path / "bad.zip"
    path.write_bytes(b"partial")
    with pytest.raises(ValueError, match="hash differs"):
        download("https://example.test/bad.zip", path, "0" * 64)
    assert path.read_bytes() == b"partial"


def test_import_counts_identity_and_repeat(local_source):
    layout, cache, output = local_source
    report = import_year(
        layout, cache=cache, output=output, batch_size=2, with_derivations=False
    )
    assert report["rows"] == 3 and report["counts"]["resident"] == 2
    assert report["status"] == "source_validated"
    with duckdb.connect(str(output), read_only=True) as con:
        assert con.execute(
            "SELECT min(id), max(id), count(*) FROM us_births"
        ).fetchone() == (2024000000001, 2024000000003, 3)
    repeated = import_year(
        layout, cache=cache, output=output, batch_size=1, with_derivations=False
    )
    assert repeated["reused"]


def test_failed_year_is_unpublished_and_restart_does_not_duplicate(
    local_source, monkeypatch
):
    layout, cache, output = local_source
    real = database.iter_batches

    def interrupted(*args, **kwargs):
        iterator = real(*args, **kwargs)
        yield next(iterator)
        raise RuntimeError("simulated interruption")

    monkeypatch.setattr(database, "iter_batches", interrupted)
    with pytest.raises(RuntimeError, match="simulated"):
        import_year(
            layout, cache=cache, output=output, batch_size=1, with_derivations=False
        )
    with duckdb.connect(str(output), read_only=True) as con:
        assert "us_births" not in {r[0] for r in con.execute("SHOW TABLES").fetchall()}
        assert con.execute("SELECT count(*) FROM __incoming").fetchone()[0] == 1
    monkeypatch.setattr(database, "iter_batches", real)
    with pytest.raises(ValueError, match="--restart"):
        import_year(layout, cache=cache, output=output, with_derivations=False)
    report = import_year(
        layout, cache=cache, output=output, restart=True, with_derivations=False
    )
    assert report["rows"] == 3


def test_wrong_counts_do_not_publish(local_source):
    layout, cache, output = local_source
    source = copy.deepcopy(layout.source)
    source["expected_counts"]["occurrence"] = 4
    with pytest.raises(ValueError, match="controls failed"):
        import_year(
            Layout(layout.data, source, layout.digest),
            cache=cache,
            output=output,
            with_derivations=False,
        )
    with duckdb.connect(str(output), read_only=True) as con:
        assert "us_births" not in {r[0] for r in con.execute("SHOW TABLES").fetchall()}


def test_protected_paths_aliases_and_unmanaged_database(local_source, tmp_path):
    with pytest.raises(ValueError, match="protected"):
        protect_output(ROOT / "data/us_births.db")
    baseline = tmp_path / "baseline.db"
    with duckdb.connect(str(baseline)) as con:
        con.execute("CREATE TABLE keep_me AS SELECT 42 value")
    before = sha256(baseline)
    symlink = tmp_path / "link.db"
    symlink.symlink_to(baseline)
    with pytest.raises(ValueError, match="protected"):
        protect_output(symlink, baseline)
    hardlink = tmp_path / "hardlink.db"
    hardlink.hardlink_to(baseline)
    with pytest.raises(ValueError, match="protected"):
        protect_output(hardlink, baseline)
    layout, cache, _ = local_source
    with pytest.raises(ValueError, match="not a managed"):
        import_year(layout, cache=cache, output=baseline, with_derivations=False)
    assert sha256(baseline) == before


def test_shared_derivations_keep_pending_and_multirace(local_source):
    layout, cache, output = local_source
    import_year(layout, cache=cache, output=output, with_derivations=False)
    with duckdb.connect(str(output)) as con:
        derive_columns(con, ROOT / "data")
        assert con.execute(
            "SELECT DISTINCT year,mage_c,ca_down_c,down_ind,mracehisp_c FROM us_births"
        ).fetchall() == [(2024, 35, "P", 1, 6)]
        assert con.execute(
            "SELECT count(p_ds_lb_wt_mage),count(p_ds_lb_wt) FROM us_births"
        ).fetchone() == (0, 3)


def test_comparison_explains_losses_and_detects_joint_differences(
    local_source, tmp_path
):
    layout, cache, output = local_source
    import_year(layout, cache=cache, output=output)
    # Give records different weights so a permutation can preserve every
    # marginal count while changing the joint record values.
    with duckdb.connect(str(output)) as con:
        con.execute("UPDATE us_births SET dbwt = 3000 + restatus")
    baseline = tmp_path / "old.db"
    with duckdb.connect(str(baseline)) as con:
        con.execute(
            "ATTACH '" + str(output).replace("'", "''") + "' AS pilot (READ_ONLY)"
        )
        con.execute("CREATE TABLE us_births AS SELECT * FROM pilot.us_births")
        con.execute(
            "UPDATE us_births SET dob_tt=NULL, no_risks=NULL, no_congen=NULL, imp_plural=NULL, bmi=26.59375"
        )
        # The baseline's BMI type needs to hold the old binary16 result.
        con.execute("ALTER TABLE us_births ALTER bmi TYPE DOUBLE")
        con.execute("UPDATE us_births SET bmi=26.59375")
    before = sha256(baseline)
    report = compare_database(layout, output, baseline)
    assert report["passed"] and report["baseline_unchanged"]
    assert sha256(baseline) == before
    with duckdb.connect(str(baseline)) as con:
        con.execute(
            "UPDATE us_births SET dbwt=CASE restatus WHEN 1 THEN 3003 WHEN 3 THEN 3001 ELSE dbwt END"
        )
    report = compare_database(layout, output, baseline)
    assert not report["passed"]
    assert not report["field_comparisons"]["dbwt"][
        "frequency_count_absolute_difference"
    ]
    assert sum(r["candidate_only"] for r in report["row_comparisons"]) == 2


def test_2025_layout_has_new_flags_and_updated_links():
    layout = load_layout(2025)
    fields = {f["name"]: f for f in layout.fields}
    assert len(fields) == 190
    for name, position in {
        "f_mbcntry": 127,
        "f_otno": 219,
        "f_tbo": 220,
        "f_iloo": 221,
        "f_ilp": 222,
        "f_attend": 439,
        "f_lmpyy": 512,
    }.items():
        assert fields[name]["start"] == position
        assert fields[name]["width"] == 1
        assert allowed_tokens(fields[name]) == ["0", "1"]
    assert fields["mbstate_rec"]["reporting_flag"] == "f_mbcntry"
    assert fields["ilop_r11"]["reporting_flag"] == "f_iloo"
    assert fields["ilp_r11"]["reporting_flag"] == "f_ilp"
    assert "reporting_flag" not in fields["apgar10"]
    assert fields["ca_down"]["guide_page"] == 39
    assert fields["ca_down"]["start"] == 552
    assert fields["f_ca_downs"]["start"] == 558
    assert allowed_tokens(fields["dob_yy"]) == ["2025"]
    assert load_controls(layout)["controls"][0]["expected"] == 3_619_990
    # Set all new flags independently, including zero beside a populated value.
    data = bytearray(record(layout))
    for position in (127, 219, 220, 221, 222, 439, 512):
        data[position - 1] = ord("1")
    data[218] = ord("0")
    row = list(iter_batches(io.BytesIO(data), layout, 2, {}))[0].to_pylist()[0]
    assert row["f_otno"] == 0 and row["priorterm"] is not None
    assert row["f_ilp"] == 1 and row["f_lmpyy"] == 1
    with pytest.raises(ValueError, match="dob_yy"):
        list(iter_batches(io.BytesIO(record(load_layout(2024))), layout, 2, {}))


@pytest.fixture
def published_pilot(tmp_path, monkeypatch):
    layout, cache, output = make_local_source(tmp_path, 2025)
    controls = {
        "controls": [
            {
                "name": "resident_recorded",
                "where": "restatus IN (1,2,3) AND ca_down IN ('C','P')",
                "expected": 2,
            },
            {"name": "new_flag", "where": "f_mbcntry=0", "expected": 3},
        ]
    }
    monkeypatch.setattr(published_validation, "load_controls", lambda _: controls)
    baseline = tmp_path / "live.db"
    with duckdb.connect(str(baseline)) as con:
        con.execute("CREATE TABLE us_births AS SELECT 2024 AS year")
    import_year(layout, cache=cache, output=output, baseline=baseline)
    return layout, output, baseline, controls


def test_published_validation_accepts_data_but_exposes_missing_references(
    published_pilot,
):
    layout, output, baseline, _ = published_pilot
    before = sha256(baseline)
    report = validate_published(layout, output, baseline)
    assert report["passed"] and report["baseline_unchanged"]
    assert sha256(baseline) == before
    coverage = report["reference_coverage"]
    assert not coverage["all_reference_inputs_available"]
    assert coverage["derived_values_match_available_references"]
    assert len(coverage["tables"]) == 5
    for result in coverage["tables"].values():
        assert result["rows_for_year"] == 0
        assert result["births_without_reference_key"] == 3
        if result["derived_column"]:
            assert result["missing_derived_values"] == 3
    with duckdb.connect(str(output), read_only=True) as con:
        with pytest.raises(ValueError, match="2025"):
            require_complete_estimates(con, ["p_ds_lb_nt_reduc", "ds_case_weight"])
        require_complete_estimates(con, ["p_ds_lb_nt"])


@pytest.mark.parametrize(
    "change",
    ["published_count", "field", "derivation", "carry_forward", "duplicate_id"],
)
def test_published_validation_rejects_changed_candidate(published_pilot, change):
    layout, output, baseline, controls = published_pilot
    if change == "published_count":
        controls["controls"][0]["expected"] += 1
    else:
        sql = {
            "field": "UPDATE us_births SET f_mbcntry=1 WHERE source_record=1",
            "derivation": "UPDATE us_births SET mracehisp_c=4 WHERE source_record=1",
            "carry_forward": "UPDATE us_births SET p_ds_lb_wt=0.0014,ds_case_weight=0",
            "duplicate_id": "UPDATE us_births SET source_record=1 WHERE source_record=2",
        }[change]
        with duckdb.connect(str(output)) as con:
            con.execute(sql)
    report = validate_published(layout, output, baseline)
    assert not report["passed"]


def test_published_validation_rejects_changed_frequency_report(published_pilot):
    layout, output, baseline, _ = published_pilot
    output.with_suffix(".frequencies.json").write_text("{}")
    with pytest.raises(ValueError, match="frequency report"):
        validate_published(layout, output, baseline)


def test_reference_review_rejects_duplicate_keys(published_pilot):
    _, output, _, _ = published_pilot
    with duckdb.connect(str(output)) as con:
        con.execute("INSERT INTO prevalence_year SELECT * FROM prevalence_year LIMIT 1")
        with pytest.raises(ValueError, match="duplicate reference keys"):
            reference_coverage(con, 2025)


def test_estimate_guard_detects_partial_missing_year():
    with duckdb.connect() as con:
        con.execute(
            "CREATE TABLE us_births AS SELECT * FROM (VALUES (2024,0.01),(2025,0.01),(2025,NULL)) t(year,p_ds_lb_wt)"
        )
        with pytest.raises(ValueError, match="2025.*1"):
            require_complete_estimates(con, ["p_ds_lb_wt"])


def test_reviewed_lexical_alias_preserves_raw_evidence_and_stays_year_specific():
    layout = load_layout(2025)
    raw = bytearray(record(layout))
    raw[104:106] = b"1 "
    raw[107:109] = b"1 "
    frequencies = {}
    row = list(iter_batches(io.BytesIO(raw), layout, 1, frequencies))[0].to_pylist()[0]
    assert row["mrace31"] == row["mrace15"] == 1
    assert frequencies["mrace31"] == {"1 ": 1}
    raw[8:12] = b"2024"
    with pytest.raises(ValueError, match="mrace31"):
        list(iter_batches(io.BytesIO(raw), load_layout(2024), 1, {}))
    raw[8:12] = b"2025"
    raw[104:106] = b"2 "
    with pytest.raises(ValueError, match="mrace31"):
        list(iter_batches(io.BytesIO(raw), layout, 1, {}))


def test_changed_lexical_exception_count_does_not_publish(tmp_path):
    layout, cache, output = make_local_source(tmp_path, 2025)
    data = copy.deepcopy(layout.data)
    field = next(f for f in data["fields"] if f["name"] == "mrace31")
    field["lexical_aliases"]["1 "]["expected_count"] = 2
    with pytest.raises(ValueError, match="Lexical exception count changed"):
        import_year(
            Layout(data, layout.source, "changed-count"), cache=cache, output=output
        )
    with duckdb.connect(str(output), read_only=True) as con:
        assert "us_births" not in {r[0] for r in con.execute("SHOW TABLES").fetchall()}


def test_lexical_alias_cannot_change_numeric_meaning():
    layout = load_layout(2025)
    data = copy.deepcopy(layout.data)
    field = next(f for f in data["fields"] if f["name"] == "mrace31")
    field["lexical_aliases"]["1 "]["canonical"] = "02"
    with pytest.raises(ValueError, match="Invalid lexical exception"):
        validate_layout(data, layout.source)


def test_descriptive_report_rejects_2025_before_writing_figures(
    published_pilot, tmp_path
):
    from dspopulations_us_birth_certificates.descriptive_analyses import build_all

    _, output, _, _ = published_pilot
    destination = tmp_path / "report"
    with pytest.raises(ValueError, match="2025"):
        build_all(output, destination)
    assert not list(destination.iterdir())
