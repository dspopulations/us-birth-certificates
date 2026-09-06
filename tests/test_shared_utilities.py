"""Consumer checks for shared console, environment and plotting utilities."""

from __future__ import annotations

from importlib.metadata import version
from io import StringIO

import duckdb
import matplotlib
import pandas as pd
import pytest
from dse_research_utils.console.console import get_console, set_console
from dse_research_utils.environment import setup
from dse_research_utils.metadata import packages
from dse_research_utils.plot import styles
from rich.console import Console

from dspopulations_us_birth_certificates import PACKAGE_LIST, cli_output, repl_utils


@pytest.fixture
def console_output():
    """Capture the real shared console and restore it after each check."""
    previous = get_console()
    output = StringIO()
    set_console(Console(file=output, width=160, color_system=None))
    try:
        yield output
    finally:
        set_console(previous)


def test_cli_wrappers_render_with_shared_console(console_output) -> None:
    cli_output.banner("Compatibility check", "Synthetic run")
    cli_output.section("Data")
    cli_output.subsection("Counts")
    cli_output.print_data_summary(1000, "synthetic_target", 2)
    cli_output.print_params("Parameters", {"optional": None, "enabled": True})
    cli_output.print_metrics_table({"average_precision": 0.125})

    output = console_output.getvalue()
    for expected in (
        "Compatibility check",
        "Synthetic run",
        "Counts",
        "synthetic_target",
        "0.200%",
        "optional",
        "enabled",
        "average_precision",
        "0.125",
    ):
        assert expected in output


def test_notebook_shim_initialises_style_and_reports_versions(console_output) -> None:
    with matplotlib.rc_context({"font.size": 1}):
        repl_utils.print_environment_info()
        assert matplotlib.rcParams["font.size"] == styles.FONT_SIZE_DEFAULT

    output = console_output.getvalue()
    assert "Environment" in output
    assert "Package Versions" in output
    assert "Not found" not in output
    for name in PACKAGE_LIST:
        assert name in output
        assert version(name) in output
    assert packages.get_package_versions(["dse-research-utils"]) == {
        "dse-research-utils": version("dse-research-utils")
    }


def test_descriptive_report_uses_shared_style_and_saves_figures(tmp_path) -> None:
    import matplotlib.pyplot as plt

    from dspopulations_us_birth_certificates.descriptive_analyses import (
        section_a_counts,
    )

    # Synthetic rows exercise all status categories without natality microdata.
    with duckdb.connect() as con, matplotlib.rc_context():
        con.execute("""
            CREATE TABLE us_births AS
            SELECT * FROM (VALUES
                (2020, 'C', 1), (2020, 'N', 0), (2020, 'U', 0),
                (2021, 'C', 1), (2021, 'P', 1), (2021, NULL, 0)
            ) AS births(year, ca_down_c, down_ind)
        """)
        setup.init_script()
        assert matplotlib.rcParams["font.size"] == styles.FONT_SIZE_DEFAULT
        unrelated = plt.figure()
        try:
            summary = section_a_counts(con, tmp_path)
            assert plt.fignum_exists(unrelated.number)
            assert summary == {
                "total_births": 6,
                "total_recorded": 3,
                "year_min": 2020,
                "year_max": 2021,
            }
            for stem in (
                "recorded_confirmed_pending",
                "recorded_rate_per_10k",
                "ds_status_unknown",
            ):
                for suffix in ("png", "svg", "csv"):
                    assert (tmp_path / f"{stem}.{suffix}").stat().st_size > 0
            rates = pd.read_csv(tmp_path / "recorded_rate_per_10k.csv")
            assert rates["year"].tolist() == [2020, 2021]
            assert rates["recorded_per_10k"].tolist() == pytest.approx(
                [10000 / 3, 20000 / 3]
            )
        finally:
            plt.close(unrelated)
