"""Sequentially fit the four selection-model variants (A/B/C/D).

Wraps ``scripts/fit_selection_model.py`` as a Python subprocess runner
so each variant's stdout/stderr streams into a per-variant log file
under ``output/selection/_run_logs/`` and failures in one variant do
not kill the others. Intended for overnight batch runs at
``--profile reporting`` but works for any profile.

Examples
--------
    # Overnight reporting run, all four variants, full spec.
    python scripts/run_all_selection_variants.py --profile reporting

    # Rerun just C and D, skipping anything already on disk.
    python scripts/run_all_selection_variants.py \\
        --variants C D --skip-existing

    # Pass extra flags through to each fit.
    python scripts/run_all_selection_variants.py \\
        --profile reporting \\
        --extra-args "--target-accept 0.98 --random-seed 7"

Resumability
------------
``--skip-existing`` skips a variant when
``output/selection/<V>/<spec>/latest/idata.nc`` is already present
(``latest`` is a symlink or the most recent timestamped run under
``output/selection/<V>/<spec>/``). This lets you resume an interrupted
overnight batch without re-running completed variants.
"""

from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import dse_research_utils.environment.setup as setup

from dspopulations_us_birth_certificates import cli_output
from dspopulations_us_birth_certificates.selection import SPECS, VARIANTS

LOG_ROOT = Path("output/selection/_run_logs")
OUTPUT_ROOT = Path("output/selection")


@dataclass
class RunnerCliConfig:
    variants: tuple[str, ...]
    spec: str
    profile: str
    years: str | None
    render: bool
    skip_existing: bool
    extra_args: tuple[str, ...]


def _parse_args(argv: list[str] | None) -> RunnerCliConfig:
    p = argparse.ArgumentParser(
        description=(
            "Batch-run the four selection-model variants sequentially. "
            "Intended for overnight reporting-quality runs."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--variants",
        nargs="+",
        default=sorted(VARIANTS),
        choices=sorted(VARIANTS),
        help="Variants to run, in order.",
    )
    p.add_argument(
        "--spec",
        default="full",
        choices=list(SPECS),
        help="Staged spec (same across all variants).",
    )
    p.add_argument(
        "--profile",
        default="reporting",
        choices=("dev", "reporting"),
        help="Run-config preset.",
    )
    p.add_argument(
        "--years",
        default=None,
        help="Year range as 'YYYY-YYYY'. Defaults to the plan's 2016-2024.",
    )
    p.add_argument(
        "--render",
        action="store_true",
        help="Pass --render to each fit_selection_model.py invocation.",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "Skip a variant when output/selection/<V>/<spec>/ already "
            "contains a run with an idata.nc on disk."
        ),
    )
    p.add_argument(
        "--extra-args",
        default="",
        help=(
            "Quoted string of extra flags passed verbatim to each "
            'fit_selection_model.py invocation (e.g. --extra-args '
            "'--target-accept 0.98 --draws 2000')."
        ),
    )
    ns = p.parse_args(argv)
    return RunnerCliConfig(
        variants=tuple(ns.variants),
        spec=ns.spec,
        profile=ns.profile,
        years=ns.years,
        render=ns.render,
        skip_existing=ns.skip_existing,
        extra_args=tuple(shlex.split(ns.extra_args)),
    )


def _has_completed_fit(variant: str, spec: str) -> bool:
    """True when at least one run dir under <root>/<variant>/<spec>/ has idata.nc."""
    parent = OUTPUT_ROOT / variant / spec
    if not parent.exists():
        return False
    for run_dir in parent.iterdir():
        if run_dir.is_dir() and (run_dir / "idata.nc").is_file():
            return True
    return False


def _build_cmd(
    variant: str,
    cli: RunnerCliConfig,
    output_dir: Path,
) -> list[str]:
    cmd = [
        sys.executable,
        "-u",
        "scripts/fit_selection_model.py",
        "--variant",
        variant,
        "--spec",
        cli.spec,
        "--profile",
        cli.profile,
        "--output-dir",
        str(output_dir),
    ]
    if cli.years:
        cmd += ["--years", cli.years]
    if cli.render:
        cmd.append("--render")
    cmd += list(cli.extra_args)
    return cmd


def _run_one(
    variant: str,
    cli: RunnerCliConfig,
    log_path: Path,
) -> tuple[str, int, Path]:
    """Run fit_selection_model.py for one variant; return (variant, rc, log)."""
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = OUTPUT_ROOT / variant / cli.spec / ts
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = _build_cmd(variant, cli, output_dir)

    cli_output.info(f"[variant {variant}] launching: {' '.join(cmd)}")
    cli_output.info(f"[variant {variant}] log -> {log_path}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.now()
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"# started {started.isoformat()}\n")
        log.write(f"# cmd    {' '.join(cmd)}\n")
        log.flush()
        proc = subprocess.run(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
        elapsed = datetime.now() - started
        log.write(f"\n# finished rc={proc.returncode} elapsed={elapsed}\n")
    return variant, proc.returncode, output_dir


def _iter_variants_to_run(
    cli: RunnerCliConfig,
) -> Iterable[str]:
    for v in cli.variants:
        if cli.skip_existing and _has_completed_fit(v, cli.spec):
            cli_output.info(
                f"[variant {v}] skipping — existing idata.nc under "
                f"{OUTPUT_ROOT / v / cli.spec}/"
            )
            continue
        yield v


def main(argv: list[str] | None = None) -> int:
    cli = _parse_args(argv)
    setup.init_script()

    cli_output.banner(
        "run_all_selection_variants",
        f"variants={','.join(cli.variants)}  spec={cli.spec}  profile={cli.profile}",
    )

    if shutil.which("quarto") is None and cli.render:
        cli_output.warning(
            "--render requested but `quarto` not on PATH; "
            "fits will still run, rendering will emit a warning per variant."
        )

    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    batch_ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    results: list[tuple[str, int, Path]] = []
    for variant in _iter_variants_to_run(cli):
        log_path = LOG_ROOT / f"{batch_ts}_{variant}.log"
        result = _run_one(variant, cli, log_path)
        results.append(result)
        status = "ok" if result[1] == 0 else f"FAILED(rc={result[1]})"
        cli_output.info(
            f"[variant {variant}] {status}  output -> {result[2]}"
        )

    cli_output.section("Summary")
    if not results:
        cli_output.warning("No variants were run.")
        return 0
    ok = [v for v, rc, _ in results if rc == 0]
    failed = [(v, rc) for v, rc, _ in results if rc != 0]
    cli_output.print_kv(
        "Batch result",
        [
            ("batch_ts", batch_ts),
            ("profile", cli.profile),
            ("spec", cli.spec),
            ("variants ok", ", ".join(ok) or "(none)"),
            ("variants failed", ", ".join(f"{v}(rc={rc})" for v, rc in failed) or "(none)"),
            ("logs", str(LOG_ROOT / f"{batch_ts}_*.log")),
        ],
    )
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
