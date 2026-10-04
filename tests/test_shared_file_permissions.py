# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Cached post-write permission choices remain local to this project."""

import json
import os
import subprocess
import sys

import pytest

from dspopulations_us_birth_certificates import file_io


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
@pytest.mark.parametrize("mask", [0o022, 0o027, 0o077])
def test_the_first_post_write_mode_is_retained_for_new_and_replaced_files(
    tmp_path, mask
):
    script = """
import json
import os
import stat
import sys
from pathlib import Path
from dspopulations_us_birth_certificates.file_io import default_file_mode, write_atomically
root = Path(sys.argv[1])
mask = int(sys.argv[2])
os.umask(0o022)
first = root / 'replaced.json'
first.write_text('old')
first.chmod(0o400)
def write_first(temporary):
    os.umask(mask)
    temporary.write_text('new')
    temporary.chmod(0o400)
write_atomically(first, write_first)
os.umask(0o077)
write_atomically(root / 'new.json', lambda temporary: temporary.write_text('new'))
print(json.dumps({p.name: stat.S_IMODE(p.stat().st_mode) for p in root.iterdir()}))
assert default_file_mode() == 0o666 & ~mask
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), str(mask)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == {
        "new.json": 0o666 & ~mask,
        "replaced.json": 0o666 & ~mask,
    }


def test_a_failed_first_mode_probe_keeps_the_previous_file(tmp_path, monkeypatch):
    target = tmp_path / "fit.json"
    target.write_text("old", encoding="utf-8")

    def fail(_directory):
        raise PermissionError("mode probe failed")

    monkeypatch.setattr(file_io, "_FILE_MODE", None)
    monkeypatch.setattr(file_io, "_shared_default_file_mode", fail)
    with pytest.raises(PermissionError, match="mode probe failed"):
        file_io.write_text_atomically(target, "new", encoding="utf-8")

    assert target.read_text(encoding="utf-8") == "old"
    assert sorted(path.name for path in tmp_path.iterdir()) == [target.name]
