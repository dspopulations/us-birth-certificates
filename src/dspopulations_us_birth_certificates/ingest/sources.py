"""Verified downloads and bounded ZIP member streams, including Deflate64."""

from __future__ import annotations

import json
import os
import shutil
import ssl
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
import zlib
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import truststore

from dspopulations_us_birth_certificates.file_io import write_text_atomically

from .layouts import Layout, sha256


def save_json(path: Path, value: dict) -> None:
    write_text_atomically(path, json.dumps(value, indent=2, allow_nan=False) + "\n")


def download(
    url: str, destination: Path, expected_hash: str, *, attempts: int = 3
) -> dict:
    """Never overwrite a different cached source or accept a partial download."""
    if not url.startswith("https://"):
        raise ValueError("Source downloads require HTTPS")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256(destination) != expected_hash:
            raise ValueError(f"Cached source hash differs: {destination}")
        metadata_path = destination.with_suffix(destination.suffix + ".json")
        metadata = (
            json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        )
        return {
            **metadata,
            "url": url,
            "sha256": expected_hash,
            "bytes": destination.stat().st_size,
            "cached": True,
        }
    context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    for attempt in range(attempts):
        fd, name = tempfile.mkstemp(
            prefix=destination.name + ".", suffix=".part", dir=destination.parent
        )
        temporary = Path(name)
        try:
            with (
                os.fdopen(fd, "wb") as out,
                urllib.request.urlopen(url, timeout=60, context=context) as response,
            ):
                if not response.url.startswith("https://"):
                    raise ValueError("Source redirected away from HTTPS")
                length = response.headers.get("Content-Length")
                metadata = {
                    "url": url,
                    "final_url": response.url,
                    "retrieved_at": datetime.now(UTC).isoformat(),
                    "etag": response.headers.get("ETag"),
                    "last_modified": response.headers.get("Last-Modified"),
                }
                shutil.copyfileobj(response, out, length=1024 * 1024)
                out.flush()
                os.fsync(out.fileno())
            size = temporary.stat().st_size
            if length is not None and size != int(length):
                raise OSError("Incomplete HTTP transfer")
            if sha256(temporary) != expected_hash:
                raise ValueError(
                    "Downloaded bytes differ from the reviewed source hash; review the new release"
                )
            temporary.replace(destination)
            metadata.update(sha256=expected_hash, bytes=size, cached=False)
            save_json(destination.with_suffix(destination.suffix + ".json"), metadata)
            return metadata
        except OSError, urllib.error.URLError:
            if attempt + 1 == attempts:
                raise
            time.sleep(min(2**attempt, 8))
        finally:
            temporary.unlink(missing_ok=True)
    raise ValueError("attempts must be positive")


def inspect_archive(path: Path, layout: Layout) -> zipfile.ZipInfo:
    with zipfile.ZipFile(path) as archive:
        members = [m for m in archive.infolist() if not m.is_dir()]
    if len(members) != 1 or members[0].filename != layout.data["member"]:
        raise ValueError("Archive member list differs from the reviewed source")
    member = members[0]
    if (
        member.flag_bits & 1
        or member.compress_type != layout.data["compression_method"]
    ):
        raise ValueError("Encrypted or unexpected ZIP compression")
    return member


def decompressor_command(path: Path, member: str, method: int = 9) -> list[str]:
    """Stream Deflate64 or PPMd using a system tool."""
    for name in ("7zz", "7z"):
        if executable := shutil.which(name):
            return [executable, "x", "-so", "-bd", "-y", str(path.resolve()), member]
    if method == 98:
        if executable := shutil.which("bsdtar"):
            return [executable, "-xOf", str(path.resolve()), member]
        raise RuntimeError(
            "This ZIP uses PPMd. Install 7-Zip (7zz/7z), or bsdtar with PPMd support."
        )
    if executable := shutil.which("unzip"):
        info = subprocess.run(
            [executable, "-v"], capture_output=True, text=True, check=True
        )
        if "USE_DEFLATE64" in info.stdout and "ZIP64_SUPPORT" in info.stdout:
            return [executable, "-p", str(path.resolve()), member]
    raise RuntimeError(
        "This ZIP uses Deflate64. Install 7-Zip (7zz/7z), or unzip built with USE_DEFLATE64 and ZIP64_SUPPORT."
    )


class CheckedReader:
    def __init__(self, stream, member: zipfile.ZipInfo):
        self.stream, self.member = stream, member
        self.size = 0
        self.crc = 0
        self.eof = False

    def read(self, size: int) -> bytes:
        if size < 1:
            raise ValueError("Reads must have a positive bound")
        block = self.stream.read(size)
        self.size += len(block)
        self.crc = zlib.crc32(block, self.crc)
        self.eof = not block
        return block

    def verify(self) -> None:
        if (
            not self.eof
            or self.size != self.member.file_size
            or self.crc != self.member.CRC
        ):
            raise ValueError(
                "ZIP member was not fully read or failed size/CRC validation"
            )


@contextmanager
def open_member(path: Path, layout: Layout):
    member = inspect_archive(path, layout)
    if member.compress_type not in (9, 98):
        with zipfile.ZipFile(path) as archive, archive.open(member) as stream:
            reader = CheckedReader(stream, member)
            yield reader, "python.zipfile"
            reader.verify()
        return
    command = decompressor_command(path, member.filename, member.compress_type)
    # A file for stderr prevents a full pipe from stopping the decompressor.
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors)
        try:
            reader = CheckedReader(process.stdout, member)
            yield reader, Path(command[0]).name
            code = process.wait(timeout=60)
            if code:
                raise ValueError(f"ZIP decompressor failed with exit status {code}")
            reader.verify()
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            process.stdout.close()
