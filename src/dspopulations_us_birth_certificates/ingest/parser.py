"""Parse explicit fixed-width records with bounded memory and strict domains."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from .layouts import TYPES, Layout, allowed_tokens


def _first_bad(mask) -> int:
    return int(np.flatnonzero(np.asarray(mask))[0])


def parse_block(
    block: bytes, layout: Layout, first_record: int, frequencies: dict
) -> pa.RecordBatch:
    length = layout.data["record_length"]
    separator = {"CRLF": b"\r\n", "LF": b"\n", "NONE": b""}[layout.data["separator"]]
    stride = length + len(separator)
    if len(block) % stride or not block.isascii():
        raise ValueError(
            f"Invalid record framing or encoding near record {first_record}"
        )
    count = len(block) // stride
    payload = np.frombuffer(block, dtype=np.uint8).reshape(count, stride)[:, :length]
    invalid_bytes = (payload < 32) | (payload > 126)
    for exception in layout.data.get("record_byte_exceptions", []):
        position, value = exception["position"], exception["byte"]
        matches = payload[:, position - 1] == value
        frequencies.setdefault("__record_bytes", Counter())[
            f"{position}:{value:02x}"
        ] += int(np.count_nonzero(matches))
        invalid_bytes[matches, position - 1] = False
    bad_payload = np.any(invalid_bytes, axis=1)
    if bad_payload.any():
        raise ValueError(
            f"Nonprinting source byte at record {first_record + _first_bad(bad_payload)}"
        )
    if separator:
        endings = np.ndarray(
            (count,),
            dtype=f"S{len(separator)}",
            buffer=block,
            offset=length,
            strides=(stride,),
        )
        bad = endings != separator
        if bad.any():
            raise ValueError(
                f"Wrong record separator at record {first_record + _first_bad(bad)}"
            )
    padding = layout.data["padding"]
    if padding["width"]:
        tail = np.ndarray(
            (count,),
            dtype=f"S{padding['width']}",
            buffer=block,
            offset=padding["start"] - 1,
            strides=(stride,),
        )
        bad = tail != b" " * padding["width"]
        if bad.any():
            raise ValueError(
                f"Nonblank undocumented padding at record {first_record + _first_bad(bad)}"
            )
    arrays = []
    for field in layout.fields:
        dtype = TYPES[field["type"]][0]
        if field["status"] == "absent":
            arrays.append(pa.nulls(count, type=dtype))
            continue
        raw = np.ndarray(
            (count,),
            dtype=f"S{field['width']}",
            buffer=block,
            offset=field["start"] - 1,
            strides=(stride,),
        )
        # Arrow copies only this field, not the full record, then checks every token.
        tokens = pc.cast(pa.array(raw), pa.string())
        good = pc.is_in(tokens, value_set=pa.array(allowed_tokens(field)))
        if not pc.all(good).as_py():
            number = first_record + _first_bad(pc.invert(good))
            raise ValueError(
                f"Invalid source code for {field['name']} at record {number}"
            )
        freq = frequencies.setdefault(field["name"], Counter())
        for item in pc.value_counts(tokens).to_pylist():
            freq[item["values"]] += item["counts"]
        for raw_token, alias in {
            **field.get("lexical_aliases", {}),
            **field.get("source_code_exceptions", {}),
        }.items():
            tokens = pc.if_else(pc.equal(tokens, raw_token), alias["canonical"], tokens)
        tokens = pc.if_else(pc.equal(tokens, " " * field["width"]), None, tokens)
        if field["type"] == "integer_string":
            values = pc.cast(pc.cast(tokens, pa.uint16()), pa.string())
        else:
            values = pc.cast(tokens, dtype)
        arrays.append(values)
    arrays.append(
        pa.array(np.arange(first_record, first_record + count, dtype=np.int64))
    )
    return pa.RecordBatch.from_arrays(arrays, schema=layout.schema)


def iter_batches(reader, layout: Layout, batch_size: int, frequencies: dict):
    if not 1 <= batch_size <= 1_000_000:
        raise ValueError("batch_size must be between 1 and 1,000,000")
    length = layout.data["record_length"]
    separator = {"CRLF": b"\r\n", "LF": b"\n", "NONE": b""}[layout.data["separator"]]
    stride = length + len(separator)
    target = stride * batch_size
    pending = bytearray()
    first_record = 1
    while chunk := reader.read(target - len(pending)):
        pending.extend(chunk)
        if len(pending) == target:
            yield parse_block(bytes(pending), layout, first_record, frequencies)
            first_record += batch_size
            pending.clear()
    if pending:
        if (
            layout.data["allow_final_record_without_separator"]
            and separator
            and len(pending) % stride == length
        ):
            pending.extend(separator)
        if len(pending) % stride:
            raise ValueError(
                f"Truncated record at record {first_record + len(pending) // stride}"
            )
        yield parse_block(bytes(pending), layout, first_record, frequencies)
