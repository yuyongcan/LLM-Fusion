#!/usr/bin/env python3
"""Sample complete SFT rows by verified source, length and conversation features.

Sources are recovered from SHA-256 fingerprints of canonical messages/tools, not
prompt heuristics. The input Parquet and source files are never modified. Only
NumPy and PyArrow are required; no model, GPU, network or pickle is used.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

CONTENT_COLUMNS = ["messages", "tools"]
FINGERPRINT_DTYPE = np.dtype("V32")


def fingerprint(row: dict) -> bytes:
    payload = {name: row[name] for name in CONTENT_COLUMNS}
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).digest()


def nonempty_json(value) -> bool:
    for _ in range(3):
        if not isinstance(value, str):
            break
        if not value.strip():
            return False
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            break
    return bool(value)


def features(row: dict, boundaries: tuple[int, ...]) -> int:
    length = row["token_length"]
    if length is None or not 0 < length <= boundaries[-1]:
        raise ValueError(f"Invalid token_length: {length}")
    messages = row["messages"] or []
    if not all(isinstance(message, dict) for message in messages):
        raise ValueError("messages must contain objects")
    reasoning = any(
        message.get("role") == "assistant"
        and (bool(message.get("reasoning_content")) or "<think>" in (message.get("content") or ""))
        for message in messages
    )
    multi_turn = any(sum(message.get("role") == role for message in messages) > 1 for role in ("user", "assistant"))
    tools = nonempty_json(row["tools"]) or any(
        message.get("role") == "tool" or nonempty_json(message.get("tool_calls")) for message in messages
    )
    bucket = int(np.searchsorted(boundaries, length, side="left"))
    return bucket * 8 + int(reasoning) * 4 + int(multi_turn) * 2 + int(tools)


def scan_task(task):
    path, groups, boundaries, with_features = task
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    parquet = pq.ParquetFile(path)
    count = sum(parquet.metadata.row_group(group).num_rows for group in groups)
    hashes = np.empty(count, dtype=FINGERPRINT_DTYPE)
    codes = np.empty(count, dtype=np.int16) if with_features else None
    lengths = np.empty(count, dtype=np.int32) if with_features else None
    columns = CONTENT_COLUMNS + (["token_length"] if with_features else [])
    offset = 0
    for batch in parquet.iter_batches(batch_size=512, row_groups=groups, columns=columns, use_threads=False):
        for row in batch.to_pylist():
            hashes[offset] = np.void(fingerprint(row))
            if with_features:
                codes[offset] = features(row, boundaries)
                lengths[offset] = row["token_length"]
            offset += 1
    if offset != count:
        raise RuntimeError("Parquet scan row count changed")
    return hashes, codes, lengths


def scan_files(paths, boundaries, with_features, workers, label):
    metadata = [pq.ParquetFile(path) for path in paths]
    total = sum(parquet.metadata.num_rows for parquet in metadata)
    hashes = np.empty(total, dtype=FINGERPRINT_DTYPE)
    codes = np.empty(total, dtype=np.int16) if with_features else None
    lengths = np.empty(total, dtype=np.int32) if with_features else None
    tasks = []
    for path, parquet in zip(paths, metadata, strict=True):
        for start in range(0, parquet.num_row_groups, 8):
            tasks.append((str(path), list(range(start, min(start + 8, parquet.num_row_groups))), boundaries, with_features))
    offset = 0
    started = time.monotonic()
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        for number, (chunk, chunk_codes, chunk_lengths) in enumerate(pool.map(scan_task, tasks), 1):
            stop = offset + len(chunk)
            hashes[offset:stop] = chunk
            if with_features:
                codes[offset:stop] = chunk_codes
                lengths[offset:stop] = chunk_lengths
            offset = stop
            if number % 10 == 0 or number == len(tasks):
                print(f"{label}: {offset:,}/{total:,} rows; {time.monotonic() - started:.1f}s", flush=True)
    if offset != total:
        raise RuntimeError("Incomplete scan")
    return hashes, codes, lengths


def recover_sources(reference_hashes, reference_counts, input_hashes):
    source_ids = np.repeat(np.arange(len(reference_counts), dtype=np.int16), reference_counts)
    order = np.argsort(reference_hashes, kind="stable")
    sorted_hashes = reference_hashes[order]
    sorted_sources = source_ids[order]
    starts = np.r_[0, np.flatnonzero(sorted_hashes[1:] != sorted_hashes[:-1]) + 1]
    unique_hashes = sorted_hashes[starts]
    minimum = np.minimum.reduceat(sorted_sources, starts)
    maximum = np.maximum.reduceat(sorted_sources, starts)
    positions = np.searchsorted(unique_hashes, input_hashes)
    valid = positions < len(unique_hashes)
    valid[valid] &= unique_hashes[positions[valid]] == input_hashes[valid]
    missing = int((~valid).sum())
    if missing:
        raise ValueError(f"{missing} input rows have no exact-content SHA-256 match in the sources")
    ambiguous = minimum[positions] != maximum[positions]
    if ambiguous.any():
        raise ValueError(f"{int(ambiguous.sum())} input rows match multiple sources; refusing to guess their source")
    return minimum[positions], {
        "reference_rows": len(reference_hashes),
        "reference_unique_contents": len(unique_hashes),
        "reference_cross_source_contents": int((minimum != maximum).sum()),
        "unmatched_input_rows": missing,
        "ambiguous_input_rows": int(ambiguous.sum()),
    }


def quotas(counts, target):
    counts = np.asarray(counts, dtype=np.int64)
    if target < 0 or target > int(counts.sum()):
        raise ValueError("Invalid sampling target")
    if not counts.sum():
        return np.zeros_like(counts)
    products = counts * target
    result = products // counts.sum()
    remainder = target - int(result.sum())
    order = np.argsort(-(products % counts.sum()), kind="stable")
    result[order[:remainder]] += 1
    return result


def choose_rows(source_ids, feature_codes, source_names, boundaries, fraction, seed):
    total = len(source_ids)
    target = (2 * total * fraction.numerator + fraction.denominator) // (2 * fraction.denominator)
    if not target:
        raise ValueError("Requested fraction rounds to zero rows")
    width = len(boundaries) * 8
    strata_ids = source_ids.astype(np.int64) * width + feature_codes
    counts = np.bincount(strata_ids, minlength=len(source_names) * width).reshape(len(source_names), width)
    source_targets = quotas(counts.sum(axis=1), target)
    allocations = np.stack([quotas(row, int(n)) for row, n in zip(counts, source_targets, strict=True)])
    grouped_indices = np.argsort(strata_ids, kind="stable")
    ends = np.cumsum(counts.ravel())
    starts = np.r_[0, ends[:-1]]
    rng = np.random.Generator(np.random.PCG64(seed))
    selected = []
    report = []
    for stratum, requested in enumerate(allocations.ravel()):
        count = int(counts.ravel()[stratum])
        if not count:
            continue
        source_id, code = divmod(stratum, width)
        bucket, flags = divmod(code, 8)
        candidates = grouped_indices[starts[stratum]:ends[stratum]]
        if requested:
            selected.append(rng.choice(candidates, size=int(requested), replace=False))
        report.append({
            "source": source_names[source_id],
            "token_length_min": 1 if bucket == 0 else boundaries[bucket - 1] + 1,
            "token_length_max": boundaries[bucket],
            "has_reasoning": bool(flags & 4),
            "multi_turn": bool(flags & 2),
            "has_tools": bool(flags & 1),
            "input_rows": count,
            "selected_rows": int(requested),
        })
    indices = np.sort(np.concatenate(selected))
    if len(indices) != target or np.any(indices[1:] <= indices[:-1]):
        raise RuntimeError("Sampling produced duplicate indices or the wrong row count")
    sources = [
        {"source": name, "input_rows": int(count), "selected_rows": int(requested)}
        for name, count, requested in zip(source_names, counts.sum(axis=1), source_targets, strict=True)
    ]
    return indices, sources, report


def row_group_sizes(count, max_rows, shard_multiple):
    multiple = min(count, shard_multiple)
    groups = min(count, math.ceil(math.ceil(count / max_rows) / multiple) * multiple)
    base, extra = divmod(count, groups)
    return [base + int(group < extra) for group in range(groups)]


def write_subset(source, output, indices, max_rows, shard_multiple):
    parquet = pq.ParquetFile(source)
    group_sizes = row_group_sizes(len(indices), max_rows, shard_multiple)
    selected_offset = 0
    source_offset = 0
    buffered = []
    buffered_rows = 0
    written_groups = 0
    with pq.ParquetWriter(output, parquet.schema_arrow, compression="zstd") as writer:
        for group in range(parquet.num_row_groups):
            rows = parquet.metadata.row_group(group).num_rows
            end = int(np.searchsorted(indices, source_offset + rows))
            if end > selected_offset:
                table = parquet.read_row_group(group, use_threads=False)
                chosen = table.take(pa.array(indices[selected_offset:end] - source_offset))
                buffered.append(chosen)
                buffered_rows += len(chosen)
            source_offset += rows
            selected_offset = end
            while written_groups < len(group_sizes) and buffered_rows >= group_sizes[written_groups]:
                table = pa.concat_tables(buffered)
                size = group_sizes[written_groups]
                writer.write_table(table.slice(0, size), row_group_size=size)
                remainder = table.slice(size)
                buffered = [remainder] if len(remainder) else []
                buffered_rows -= size
                written_groups += 1
            if (group + 1) % 100 == 0 or group + 1 == parquet.num_row_groups:
                print(f"write: {selected_offset:,}/{len(indices):,} rows", flush=True)
    if buffered_rows or selected_offset != len(indices) or written_groups != len(group_sizes):
        raise RuntimeError("Output buffering lost rows")
    return group_sizes


def file_identity(path):
    stat = path.stat()
    parquet = pq.ParquetFile(path)
    return {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "rows": parquet.metadata.num_rows, "row_groups": parquet.num_row_groups}


def sample(args):
    started = time.monotonic()
    source = Path(args.input).resolve(strict=True)
    output = Path(args.output).resolve()
    names, references = [], []
    for spec in args.source:
        name, separator, path = spec.partition("=")
        if not separator or not name or name in names:
            raise ValueError("Each --source must be a unique NAME=/absolute/path.parquet")
        names.append(name)
        references.append(Path(path).resolve(strict=True))
    boundaries = tuple(args.length_boundaries)
    if not boundaries or boundaries[0] <= 0 or tuple(sorted(set(boundaries))) != boundaries:
        raise ValueError("Length boundaries must be positive and strictly increasing")
    fraction = Fraction(args.fraction)
    if not 0 < fraction <= 1 or min(args.workers, args.row_group_size, args.shard_multiple) <= 0 or args.seed < 0:
        raise ValueError("Invalid fraction, worker count, row group size, shard multiple or seed")
    summary_path = output.with_suffix(output.suffix + ".json")
    indices_path = output.with_suffix(output.suffix + ".indices.npy")
    targets = [output, summary_path, indices_path]
    if any(path.exists() for path in targets) or output in [source, *references]:
        raise FileExistsError("Refusing to overwrite an input or existing output artifact")
    schema = pq.ParquetFile(source).schema_arrow
    for name in CONTENT_COLUMNS + ["token_length"]:
        if name not in schema.names:
            raise ValueError(f"Input is missing {name}")
    for path in references:
        if any(name not in pq.ParquetFile(path).schema_arrow.names for name in CONTENT_COLUMNS):
            raise ValueError(f"Reference is missing content columns: {path}")
    identities = [file_identity(path) for path in [source, *references]]
    print(f"input: {identities[0]['rows']:,} rows; fraction={fraction}; seed={args.seed}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.with_suffix(output.suffix + ".lock")
    lock_owned = False
    created = []
    try:
        with lock.open("x") as handle:
            lock_owned = True
            handle.write(str(os.getpid()) + "\n")
        reference_hashes, _, _ = scan_files(references, boundaries, False, args.workers, "reference fingerprints")
        input_hashes, input_features, input_lengths = scan_files([source], boundaries, True, args.workers, "input fingerprints")
        source_ids, match_report = recover_sources(reference_hashes, [item["rows"] for item in identities[1:]], input_hashes)
        del reference_hashes
        indices, sources, strata = choose_rows(source_ids, input_features, names, boundaries, fraction, args.seed)
        print("source allocations: " + json.dumps(sources, ensure_ascii=False), flush=True)
        with tempfile.TemporaryDirectory(prefix=".sft-sample-", dir=output.parent) as staging:
            staged_output = Path(staging) / "subset.parquet"
            sizes = write_subset(source, staged_output, indices, args.row_group_size, args.shard_multiple)
            output_hashes, output_features, output_lengths = scan_files([staged_output], boundaries, True, args.workers, "verify output")
            if not pq.ParquetFile(staged_output).schema_arrow.equals(schema, check_metadata=True):
                raise RuntimeError("Output schema or metadata changed")
            for actual, expected in ((output_hashes, input_hashes[indices]), (output_features, input_features[indices]),
                                     (output_lengths, input_lengths[indices])):
                if not np.array_equal(actual, expected):
                    raise RuntimeError("Output verification failed: selected rows changed")
            if identities != [file_identity(path) for path in [source, *references]]:
                raise RuntimeError("An input file changed during sampling")
            rank_rows = [sum(sizes[rank::args.shard_multiple]) for rank in range(min(len(sizes), args.shard_multiple))]
            summary = {
                "input": str(source), "output": str(output), "fraction": str(fraction), "seed": args.seed,
                "rng": "numpy.PCG64", "numpy_version": np.__version__, "pyarrow_version": pa.__version__,
                "sampler_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "source_files": identities, "source_matching": match_report,
                "fingerprint": "SHA-256 of UTF-8 canonical JSON of messages/tools (sorted keys, no whitespace)",
                "sampling": "without replacement by input row index; source quota then feature quota, largest remainder",
                "stratification": ["source", "token_length_bucket", "has_reasoning", "multi_turn", "has_tools"],
                "length_boundaries": boundaries, "input_rows": len(input_hashes), "selected_rows": len(indices),
                "retained_fraction": len(indices) / len(input_hashes),
                "input_tokens": int(input_lengths.sum()), "selected_tokens": int(output_lengths.sum()),
                "source_allocations": sources, "strata": strata,
                "selected_input_indices": str(indices_path), "row_groups": len(sizes),
                "shard_multiple": args.shard_multiple, "rank_rows_min": min(rank_rows), "rank_rows_max": max(rank_rows),
                "all_selected_contents_and_lengths_verified": True,
                "selected_content_fingerprints_sha256": hashlib.sha256(output_hashes.tobytes()).hexdigest(),
                "elapsed_seconds": time.monotonic() - started,
            }
            staged_indices = Path(staging) / "indices.npy"
            staged_summary = Path(staging) / "summary.json"
            with staged_indices.open("xb") as handle:
                np.save(handle, indices, allow_pickle=False)
            with staged_summary.open("x") as handle:
                json.dump(summary, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            for staged, target in ((staged_indices, indices_path), (staged_output, output),
                                   (staged_summary, summary_path)):
                os.link(staged, target)
                created.append(target)
            print(f"COMPLETE: {output}; rows={len(indices):,}; summary={summary_path}", flush=True)
    except BaseException:
        for path in reversed(created):
            path.unlink(missing_ok=True)
        raise
    finally:
        if lock_owned:
            lock.unlink(missing_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source", action="append", required=True, metavar="NAME=PARQUET")
    parser.add_argument("--fraction", default="0.2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--length-boundaries", type=int, nargs="+", default=[2048, 4096, 8192, 16384])
    parser.add_argument("--row-group-size", type=int, default=2048)
    parser.add_argument("--shard-multiple", type=int, default=128)
    return parser.parse_args()


if __name__ == "__main__":
    sample(parse_args())
