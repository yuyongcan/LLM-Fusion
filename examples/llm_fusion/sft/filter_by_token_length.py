#!/usr/bin/env python3
"""Filter the SFT parquet by the exact pre-truncation training sequence length.

The length that matters is the one ``M2RLSFTDataset`` actually produces, so this imports
that very class and reads ``len(input_ids)`` per row. Estimating from character counts, or
truncating first and then measuring, both give the wrong answer -- the chat template and
the preserved ``reasoning_content`` are a large part of the total.

Run one process per rank with a distinct ``--rank``; each reads only its own row groups
(``SFT_SHARD_BY_RANK``), writes ``part-<rank>.parquet``, and ``merge`` concatenates them.
Each kept row gains a ``token_length: int32`` column for later auditing.

    # per rank, 32 or 64 processes in parallel across the cluster
    python filter_by_token_length.py filter \\
        --source "${SFT_DATA_ROOT}/train.parquet" \\
        --output-dir "${SFT_DATA_ROOT}/.train.le16384.parts" \\
        --model-path "${SFT_BASE_MODEL}" --rank 0 --world-size 64 --max-length 16384

    python filter_by_token_length.py merge \\
        --source "${SFT_DATA_ROOT}/train.parquet" \\
        --parts-dir "${SFT_DATA_ROOT}/.train.le16384.parts" \\
        --output "${SFT_DATA_ROOT}/train.le16384.parquet" --world-size 64
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from omegaconf import OmegaConf
from transformers import AutoTokenizer

# dataset.py sits next to this file and is not an installed module (it is loaded by path
# through verl's data.custom_cls hook), so make this directory importable.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import M2RLSFTDataset  # noqa: E402


def atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def filter_rank(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"part-{args.rank:05d}.parquet"
    summary_path = output_dir / f"part-{args.rank:05d}.json"
    if output.exists() and summary_path.exists() and not args.force:
        print(f"rank {args.rank}: output already complete, skipping", flush=True)
        return

    os.environ["SFT_SHARD_BY_RANK"] = "1"
    os.environ["RANK"] = str(args.rank)
    os.environ["WORLD_SIZE"] = str(args.world_size)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    logging.getLogger().setLevel(logging.ERROR)
    logging.getLogger("dataset").setLevel(logging.ERROR)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    config = OmegaConf.create(
        {
            "pad_mode": "no_padding",
            "truncation": "error",
            "max_length": 1_000_000_000,
            "messages_key": "messages",
            "tools_key": "tools",
            "enable_thinking_key": "enable_thinking",
            "enable_thinking_default": None,
            "apply_chat_template_kwargs": {},
            "shuffle": False,
            "ignore_input_ids_mismatch": False,
        }
    )
    dataset = M2RLSFTDataset(
        parquet_files=args.source,
        tokenizer=tokenizer,
        config=config,
        processor=None,
        max_samples=-1,
    )
    source_rows = len(dataset.messages)
    if args.max_rows is not None:
        source_rows = min(source_rows, args.max_rows)
    tmp_output = output.with_suffix(".parquet.tmp")
    tmp_output.unlink(missing_ok=True)
    writer: pq.ParquetWriter | None = None
    kept_indices: list[int] = []
    kept_lengths: list[int] = []
    kept = 0
    removed = 0
    started = time.time()

    def flush() -> None:
        nonlocal writer, kept_indices, kept_lengths
        if not kept_indices:
            return
        table = pa.Table.from_pandas(dataset.dataframe.iloc[kept_indices], preserve_index=False)
        table = table.append_column("token_length", pa.array(kept_lengths, type=pa.int32()))
        if writer is None:
            writer = pq.ParquetWriter(tmp_output, table.schema, compression="zstd")
        writer.write_table(table, row_group_size=args.row_group_size)
        kept_indices = []
        kept_lengths = []

    try:
        for index in range(source_rows):
            token_length = int(dataset[index]["input_ids"].numel())
            if token_length <= args.max_length:
                kept_indices.append(index)
                kept_lengths.append(token_length)
                kept += 1
                if len(kept_indices) >= args.write_batch_size:
                    flush()
            else:
                removed += 1
            if (index + 1) % args.log_every == 0:
                elapsed = max(time.time() - started, 1e-6)
                print(
                    f"rank {args.rank}: {index + 1}/{source_rows}, kept={kept}, "
                    f"removed={removed}, rows/s={(index + 1) / elapsed:.2f}",
                    flush=True,
                )
        flush()
    finally:
        if writer is not None:
            writer.close()

    if kept == 0:
        tmp_output.unlink(missing_ok=True)
        raise RuntimeError(f"rank {args.rank}: no rows remained after filtering")
    tmp_output.replace(output)
    summary = {
        "rank": args.rank,
        "world_size": args.world_size,
        "source": str(Path(args.source).resolve()),
        "model_path": str(Path(args.model_path).resolve()),
        "max_length": args.max_length,
        "source_rows": source_rows,
        "kept_rows": kept,
        "removed_rows": removed,
        "elapsed_seconds": time.time() - started,
        "output": str(output),
    }
    atomic_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def merge_parts(args: argparse.Namespace) -> None:
    parts_dir = Path(args.parts_dir).resolve()
    output = Path(args.output).resolve()
    parts = [parts_dir / f"part-{rank:05d}.parquet" for rank in range(args.world_size)]
    summaries = [parts_dir / f"part-{rank:05d}.json" for rank in range(args.world_size)]
    missing = [str(path) for path in parts + summaries if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} filter artifacts: {missing[:5]}")
    if output == Path(args.source).resolve():
        raise ValueError("refusing to overwrite the source parquet")

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = output.with_suffix(output.suffix + ".tmp")
    tmp_output.unlink(missing_ok=True)
    writer: pq.ParquetWriter | None = None
    total_rows = 0
    try:
        for part in parts:
            parquet = pq.ParquetFile(part)
            for batch in parquet.iter_batches(batch_size=args.row_group_size):
                table = pa.Table.from_batches([batch])
                if writer is None:
                    writer = pq.ParquetWriter(tmp_output, table.schema, compression="zstd")
                writer.write_table(table, row_group_size=args.row_group_size)
                total_rows += table.num_rows
    finally:
        if writer is not None:
            writer.close()
    if total_rows == 0:
        tmp_output.unlink(missing_ok=True)
        raise RuntimeError("merged output would be empty")
    tmp_output.replace(output)

    rank_summaries = [json.loads(path.read_text()) for path in summaries]
    metadata = {
        "source": str(Path(args.source).resolve()),
        "output": str(output),
        "max_length": rank_summaries[0]["max_length"],
        "source_rows": sum(item["source_rows"] for item in rank_summaries),
        "kept_rows": total_rows,
        "removed_rows": sum(item["removed_rows"] for item in rank_summaries),
        "world_size": args.world_size,
        "row_groups": pq.ParquetFile(output).num_row_groups,
    }
    atomic_json(output.with_suffix(output.suffix + ".json"), metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    filter_parser = subparsers.add_parser("filter")
    filter_parser.add_argument("--source", required=True)
    filter_parser.add_argument("--output-dir", required=True)
    filter_parser.add_argument("--model-path", required=True)
    filter_parser.add_argument("--rank", type=int, required=True)
    filter_parser.add_argument("--world-size", type=int, required=True)
    filter_parser.add_argument("--max-length", type=int, default=16384)
    filter_parser.add_argument("--write-batch-size", type=int, default=10000)
    filter_parser.add_argument("--row-group-size", type=int, default=10000)
    filter_parser.add_argument("--log-every", type=int, default=1000)
    filter_parser.add_argument("--max-rows", type=int)
    filter_parser.add_argument("--force", action="store_true")

    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument("--source", required=True)
    merge_parser.add_argument("--parts-dir", required=True)
    merge_parser.add_argument("--output", required=True)
    merge_parser.add_argument("--world-size", type=int, required=True)
    merge_parser.add_argument("--row-group-size", type=int, default=10000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "filter":
        filter_rank(args)
    else:
        merge_parts(args)


if __name__ == "__main__":
    main()
