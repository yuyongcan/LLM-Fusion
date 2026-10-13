import argparse
from fractions import Fraction
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


SCRIPT = Path(__file__).resolve().parents[3] / "examples/llm_fusion/sft/stratified_sample.py"
SPEC = importlib.util.spec_from_file_location("sft_stratified_sample", SCRIPT)
SAMPLER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SAMPLER)


def test_quotas_and_reproducibility():
    sources = np.repeat([0, 1, 2], [101, 77, 22])
    codes = np.arange(len(sources), dtype=np.int16) % 32
    first, report, strata = SAMPLER.choose_rows(sources, codes, ["a", "b", "c"], (2048, 4096, 8192, 16384), Fraction(1, 5), 42)
    second, _, _ = SAMPLER.choose_rows(sources, codes, ["a", "b", "c"], (2048, 4096, 8192, 16384), Fraction(1, 5), 42)
    assert np.array_equal(first, second)
    assert len(np.unique(first)) == 40
    assert sum(row["selected_rows"] for row in strata) == 40
    assert all(abs(row["selected_rows"] - row["input_rows"] / 5) <= 1 for row in report)
    assert SAMPLER.quotas([0, 1, 1], 1).tolist() == [0, 1, 0]


def test_source_matching_rejects_missing_and_ambiguous():
    h = lambda text: np.void(SAMPLER.fingerprint({"messages": [], "tools": text}))
    reference = np.array([h("a"), h("a"), h("b")], dtype=SAMPLER.FINGERPRINT_DTYPE)
    labels, report = SAMPLER.recover_sources(reference, [2, 1], reference[[2, 0]])
    assert labels.tolist() == [1, 0]
    assert report["reference_unique_contents"] == 2
    with pytest.raises(ValueError, match="no exact-content"):
        SAMPLER.recover_sources(reference, [2, 1], np.array([h("missing")], dtype=SAMPLER.FINGERPRINT_DTYPE))
    with pytest.raises(ValueError, match="multiple sources"):
        SAMPLER.recover_sources(reference, [1, 2], reference[:1])


def test_fingerprint_and_conversation_features():
    row = {"tools": "\"[]\"", "messages": [{"role": "assistant", "reasoning_content": "reason", "content": "answer"}], "token_length": 2048}
    assert SAMPLER.features(row, (2048, 4096)) == 4
    reordered = {"messages": [{"content": "answer", "reasoning_content": "reason", "role": "assistant"}], "tools": "\"[]\""}
    assert SAMPLER.fingerprint(row) == SAMPLER.fingerprint(reordered)
    row["messages"].append({"role": "assistant", "tool_calls": "[{\"name\":\"x\"}]"})
    row["token_length"] = 2049
    assert SAMPLER.features(row, (2048, 4096)) == 15
    row["token_length"] = 5000
    with pytest.raises(ValueError, match="token_length"):
        SAMPLER.features(row, (2048, 4096))


def test_row_groups_balance_training_ranks():
    sizes = SAMPLER.row_group_sizes(2_444_168, 2048, 128)
    assert sum(sizes) == 2_444_168
    assert len(sizes) % 128 == 0
    ranks = [sum(sizes[rank::128]) for rank in range(128)]
    assert max(ranks) - min(ranks) <= 1
    assert max(sizes) <= 2048


def test_end_to_end_preserves_rows_and_refuses_overwrite(tmp_path):
    messages_type = pa.list_(pa.struct([
        ("content", pa.string()), ("reasoning_content", pa.string()),
        ("role", pa.string()), ("tool_calls", pa.string()),
    ]))
    schema = pa.schema([("messages", messages_type), ("tools", pa.string())])
    source_rows = []
    references = []
    for name in ["a", "b"]:
        rows = [{"messages": [{"content": f"{name}-{i}", "reasoning_content": "trace" if i % 2 else None,
                               "role": "assistant", "tool_calls": None}], "tools": "[]"} for i in range(100)]
        path = tmp_path / f"{name}.parquet"
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), path, row_group_size=25)
        references.append(f"{name}={path}")
        source_rows.extend(rows)
    order = np.random.default_rng(17).permutation(len(source_rows))[:150]
    rows = [{**source_rows[i], "token_length": int(100 + i * 20)} for i in order]
    source = tmp_path / "train.parquet"
    output = tmp_path / "subset.parquet"
    full_schema = schema.append(pa.field("token_length", pa.int32())).with_metadata({b"test": b"preserved"})
    pq.write_table(pa.Table.from_pylist(rows, schema=full_schema), source, row_group_size=25)
    command = [sys.executable, str(SCRIPT), "--input", str(source), "--output", str(output),
               "--workers", "2", "--shard-multiple", "4", "--row-group-size", "8"]
    for reference in references:
        command.extend(["--source", reference])
    completed = subprocess.run(command, capture_output=True, text=True, timeout=120)
    assert completed.returncode == 0, completed.stderr
    summary = json.loads(output.with_suffix(".parquet.json").read_text())
    indices = np.load(output.with_suffix(".parquet.indices.npy"), allow_pickle=False)
    assert summary["selected_rows"] == 30
    assert summary["all_selected_contents_and_lengths_verified"]
    actual = pq.read_table(output)
    expected = pq.read_table(source).take(pa.array(indices))
    assert actual.equals(expected, check_metadata=True)
    again = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert again.returncode != 0
    assert "Refusing to overwrite" in again.stderr
    assert pq.read_table(output).equals(actual, check_metadata=True)
    assert not list(tmp_path.glob(".sft-sample-*"))
    assert not list(tmp_path.glob("*.lock"))
