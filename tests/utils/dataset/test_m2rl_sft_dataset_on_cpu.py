# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for M2RLSFTDataset in examples/llm_fusion/sft/dataset.py.

This is the gate that matters most for the SFT stage, because the failure it guards
against is silent: with ``reasoning_content`` dropped, training still converges and the
loss curve looks fine. The stock MultiTurnSFTDataset loses it on thinking models, so
``test_stock_dataset_loses_reasoning`` pins the bug down first and the rest prove the fix.

Runs on synthetic parquet written to a temp dir, so it needs no site data. It does need a
Qwen3 tokenizer for the ``<think>``-emitting chat template; set ``SFT_TEST_TOKENIZER`` to
override the default path, and the module skips if none is reachable.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "examples" / "llm_fusion" / "sft"))

from dataset import M2RLSFTDataset, decode_json_field, normalize_messages  # noqa: E402

TOKENIZER_PATH = os.environ.get(
    "SFT_TEST_TOKENIZER",
    "/taijifs_sz34/share_304169095/hunyuan/yongcanyu/checkpoints/Qwen/Qwen3-4B-Base",
)

# A reasoning trace long and distinctive enough that finding it in the decoded trained
# region cannot be a coincidence.
REASONING = "Let me work through this. First 17*20=340, then 17*3=51, so 340+51=391."
ANSWER = "The answer is 391."


def _base_config(**overrides):
    config = {
        "messages_key": "messages",
        "tools_key": "tools",
        "enable_thinking_key": "enable_thinking",
        "enable_thinking_default": None,
        "apply_chat_template_kwargs": {},
        "pad_mode": "no_padding",
        "truncation": "error",
        "max_length": 8192,
        "shuffle": False,
        "ignore_input_ids_mismatch": False,
    }
    config.update(overrides)
    return OmegaConf.create(config)


def _write_parquet(path, rows, row_group_size=None):
    frame = pd.DataFrame(rows)
    if row_group_size is None:
        frame.to_parquet(path)
    else:
        frame.to_parquet(path, row_group_size=row_group_size)
    return str(path)


def _thinking_row(reasoning=REASONING, answer=ANSWER, tools=None):
    """One user/assistant exchange whose assistant turn carries reasoning_content."""
    return {
        "messages": [
            {"role": "user", "content": "What is 17*23?"},
            {"role": "assistant", "content": answer, "reasoning_content": reasoning},
        ],
        "tools": tools,
    }


def _multiturn_row():
    """Two assistant turns, each with its own reasoning.

    Qwen3's template only emits <think> for the *last* assistant turn, so the first turn's
    reasoning is exactly what the stock per-turn render throws away.
    """
    return {
        "messages": [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "It is 4.", "reasoning_content": "Two plus two is four."},
            {"role": "user", "content": "And times three?"},
            {"role": "assistant", "content": "That is 12.", "reasoning_content": "Four times three is twelve."},
        ],
        "tools": None,
    }


class M2RLSFTDatasetTestCase(unittest.TestCase):
    """Shared tokenizer + temp dir for every test below."""

    @classmethod
    def setUpClass(cls):
        try:
            from verl.utils import hf_tokenizer

            cls.tokenizer = hf_tokenizer(TOKENIZER_PATH)
        except Exception as exc:  # noqa: BLE001 - any failure means we cannot test
            raise unittest.SkipTest(f"no Qwen3 tokenizer at {TOKENIZER_PATH}: {exc}") from exc
        if "<think>" not in (cls.tokenizer.chat_template or ""):
            raise unittest.SkipTest(f"{TOKENIZER_PATH} has no thinking chat template")

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def build(self, rows, *, cls=M2RLSFTDataset, config=None, max_samples=-1, name="data.parquet"):
        path = _write_parquet(self.tmp_path / name, rows)
        return cls(
            parquet_files=path,
            tokenizer=self.tokenizer,
            config=config if config is not None else _base_config(),
            processor=None,
            max_samples=max_samples,
        )

    def trained_text(self, row):
        """Decode only the tokens the loss is actually computed on."""
        ids = row["input_ids"].tolist()
        mask = row["loss_mask"].tolist()
        return self.tokenizer.decode([i for i, m in zip(ids, mask, strict=True) if m == 1])


class TestReasoningPreserved(M2RLSFTDatasetTestCase):
    def test_stock_dataset_loses_reasoning(self):
        """The bug this class exists to fix must be real, not hypothetical.

        If this ever starts failing, upstream fixed the per-turn render and the whole
        prefix-difference override may be removable.
        """
        stock = self.build([_thinking_row()], cls=MultiTurnSFTDataset, config=_base_config(
            ignore_input_ids_mismatch=True
        ))
        self.assertNotIn(REASONING, self.tokenizer.decode(stock[0]["input_ids"].tolist()))

    def test_reasoning_reaches_the_trained_region(self):
        """Reasoning must land where loss_mask == 1, not merely somewhere in input_ids.

        Text present but masked out teaches the model nothing, and is indistinguishable
        from the fix working if you only grep the decoded ids.
        """
        dataset = self.build([_thinking_row()])
        self.assertIn(REASONING, self.trained_text(dataset[0]))

    def test_every_turn_keeps_its_own_reasoning(self):
        """Including the non-final assistant turn, which the template alone would strip."""
        dataset = self.build([_multiturn_row()])
        trained = self.trained_text(dataset[0])
        self.assertIn("Two plus two is four.", trained)
        self.assertIn("Four times three is twelve.", trained)

    def test_think_tag_survives(self):
        dataset = self.build([_thinking_row()])
        self.assertIn("<think>", self.tokenizer.decode(dataset[0]["input_ids"].tolist()))


class TestLossMask(M2RLSFTDatasetTestCase):
    def test_user_turns_carry_no_loss(self):
        """No label leakage: training on the prompt would teach the model to ask questions."""
        dataset = self.build([_multiturn_row()])
        row = dataset[0]
        ids = row["input_ids"].tolist()
        mask = row["loss_mask"].tolist()

        user_start = self.tokenizer.encode("<|im_start|>user", add_special_tokens=False)
        leaked = 0
        for i in range(len(ids) - len(user_start) + 1):
            if ids[i : i + len(user_start)] == user_start:
                # Walk to the end of this user span and count any trained token in it.
                end = i
                im_end = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
                while end < len(ids) and ids[end] != im_end:
                    end += 1
                leaked += sum(mask[i:end])
        self.assertEqual(leaked, 0, "user tokens must never carry loss_mask == 1")

    def test_generation_prompt_is_masked(self):
        """'<|im_start|>assistant' is the cue the model is *given*, not what it produces."""
        dataset = self.build([_thinking_row()])
        self.assertNotIn("<|im_start|>assistant", self.trained_text(dataset[0]))

    def test_turn_ending_token_is_trained(self):
        """The model has to learn to emit <|im_end|>.

        If it is masked out, generation never stops -- at eval time that shows up as
        finish_reason=length on every sample, not as a bad score. Note the trained region
        ends '<|im_end|>\\n': the template's trailing newline is inside the turn, so this
        checks membership rather than that it is the very last token.
        """
        dataset = self.build([_thinking_row()])
        row = dataset[0]
        im_end = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        trained_ids = [i for i, m in zip(row["input_ids"].tolist(), row["loss_mask"].tolist(), strict=True) if m == 1]
        self.assertIn(im_end, trained_ids)


class TestToolsDecoding(M2RLSFTDatasetTestCase):
    """The stored `tools` column is a JSON string, sometimes doubly encoded."""

    def test_decode_json_field_handles_double_encoding(self):
        self.assertEqual(decode_json_field('"[]"'), [])
        self.assertEqual(decode_json_field("[]"), [])
        self.assertEqual(decode_json_field(""), [])
        self.assertEqual(decode_json_field('[{"name": "f"}]'), [{"name": "f"}])
        # Not JSON at all -- pass through rather than raising.
        self.assertEqual(decode_json_field("not json"), "not json")

    def test_double_encoded_empty_tools_does_not_reach_the_template(self):
        """Left as the string '"[]"', apply_chat_template raises ValueError: Tools should
        either be a JSON schema, or a callable function."""
        dataset = self.build([_thinking_row(tools='"[]"')])
        self.assertIsNone(dataset.tools[0])
        self.assertIn(REASONING, self.trained_text(dataset[0]))

    def test_real_tool_schema_is_decoded_to_a_list(self):
        schema = [
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": "Do arithmetic.",
                    "parameters": {"type": "object", "properties": {"expr": {"type": "string"}}},
                },
            }
        ]
        dataset = self.build([_thinking_row(tools=json.dumps(schema))])
        self.assertEqual(dataset.tools[0], schema)

    def test_normalize_messages_drops_empty_tool_calls(self):
        """Qwen3's template branches on `{%- if message.tool_calls %}`; a stray '[]' makes
        `tojson` hit a Jinja Undefined and raise TypeError."""
        normalized = normalize_messages([{"role": "assistant", "content": "hi", "tool_calls": "[]"}])
        self.assertNotIn("tool_calls", normalized[0])

    def test_normalize_messages_drops_none_values(self):
        normalized = normalize_messages([{"role": "user", "content": "hi", "reasoning_content": None}])
        self.assertEqual(normalized[0], {"role": "user", "content": "hi"})


class TestRankSharding(M2RLSFTDatasetTestCase):
    """SFT_SHARD_BY_RANK=1 is mandatory on the full mixture (~700GB per process without
    it), so its contract with the trainer's sampler needs pinning down."""

    def setUp(self):
        super().setUp()
        self.rows = [_thinking_row(reasoning=f"Reasoning number {i}.") for i in range(12)]
        # One row per group, so sharding is observable at this size.
        self.path = _write_parquet(self.tmp_path / "sharded.parquet", self.rows, row_group_size=1)
        for name in ("SFT_SHARD_BY_RANK", "RANK", "WORLD_SIZE"):
            self.addCleanup(os.environ.pop, name, None)

    def _shard(self, rank, world_size):
        os.environ["SFT_SHARD_BY_RANK"] = "1"
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        return M2RLSFTDataset(
            parquet_files=self.path,
            tokenizer=self.tokenizer,
            config=_base_config(),
            processor=None,
            max_samples=-1,
        )

    def test_shards_are_disjoint_and_complete(self):
        world_size = 4
        seen = []
        for rank in range(world_size):
            dataset = self._shard(rank, world_size)
            seen.append({m[0]["content"] + m[1]["reasoning_content"] for m in dataset.messages})
        union = set().union(*seen)
        self.assertEqual(sum(len(s) for s in seen), len(union), "shards overlap")
        self.assertEqual(len(union), len(self.rows), "shards do not cover the data")

    def test_len_is_equal_across_ranks(self):
        """Every rank must run the same number of collectives or the job deadlocks.

        Shard sizes differ when row groups do not divide evenly, so __len__ reports the
        largest shard and __getitem__ wraps; 12 rows over 5 ranks is such a case.
        """
        lengths = {len(self._shard(rank, 5)) for rank in range(5)}
        self.assertEqual(len(lengths), 1, f"ranks disagree on __len__: {lengths}")

    def test_rank_sharded_flag_is_set_for_the_trainer(self):
        """sft_trainer.py reads this attribute to pick a one-replica sampler. If it is
        missing or False, the sampler shards the already-disjoint shard again and the run
        silently trains 1/world_size of the data."""
        self.assertTrue(self._shard(0, 4).rank_sharded)

    def test_single_process_is_not_rank_sharded(self):
        os.environ["SFT_SHARD_BY_RANK"] = "1"
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        dataset = M2RLSFTDataset(
            parquet_files=self.path,
            tokenizer=self.tokenizer,
            config=_base_config(),
            processor=None,
            max_samples=-1,
        )
        self.assertFalse(dataset.rank_sharded)
        self.assertEqual(len(dataset), len(self.rows))

    def test_more_ranks_than_row_groups_fails_loudly(self):
        """A rank with zero row groups would otherwise train on an empty shard."""
        path = _write_parquet(self.tmp_path / "tiny.parquet", self.rows[:2], row_group_size=1)
        os.environ["SFT_SHARD_BY_RANK"] = "1"
        os.environ["RANK"] = "3"
        os.environ["WORLD_SIZE"] = "4"
        with self.assertRaises(ValueError):
            M2RLSFTDataset(
                parquet_files=path,
                tokenizer=self.tokenizer,
                config=_base_config(),
                processor=None,
                max_samples=-1,
            )


class TestSanityCheck(M2RLSFTDatasetTestCase):
    def test_prefix_path_keeps_the_parent_check(self):
        """On rows that render monotonically, the parent's whole-conversation equality
        check still runs -- it is the only remaining detector for real tokenization bugs,
        which is why ignore_input_ids_mismatch must stay False in the launcher."""
        dataset = self.build([_thinking_row()])
        row = dataset[0]  # would raise inside __getitem__ if the check failed
        self.assertFalse(dataset._last_row_used_fallback)
        self.assertGreater(int(row["loss_mask"].sum()), 0)

    def test_truncation_error_is_reachable(self):
        """The launcher runs truncation=error against pre-filtered data, so an overlong row
        must actually raise rather than being silently cut."""
        dataset = self.build([_thinking_row()], config=_base_config(max_length=16, truncation="error"))
        with self.assertRaises(Exception):
            _ = dataset[0]


if __name__ == "__main__":
    unittest.main()
