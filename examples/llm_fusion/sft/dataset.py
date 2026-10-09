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

"""SFT dataset for the Nemotron reasoning mixture, mounted via ``data.custom_cls``.

Used by ``train_sft_qwen3_4b_base.sh`` to turn Qwen3-4B-*Base* into the chat/reasoning
model the RL stages start from. The measurements below were taken on
``${SFT_DATA_ROOT}/Nemotron-Math-v2.parquet``; ``overview.md`` has the full record.

Two problems with feeding this parquet to verl's stock ``MultiTurnSFTDataset``:

1. **Thinking traces are silently dropped.** The stock dataset renders each turn in
   isolation (``apply_chat_template(messages=[message])``). Qwen3's template only emits the
   ``<think>`` block when ``loop.index0 > ns.last_query_index`` -- a lone assistant message
   fails that test, so the block is discarded. Measured: 6/6 rows lost 100% of
   ``reasoning_content``, up to 15,369 chars in a single row. Training still converges, so
   nothing surfaces the loss. verl's own ``sanity_check`` does catch the resulting mismatch,
   but its suggested remedy (``ignore_input_ids_mismatch=True``) just silences the alarm.

   Fix: render *cumulative prefixes* and take the string difference, so every turn carries
   its preceding context. This mirrors what the reference implementation does -- slime's
   ``gen_multi_turn_loss_mask_qwen3`` splices in a sentinel message for the same reason --
   and is verified token-identical to it (ids **and** loss_mask, 6/6 rows).

2. **``tools`` / ``tool_calls`` are JSON strings, sometimes double-encoded** (the stored
   value is literally ``'"[]"'``). verl's ``convert_nested_value_to_list_recursive`` passes
   strings through untouched, so ``apply_chat_template`` raises
   ``ValueError: Tools should either be a JSON schema, or a callable function...``.

Only ``_process_single_message`` and ``sanity_check`` are overridden. Everything else --
``__getitem__``, pad modes, truncation, multimodal handling -- is inherited, so this keeps
tracking upstream verl instead of forking it. The ``sanity_check`` override exempts *only*
rows that took the per-turn fallback, because on those the parent compares against the very
reasoning-stripping render being fixed; see that method for the measurements.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Optional

import torch

from verl.utils.chat_template import extract_system_prompt_and_generation
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset
from verl.utils.py_functional import convert_nested_value_to_list_recursive

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

__all__ = [
    "M2RLSFTDataset",
    "align_user_prompt",
    "decode_json_field",
    "normalize_messages",
]

#: Suffix the RL datasets and the eval suite append to every math/science question
#: (``data/LLM-Fusion/train/{Math,Science}``, ``data/LLM-Fusion/test/AIME2025``).
RL_REASONING_SUFFIX = "\n\nLet's think step by step and output the final answer within \\boxed{}."

#: Wrapper the RL/eval code sets put around every competitive-programming question
#: (``data/LLM-Fusion/train/Code``, ``test/LCB_v5``). The SFT code mixture carries no
#: equivalent instruction at all -- measured 0/400 sampled rows -- so here the SFT data
#: moves toward RL by gaining it, the opposite direction from math/science.
RL_CODE_PREFIX = (
    "You will be given a question (problem specification) and will generate a correct "
    "Python program that matches the specification and passes all tests.\n\nQuestion: "
)
RL_CODE_SUFFIX = (
    "\n\nRead the inputs from stdin solve the problem and write the answer to stdout "
    "(do not directly test on the sample inputs). Enclose your code within delimiters "
    "as follows. Ensure that when the python program runs, it reads the inputs, runs "
    "the algorithm and writes output to STDOUT.```python\n# YOUR CODE HERE\n```"
)

#: Instruction prefixes the Nemotron SFT mixture puts *before* the question. Each is
#: stripped and replaced by the RL phrasing. Matching is anchored at the start and
#: tolerant of the ``\boxed`` -> ``\x08oxed`` (backspace) mangling present in part of the
#: science split, which is why these are regexes rather than literals.
_SFT_REASONING_PREFIXES = (
    # Math-v2: 800/800 sampled rows, verbatim.
    re.compile(r"^Solve the following math problem\.[^\n]*?inside (?:\\b|\x08)oxed\{\}\.\s*\n+"),
    # Science-v1 free-form: 191/800. Same sentence without the word "math".
    re.compile(r"^Solve the following problem\.[^\n]*?inside (?:\\b|\x08)oxed\{\}\.\s*\n+"),
    # Science-v1 multiple choice: 530/800 across three option-count variants.
    re.compile(r"^Answer the following multiple choice question\.[^\n]*?\)\.\s*\n+"),
    re.compile(r"^The following is a multiple-choice question\.[^\n]*?\n+"),
    re.compile(r"^Help me with this mutliple-choice question\.\s*\n*"),
)


def align_user_prompt(content: str, domain: str) -> str:
    """Rewrite one SFT question into the phrasing its RL/eval counterpart uses.

    The three stages disagreed on how to ask for an answer: SFT put the format
    instruction *before* the question, RL and the eval suite append a different one
    *after* it. Measured on ``Jackwang111/M2RL-SFT``, serving AIME'25 the SFT-style
    prompt instead of ours moved acc .3208 -> .4500 and cut median response length
    17630 -> 5524 tokens, because the model only emits ``\\boxed{}`` reliably for the
    phrasing it was trained on. RL and eval already agree, so SFT is the side that moves.

    ``domain`` comes from the source filename, or is ``"auto"`` for the merged mixture,
    where the filename says nothing. Under ``"auto"`` only rows whose *own* text starts
    with a recognised reasoning prefix are rewritten (37.6% of 1000 sampled rows from
    ``train.le16384.parquet``); everything else -- code, IF, agent, and any unrecognised
    phrasing -- is left exactly as stored. That asymmetry is deliberate: appending a
    \\boxed instruction to an IF or agent row would be actively wrong, so "auto" never
    guesses, it only acts on positive evidence.

    The IF split is excluded even in per-domain mode (its SFT rows carry a system message
    that RL/eval do not, and IFEval already scores 81.0 against the paper's 79.48).
    """
    if domain == "auto":
        if content.rstrip().endswith(RL_REASONING_SUFFIX.strip()):
            return content
        for pattern in _SFT_REASONING_PREFIXES:
            stripped, n = pattern.subn("", content, count=1)
            if n:
                return stripped.rstrip() + RL_REASONING_SUFFIX
        return content
    if domain in ("math", "science"):
        if content.rstrip().endswith(RL_REASONING_SUFFIX.strip()):
            return content
        for pattern in _SFT_REASONING_PREFIXES:
            stripped, n = pattern.subn("", content, count=1)
            if n:
                return stripped.rstrip() + RL_REASONING_SUFFIX
        # No known prefix: still append the suffix so the answer format is requested.
        return content.rstrip() + RL_REASONING_SUFFIX
    if domain == "code":
        if "Enclose your code within delimiters" in content:
            return content
        return RL_CODE_PREFIX + content.strip() + RL_CODE_SUFFIX
    return content


def decode_json_field(value: Any) -> Any:
    """Decode a possibly (multiply) JSON-encoded field into a real Python object.

    The mixture stores ``tools`` as ``'"[]"'`` -- a JSON string *containing* a JSON string -- so a
    single ``json.loads`` is not always enough. Returns the value unchanged if it does not
    parse, and ``[]`` for empty input.
    """
    for _ in range(3):
        if not isinstance(value, str):
            break
        text = value.strip()
        if not text:
            return []
        try:
            value = json.loads(text)
        except (ValueError, TypeError):
            return value
    return value


def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop null-valued keys and turn ``tool_calls`` into a real list.

    An empty ``tool_calls`` is removed rather than left as ``[]``: Qwen3's template branches
    on ``{%- if message.tool_calls %}``, and a stray falsy value is noise. ``pandas`` may
    hand us ``NAType`` for missing cells, which is not iterable -- treat it as absent.
    """
    normalized = []
    for message in messages:
        clean: dict[str, Any] = {}
        for key, value in message.items():
            if value is None:
                continue
            # pandas NA / NaN sentinels are not None but are still "missing"
            if not isinstance(value, str | list | dict | int | float | bool):
                continue
            if isinstance(value, float) and value != value:  # NaN
                continue
            clean[key] = value

        tool_calls = decode_json_field(clean.pop("tool_calls", None))
        if tool_calls:
            clean["tool_calls"] = tool_calls
        normalized.append(clean)
    return normalized


#: Source-filename fragment -> domain, for prompt alignment. Matched case-insensitively
#: against every parquet path. ``if`` and ``agent`` are intentionally absent: their SFT
#: and RL phrasings already agree closely enough that rewriting would risk a regression
#: (see ``align_user_prompt``).
_DOMAIN_MARKERS = (
    ("math", "math"),
    ("competitive-programming", "code"),
    ("science", "science"),
)


def _infer_domain(parquet_files) -> Optional[str]:
    """Infer the mixture domain from the source filename(s).

    Returns ``None`` -- meaning "leave prompts alone" -- when the files span more than
    one domain (e.g. the merged ``train.le16384.parquet``), because a single rewrite rule
    cannot be correct for all of them. Per-domain runs, which is what
    ``run_sft_5expert_serial.sh`` does, are the case this supports.
    """
    paths = [parquet_files] if isinstance(parquet_files, str) else list(parquet_files)
    found = set()
    for path in paths:
        name = str(path).lower()
        for marker, domain in _DOMAIN_MARKERS:
            if marker in name:
                found.add(domain)
    if len(found) == 1:
        return found.pop()
    # Mixed or unrecognised filename (the merged train.le16384.parquet is both): fall
    # back to per-row detection rather than giving up, so the merged mixture still gets
    # its math/science rows aligned. See align_user_prompt's "auto" branch.
    if len(found) > 1:
        logger.warning("Mixed domains %s in %s; using per-row prompt alignment.", sorted(found), paths)
    return "auto"


class M2RLSFTDataset(MultiTurnSFTDataset):
    """``MultiTurnSFTDataset`` that preserves ``reasoning_content`` on thinking models."""

    #: Domain whose RL phrasing this dataset's questions are rewritten into, or ``None``
    #: to leave them as stored. Set in ``_read_files_and_process``.
    align_domain: Optional[str] = None

    _row_tools: Optional[list[dict[str, Any]]] = None
    _row_used_fallback: bool = False
    #: Whether the row returned by the most recent ``__getitem__`` took the per-turn
    #: fallback. Survives past the call (unlike ``_row_used_fallback``, which is reset) so
    #: tests and data audits can select fallback rows without re-rendering them.
    _last_row_used_fallback: bool = False

    def _read_files_and_process(self):
        """Load rows via Arrow row groups, then decode the JSON-string ``tools`` column.

        **Why not just call super().** The parent does
        ``pd.read_parquet(file, dtype_backend="pyarrow")`` over the whole file and then
        subsamples with ``.iloc[]`` (``multiturn_sft_dataset.py:147,160``). On these files
        that raises ``pyarrow.lib.ArrowInvalid: offset overflow while concatenating arrays``
        -- Arrow's per-chunk 2GB offset limit, hit because these rows carry very long
        reasoning traces (Nemotron-Math-v2 is 28GB over 2.95M rows). Reproducible without any
        of our code:

            df = pd.read_parquet(src, dtype_backend="pyarrow")   # 2,950,525 rows, fine
            df.iloc[rng.choice(len(df), 64).tolist()]            # ArrowInvalid

        Reading only the row groups we need also sidesteps the parent's other scaling
        problem: it materializes every row through ``.apply(...).tolist()`` regardless of
        ``train_max_samples``, so a 104GB mixed file would balloon in RAM just to train on a
        subsample.

        The parent's ``max_samples`` semantics are preserved: ``shuffle`` picks random row
        *groups* (cheap and still well-mixed, since group assignment is unrelated to
        content), otherwise groups are taken in order.

        **Rank sharding (``SFT_SHARD_BY_RANK=1``).** ``sft_trainer.py:236`` has every
        rank build the whole dataset and then carve it up with a ``DistributedSampler``. That
        is fine for small files and fatal for this one: measured on ``train.parquet``, one
        process holds ~7.7GB per 110k rows, i.e. **~700GB for all 14.1M rows** (linear,
        checked at 110k/220k/440k rows). Eight ranks per node would need ~5.6TB against
        2.2TB of RAM, so a full-data run OOMs during loading regardless of GPU count.

        With sharding on, each rank reads only ``row_groups[rank::world_size]``. The trainer
        must then use a one-replica sampler on each already-distinct local shard; applying its
        normal ``DistributedSampler(num_replicas=world_size)`` again would retain only
        1/world_size of every shard (double sharding). Local shard lengths are padded to the
        largest shard by repeating a few local rows so every rank executes the same number of
        collectives. Row groups are content-independent, so the shards remain well mixed.
        """
        import numpy as np
        import pandas as pd
        import pyarrow.parquet as pq

        # On by default: the three stages disagreeing on prompt phrasing is a measured
        # 13-point swing (see align_user_prompt). Set SFT_ALIGN_PROMPTS=0 to train on the
        # questions exactly as stored, which is what every run before this change did.
        if os.environ.get("SFT_ALIGN_PROMPTS", "1") == "1":
            self.align_domain = _infer_domain(self.parquet_files)
            if self.align_domain:
                print(f"prompt alignment: rewriting {self.align_domain} questions to RL phrasing")

        # Opt-in because it changes what a rank sees; harmless single-process default.
        shard = os.environ.get("SFT_SHARD_BY_RANK", "0") == "1"
        rank = int(os.environ.get("RANK", "0"))
        world = int(os.environ.get("WORLD_SIZE", "1"))
        rank_sharded = shard and world > 1
        global_max_samples = self.max_samples
        shard_row_counts = [0] * world if rank_sharded else None
        if rank_sharded and global_max_samples and global_max_samples > 0:
            if global_max_samples < world:
                raise ValueError(f"max_samples={global_max_samples} must be >= world_size={world}")
            base, extra = divmod(global_max_samples, world)
            self.max_samples = base + int(rank < extra)
        elif not rank_sharded:
            rank, world = 0, 1

        frames = []
        remaining = self.max_samples if self.max_samples and self.max_samples > 0 else None

        for path in self.parquet_files:
            handle = pq.ParquetFile(path)
            group_order = list(range(handle.num_row_groups))
            if remaining is not None and self.shuffle:
                rng = np.random.default_rng(*( (self.seed,) if self.seed is not None else () ))
                rng.shuffle(group_order)
            if rank_sharded:
                # Every rank can derive the largest shard from Parquet metadata. Padding all
                # local dataset lengths to that value keeps collective counts identical.
                for shard_rank in range(world):
                    shard_row_counts[shard_rank] += sum(
                        handle.metadata.row_group(group_idx).num_rows
                        for group_idx in group_order[shard_rank::world]
                    )
                shard_groups = group_order[rank::world]
                if not shard_groups:
                    raise ValueError(
                        f"rank {rank}/{world} got 0 row groups from {path} "
                        f"({handle.num_row_groups} total). Use fewer ranks than row groups, "
                        f"or unset SFT_SHARD_BY_RANK."
                    )
                group_order = shard_groups

            for group_idx in group_order:
                table = handle.read_row_group(group_idx)
                if remaining is not None and table.num_rows > remaining:
                    table = table.slice(0, remaining)
                frames.append(table.to_pandas())
                if remaining is not None:
                    remaining -= table.num_rows
                    if remaining <= 0:
                        break
            if remaining is not None and remaining <= 0:
                break

        self.dataframe = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
        total = len(self.dataframe)
        self.rank_sharded = rank_sharded
        if rank_sharded:
            if global_max_samples and global_max_samples > 0:
                base, extra = divmod(global_max_samples, world)
                budgets = [base + int(shard_rank < extra) for shard_rank in range(world)]
                self.rank_shard_size = max(
                    min(shard_row_counts[shard_rank], budgets[shard_rank]) for shard_rank in range(world)
                )
            else:
                self.rank_shard_size = max(shard_row_counts)
            if total == 0:
                raise ValueError(f"rank {rank}/{world} has an empty dataset shard")
            print(
                f"dataset len: {total} (rank {rank}/{world} shard, "
                f"sampler len: {self.rank_shard_size})"
            )
        else:
            self.rank_shard_size = total
            print(f"dataset len: {total}")
        logger.info("dataset len: %d", total)

        # Mirror the parent's post-load extraction (multiturn_sft_dataset.py:164-180).
        self.messages = [
            convert_nested_value_to_list_recursive(m) for m in self.dataframe[self.messages_key]
        ]
        if self.tools_key in self.dataframe.columns:
            self.tools = [
                convert_nested_value_to_list_recursive(t) for t in self.dataframe[self.tools_key]
            ]
        else:
            self.tools = None
        if self.enable_thinking_key in self.dataframe.columns:
            self.enable_thinking = self.dataframe[self.enable_thinking_key].tolist()
        else:
            self.enable_thinking = None

        # Decode tools here, not in __getitem__: the parent's __getitem__ reads
        # self.tools[item] directly (:293) and hands it to apply_chat_template, so the raw
        # '"[]"' string would reach the template and raise
        # "ValueError: Tools should either be a JSON schema...". Empty -> None, which is what
        # "no tools" means to apply_chat_template.
        if self.tools is not None:
            self.tools = [
                value if isinstance(value := decode_json_field(raw), list) and value else None
                for raw in self.tools
            ]

        # Also from the parent's tail (:180-182). Our prefix-difference path does not need
        # these, but the per-turn fallback in _process_single_message calls super(), which
        # slices by len(self.system_prompt) / len(self.generation_prompt). Omitting them
        # only fails on rows that take the fallback -- i.e. long after startup, mid-training.
        self.system_prompt, self.generation_prompt = extract_system_prompt_and_generation(
            self.tokenizer, **self.apply_chat_template_kwargs
        )

    def _build_messages(self, example: dict):
        """Normalize the row's messages before anything renders them.

        The parent calls this once per row and passes the result to ``_process_single_message``,
        ``print_assembled_message`` *and* ``sanity_check``, so normalizing here fixes every
        consumer at once. Without it, ``tool_calls`` arrives as the string ``'[]'`` and Qwen3's
        template hits ``tojson`` on a Jinja ``Undefined``, raising
        ``TypeError: Object of type Undefined is not JSON serializable``.

        This is also where the question is rewritten into the RL/eval phrasing (see
        ``align_user_prompt``), so the parquet on disk stays untouched and the alignment
        can be A/B'd with ``SFT_ALIGN_PROMPTS=0``.
        """
        messages = normalize_messages(super()._build_messages(example))
        if self.align_domain is None:
            return messages
        # Only the first user turn carries the format instruction; later turns in a
        # multi-turn row are follow-ups and must not gain a second copy of the suffix.
        for message in messages:
            if message.get("role") == "user":
                content = message.get("content")
                if isinstance(content, str):
                    message["content"] = align_user_prompt(content, self.align_domain)
                break
        return messages

    def _render(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]],
        add_generation_prompt: bool,
        enable_thinking: Optional[bool],
    ) -> str:
        """Render messages to text. ``tokenize=False`` deliberately.

        Under transformers 5.x ``apply_chat_template(tokenize=True)`` returns a
        ``BatchEncoding``, not a flat id list -- code assuming the latter computes garbage
        from ``len()``. Rendering to text and encoding separately sidesteps that entirely.
        """
        processor = self.processor if self.processor is not None else self.tokenizer
        kwargs = {**self.apply_chat_template_kwargs}
        if enable_thinking is not None:
            kwargs["enable_thinking"] = enable_thinking
        return processor.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=add_generation_prompt,
            tokenize=False,
            **kwargs,
        )

    def _encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _process_single_message(
        self,
        index: int,
        message: dict[str, Any],
        full_message: list,
        tools: Optional[list[dict[str, Any]]] = None,
        enable_thinking: Optional[bool] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Tokenize turn ``index`` by prefix difference against the whole conversation.

        ``render(messages[:index])`` vs ``render(messages[:index + 1])`` isolates this turn's
        text while the template still sees full context. For assistant turns the generation
        prompt (``<|im_start|>assistant\\n``) is split off and masked out, so loss covers only
        the model's own output -- including its ``<think>`` block.

        Note ``tools`` is passed for *every* prefix render, not just turn 0. The tool schema
        belongs to the whole conversation; omitting it mid-way would shift the prefix and
        corrupt the difference.
        """
        # Already normalized by _build_messages; this is just a rename for readability.
        messages = full_message
        # The caller only supplies tools on turn 0, but the tool schema belongs to the whole
        # conversation: omitting it on later turns would shift the prefix and corrupt the
        # difference. self.tools was decoded in _read_files_and_process.
        row_tools = tools if tools is not None else self._row_tools

        prev_text = (
            self._render(messages[:index], row_tools, False, enable_thinking) if index > 0 else ""
        )
        cur_text = self._render(messages[: index + 1], row_tools, False, enable_thinking)

        if not cur_text.startswith(prev_text):
            # Qwen3's template emits an empty '<think>\n\n</think>' placeholder for whatever
            # is currently the last assistant turn, then omits it once a later turn exists.
            # So extending the prefix can rewrite earlier text and the difference is
            # undefined. Fall back to per-turn tokenization rather than emit a wrong slice.
            #
            # Frequency is dataset-dependent: ~0.2% on Nemotron-Math-v2 (mostly single-turn),
            # but 25% on the full mixed train.parquet, concentrated in the multi-turn IF and
            # Agentic subsets (84% of rows with 7-20 turns). See _row_used_fallback below for
            # why this does not mean 25% of rows are damaged.
            logger.warning_once(
                "Prefix render is not monotonic at turn %d (usually Qwen3's empty <think> "
                "placeholder on the trailing assistant turn); using per-turn tokenization "
                "for this row.",
                index,
            )
            self._row_used_fallback = True
            return super()._process_single_message(
                index=index,
                message=message,
                full_message=full_message,
                tools=tools,
                enable_thinking=enable_thinking,
            )

        is_assistant = messages[index].get("role") == "assistant"
        if is_assistant:
            gen_text = self._render(messages[:index], row_tools, True, enable_thinking)
            if gen_text.startswith(prev_text) and cur_text.startswith(gen_text):
                prompt_ids = self._encode(gen_text[len(prev_text) :])
                body_ids = self._encode(cur_text[len(gen_text) :])
                input_ids = prompt_ids + body_ids
                loss_mask = [0] * len(prompt_ids) + [1] * len(body_ids)
            else:
                # Template does not expose a clean generation prompt; train the whole turn.
                input_ids = self._encode(cur_text[len(prev_text) :])
                loss_mask = [1] * len(input_ids)
        else:
            input_ids = self._encode(cur_text[len(prev_text) :])
            loss_mask = [0] * len(input_ids)

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(loss_mask, dtype=torch.long),
            torch.ones(len(input_ids), dtype=torch.long),
            {},
        )

    def sanity_check(self, input_ids, messages, tools, enable_thinking):
        """Skip the parent's whole-conversation equality check on fallback rows only.

        The parent (``multiturn_sft_dataset.py:425-456``) treats
        ``apply_chat_template(messages)`` as ground truth. On thinking models that render is
        exactly the bug this class exists to fix: Qwen3's template only emits the ``<think>``
        block for the *last* assistant turn (``loop.index0 > ns.last_query_index``), so every
        earlier turn's reasoning is stripped from the "truth". Any row that genuinely
        preserves multi-turn reasoning therefore *must* differ from it, and the assertion
        fires on correct output.

        The prefix-difference path keeps the check: there the two renders agree unless
        something is actually wrong, so it still guards real regressions. Only rows that took
        the per-turn fallback are exempt.

        Why the exemption is safe -- measured over 700 sampled rows of the full
        ``train.parquet`` (184 fallback rows, 516 differing regions), see
        ``tests/test_reasoning_preserved.py::check_fallback_rows_preserve_reasoning``:

        * Every difference is one-directional (ours ⊃ template's): 398 regions were chat
          markup/whitespace, 112 were text appearing *only* in ours and matching the row's
          source ``reasoning_content`` verbatim. Regions carrying content only the template
          had -- i.e. anything we dropped -- numbered **0**.
        * Reasoning survives into the trained region on 184/184 fallback rows.
        * No label leakage: walking the id sequence by ``<|im_start|>user`` span, user turns
          carry ``loss_mask == 1`` on 0 tokens.

        This is deliberately *not* ``ignore_input_ids_mismatch=True``. That config flag is
        global and would also silence the prefix path, which is the only remaining detector
        for real tokenization bugs (see ``overview.md``); it must stay ``False``.
        """
        if self._row_used_fallback:
            return
        return super().sanity_check(input_ids, messages, tools, enable_thinking)

    def __len__(self):
        """Return an equal length on every pre-sharded rank to keep collectives aligned."""
        if getattr(self, "rank_sharded", False):
            return self.rank_shard_size
        return super().__len__()

    def __getitem__(self, item):
        """Record row tools and repeat local tail rows when rank shards need padding."""
        if getattr(self, "rank_sharded", False):
            item %= len(self.messages)
        self._row_tools = self.tools[item] if self.tools is not None else None
        self._row_used_fallback = False
        try:
            return super().__getitem__(item)
        finally:
            self._row_tools = None
            self._last_row_used_fallback = self._row_used_fallback
            self._row_used_fallback = False
