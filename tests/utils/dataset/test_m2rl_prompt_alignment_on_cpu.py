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

"""Tests for the SFT -> RL prompt alignment in ``examples/llm_fusion/sft/dataset.py``.

Why this exists: the three stages asked for answers in different words. SFT put the
format instruction before the question ("Solve the following math problem. Make sure to
put the answer ... inside \\boxed{}."), while the RL datasets and the eval suite append
"Let's think step by step ..." after it. Measured on ``Jackwang111/M2RL-SFT`` over
AIME'25 at n=16, serving the SFT-style prompt instead of ours moved acc .3208 -> .4500
and cut median response length 17630 -> 5524 tokens. RL and eval already agree, so SFT
is the side that moves.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "examples" / "llm_fusion" / "sft"))

from dataset import (  # noqa: E402
    RL_CODE_SUFFIX,
    RL_REASONING_SUFFIX,
    _infer_domain,
    align_user_prompt,
)

MATH_PREFIX = "Solve the following math problem. Make sure to put the answer (and only answer) inside \\boxed{}."
# Part of the science split stores \boxed with the backslash-b collapsed to a literal
# backspace (0x08). Both spellings must strip.
SCIENCE_FREEFORM_PREFIX = "Solve the following problem. Make sure to put the answer (and only answer) inside \x08oxed{}."
SCIENCE_MCQ_PREFIX = (
    "Answer the following multiple choice question. The last line of your response "
    "should be in the following format: 'Answer: A/B/C/D' (e.g. 'Answer: A')."
)


@pytest.mark.parametrize(
    ("prefix", "domain"),
    [
        pytest.param(MATH_PREFIX, "math", id="math"),
        pytest.param(SCIENCE_FREEFORM_PREFIX, "science", id="science_freeform_mangled_boxed"),
        pytest.param(SCIENCE_MCQ_PREFIX, "science", id="science_multiple_choice"),
    ],
)
def test_reasoning_prefix_is_replaced_by_the_rl_suffix(prefix, domain):
    question = "What is the value of x?"
    aligned = align_user_prompt(f"{prefix}\n\n{question}", domain)

    assert aligned == question + RL_REASONING_SUFFIX
    assert not aligned.startswith("Solve the following")
    assert not aligned.startswith("Answer the following")


def test_reasoning_suffix_is_added_even_without_a_known_prefix():
    """An unrecognised phrasing still has to end up requesting \\boxed{}."""
    aligned = align_user_prompt("Some question with no instruction.", "math")
    assert aligned.endswith(RL_REASONING_SUFFIX)


def test_code_questions_gain_the_rl_delimiter_instruction():
    """Code moves the other way: the SFT split has no such instruction (0/400 sampled)."""
    aligned = align_user_prompt("Sum two integers from stdin.", "code")
    assert aligned.startswith("You will be given a question (problem specification)")
    assert aligned.endswith(RL_CODE_SUFFIX)
    assert "Sum two integers from stdin." in aligned


def test_code_questions_that_already_have_the_instruction_are_untouched():
    original = "Question: foo\n\nEnclose your code within delimiters as follows.```python\n# YOUR CODE HERE\n```"
    assert align_user_prompt(original, "code") == original


@pytest.mark.parametrize("domain", ["if", "agent", "unknown"])
def test_untargeted_domains_pass_through_unchanged(domain):
    """IF keeps its system-message convention; rewriting it risks a healthy benchmark.

    IFEval already scores 81.0 against the paper's 79.48, so there is nothing to fix.
    """
    original = "Write a limerick about ducks."
    assert align_user_prompt(original, domain) == original


def test_alignment_is_not_applied_twice():
    """``_build_messages`` only rewrites the first user turn, but guard the pure fn too."""
    once = align_user_prompt(f"{MATH_PREFIX}\n\nFind x.", "math")
    twice = align_user_prompt(once, "math")
    assert twice.count("Let's think step by step") == 1


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("Nemotron-Math-v2.parquet", "math"),
        ("Nemotron-Math-Proofs-v1.parquet", "math"),
        ("Nemotron-Competitive-Programming-v1.parquet", "code"),
        ("Nemotron-Science-v1.parquet", "science"),
        # No domain in the name -> per-row detection rather than giving up.
        ("Nemotron-Instruction-Following-Chat-v1.parquet", "auto"),
        ("Nemotron-Agentic-v1.parquet", "auto"),
        ("train.le16384.parquet", "auto"),
    ],
)
def test_domain_inference_from_filename(filename, expected):
    assert _infer_domain(f"/data/sft_train/{filename}") == expected


def test_mixed_domain_file_list_falls_back_to_per_row_detection():
    """No single file-level rule fits the merged mixture, so decide per row instead."""
    assert _infer_domain(["/d/Nemotron-Math-v2.parquet", "/d/Nemotron-Science-v1.parquet"]) == "auto"


@pytest.mark.parametrize(
    ("content", "should_change"),
    [
        pytest.param(f"{MATH_PREFIX}\n\nFind x.", True, id="math_row_is_aligned"),
        pytest.param(f"{SCIENCE_MCQ_PREFIX}\n\nWhich?", True, id="science_row_is_aligned"),
        # Everything without a recognised reasoning prefix must survive untouched --
        # appending a \boxed instruction to an IF or agent row would be actively wrong.
        pytest.param("Write a limerick about ducks.", False, id="if_row_untouched"),
        pytest.param("You will be given a question (problem specification)...", False, id="code_row_untouched"),
    ],
)
def test_auto_mode_only_rewrites_rows_it_recognises(content, should_change):
    aligned = align_user_prompt(content, "auto")
    if should_change:
        assert aligned.endswith(RL_REASONING_SUFFIX)
    else:
        assert aligned == content
