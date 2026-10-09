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

"""Regression tests for the science scorer (GPQA eval and ``LLM-Fusion/Science`` RL).

Same defect and same fix as ``test_math_verify_enhanced_on_cpu.py``: the scorer used to
clip ``model_output[-300:]`` before looking for ``\\boxed{}``, so a correct answer
followed by a verification paragraph scored 0. Measured on the GPQA dump, 104/2142
(4.9%) of boxed answers sat beyond that window; restoring them moved GPQA from .3977 to
.4050. Train and val share one reward manager, so this fed the GRPO advantage too.
"""

import pytest

from verl.utils.reward_score import science_verify_enhanced

TRAILING_PROSE = (
    "\n\nLet me verify this choice against each of the other options to be sure it is "
    "the only one consistent with the stated experimental constraints. "
) * 6


@pytest.mark.parametrize(
    "solution",
    [
        pytest.param("The answer is \\boxed{C}.", id="boxed_at_end"),
        pytest.param("The answer is \\boxed{C}." + TRAILING_PROSE, id="boxed_followed_by_long_prose"),
        pytest.param(
            "I first considered \\boxed{A}, then settled on \\boxed{C}." + TRAILING_PROSE,
            id="last_boxed_wins_despite_trailing_prose",
        ),
    ],
)
def test_correct_boxed_choice_scores_one(solution):
    assert science_verify_enhanced.compute_score(solution, "C") == 1.0


def test_wrong_boxed_choice_scores_zero():
    assert science_verify_enhanced.compute_score("The answer is \\boxed{A}.", "C") == 0.0


def test_superseded_earlier_boxed_does_not_rescue_a_wrong_final_answer():
    solution = "I first found \\boxed{C}, but on review the answer is \\boxed{A}."
    assert science_verify_enhanced.compute_score(solution, "C") == 0.0


def test_no_boxed_answer_scores_zero():
    """Documents current behaviour: the bare ``Answer: C`` form is not extracted.

    Part of the science SFT mixture asks for exactly that form (235/800 sampled rows use
    "The last line ... should be 'Answer: A/B/C/D'"), so this is a live mismatch worth
    pinning rather than a hypothetical. 32.4% of GPQA rollouts carry no ``\\boxed`` at all.
    """
    assert science_verify_enhanced.compute_score("Reasoning...\n\nAnswer: C", "C") == 0.0
