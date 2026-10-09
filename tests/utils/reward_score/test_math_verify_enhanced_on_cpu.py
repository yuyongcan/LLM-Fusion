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

"""Regression tests for the math scorer used by both RL training and eval.

``verl/utils/reward_score/__init__.py:45`` routes every math data_source --
``LLM-Fusion/Math`` (the RL training set) as well as ``aime2025``/``aime2026`` (the eval
suite) -- to ``math_verify_enhanced.compute_score``. Train and val share one
``RewardLoopManager``, so a scoring defect here corrupts the GRPO advantage, not just
the reported number.

The case these tests exist for: the scorer used to clip ``model_output[-300:]`` before
looking for ``\\boxed{}``. A model that answered correctly and then appended a
verification paragraph -- routine for thinking models -- had its answer sliced out of
the window and scored 0, indistinguishable from a wrong answer. Measured on the
AIME'25 dump, 7/180 (3.9%) of boxed answers sat beyond that window.
"""

import pytest

from verl.utils.reward_score import math_verify_enhanced

# Long enough to clear the old 300-char window several times over.
TRAILING_PROSE = (
    "\n\nLet me double-check this result. Substituting back into the original equation "
    "confirms the value satisfies every constraint stated in the problem. "
) * 6


@pytest.mark.parametrize(
    "solution",
    [
        pytest.param("The answer is \\boxed{588}.", id="boxed_at_end"),
        pytest.param(
            "The answer is \\boxed{588}." + TRAILING_PROSE,
            id="boxed_followed_by_long_prose",
        ),
        pytest.param(
            "First I guessed \\boxed{42}, then corrected it to \\boxed{588}." + TRAILING_PROSE,
            id="last_boxed_wins_despite_trailing_prose",
        ),
    ],
)
def test_correct_boxed_answer_scores_one(solution):
    """A correct \\boxed answer must score 1.0 regardless of what follows it."""
    assert math_verify_enhanced.compute_score(solution, "588") == 1.0


def test_wrong_boxed_answer_scores_zero():
    assert math_verify_enhanced.compute_score("The answer is \\boxed{42}.", "588") == 0.0


def test_superseded_earlier_boxed_does_not_rescue_a_wrong_final_answer():
    """Only the *last* \\boxed counts -- an earlier correct guess must not score 1.0."""
    solution = "I first found \\boxed{588}, but on review the answer is \\boxed{42}."
    assert math_verify_enhanced.compute_score(solution, "588") == 0.0


def test_no_boxed_answer_scores_zero():
    """Documents current behaviour: without \\boxed there is no fallback extraction.

    Trailing ```python blocks are a known source of false negatives on this data
    (49/480 correct-but-unscored rollouts on the AIME'25 dump). Adding a fallback is a
    separate change; this test pins today's semantics so that change is deliberate.
    """
    assert math_verify_enhanced.compute_score("### Final Answer\n```python\n588\n```", "588") == 0.0
