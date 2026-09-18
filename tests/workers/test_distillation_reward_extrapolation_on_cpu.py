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
"""Tests for reward extrapolation (ExOPD) in the estimator distillation loss.

ExOPD (https://arxiv.org/abs/2602.12125) generalizes on-policy distillation by
weighting the implicit reward against the KL regularization, which standard OPD
pins at 1:1. The per-token advantage becomes

    A_t = (log pi_teacher - log pi_student)
        + (lambda - 1) * (log pi_teacher - log pi_reference)

and since the caller forms ``advantages = -distillation_losses``, the loss
carries the negated extrapolation term.

The property that matters most here is the first test: lambda == 1.0 must be
bit-for-bit identical to the behaviour before the knob existed, whether or not
reference log probs happen to be present in the batch. Every existing MOPD run
leaves the lambda at its default, so a regression there would silently change
results of runs that never asked for ExOPD.

Tensor layout mirrors what the loss actually receives after
``left_right_2_no_padding``: student/teacher log probs are nested (jagged over
the full prompt+response sequence), while ``ref_log_prob`` is a plain padded
``(bsz, response_len)`` tensor because ``_compute_ref_log_prob`` already sliced
it to the response and nothing converts it afterwards.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pytest
import torch
from tensordict import TensorDict

from verl.trainer.distillation.losses import compute_distillation_loss_reverse_kl_estimator
from verl.workers.config import DistillationConfig, DistillationLossConfig
from verl.workers.utils.padding import no_padding_2_padding

BSZ, PROMPT_LEN, RESPONSE_LEN = 3, 5, 6
PROMPT_LENS = torch.tensor([5, 4, 3])
RESPONSE_LENS = torch.tensor([6, 4, 5])


def _make_masks():
    attention_mask = torch.zeros(BSZ, PROMPT_LEN + RESPONSE_LEN, dtype=torch.long)
    response_mask = torch.zeros(BSZ, RESPONSE_LEN, dtype=torch.long)
    for i in range(BSZ):
        attention_mask[i, PROMPT_LEN - PROMPT_LENS[i] : PROMPT_LEN] = 1
        attention_mask[i, PROMPT_LEN : PROMPT_LEN + RESPONSE_LENS[i]] = 1
        response_mask[i, : RESPONSE_LENS[i]] = 1
    return attention_mask, response_mask


def _nested_logprobs(generator):
    """A jagged (bsz, seq_len) tensor, as produced by left_right_2_no_padding."""
    seq_lens = PROMPT_LENS + RESPONSE_LENS
    offsets = torch.cat([torch.zeros(1, dtype=torch.long), seq_lens.cumsum(0)])
    values = torch.randn(int(seq_lens.sum()), generator=generator)
    return torch.nested.nested_tensor_from_jagged(values, offsets=offsets)


def _build_data(teacher_log_probs, ref_log_probs=None):
    attention_mask, response_mask = _make_masks()
    data = {
        "prompts": torch.zeros(BSZ, PROMPT_LEN, dtype=torch.long),
        "responses": torch.zeros(BSZ, RESPONSE_LEN, dtype=torch.long),
        "attention_mask": attention_mask,
        "response_mask": response_mask,
        "teacher_logprobs": teacher_log_probs,
    }
    if ref_log_probs is not None:
        data["ref_log_prob"] = ref_log_probs
    # batch_size=[] so the padded and jagged entries can coexist, as they do in
    # the real micro batch.
    return TensorDict(data, batch_size=[])


def _make_config(lambda_value, loss_mode="k1", use_policy_gradient=True):
    loss_config = DistillationLossConfig(
        loss_mode=loss_mode,
        use_policy_gradient=use_policy_gradient,
        use_task_rewards=False,
        reward_extrapolation_lambda=lambda_value,
    )
    return DistillationConfig(enabled=False, distillation_loss=loss_config)


@pytest.fixture
def tensors():
    generator = torch.Generator().manual_seed(0)
    return {
        "student": _nested_logprobs(generator),
        "teacher": _nested_logprobs(generator),
        "ref": torch.randn(BSZ, RESPONSE_LEN, generator=generator),
    }


def test_lambda_one_is_unchanged_with_and_without_reference(tensors):
    """lambda=1.0 must reproduce standard OPD exactly, ref present or not."""
    model_output = {"log_probs": tensors["student"]}
    config = _make_config(1.0)

    without_ref, metrics = compute_distillation_loss_reverse_kl_estimator(
        None, config, model_output, _build_data(tensors["teacher"])
    )
    with_ref, _ = compute_distillation_loss_reverse_kl_estimator(
        None, config, model_output, _build_data(tensors["teacher"], tensors["ref"])
    )

    assert without_ref.shape == (BSZ, RESPONSE_LEN)
    # Bitwise, not approximate: the extrapolation branch must not run at all.
    assert torch.equal(without_ref, with_ref)
    # No extrapolation metrics are emitted, so dashboards for plain OPD runs are untouched.
    assert set(metrics) == {"distillation/abs_loss"}


def test_extrapolation_matches_closed_form(tensors):
    """lambda != 1 subtracts exactly (lambda - 1) * (teacher - reference)."""
    model_output = {"log_probs": tensors["student"]}
    lambda_value = 1.25

    baseline, baseline_metrics = compute_distillation_loss_reverse_kl_estimator(
        None, _make_config(1.0), model_output, _build_data(tensors["teacher"])
    )
    data = _build_data(tensors["teacher"], tensors["ref"])
    extrapolated, metrics = compute_distillation_loss_reverse_kl_estimator(
        None, _make_config(lambda_value), model_output, data
    )

    teacher_padded = no_padding_2_padding(tensors["teacher"], data)
    expected = baseline - (lambda_value - 1.0) * (teacher_padded - tensors["ref"])
    torch.testing.assert_close(extrapolated, expected)

    # The resulting advantage is the paper's, up to the caller's negation.
    student_padded = no_padding_2_padding(tensors["student"], data)
    advantage = -extrapolated
    paper_advantage = (teacher_padded - student_padded) + (lambda_value - 1.0) * (teacher_padded - tensors["ref"])
    torch.testing.assert_close(advantage, paper_advantage)

    assert set(metrics) - set(baseline_metrics) == {
        "distillation/extrapolation_term",
        "distillation/teacher_ref_gap",
    }


def test_missing_reference_log_probs_raises(tensors):
    """Falling back to standard OPD here would be invisible, so it must raise."""
    model_output = {"log_probs": tensors["student"]}
    with pytest.raises(ValueError, match="ref_log_prob"):
        compute_distillation_loss_reverse_kl_estimator(
            None, _make_config(1.25), model_output, _build_data(tensors["teacher"])
        )


@pytest.mark.parametrize(
    ("loss_mode", "use_policy_gradient", "match"),
    [
        # top-k losses are not a difference of log probs, so the term is undefined
        ("forward_kl_topk", True, "log-prob-difference"),
        # the term is detached, so without the policy-gradient path it moves no gradient
        ("k3", False, "use_policy_gradient"),
    ],
)
def test_incompatible_config_is_rejected_at_startup(loss_mode, use_policy_gradient, match):
    with pytest.raises(ValueError, match=match):
        DistillationLossConfig(
            loss_mode=loss_mode,
            use_policy_gradient=use_policy_gradient,
            reward_extrapolation_lambda=1.25,
        )


def test_lambda_one_accepts_any_loss_mode():
    """The validation above must not constrain runs that never asked for ExOPD."""
    for loss_mode, use_policy_gradient in [("forward_kl_topk", False), ("k3", False), ("k1", True)]:
        config = DistillationLossConfig(loss_mode=loss_mode, use_policy_gradient=use_policy_gradient)
        assert config.reward_extrapolation_lambda == 1.0
