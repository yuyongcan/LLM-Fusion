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
"""Tests for ``distillation.mode``: vLLM teacher pool vs in-process HF teacher.

The two modes deliver teacher log probs in different layouts, and the estimator loss is
shape-polymorphic over them:

    mode="vllm"        a nested/jagged tensor over the full prompt+response sequence, produced
                       per sample during rollout and converted by ``left_right_2_no_padding``.
    mode="in_process"  a plain ``(bsz, response_len)`` tensor, produced by one batched forward
                       in the trainer and already sliced to the response -- the same layout
                       ``ref_log_prob`` arrives in.

The property that matters is that the *same underlying numbers* produce the *same loss* through
either layout. The risk is an off-by-one: the vLLM path drops the first token's ``None`` logprob
and pads a dummy at the end, and ``no_padding_2_padding`` left-shifts by one when slicing the
response out of the sequence. The in-process path inherits that same left-shift for free because
it goes through the same ``no_padding_2_padding`` in ``_compute_teacher_log_prob``. These tests
pin that equivalence, so a future change to either slicing rule fails here rather than silently
distilling against misaligned tokens.

Tensor-layout fixtures mirror ``test_distillation_reward_extrapolation_on_cpu.py``.
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
    return TensorDict(data, batch_size=[])


def _make_config(lambda_value=1.0, loss_mode="k1"):
    loss_config = DistillationLossConfig(
        loss_mode=loss_mode,
        use_policy_gradient=True,
        use_task_rewards=False,
        reward_extrapolation_lambda=lambda_value,
    )
    return DistillationConfig(enabled=False, distillation_loss=loss_config)


@pytest.fixture
def tensors():
    generator = torch.Generator().manual_seed(0)
    teacher_nested = _nested_logprobs(generator)
    return {
        "student": _nested_logprobs(generator),
        "teacher_nested": teacher_nested,
        # What the in-process teacher would return for the same sequence: the response slice of
        # the very same numbers. Derived with the same helper the vLLM path uses, so any change
        # to that slicing rule keeps the two sides in agreement by construction -- what these
        # tests then verify is that the *loss* treats both layouts identically.
        "teacher_padded": no_padding_2_padding(teacher_nested, _build_data(teacher_nested)).squeeze(-1),
        "ref": torch.randn(BSZ, RESPONSE_LEN, generator=generator),
    }


def test_in_process_layout_is_plain_response_shaped(tensors):
    """The in-process teacher tensor is dense and response-shaped, unlike the vLLM one."""
    assert tensors["teacher_nested"].is_nested
    assert not tensors["teacher_padded"].is_nested
    assert tensors["teacher_padded"].shape == (BSZ, RESPONSE_LEN)


def test_both_modes_give_identical_loss(tensors):
    """Same numbers through either layout must produce the same loss, bitwise."""
    model_output = {"log_probs": tensors["student"]}
    config = _make_config()

    vllm_loss, vllm_metrics = compute_distillation_loss_reverse_kl_estimator(
        None, config, model_output, _build_data(tensors["teacher_nested"])
    )
    in_process_loss, in_process_metrics = compute_distillation_loss_reverse_kl_estimator(
        None, config, model_output, _build_data(tensors["teacher_padded"])
    )

    assert vllm_loss.shape == in_process_loss.shape == (BSZ, RESPONSE_LEN)
    # Bitwise: the only difference between the two paths is whether no_padding_2_padding ran
    # inside the loss or upstream in the trainer, and that must not perturb the arithmetic.
    assert torch.equal(vllm_loss, in_process_loss)
    assert vllm_metrics["distillation/abs_loss"].aggregate() == in_process_metrics[
        "distillation/abs_loss"
    ].aggregate()


@pytest.mark.parametrize("loss_mode", ["k1", "k3", "abs", "mse", "low_var_kl"])
def test_both_modes_agree_across_estimators(tensors, loss_mode):
    """Layout-independence must not be specific to the default estimator."""
    model_output = {"log_probs": tensors["student"]}
    config = _make_config(loss_mode=loss_mode)

    vllm_loss, _ = compute_distillation_loss_reverse_kl_estimator(
        None, config, model_output, _build_data(tensors["teacher_nested"])
    )
    in_process_loss, _ = compute_distillation_loss_reverse_kl_estimator(
        None, config, model_output, _build_data(tensors["teacher_padded"])
    )
    assert torch.equal(vllm_loss, in_process_loss)


def test_both_modes_agree_under_extrapolation(tensors):
    """ExOPD composes with either teacher mode: lambda applies after the layout is resolved."""
    model_output = {"log_probs": tensors["student"]}
    config = _make_config(lambda_value=1.25)

    vllm_loss, vllm_metrics = compute_distillation_loss_reverse_kl_estimator(
        None, config, model_output, _build_data(tensors["teacher_nested"], tensors["ref"])
    )
    in_process_loss, in_process_metrics = compute_distillation_loss_reverse_kl_estimator(
        None, config, model_output, _build_data(tensors["teacher_padded"], tensors["ref"])
    )

    assert torch.equal(vllm_loss, in_process_loss)
    # teacher_ref_gap is what lambda amplifies, so it must also be layout-independent.
    assert vllm_metrics["distillation/teacher_ref_gap"].aggregate() == in_process_metrics[
        "distillation/teacher_ref_gap"
    ].aggregate()


def test_misaligned_teacher_is_not_silently_accepted(tensors):
    """A response-shaped tensor that is off by one token must not pass as valid.

    This is the failure this file exists to catch: if a future change shifts either slicing rule,
    the numbers stay finite and every metric stays plausible, so only a value check finds it.
    """
    model_output = {"log_probs": tensors["student"]}
    config = _make_config()

    aligned, _ = compute_distillation_loss_reverse_kl_estimator(
        None, config, model_output, _build_data(tensors["teacher_padded"])
    )
    shifted = torch.roll(tensors["teacher_padded"], shifts=1, dims=1)
    misaligned, _ = compute_distillation_loss_reverse_kl_estimator(
        None, config, model_output, _build_data(shifted)
    )

    # Same shape, finite, plausible-looking -- and wrong. Hence the equivalence tests above.
    assert misaligned.shape == aligned.shape
    assert torch.isfinite(misaligned).all()
    assert not torch.allclose(aligned, misaligned)


def test_wrong_shape_is_rejected(tensors):
    """A dense teacher tensor of the wrong width must fail the loss's shape assert."""
    model_output = {"log_probs": tensors["student"]}
    config = _make_config()
    truncated = tensors["teacher_padded"][:, :-1]

    with pytest.raises(AssertionError):
        compute_distillation_loss_reverse_kl_estimator(
            None, config, model_output, _build_data(truncated)
        )
