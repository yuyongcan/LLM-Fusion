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
"""CPU tests for loss_mode="argmax_ce": CE against the teacher's argmax token.

The mode is the topk=1 limit of ``forward_kl_topk`` with the ``q = pi_teacher(argmax)``
weighting and the zero clamp removed, so the tests below pin that relationship down
numerically instead of just re-implementing the loss:

    L_top1 = q * (log q - log pi_S(a)) = q * log q + q * L_ce
    => L_ce = L_top1 / q - log q

and assert the two places the modes diverge: the clamp, and the normalisation of
``overlap_ratio``.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from verl.trainer.distillation.fsdp.losses import _gather_log_softmax
from verl.trainer.distillation.fsdp.losses import compute_argmax_ce as compute_fsdp_argmax_ce
from verl.trainer.distillation.fsdp.losses import compute_forward_kl_topk as compute_fsdp_forward_kl_topk
from verl.trainer.distillation.losses import compute_argmax_ce as collect_argmax_ce_metrics
from verl.trainer.distillation.losses import compute_forward_kl_topk as collect_forward_kl_topk_metrics
from verl.trainer.distillation.losses import compute_topk_loss
from verl.utils import tensordict_utils as tu
from verl.workers.config import DistillationLossConfig

_VOCAB_SIZE = 6


def _nested_from_rows(rows):
    """Jagged tensor of one sequence, mirroring what the rollout writes into the batch.

    `rows` must be rectangular: `nested_tensor_from_jagged` raggedness applies to dim 1, so a
    varying top-k is not representable. The last (top-k) dimension is fixed.
    """
    values = torch.tensor(rows)
    offsets = torch.tensor([0, len(rows)], dtype=torch.int64)
    return torch.nested.nested_tensor_from_jagged(values, offsets=offsets)


def _make_config(log_prob_min_clamp=None):
    return SimpleNamespace(distillation_loss=SimpleNamespace(log_prob_min_clamp=log_prob_min_clamp))


def test_argmax_ce_is_cross_entropy_against_teacher_argmax():
    logits = torch.tensor(
        [[0.0, 9.0, 8.0, 1.0, 0.0, 0.0], [8.0, 7.0, 0.0, 0.0, 9.0, 0.0], [9.0, 8.0, 7.0, 0.0, 0.0, 0.0]],
        dtype=torch.float32,
    ).unsqueeze(0)
    teacher_ids = _nested_from_rows([[1], [4], [3]]).to(torch.int64)
    teacher_logprobs = _nested_from_rows(
        [[torch.log(torch.tensor(0.7))], [torch.log(torch.tensor(0.6))], [torch.log(torch.tensor(0.5))]]
    ).to(torch.float32)

    output = compute_fsdp_argmax_ce(
        student_logits=logits,
        teacher_topk_log_probs=teacher_logprobs,
        teacher_topk_ids=teacher_ids,
        config=_make_config(),
        data_format="thd",
    )

    student_log_probs = torch.log_softmax(logits, dim=-1)
    expected = -torch.stack(
        [student_log_probs[0, 0, 1], student_log_probs[0, 1, 4], student_log_probs[0, 2, 3]]
    ).unsqueeze(0)
    torch.testing.assert_close(output["distillation_losses"], expected)

    # student_mass / teacher_mass are the two models' probabilities on that same token.
    torch.testing.assert_close(output["student_mass"], (-expected).exp())
    torch.testing.assert_close(output["teacher_mass"], torch.tensor([[0.7, 0.6, 0.5]]))


def test_argmax_is_derived_from_log_probs_not_from_slot_zero():
    """Defensive: a backend that returns an unsorted top-k slice must still give the argmax.

    Unreachable under the shipped config (argmax_ce pins topk=1), but the function does not
    re-derive the argmax only to be wrong the moment the restriction is lifted.
    """
    logits = torch.zeros(1, 1, _VOCAB_SIZE, dtype=torch.float32)
    # Slot 1 carries the larger probability, so slot 0 is *not* the argmax.
    teacher_ids = _nested_from_rows([[2, 4]]).to(torch.int64)
    teacher_logprobs = _nested_from_rows([[torch.log(torch.tensor(0.3)), torch.log(torch.tensor(0.6))]]).to(
        torch.float32
    )

    output = compute_fsdp_argmax_ce(
        student_logits=logits,
        teacher_topk_log_probs=teacher_logprobs,
        teacher_topk_ids=teacher_ids,
        config=_make_config(),
        data_format="thd",
    )

    uniform_ce = torch.log(torch.tensor(float(_VOCAB_SIZE)))
    torch.testing.assert_close(output["distillation_losses"], uniform_ce.reshape(1, 1))
    assert output["teacher_mass"].item() == pytest.approx(0.6), "must have picked slot 1, not slot 0"

    # A uniform student has no argmax tie-break guarantee, so pin the id directly instead.
    assert torch.equal(output["overlap_count"], torch.tensor([[1.0]] if logits[0, 0].argmax() == 4 else [[0.0]]))


def test_argmax_ce_equals_top1_forward_kl_without_the_q_weighting():
    """L_top1 = q * log q + q * L_ce -- so the modes coincide only at q == 1."""
    torch.manual_seed(0)
    logits = torch.randn(1, 4, _VOCAB_SIZE, dtype=torch.float32)
    teacher_ids = torch.randint(0, _VOCAB_SIZE, (4, 1)).to(torch.int64)
    q = torch.tensor([[0.2], [0.5], [0.9], [1.0]], dtype=torch.float32)

    teacher_ids_nested = _nested_from_rows(teacher_ids.tolist()).to(torch.int64)
    teacher_logprobs_nested = _nested_from_rows(torch.log(q).tolist()).to(torch.float32)
    q_flat = q.reshape(4)

    ce = compute_fsdp_argmax_ce(
        student_logits=logits,
        teacher_topk_log_probs=teacher_logprobs_nested,
        teacher_topk_ids=teacher_ids_nested,
        config=_make_config(),
        data_format="thd",
    )["distillation_losses"]

    top1 = compute_fsdp_forward_kl_topk(
        student_logits=logits,
        teacher_topk_log_probs=teacher_logprobs_nested,
        teacher_topk_ids=teacher_ids_nested,
        config=_make_config(),
        data_format="thd",
    )["distillation_losses"]

    torch.testing.assert_close(top1, q_flat * torch.log(q_flat) + q_flat * ce)

    # At q == 1 the teacher is certain, the weighting and the constant vanish, and the two modes
    # are identical. Everywhere else argmax_ce is strictly larger, by (1 - q) * L_ce - q * log q.
    torch.testing.assert_close(top1[0, 3], ce[0, 3])
    assert (ce[0, :3] > top1[0, :3]).all(), "with q < 1 the unweighted CE must exceed the weighted one"
    assert (ce >= top1).all()


def test_log_prob_min_clamp_bounds_the_loss():
    logits = torch.zeros(1, 1, _VOCAB_SIZE, dtype=torch.float32)
    # Uniform student: log pi_S(a) = -log(6) ~ -1.79, so the unclamped loss is 1.79.
    teacher_ids = _nested_from_rows([[2]]).to(torch.int64)
    teacher_logprobs = _nested_from_rows([[torch.log(torch.tensor(0.5))]]).to(torch.float32)

    unclamped = compute_fsdp_argmax_ce(
        student_logits=logits,
        teacher_topk_log_probs=teacher_logprobs,
        teacher_topk_ids=teacher_ids,
        config=_make_config(log_prob_min_clamp=None),
        data_format="thd",
    )["distillation_losses"]

    torch.testing.assert_close(unclamped, torch.log(torch.tensor(6.0)).reshape(1, 1))

    clamped = compute_fsdp_argmax_ce(
        student_logits=logits,
        teacher_topk_log_probs=teacher_logprobs,
        teacher_topk_ids=teacher_ids,
        config=_make_config(log_prob_min_clamp=-0.5),
        data_format="thd",
    )["distillation_losses"]

    assert clamped.item() == pytest.approx(0.5), "log_prob_min_clamp must cap the CE from above"


def _make_batch(resp_len=3):
    return TensorDict(
        {
            "prompts": torch.tensor([[101]]),
            "responses": torch.tensor([[11, 12, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 0]]),
            "response_mask": torch.tensor([[1, 1, 0]], dtype=torch.bool),
        },
        batch_size=[1],
    )


def test_collector_does_not_apply_the_topk_zero_clamp():
    """forward_kl_topk clamps its divergence at 0; argmax_ce must not.

    topk=1 goes negative exactly when the student is already more confident on the argmax than
    the teacher is, and that is when the clamp stops supervising the position. argmax_ce keeps
    pushing instead, which is the behavioural difference between the two modes.
    """
    data = _make_batch()
    model_output = {
        "distillation_losses": torch.tensor([0.1, -0.4, 0.3]),
        "student_mass": torch.tensor([0.9, 0.8, 0.7]),
        "teacher_mass": torch.tensor([0.95, 0.85, 0.75]),
        "overlap_count": torch.tensor([1.0, 0.0, 1.0]),
        "overlap_token_advantage": torch.tensor([0.1, 0.0, 0.3]),
    }
    distillation_config = SimpleNamespace(distillation_loss=SimpleNamespace(topk=1))

    ce_losses, _ = collect_argmax_ce_metrics(
        config=SimpleNamespace(),
        distillation_config=distillation_config,
        model_output=model_output,
        data=data,
    )
    topk_losses, _ = collect_forward_kl_topk_metrics(
        config=SimpleNamespace(),
        distillation_config=distillation_config,
        model_output=model_output,
        data=data,
    )

    # response_mask is [1, 1, 0], so only the first two positions are read.
    assert ce_losses[0, 1].item() == pytest.approx(-0.4), "argmax_ce must not clamp at zero"
    assert topk_losses[0, 1].item() == pytest.approx(0.0), "forward_kl_topk must clamp at zero"
    torch.testing.assert_close(ce_losses[0, 0], topk_losses[0, 0])


def test_overlap_ratio_is_argmax_agreement_over_response_tokens_only():
    data = _make_batch()
    model_output = {
        "distillation_losses": torch.tensor([0.1, 0.2, 99.0]),
        "student_mass": torch.tensor([0.9, 0.8, 0.7]),
        "teacher_mass": torch.tensor([0.95, 0.85, 0.75]),
        "overlap_count": torch.tensor([1.0, 0.0, 0.0]),
        "overlap_token_advantage": torch.tensor([0.1, 0.0, 0.0]),
    }
    distillation_config = SimpleNamespace(distillation_loss=SimpleNamespace(topk=1))

    _, metrics = collect_argmax_ce_metrics(
        config=SimpleNamespace(),
        distillation_config=distillation_config,
        model_output=model_output,
        data=data,
    )

    # 1 of the 2 response positions agrees, and the padded third position must not count.
    assert metrics["distillation/overlap_ratio"] == pytest.approx(0.5)
    assert metrics["distillation/overlap_token_advantage"] == pytest.approx(0.1)


def test_argmax_ce_requires_topk_one():
    # use_policy_gradient defaults to True in the dataclass, so it has to be turned off here --
    # which is exactly the trap the next test guards: train_mopd.sh ships that default.
    DistillationLossConfig(loss_mode="argmax_ce", topk=1, use_policy_gradient=False)

    with pytest.raises(ValueError, match="topk"):
        DistillationLossConfig(loss_mode="argmax_ce", topk=32, use_policy_gradient=False)


def test_argmax_ce_rejects_policy_gradient():
    """train_mopd.sh defaults to use_policy_gradient=True; that inverts this loss.

    Under policy gradient the advantage becomes log pi_S(argmax_teacher) and the gradient flows
    only through the sampled token, so the update reinforces the student's own token hardest
    where it is furthest from the teacher. Fail at config time rather than ship a run whose
    reward curve looks plausible while moving the wrong way.
    """
    DistillationLossConfig(loss_mode="argmax_ce", topk=1, use_policy_gradient=False)

    with pytest.raises(ValueError, match="use_policy_gradient"):
        DistillationLossConfig(loss_mode="argmax_ce", topk=1, use_policy_gradient=True)


def test_loss_backpropagates_to_the_logits():
    """The loss must stay differentiable w.r.t. student_logits, and match the dense gradient.

    On GPU this routes through flash-attn's fused CE, which backpropagates in place; the gradient
    it produces is the thing worth pinning, since a forward-only equivalence would not catch a
    detached or mis-scaled backward.
    """
    base = torch.tensor(
        [[0.0, 2.0, 1.0, 0.5, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 3.0, 0.0], [2.0, 1.0, 0.5, 0.0, 0.0, 0.0]],
        dtype=torch.float32,
    ).unsqueeze(0)
    teacher_ids = _nested_from_rows([[1], [4], [3]]).to(torch.int64)
    teacher_logprobs = _nested_from_rows([[-0.3], [-0.5], [-0.7]]).to(torch.float32)

    chunked_logits = base.clone().requires_grad_(True)
    output = compute_fsdp_argmax_ce(
        student_logits=chunked_logits,
        teacher_topk_log_probs=teacher_logprobs,
        teacher_topk_ids=teacher_ids,
        config=_make_config(),
        data_format="thd",
    )
    output["distillation_losses"].sum().backward()

    dense_logits = base.clone().requires_grad_(True)
    dense_loss = -torch.log_softmax(dense_logits, dim=-1).gather(dim=-1, index=torch.tensor([[[1], [4], [3]]]))
    dense_loss.sum().backward()

    assert chunked_logits.grad is not None
    torch.testing.assert_close(chunked_logits.grad, dense_logits.grad)


def test_argmax_diagnostic_does_not_retain_a_graph():
    """overlap_count is a metric, not a loss: it must not hold the logits graph alive.

    If the argmax path kept requires_grad, every micro batch would pin its logits until the
    metric tensor was freed -- reintroducing the memory this change removes, and silently.
    It also has to be read before the CE call, which may overwrite the logits in place.
    """
    logits = torch.randn(1, 5, _VOCAB_SIZE, requires_grad=True)
    teacher_ids = _nested_from_rows([[0], [1], [2], [3], [4]]).to(torch.int64)
    teacher_logprobs = _nested_from_rows([[-0.1], [-0.2], [-0.3], [-0.4], [-0.5]]).to(torch.float32)

    output = compute_fsdp_argmax_ce(
        student_logits=logits,
        teacher_topk_log_probs=teacher_logprobs,
        teacher_topk_ids=teacher_ids,
        config=_make_config(),
        data_format="thd",
    )

    assert not output["overlap_count"].requires_grad
    assert output["distillation_losses"].requires_grad


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
@pytest.mark.parametrize("requires_grad", [True, False])
def test_log_probs_fallback_matches_dense_across_chunks(dtype: torch.dtype, requires_grad: bool):
    torch.manual_seed(0)
    logits = torch.randn(1, 259, 31, dtype=dtype, requires_grad=requires_grad)
    ids = torch.randint(0, 31, (1, 259, 1))
    weights = torch.linspace(0.1, 1.0, 259).unsqueeze(0)
    dense_logits = logits.detach().clone().requires_grad_(requires_grad)
    normalized_logits = dense_logits.float() if dtype == torch.bfloat16 else dense_logits
    expected = torch.log_softmax(normalized_logits, dim=-1).gather(-1, ids).squeeze(-1)

    actual = _gather_log_softmax(logits, ids)
    torch.testing.assert_close(actual, expected)
    if requires_grad:
        (actual * weights).sum().backward()
        (expected * weights).sum().backward()
        torch.testing.assert_close(logits.grad, dense_logits.grad)


def test_fallback_does_not_save_full_vocab_normalization_tables():
    logits = torch.randn(1, 259, 31, requires_grad=True)
    ids = torch.randint(0, 31, (1, 259, 1))
    saved = []

    def record_saved(tensor: torch.Tensor) -> torch.Tensor:
        saved.append(tensor)
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(record_saved, lambda tensor: tensor):
        log_probs = _gather_log_softmax(logits, ids)

    # Checkpoint may save input views, but not independent vocabulary-sized intermediates.
    vocab_tensors = [tensor for tensor in saved if tensor.ndim >= 2 and tensor.shape[-1] == 31]
    assert vocab_tensors
    assert all(tensor.untyped_storage().data_ptr() == logits.untyped_storage().data_ptr() for tensor in vocab_tensors)
    log_probs.sum().backward()
    assert logits.grad is not None


@pytest.mark.parametrize(
    ("metadata", "expected_inplace"),
    [
        ({}, False),
        ({"log_probs_requires_grad": True}, False),
        ({"log_probs_requires_grad": False}, True),
        ({"log_probs_requires_grad": False, "calculate_entropy": True}, False),
        ({"log_probs_requires_grad": False, "calculate_sum_pi_squared": True}, False),
    ],
)
def test_argmax_ce_inplace_backward_requires_exclusive_logits(monkeypatch, metadata, expected_inplace):
    import verl.trainer.distillation.fsdp.losses as fsdp_losses

    def fake_argmax_ce(*, student_logits, inplace_backward, **kwargs):
        assert inplace_backward is expected_inplace
        return {"distillation_losses": student_logits.sum(dim=-1)}

    monkeypatch.setattr(fsdp_losses, "compute_argmax_ce", fake_argmax_ce)
    data = TensorDict({"teacher_logprobs": torch.zeros(1), "teacher_ids": torch.zeros(1)}, batch_size=[])
    tu.assign_non_tensor(data, **metadata)
    compute_topk_loss(
        config=SimpleNamespace(strategy="fsdp"),
        distillation_config=SimpleNamespace(distillation_loss=SimpleNamespace(loss_mode="argmax_ce")),
        data=data,
        student_logits=torch.zeros(1, 1, _VOCAB_SIZE),
        data_format="thd",
    )
