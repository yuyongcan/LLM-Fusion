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
"""Tests for the representation-distillation objective (OPRD).

These pin the arithmetic of the loss itself -- layer averaging, position masking, the 1/d
factor, normalization, and teacher detachment -- independently of how the hidden states are
produced. The properties that would otherwise fail silently:

* a zero loss exactly when the representations match, so "it trained" can be distinguished
  from "the target was vacuous";
* padding excluded from the average, since response_mask is right-padded and the slice this
  loss takes is right-aligned, so the tail of a short response is padding;
* no gradient to the teacher, which is a target rather than a parameter.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pytest
import torch

from verl.trainer.distillation.representation import (
    build_last_k_mask,
    representation_distillation_loss,
    select_hidden_states,
)

BSZ, SEQ_LEN, RESPONSE_LEN, HIDDEN = 3, 12, 6, 8
NUM_LAYERS = 5  # so hidden_states has NUM_LAYERS + 1 = 6 entries


def _hidden_states(generator):
    return tuple(
        torch.randn(BSZ, SEQ_LEN, HIDDEN, generator=generator) for _ in range(NUM_LAYERS + 1)
    )


def _response_mask(valid_lens=(6, 4, 5)):
    mask = torch.zeros(BSZ, RESPONSE_LEN, dtype=torch.long)
    for i, n in enumerate(valid_lens):
        mask[i, :n] = 1
    return mask


# --------------------------------------------------------------------------- selection


def test_select_picks_requested_layers_and_tail():
    """Layer order follows the request; positions are each sample's own response tail."""
    generator = torch.Generator().manual_seed(0)
    hs = _hidden_states(generator)
    # All responses full-length, so the per-sample window is also the batch tail.
    full = torch.ones(BSZ, RESPONSE_LEN, dtype=torch.long)
    out, mask = select_hidden_states(hs, layers=(1, 3), response_mask=full, last_k=4)

    assert out.shape == (BSZ, 2, 4, HIDDEN)
    assert mask.shape == (BSZ, 4)
    assert torch.equal(mask, torch.ones(BSZ, 4))
    response = slice(SEQ_LEN - RESPONSE_LEN, SEQ_LEN)
    assert torch.equal(out[:, 0], hs[1][:, response, :][:, -4:, :])
    assert torch.equal(out[:, 1], hs[3][:, response, :][:, -4:, :])


def test_select_is_per_sample_not_a_batch_tail_slice():
    """The window must track each response's own end, not the padded batch width.

    response_mask is right-padded, so for a short response the trailing batch columns are
    padding. A global tail slice would align onto that padding -- and for a response shorter
    than last_k it would supervise nothing while still returning a finite loss.
    """
    generator = torch.Generator().manual_seed(0)
    hs = _hidden_states(generator)
    response_mask = _response_mask(valid_lens=(6, 2, 5))
    out, mask = select_hidden_states(hs, layers=(1,), response_mask=response_mask, last_k=3)

    response = hs[1][:, SEQ_LEN - RESPONSE_LEN :, :]

    # Sample 0: 6 valid tokens, last 3 are response positions 3,4,5.
    assert mask[0].tolist() == [1.0, 1.0, 1.0]
    assert torch.equal(out[0, 0], response[0, 3:6, :])

    # Sample 1: only 2 valid tokens, so one leading slot is empty and the real tokens are
    # response positions 0,1 -- right-aligned into the last two slots.
    assert mask[1].tolist() == [0.0, 1.0, 1.0]
    assert torch.equal(out[1, 0, 1:], response[1, 0:2, :])

    # Sample 2: 5 valid tokens, last 3 are positions 2,3,4 -- note NOT 3,4,5, which is what a
    # batch-tail slice would have taken.
    assert mask[2].tolist() == [1.0, 1.0, 1.0]
    assert torch.equal(out[2, 0], response[2, 2:5, :])


def test_select_clamps_k_to_response_length():
    """A last_k larger than the response must not reach back into the prompt."""
    generator = torch.Generator().manual_seed(0)
    hs = _hidden_states(generator)
    full = torch.ones(BSZ, RESPONSE_LEN, dtype=torch.long)
    out, mask = select_hidden_states(hs, layers=(1,), response_mask=full, last_k=10_000)

    assert out.shape == (BSZ, 1, RESPONSE_LEN, HIDDEN)
    assert torch.equal(mask, torch.ones(BSZ, RESPONSE_LEN))
    assert torch.equal(out[:, 0], hs[1][:, SEQ_LEN - RESPONSE_LEN :, :])


@pytest.mark.parametrize("bad_layer", [0, NUM_LAYERS + 1, -1])
def test_select_rejects_out_of_range_layers(bad_layer):
    generator = torch.Generator().manual_seed(0)
    hs = _hidden_states(generator)
    full = torch.ones(BSZ, RESPONSE_LEN, dtype=torch.long)
    with pytest.raises(ValueError, match="out of range"):
        select_hidden_states(hs, layers=(bad_layer,), response_mask=full, last_k=4)


def test_mask_helper_agrees_with_selection():
    """The standalone mask must equal the one selection returns, or student and teacher
    would be masked differently."""
    generator = torch.Generator().manual_seed(0)
    hs = _hidden_states(generator)
    response_mask = _response_mask(valid_lens=(6, 2, 5))

    _, from_select = select_hidden_states(hs, layers=(1,), response_mask=response_mask, last_k=3)
    standalone = build_last_k_mask(response_mask, last_k=3)
    assert torch.equal(from_select, standalone)


def test_mask_marks_only_real_tokens():
    response_mask = _response_mask(valid_lens=(6, 2, 5))
    mask = build_last_k_mask(response_mask, last_k=4)

    assert mask.shape == (BSZ, 4)
    # Sample 1 has just 2 tokens, so 2 of its 4 slots are empty -- and crucially it still has
    # 2 supervised positions, which a batch-tail slice would have reduced to zero.
    assert mask[1].tolist() == [0.0, 0.0, 1.0, 1.0]
    assert mask[0].tolist() == [1.0, 1.0, 1.0, 1.0]
    assert mask.sum(dim=1).tolist() == [4.0, 2.0, 4.0]


# --------------------------------------------------------------------------- loss


def test_identical_representations_give_zero_loss():
    generator = torch.Generator().manual_seed(0)
    hidden = torch.randn(BSZ, 2, 4, HIDDEN, generator=generator)
    mask = torch.ones(BSZ, 4)

    for normalize in (True, False):
        loss, metrics = representation_distillation_loss(hidden, hidden.clone(), mask, normalize)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)
        assert metrics["distillation/rep_cosine"].item() == pytest.approx(1.0, abs=1e-5)


def test_unnormalized_loss_matches_closed_form():
    """The 1/d factor and the masked mean must be exactly as documented."""
    student = torch.zeros(1, 1, 2, 4)
    teacher = torch.ones(1, 1, 2, 4)
    mask = torch.ones(1, 2)

    loss, _ = representation_distillation_loss(student, teacher, mask, normalize=False)
    # Every element differs by 1, so (1/d) * sum of squares = 1 at every position.
    assert loss.item() == pytest.approx(1.0)


def test_normalize_makes_loss_scale_invariant():
    """With normalize=True, scaling the student must not change the loss."""
    generator = torch.Generator().manual_seed(0)
    student = torch.randn(BSZ, 2, 4, HIDDEN, generator=generator)
    teacher = torch.randn(BSZ, 2, 4, HIDDEN, generator=generator)
    mask = torch.ones(BSZ, 4)

    base, _ = representation_distillation_loss(student, teacher, mask, normalize=True)
    scaled, _ = representation_distillation_loss(student * 7.5, teacher, mask, normalize=True)
    assert scaled.item() == pytest.approx(base.item(), rel=1e-5)

    # Without normalization the magnitude matters, which is the whole difference between the
    # two settings and why a coefficient tuned for one is wrong for the other.
    raw_base, _ = representation_distillation_loss(student, teacher, mask, normalize=False)
    raw_scaled, _ = representation_distillation_loss(student * 7.5, teacher, mask, normalize=False)
    assert raw_scaled.item() > raw_base.item() * 2


def test_padding_is_excluded_from_the_average():
    """Differences at empty slots must not reach the loss."""
    generator = torch.Generator().manual_seed(0)
    student = torch.randn(BSZ, 2, 4, HIDDEN, generator=generator)
    teacher = student.clone()
    mask = build_last_k_mask(_response_mask(valid_lens=(6, 2, 5)), last_k=4)

    # Corrupt only slots the mask excludes: sample 1's two leading slots are empty.
    teacher[1, :, :2, :] += 1000.0
    loss, _ = representation_distillation_loss(student, teacher, mask, normalize=False)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)

    # Corrupting a supervised position does move the loss.
    teacher[0, :, -1, :] += 1.0
    loss_supervised, _ = representation_distillation_loss(student, teacher, mask, normalize=False)
    assert loss_supervised.item() > 0.0


def test_all_masked_out_does_not_divide_by_zero():
    generator = torch.Generator().manual_seed(0)
    student = torch.randn(BSZ, 2, 4, HIDDEN, generator=generator)
    teacher = torch.randn(BSZ, 2, 4, HIDDEN, generator=generator)

    loss, _ = representation_distillation_loss(student, teacher, torch.zeros(BSZ, 4), normalize=False)
    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(0.0)


def test_layers_are_averaged_not_summed():
    """Adding a perfectly-matched layer must dilute the loss, not leave it unchanged."""
    generator = torch.Generator().manual_seed(0)
    one_layer_student = torch.randn(1, 1, 2, HIDDEN, generator=generator)
    one_layer_teacher = torch.randn(1, 1, 2, HIDDEN, generator=generator)
    mask = torch.ones(1, 2)

    single, _ = representation_distillation_loss(
        one_layer_student, one_layer_teacher, mask, normalize=False
    )
    two_layer_student = torch.cat([one_layer_student, one_layer_student], dim=1)
    two_layer_teacher = torch.cat([one_layer_teacher, one_layer_student], dim=1)
    both, _ = representation_distillation_loss(two_layer_student, two_layer_teacher, mask, normalize=False)

    assert both.item() == pytest.approx(single.item() / 2, rel=1e-5)


def test_teacher_receives_no_gradient():
    """The teacher is a target. A gradient reaching it would be training the teacher."""
    generator = torch.Generator().manual_seed(0)
    student = torch.randn(BSZ, 2, 4, HIDDEN, generator=generator, requires_grad=True)
    teacher = torch.randn(BSZ, 2, 4, HIDDEN, generator=generator, requires_grad=True)
    mask = torch.ones(BSZ, 4)

    loss, _ = representation_distillation_loss(student, teacher, mask)
    loss.backward()

    assert student.grad is not None
    assert torch.any(student.grad != 0)
    assert teacher.grad is None


def test_shape_mismatches_are_rejected():
    student = torch.randn(BSZ, 2, 4, HIDDEN)
    mask = torch.ones(BSZ, 4)

    with pytest.raises(ValueError, match="!="):
        representation_distillation_loss(student, torch.randn(BSZ, 3, 4, HIDDEN), mask)
    with pytest.raises(ValueError, match="position_mask"):
        representation_distillation_loss(student, student.clone(), torch.ones(BSZ, 5))
