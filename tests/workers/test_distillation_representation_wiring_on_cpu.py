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
"""End-to-end shape and alignment tests for compute_representation_loss.

The two sides of this loss arrive in different layouts and are prepared in different places:

* the **student's** hidden states come straight out of the engine as jagged tensors over
  prompt+response, and are padded and gathered inside the loss;
* the **teacher's** are padded and gathered by the trainer, so only the last-k window crosses
  Ray -- which is what makes the whole thing affordable.

So the alignment between them is established by two separate pieces of code agreeing, which is
exactly the kind of invariant that fails silently: mismatched windows still produce a finite
loss and a plausible cosine. These tests drive the real function with the real layouts and
assert that a student identical to the teacher yields zero loss -- which holds only if both
sides selected the same positions.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pytest
import torch
from tensordict import TensorDict

from verl.trainer.distillation.losses import compute_representation_loss
from verl.trainer.distillation.representation import gather_last_k, teacher_hidden_key
from verl.workers.config import DistillationLossConfig
from verl.workers.utils.padding import no_padding_2_padding

BSZ, PROMPT_LEN, RESPONSE_LEN, HIDDEN = 3, 5, 6, 4
PROMPT_LENS = torch.tensor([5, 4, 3])
RESPONSE_LENS = torch.tensor([6, 4, 5])
LAYERS = (1, 3)
LAST_K = 3


def _masks():
    attention_mask = torch.zeros(BSZ, PROMPT_LEN + RESPONSE_LEN, dtype=torch.long)
    response_mask = torch.zeros(BSZ, RESPONSE_LEN, dtype=torch.long)
    for i in range(BSZ):
        attention_mask[i, PROMPT_LEN - PROMPT_LENS[i] : PROMPT_LEN] = 1
        attention_mask[i, PROMPT_LEN : PROMPT_LEN + RESPONSE_LENS[i]] = 1
        response_mask[i, : RESPONSE_LENS[i]] = 1
    return attention_mask, response_mask


def _jagged_hidden(generator):
    """Per-layer student hidden states as the engine emits them: jagged over the sequence."""
    seq_lens = PROMPT_LENS + RESPONSE_LENS
    offsets = torch.cat([torch.zeros(1, dtype=torch.long), seq_lens.cumsum(0)])
    values = torch.randn(int(seq_lens.sum()), HIDDEN, generator=generator)
    return torch.nested.nested_tensor_from_jagged(values, offsets=offsets)


def _build(student_hidden, teacher_hidden):
    attention_mask, response_mask = _masks()
    data = {
        "prompts": torch.zeros(BSZ, PROMPT_LEN, dtype=torch.long),
        "responses": torch.zeros(BSZ, RESPONSE_LEN, dtype=torch.long),
        "attention_mask": attention_mask,
        "response_mask": response_mask,
    }
    for layer, tensor in teacher_hidden.items():
        data[teacher_hidden_key(layer)] = tensor
    model_output = {f"rep_hidden_{layer}": tensor for layer, tensor in student_hidden.items()}
    return model_output, TensorDict(data, batch_size=[])


def _config(**kw):
    return DistillationLossConfig(
        loss_mode="k1",
        use_policy_gradient=True,
        use_task_rewards=False,
        rep_coef=kw.pop("rep_coef", 1.0),
        rep_layers=kw.pop("rep_layers", LAYERS),
        rep_last_k=kw.pop("rep_last_k", LAST_K),
        **kw,
    )


def _teacher_from_student(student_hidden, data):
    """Prepare the teacher side the way the trainer does, from the same numbers."""
    _, response_mask = _masks()
    out = {}
    for layer, jagged in student_hidden.items():
        padded = no_padding_2_padding(jagged, data)
        gathered, _ = gather_last_k(padded, response_mask, LAST_K)
        out[layer] = gathered.float()
    return out


@pytest.fixture
def student_hidden():
    generator = torch.Generator().manual_seed(0)
    return {layer: _jagged_hidden(generator) for layer in LAYERS}


def test_matching_representations_give_zero_loss(student_hidden):
    """The student and teacher paths must select the same positions.

    The teacher tensors here are built from the student's own numbers through the trainer's
    preparation path. If the two sides disagreed about the window -- the one-token left shift,
    or the per-sample last-k alignment -- this would be nonzero.
    """
    _, data_for_prep = _build(student_hidden, {})
    teacher = _teacher_from_student(student_hidden, data_for_prep)
    model_output, data = _build(student_hidden, teacher)

    loss, metrics = compute_representation_loss(_config(), model_output, data)

    assert loss.item() == pytest.approx(0.0, abs=1e-6)
    assert metrics["distillation/rep_cosine"].item() == pytest.approx(1.0, abs=1e-5)


def test_shifted_teacher_is_detected(student_hidden):
    """A teacher window off by one token must not pass as matching."""
    _, data_for_prep = _build(student_hidden, {})
    teacher = _teacher_from_student(student_hidden, data_for_prep)
    shifted = {layer: torch.roll(t, shifts=1, dims=1) for layer, t in teacher.items()}
    model_output, data = _build(student_hidden, shifted)

    loss, _ = compute_representation_loss(_config(), model_output, data)
    assert torch.isfinite(loss)
    assert loss.item() > 1e-6


def test_gradient_flows_to_student_only(student_hidden):
    _, data_for_prep = _build(student_hidden, {})
    teacher = _teacher_from_student(student_hidden, data_for_prep)
    # Rebuild the student side as leaf tensors we can inspect gradients on.
    grad_student = {}
    for layer, jagged in student_hidden.items():
        values = jagged.values().clone().requires_grad_(True)
        grad_student[layer] = (values, torch.nested.nested_tensor_from_jagged(values, jagged.offsets()))
    model_output, data = _build({k: v[1] for k, v in grad_student.items()}, teacher)

    # Perturb the teacher so the loss is nonzero and a gradient exists.
    for layer in teacher:
        data[teacher_hidden_key(layer)] = data[teacher_hidden_key(layer)] + 0.5

    loss, _ = compute_representation_loss(_config(), model_output, data)
    loss.backward()

    for layer, (values, _) in grad_student.items():
        assert values.grad is not None, f"layer {layer} received no gradient"
        assert torch.any(values.grad != 0), f"layer {layer} gradient is all zeros"


def test_missing_student_layer_is_rejected(student_hidden):
    _, data_for_prep = _build(student_hidden, {})
    teacher = _teacher_from_student(student_hidden, data_for_prep)
    model_output, data = _build(student_hidden, teacher)
    del model_output[f"rep_hidden_{LAYERS[0]}"]

    with pytest.raises(ValueError, match="from the student forward"):
        compute_representation_loss(_config(), model_output, data)


def test_missing_teacher_layer_is_rejected(student_hidden):
    _, data_for_prep = _build(student_hidden, {})
    teacher = _teacher_from_student(student_hidden, data_for_prep)
    model_output, data = _build(student_hidden, teacher)
    del data[teacher_hidden_key(LAYERS[0])]

    with pytest.raises(ValueError, match="in the batch"):
        compute_representation_loss(_config(), model_output, data)


def test_layer_subset_is_honoured(student_hidden):
    """Only the configured layers are read, so a partial teacher set is fine."""
    _, data_for_prep = _build(student_hidden, {})
    teacher = _teacher_from_student(student_hidden, data_for_prep)
    one_layer = {LAYERS[0]: teacher[LAYERS[0]]}
    model_output, data = _build(student_hidden, one_layer)

    loss, _ = compute_representation_loss(_config(rep_layers=(LAYERS[0],)), model_output, data)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)
