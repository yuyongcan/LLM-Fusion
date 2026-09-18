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
"""Representation distillation (OPRD, https://arxiv.org/abs/2606.06021).

Output-space distillation supervises the student only through the LM head. That projection
R^d -> R^|V| is low-rank and softmax-invariant to additive shifts, so hidden-state differences
lying in its effective null space produce no loss at all: the student is graded on what survives
the projection, not on how the teacher arrived at it. Aligning hidden states directly removes
that bottleneck, and because the MSE is a deterministic function of the rollout it also carries
none of the per-token sampling variance that makes sampled-token OPD stall once the student
approaches the teacher.

The objective is a layer-averaged, position-masked MSE between student and teacher hidden
states, with the teacher detached:

    L_rep = mean over selected layers of
            masked_mean over positions of  (1/d) * || h_student - sg(h_teacher) ||^2

Two facts about this codebase shape the implementation:

* **Layer indices** are indices into HF ``output_hidden_states``, where 0 is the embedding
  output and k is the output of block k. Index 0 is rejected at config time.

* **Memory is the binding constraint, and it binds on accumulation rather than on any single
  tensor.** All 36 layers of a 4B model over a full 21.5k-token sequence is ~3.7 GB per sample;
  restricted to four layers and a 2000-token suffix it is ~39 MB, and a micro-batch here is one
  sample. What does not fit is keeping that for a whole step (1056 samples) or shipping it to the
  driver. So the teacher's hidden states are consumed inside the worker that produced them, and
  only the scalar loss and its metrics leave.
"""

from typing import Optional

import torch
import torch.nn.functional as F

__all__ = [
    "select_hidden_states",
    "build_last_k_mask",
    "representation_distillation_loss",
]


def select_hidden_states(
    hidden_states: tuple[torch.Tensor, ...],
    layers: tuple[int, ...],
    response_mask: torch.Tensor,
    last_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Slice the layers and response positions to be supervised.

    The window is the last ``k`` tokens **of each sample's own response**, which is not the same
    as the last ``k`` columns of the batch: ``response_mask`` is right-padded, so a sample whose
    response is shorter than the padded width has padding in those trailing columns. A global
    tail slice would supervise that padding, and for a response much shorter than ``last_k`` it
    would supervise nothing at all while still reporting a finite loss. So positions are
    gathered per sample, right-aligned within the returned window.

    Args:
        hidden_states: HF ``output_hidden_states`` tuple, ``num_layers + 1`` entries of
            ``(bsz, seq_len, hidden)``. Entry 0 is the embedding output.
        layers: which entries to keep. Validated at config time to exclude 0.
        response_mask: ``(bsz, response_len)``, 1 for real response tokens. Assumed
            right-padded, i.e. the valid tokens are a prefix -- the same assumption the rest of
            this codebase's response handling makes.
        last_k: keep at most this many trailing response tokens per sample.

    Returns:
        ``(hidden, mask)`` where hidden is ``(bsz, n_layers, k, hidden_size)`` and mask is
        ``(bsz, k)`` marking which of those slots hold real tokens. ``k = min(last_k,
        response_len)``.
    """
    if not layers:
        raise ValueError("layers must be non-empty")
    num_entries = len(hidden_states)
    for layer in layers:
        if not 0 < layer < num_entries:
            raise ValueError(
                f"layer {layer} is out of range for a model returning {num_entries} hidden-state "
                f"entries (valid: 1..{num_entries - 1}, where 0 is the embedding output)"
            )

    bsz, response_len = response_mask.shape
    k = min(last_k, response_len)

    # (bsz, n_layers, response_len, hidden): the response is the tail of the sequence.
    stacked = torch.stack([hidden_states[layer][:, -response_len:, :] for layer in layers], dim=1)

    valid_lens = response_mask.long().sum(dim=1, keepdim=True)  # (bsz, 1)
    # Slot j holds response position valid_len - k + j, so the last valid token always lands in
    # the final slot. Samples with fewer than k valid tokens leave the leading slots empty.
    offsets = torch.arange(k, device=response_mask.device).unsqueeze(0)  # (1, k)
    positions = valid_lens - k + offsets  # (bsz, k)
    keep = (positions >= 0) & (positions < valid_lens)
    gather_positions = positions.clamp(min=0, max=max(response_len - 1, 0))

    index = gather_positions.view(bsz, 1, k, 1).expand(-1, stacked.shape[1], -1, stacked.shape[3])
    selected = stacked.gather(dim=2, index=index)
    return selected, keep.float()


def build_last_k_mask(response_mask: torch.Tensor, last_k: int) -> torch.Tensor:
    """Mask marking which of the ``k`` gathered slots hold real tokens.

    Returns the same mask :func:`select_hidden_states` produces, for callers that need it
    without re-gathering (the student and teacher share one mask).
    """
    bsz, response_len = response_mask.shape
    k = min(last_k, response_len)
    valid_lens = response_mask.long().sum(dim=1, keepdim=True)
    offsets = torch.arange(k, device=response_mask.device).unsqueeze(0)
    positions = valid_lens - k + offsets
    return ((positions >= 0) & (positions < valid_lens)).float()


def representation_distillation_loss(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    position_mask: torch.Tensor,
    normalize: bool = True,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Layer-averaged, position-masked MSE between student and teacher hidden states.

    Args:
        student_hidden: ``(bsz, n_layers, k, hidden)``, requires grad.
        teacher_hidden: ``(bsz, n_layers, k, hidden)``; detached here regardless.
        position_mask: ``(bsz, k)`` from :func:`build_last_k_mask`.
        normalize: L2-normalize both sides first, making the loss purely directional.

    Returns:
        ``(loss, metrics)``. ``metrics`` reports the cosine similarity actually being optimised
        and the fraction of positions supervised, both as detached scalars.
    """
    if student_hidden.shape != teacher_hidden.shape:
        raise ValueError(
            f"student hidden {tuple(student_hidden.shape)} != teacher hidden "
            f"{tuple(teacher_hidden.shape)}"
        )
    if position_mask.shape != student_hidden.shape[:1] + student_hidden.shape[2:3]:
        raise ValueError(
            f"position_mask {tuple(position_mask.shape)} does not match hidden states "
            f"{tuple(student_hidden.shape)} (expected (bsz, k))"
        )

    # The teacher is a target, never a parameter. It arrives from a no_grad forward already, but
    # detaching here means this function cannot be misused into training the teacher.
    teacher_hidden = teacher_hidden.detach()

    # Compute in fp32: these are bf16 activations whose norms reach the hundreds, and a squared
    # difference in bf16 loses precision exactly where the loss is small, which is where it
    # spends most of training.
    student = student_hidden.float()
    teacher = teacher_hidden.float()

    if normalize:
        student = F.normalize(student, p=2, dim=-1)
        teacher = F.normalize(teacher, p=2, dim=-1)

    # (bsz, n_layers, k): mean over the hidden dimension is the 1/d in the objective.
    per_position_mse = (student - teacher).pow(2).mean(dim=-1)

    # Broadcast the (bsz, k) mask over layers and average only over supervised positions. The
    # denominator is the mask sum rather than the element count, so the loss does not shrink
    # just because a batch happens to hold short responses.
    mask = position_mask.unsqueeze(1)
    denom = mask.sum() * per_position_mse.shape[1]
    loss = (per_position_mse * mask).sum() / denom.clamp(min=1.0)

    with torch.no_grad():
        cos = F.cosine_similarity(student, teacher, dim=-1)
        cos_mean = (cos * mask).sum() / denom.clamp(min=1.0)
        metrics = {
            "distillation/rep_loss": loss.detach(),
            "distillation/rep_cosine": cos_mean,
            "distillation/rep_supervised_frac": mask.mean().detach(),
        }
    return loss, metrics


def teacher_hidden_key(layer: int) -> str:
    """Batch key under which the teacher's hidden states for ``layer`` travel."""
    return f"teacher_hidden_{layer}"


def stack_teacher_hidden(
    data, layers: tuple[int, ...], device: Optional[torch.device] = None
) -> torch.Tensor:
    """Re-stack per-layer teacher hidden states from a batch into one tensor.

    Kept as a helper so the layer ordering used when writing the batch and when reading it
    cannot drift apart.
    """
    tensors = []
    for layer in layers:
        key = teacher_hidden_key(layer)
        if key not in data.keys():
            raise ValueError(
                f"representation distillation needs {key!r} in the batch but it is absent; "
                "check that the in-process teacher was asked for this layer."
            )
        tensor = data[key]
        tensors.append(tensor.to(device) if device is not None else tensor)
    return torch.stack(tensors, dim=1)
