# Copyright 2025 Bytedance Ltd. and/or its affiliates
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


import torch
import torch.nn.functional as F

from verl.utils.ulysses import (
    get_ulysses_sequence_parallel_world_size,
    slice_input_tensor,
)
from verl.workers.config import DistillationConfig, DistillationLossConfig


def kl_divergence(log_q: torch.Tensor, log_p: torch.Tensor) -> torch.Tensor:
    """Compute KL divergence between two distributions given their log probabilities."""
    log_p = log_p.float()
    log_q = log_q.float()
    p = log_p.exp()
    kld = p * (log_p - log_q)
    return kld.sum(dim=-1)


def compute_forward_kl_topk(
    student_logits: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    config: DistillationConfig,
    data_format: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute forward KL distillation loss using top-k log probabilities.

    Args:
        student_logits: (bsz, seqlen/sp_size, vocab_size).
        teacher_topk_log_probs: (bsz, seqlen, topk).
        teacher_topk_ids: (bsz, seqlen, topk).
        data_format: "thd" or "bshd", models not support THD format, e.g GPT-OSS, Qwen3.5

    Returns:
    - distillation_losses: (bsz, seqlen/sp_size)
    - student_mass: (bsz, seqlen/sp_size)
    - teacher_mass: (bsz, seqlen/sp_size)
    """
    assert teacher_topk_log_probs.is_nested and teacher_topk_ids.is_nested
    teacher_topk_log_probs = teacher_topk_log_probs.values().unsqueeze(0)  # (1, total_nnz, topk)
    teacher_topk_ids = teacher_topk_ids.values().unsqueeze(0)  # (1, total_nnz, topk)

    # 1. split across sp groups (bsz, seqlen, topk) => (bsz, seqlen/sp_size, topk)
    if get_ulysses_sequence_parallel_world_size() > 1:
        teacher_topk_log_probs = slice_input_tensor(teacher_topk_log_probs, dim=1)
        teacher_topk_ids = slice_input_tensor(teacher_topk_ids, dim=1)
    assert teacher_topk_log_probs.shape[:2] == teacher_topk_ids.shape[:2] == student_logits.shape[:2]

    # 2. compute token-wise KL divergence across sp groups
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    student_topk_ids = torch.topk(student_log_probs, k=teacher_topk_ids.shape[-1], dim=-1).indices
    student_topk_log_probs = torch.gather(student_log_probs, dim=-1, index=teacher_topk_ids)
    student_mass = student_topk_log_probs.exp().sum(dim=-1)
    teacher_mass = teacher_topk_log_probs.exp().sum(dim=-1)
    loss_config: DistillationLossConfig = config.distillation_loss
    if loss_config.log_prob_min_clamp is not None:
        student_topk_log_probs = student_topk_log_probs.clamp_min(loss_config.log_prob_min_clamp)
        teacher_topk_log_probs = teacher_topk_log_probs.clamp_min(loss_config.log_prob_min_clamp)
    distillation_losses = kl_divergence(log_q=student_topk_log_probs, log_p=teacher_topk_log_probs)

    # Diagnostics for tracking teacher/student top-k overlap in OPD, following
    # "Rethinking On-Policy Distillation of Large Language Models" (arXiv:2604.13016).
    overlap_mask = (teacher_topk_ids.unsqueeze(-1) == student_topk_ids.unsqueeze(-2)).any(dim=-1)
    overlap_count = overlap_mask.sum(dim=-1)
    token_kl = teacher_topk_log_probs.exp() * (teacher_topk_log_probs - student_topk_log_probs)
    overlap_token_advantage_sum = (-token_kl * overlap_mask).sum(dim=-1)
    overlap_token_advantage = overlap_token_advantage_sum / overlap_count.clamp_min(1)
    overlap_token_advantage = torch.where(
        overlap_count > 0, overlap_token_advantage, torch.zeros_like(overlap_token_advantage)
    )

    return {
        "distillation_losses": distillation_losses,
        "student_mass": student_mass,
        "teacher_mass": teacher_mass,
        "overlap_count": overlap_count,
        "overlap_token_advantage": overlap_token_advantage,
    }


def compute_argmax_ce(
    student_logits: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    config: DistillationConfig,
    data_format: str,
) -> dict[str, torch.Tensor]:
    """Cross-entropy against the teacher's argmax token: on-policy seqKD.

    The target at every response position is the single token the teacher ranks first, scored
    under the student's own on-policy context:

        L_t = -log pi_student( argmax_v pi_teacher(v | h_t) | h_t )

    This is the ``topk=1`` limit of :func:`compute_forward_kl_topk` with two properties
    removed, and both removals are the reason it exists as a separate mode:

    * **No teacher-probability weighting.** topk=1 computes ``q * (log q - log pi_S(a))`` with
      ``q = pi_teacher(a)``, so a teacher that is unsure of its own top choice contributes
      proportionally less. Here every position contributes at full strength, including the
      ones where q is barely above uniform and the argmax is decided by bf16 rounding in the
      teacher's forward pass.
    * **No zero clamp.** Once the student is *more* confident on the argmax than the teacher
      is, the topk=1 term goes negative and is clamped away, which stops supervising that
      position. Here the gradient keeps pushing ``pi_S(a) -> 1``.

    So the fixed point of this objective is the teacher's greedy decoding rather than the
    teacher's distribution: it is not consistent, and cannot converge to pi_teacher. That is
    acceptable if evaluation is greedy and a liability if it is sampled.

    Args:
        student_logits: (bsz, seqlen/sp_size, vocab_size).
        teacher_topk_log_probs: (bsz, seqlen, topk) nested. topk must be 1 (config-time check).
        teacher_topk_ids: (bsz, seqlen, topk) nested.
        data_format: "thd" or "bshd". Unused; kept for signature parity with the top-k loss.

    Returns:
        The same keys as :func:`compute_forward_kl_topk`, so the engine path is unchanged.
        Under topk == 1, ``overlap_count`` is 1 exactly where the student's own argmax equals
        the teacher's, which makes ``distillation/overlap_ratio`` the argmax-agreement rate --
        the most direct read on whether the student tracks the teacher's greedy path at all.
    """
    assert teacher_topk_log_probs.is_nested and teacher_topk_ids.is_nested
    teacher_topk_log_probs = teacher_topk_log_probs.values().unsqueeze(0)  # (1, total_nnz, topk)
    teacher_topk_ids = teacher_topk_ids.values().unsqueeze(0)  # (1, total_nnz, topk)

    # 1. split across sp groups (bsz, seqlen, topk) => (bsz, seqlen/sp_size, topk)
    if get_ulysses_sequence_parallel_world_size() > 1:
        teacher_topk_log_probs = slice_input_tensor(teacher_topk_log_probs, dim=1)
        teacher_topk_ids = slice_input_tensor(teacher_topk_ids, dim=1)
    assert teacher_topk_log_probs.shape[:2] == teacher_topk_ids.shape[:2] == student_logits.shape[:2]

    loss_config: DistillationLossConfig = config.distillation_loss
    student_log_probs = F.log_softmax(student_logits, dim=-1)

    # vLLM returns the top-k slice sorted, so slot 0 is already the argmax -- but re-deriving
    # it from the log probs keeps this correct for any backend that does not sort.
    best_slot = teacher_topk_log_probs.argmax(dim=-1, keepdim=True)  # (1, total_nnz, 1)
    argmax_ids = teacher_topk_ids.gather(dim=-1, index=best_slot)
    teacher_log_prob = teacher_topk_log_probs.gather(dim=-1, index=best_slot).squeeze(-1)

    student_log_prob = student_log_probs.gather(dim=-1, index=argmax_ids).squeeze(-1)
    if loss_config.log_prob_min_clamp is not None:
        # Bounds the loss at -log_prob_min_clamp instead of letting a single token the student
        # assigns ~0 probability contribute an unbounded term.
        student_log_prob = student_log_prob.clamp_min(loss_config.log_prob_min_clamp)

    distillation_losses = -student_log_prob

    # With topk == 1 both masses are the probability each model puts on the argmax token, so
    # student_mass / teacher_mass reads directly as "is the student as confident as the teacher".
    student_mass = student_log_prob.exp()
    teacher_mass = teacher_log_prob.exp()

    student_argmax_ids = torch.topk(student_log_probs, k=1, dim=-1).indices
    agreement = (argmax_ids == student_argmax_ids).squeeze(-1).float()
    overlap_token_advantage = torch.where(
        agreement.bool(), distillation_losses, torch.zeros_like(distillation_losses)
    )

    return {
        "distillation_losses": distillation_losses,
        "student_mass": student_mass,
        "teacher_mass": teacher_mass,
        "overlap_count": agreement,
        "overlap_token_advantage": overlap_token_advantage,
    }
