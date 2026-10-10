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
"""Tests for KDRL: fusing the teacher signal into the RL advantage.

KDRL (https://arxiv.org/abs/2506.02208) is the canonical weighted-additive combination of RLVR
and on-policy distillation. Every method in Table 1 of "Sequential Beats Joint"
(https://arxiv.org/abs/2609.04108) shares one PPO-clipped surrogate and differs only in the
per-token advantage, which for KDRL is

    A_t = A_hat + beta * (log pi_teacher - log pi_student)

The property these tests exist to pin down is *where* the teacher enters. The pre-existing
``use_task_rewards=True`` path runs two policy losses -- one on the RL advantage, one on
-distillation_losses -- and adds them, each clipped independently. Clipping the sum is not the
sum of the clips, so that path is not a parameterisation of KDRL, and a run that believes it is
using KDRL while silently taking the two-loss path would be indistinguishable in every metric.
The two tests below assert the difference directly, by recording what reaches ``ppo_loss``.

Tensor layout mirrors the real micro batch after ``left_right_2_no_padding``: student log probs
are nested (jagged over prompt+response) and so are the vLLM teacher's, while the in-process HF
teacher writes a plain padded (bsz, response_len) tensor. Both are covered.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pytest
import torch
from tensordict import TensorDict

from verl.trainer.distillation import losses as distillation_losses
from verl.trainer.distillation.losses import compute_teacher_advantage, distillation_ppo_loss
from verl.workers.config import DistillationConfig, DistillationLossConfig
from verl.workers.utils.padding import no_padding_2_padding

BSZ, PROMPT_LEN, RESPONSE_LEN = 3, 5, 6
PROMPT_LENS = torch.tensor([5, 4, 3])
RESPONSE_LENS = torch.tensor([6, 4, 5])
BETA = 0.2


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


def _build_data(teacher_log_probs, advantages):
    attention_mask, response_mask = _make_masks()
    data = {
        "prompts": torch.zeros(BSZ, PROMPT_LEN, dtype=torch.long),
        "responses": torch.zeros(BSZ, RESPONSE_LEN, dtype=torch.long),
        "attention_mask": attention_mask,
        "response_mask": response_mask,
        "teacher_logprobs": teacher_log_probs,
        "advantages": advantages,
        "old_log_probs": torch.zeros(BSZ, RESPONSE_LEN),
    }
    # batch_size=[] so the padded and jagged entries can coexist, as they do in the real micro batch.
    return TensorDict(data, batch_size=[])


def _make_config(**kwargs):
    loss_config = DistillationLossConfig(loss_mode="k1", use_task_rewards=True, **kwargs)
    return DistillationConfig(enabled=False, distillation_loss=loss_config)


class _PpoRecorder:
    """Stands in for ppo_loss and records the advantage it was asked to clip."""

    def __init__(self, policy_loss=1.0):
        self.policy_loss = policy_loss
        self.calls = []

    def __call__(self, config, model_output, data, dp_group=None, advantages=None):
        self.calls.append(advantages)
        return torch.tensor(self.policy_loss), {"actor/pg_loss": self.policy_loss}

    @property
    def call_count(self):
        return len(self.calls)

    @property
    def advantages(self):
        return self.calls[-1]


@pytest.fixture
def tensors():
    generator = torch.Generator().manual_seed(0)
    return {
        "student": _nested_logprobs(generator),
        "teacher_nested": _nested_logprobs(generator),
        "teacher_padded": torch.randn(BSZ, RESPONSE_LEN, generator=generator),
        "advantages": torch.randn(BSZ, RESPONSE_LEN, generator=generator),
    }


def _teacher_advantage_reference(student, teacher_nested, data):
    """d_t computed independently of the implementation, as log pi_T - log pi_theta."""
    student_padded = no_padding_2_padding(student, data)
    teacher_padded = no_padding_2_padding(teacher_nested, data).squeeze(-1)
    return teacher_padded - student_padded


def test_teacher_advantage_is_the_log_prob_difference(tensors):
    """d_t = log pi_teacher - log pi_student, and nothing outside the response mask."""
    data = _build_data(tensors["teacher_nested"], tensors["advantages"])
    model_output = {"log_probs": tensors["student"]}

    teacher_advantage, metrics = compute_teacher_advantage(model_output, data)

    expected = _teacher_advantage_reference(
        tensors["student"], tensors["teacher_nested"], data
    )
    mask = data["response_mask"].bool()
    torch.testing.assert_close(teacher_advantage, torch.where(mask, expected, torch.zeros_like(expected)))
    # Detached: the teacher term is a reward, not a loss, so no gradient may flow through it.
    assert not teacher_advantage.requires_grad
    assert metrics["distillation/teacher_advantage_mean"] is not None


def test_padded_teacher_log_probs_are_accepted(tensors):
    """The in-process HF teacher writes (bsz, resp_len) directly; nested handling must not be required."""
    data = _build_data(tensors["teacher_padded"], tensors["advantages"])
    model_output = {"log_probs": tensors["student"]}

    teacher_advantage, _ = compute_teacher_advantage(model_output, data)
    student_padded = no_padding_2_padding(tensors["student"], data)
    mask = data["response_mask"].bool()

    torch.testing.assert_close(
        teacher_advantage, torch.where(mask, tensors["teacher_padded"] - student_padded, torch.zeros(()))
    )


@pytest.mark.parametrize("teacher_key", ["teacher_nested", "teacher_padded"])
def test_additive_fuses_into_one_advantage_under_one_clip(tensors, monkeypatch, teacher_key):
    """KDRL: the teacher enters the RL advantage, so ppo_loss sees A_hat + beta * d_t, once."""
    data = _build_data(tensors[teacher_key], tensors["advantages"])
    model_output = {"log_probs": tensors["student"]}
    recorder = _PpoRecorder(policy_loss=1.0)
    monkeypatch.setattr(distillation_losses, "ppo_loss", recorder)

    config = _make_config(advantage_fusion="additive", teacher_advantage_coef=BETA)
    policy_loss, metrics = distillation_ppo_loss(None, config, model_output, data)

    # A single PPO surrogate: the teacher is not a second loss to be added afterwards.
    assert recorder.call_count == 1
    expected = tensors["advantages"] + BETA * torch.where(
        data["response_mask"].bool(),
        _teacher_advantage_reference(tensors["student"], tensors["teacher_nested"], data)
        if teacher_key == "teacher_nested"
        else tensors["teacher_padded"] - no_padding_2_padding(tensors["student"], data),
        torch.zeros(()),
    )
    torch.testing.assert_close(recorder.advantages, expected)
    # Nothing is added on top of the recorded loss, so the fused advantage is the whole update.
    torch.testing.assert_close(policy_loss, torch.tensor(1.0))
    assert "distillation/teacher_advantage_mean" in metrics


def test_additive_leaves_padding_at_the_rl_advantage(tensors, monkeypatch):
    """d_t is zeroed outside the response, so fused positions there stay exactly A_hat."""
    data = _build_data(tensors["teacher_nested"], tensors["advantages"])
    model_output = {"log_probs": tensors["student"]}
    recorder = _PpoRecorder()
    monkeypatch.setattr(distillation_losses, "ppo_loss", recorder)

    distillation_ppo_loss(
        None,
        _make_config(advantage_fusion="additive", teacher_advantage_coef=BETA),
        model_output,
        data,
    )

    padding = ~data["response_mask"].bool()
    torch.testing.assert_close(recorder.advantages[padding], tensors["advantages"][padding])


def test_default_path_keeps_two_independent_losses(tensors, monkeypatch):
    """The pre-existing behaviour: two policy losses, each clipped on its own advantage."""
    data = _build_data(tensors["teacher_nested"], tensors["advantages"])
    model_output = {"log_probs": tensors["student"]}
    recorder = _PpoRecorder(policy_loss=1.0)
    monkeypatch.setattr(distillation_losses, "ppo_loss", recorder)
    distill_loss = torch.tensor(0.5)
    monkeypatch.setattr(
        distillation_losses,
        "distillation_loss",
        lambda *args, **kwargs: (distill_loss, {"distillation/abs_loss": 0.1}),
    )

    coef = 0.5
    config = _make_config(distillation_loss_coef=coef)
    policy_loss, metrics = distillation_ppo_loss(None, config, model_output, data)

    # The RL loss is clipped on the untouched RL advantage -- no teacher term fused in.
    assert recorder.advantages is None
    # ...and the teacher contributes a separate, independently clipped loss on top.
    torch.testing.assert_close(policy_loss, torch.tensor(1.0) + coef * distill_loss)
    assert "distillation/loss" in metrics
    assert "distillation/teacher_advantage_mean" not in metrics


def test_defaults_are_off():
    """Nothing changes for runs that never ask for KDRL."""
    config = DistillationLossConfig()
    assert config.advantage_fusion == "none"
    assert config.teacher_advantage_coef == 0.0


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        # beta set but fusion off: the coefficient would be silently discarded
        ({"teacher_advantage_coef": BETA}, "no effect"),
        # unknown fusion mode
        ({"advantage_fusion": "modulated"}, "must be 'none', 'additive' or 'per_domain'"),
        # non-positive weight pushes the student away from the teacher everywhere
        ({"advantage_fusion": "additive", "teacher_advantage_coef": 0.0}, "must be > 0"),
        ({"advantage_fusion": "additive", "teacher_advantage_coef": -0.1}, "must be > 0"),
        # nothing to fuse into: would collapse to plain OPD scaled by beta
        (
            {"advantage_fusion": "additive", "teacher_advantage_coef": BETA, "use_task_rewards": False},
            "use_task_rewards",
        ),
        # top-k losses are not a difference of log probs
        (
            {
                "advantage_fusion": "additive",
                "teacher_advantage_coef": BETA,
                "loss_mode": "forward_kl_topk",
                "use_policy_gradient": False,
            },
            "log-prob difference",
        ),
        # both knobs act on the same term; the pairing is unvalidated
        (
            {
                "advantage_fusion": "additive",
                "teacher_advantage_coef": BETA,
                "reward_extrapolation_lambda": 1.25,
            },
            "not validated together",
        ),
    ],
)
def test_incompatible_config_is_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        DistillationLossConfig(**kwargs)
