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
"""Trainer metadata must disable sampled-token gradients only for pure supervised distillation."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from omegaconf import OmegaConf
from tensordict import TensorDict

from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils import tensordict_utils as tu


@pytest.mark.parametrize(
    ("enabled", "use_topk", "use_policy_gradient", "use_task_rewards", "expected_requires_grad"),
    [
        (True, True, False, False, False),
        (True, True, False, True, True),
        (True, True, True, False, True),
        (True, False, True, False, True),
        (False, True, False, False, True),
    ],
)
def test_update_actor_sets_sampled_token_gradient_metadata(
    monkeypatch, enabled, use_topk, use_policy_gradient, use_task_rewards, expected_requires_grad
):
    trainer = object.__new__(RayPPOTrainer)
    trainer.config = OmegaConf.create(
        {
            "distillation": {"enabled": enabled},
            "actor_rollout_ref": {
                "actor": {
                    "calculate_entropy": False,
                    "entropy_coeff": 0.0,
                    "ppo_mini_batch_size": 1,
                    "ppo_epochs": 1,
                    "data_loader_seed": 0,
                    "shuffle": False,
                },
                "rollout": {"multi_turn": {"enable": False}, "temperature": 1.0, "n": 1},
            },
        }
    )
    trainer.distillation_config = SimpleNamespace(
        distillation_loss=SimpleNamespace(
            loss_settings=SimpleNamespace(use_topk=use_topk),
            use_policy_gradient=use_policy_gradient,
            use_task_rewards=use_task_rewards,
        )
    )
    batch_td = TensorDict({}, batch_size=[])
    batch = SimpleNamespace(meta_info={}, to_tensordict=lambda: batch_td)
    monkeypatch.setattr("verl.trainer.ppo.ray_trainer.left_right_2_no_padding", lambda data: data)
    monkeypatch.setattr(trainer, "_rep_hidden_layers", lambda: ())
    metrics = TensorDict({}, batch_size=[])
    tu.assign_non_tensor(metrics, metrics={"mfu": 0.0})
    trainer.actor_rollout_wg = SimpleNamespace(update_actor=Mock(return_value=metrics))

    trainer._update_actor(batch)

    assert tu.get_non_tensor_data(batch_td, "log_probs_requires_grad", None) is expected_requires_grad
    assert tu.get_non_tensor_data(batch_td, "distillation_use_topk", None) is (enabled and use_topk)
    trainer.actor_rollout_wg.update_actor.assert_called_once_with(batch_td)
