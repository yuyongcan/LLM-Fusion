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

import pytest
import torch

from verl.trainer.ppo.core_algos import agg_loss


def test_sequence_mean_weights_responses_equally() -> None:
    """Unequal response lengths do not change per-response aggregation weight."""
    losses = torch.tensor([[2.0, 99.0, 99.0], [4.0, 4.0, 4.0]], requires_grad=True)
    mask = torch.tensor([[True, False, False], [True, True, True]])

    sequence_mean = agg_loss(losses, mask, "seq-mean-token-mean")
    token_mean = agg_loss(losses, mask, "token-mean")

    torch.testing.assert_close(sequence_mean, torch.tensor(3.0))
    torch.testing.assert_close(token_mean, torch.tensor(3.5))
    sequence_mean.backward()
    torch.testing.assert_close(losses.grad, torch.tensor([[0.5, 0.0, 0.0], [1 / 6, 1 / 6, 1 / 6]]))


def test_sequence_mean_excludes_fully_masked_responses() -> None:
    """Padding and responses without valid tokens do not affect the local mean."""
    losses = torch.tensor([[2.0, 99.0], [4.0, 4.0], [99.0, 99.0]])
    mask = torch.tensor([[True, False], [True, True], [False, False]])

    loss = agg_loss(losses, mask, "seq-mean-token-mean")

    torch.testing.assert_close(loss, torch.tensor(3.0))


def test_sequence_mean_uses_global_denominator_across_micro_batches() -> None:
    """Splitting responses across micro batches and DP ranks preserves the mean."""
    losses = torch.tensor([[2.0, 99.0], [4.0, 4.0], [6.0, 99.0]])
    mask = torch.tensor([[True, False], [True, True], [True, False]])
    expected = agg_loss(losses, mask, "seq-mean-token-mean")
    dp_size = 2
    local_losses = [
        agg_loss(losses[start:end], mask[start:end], "seq-mean-token-mean", dp_size=dp_size, global_batch_size=3)
        for start, end in ((0, 1), (1, 2), (2, 3))
    ]

    torch.testing.assert_close(sum(local_losses) / dp_size, expected)


def test_invalid_loss_aggregation_is_rejected() -> None:
    """Unknown aggregation modes fail instead of silently using a default."""
    with pytest.raises(ValueError, match="Invalid loss_agg_mode"):
        agg_loss(torch.ones(1, 1), torch.ones(1, 1), "invalid")
