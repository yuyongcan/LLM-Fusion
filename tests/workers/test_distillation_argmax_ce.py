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
"""CUDA regression tests for fused argmax CE and shared-logits backward."""

import pytest
import torch

from verl.trainer.distillation.fsdp.losses import _gather_log_softmax
from verl.utils.torch_functional import FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE, logprobs_from_logits

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE,
    reason="requires CUDA and flash-attn's fused cross-entropy",
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shared_logits", [True, False])
def test_fused_ce_backward_matches_dense(dtype: torch.dtype, shared_logits: bool):
    torch.manual_seed(0)
    hidden = torch.randn(17, 16, device="cuda", dtype=dtype, requires_grad=True)
    weight = torch.randn(513, 16, device="cuda", dtype=dtype, requires_grad=True)
    labels = torch.randint(0, 513, (17,), device="cuda")
    teacher_ids = torch.randint(0, 513, (17, 1), device="cuda")
    logits = hidden @ weight.T
    original_logits = logits.detach().clone()

    ce = -_gather_log_softmax(logits, teacher_ids, inplace_backward=not shared_logits)
    loss = ce.sum()
    if shared_logits:
        sampled_log_probs = logprobs_from_logits(logits, labels, inplace_backward=False)
        loss = loss - 0.3 * sampled_log_probs.sum()
    loss.backward()
    if shared_logits:
        torch.testing.assert_close(logits.detach(), original_logits)

    dense_hidden = hidden.detach().clone().requires_grad_(True)
    dense_weight = weight.detach().clone().requires_grad_(True)
    dense_log_probs = torch.log_softmax((dense_hidden @ dense_weight.T).float(), dim=-1)
    dense_ce = -dense_log_probs.gather(-1, teacher_ids).squeeze(-1)
    expected_loss = dense_ce.sum()
    if shared_logits:
        expected_loss = expected_loss - 0.3 * dense_log_probs.gather(-1, labels.unsqueeze(-1)).sum()
    expected_loss.backward()

    torch.testing.assert_close(ce, dense_ce, atol=1e-5, rtol=1e-5)
    tolerance = {"atol": 0.03, "rtol": 0.03} if dtype == torch.bfloat16 else {"atol": 1e-5, "rtol": 1e-5}
    torch.testing.assert_close(hidden.grad, dense_hidden.grad, **tolerance)
    torch.testing.assert_close(weight.grad, dense_weight.grad, **tolerance)
