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
"""CPU tests for per-domain OPD/RLVR routing (``advantage_fusion="per_domain"``).

Distillation is normally a global switch: with one teacher every sample is distilled, and with
several the router maps every sample to one. "per_domain" is the exception -- a sample is
supervised by OPD only if its ``data_source`` is listed in ``opd_data_sources``, and by the
verifier reward otherwise.

That makes the failure mode quiet. If routing is wrong, samples still get an advantage, the loss
still decreases, and no metric says the teacher was ignored -- which is exactly what happened
with the eos mask this repo already guards with a test. These tests pin the two halves that can
go wrong: which samples are marked, and what advantage each of them actually receives.
"""

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from verl import DataProto
from verl.trainer.distillation.losses import fuse_advantages_per_domain
from verl.trainer.ppo.ray_trainer import build_opd_mask
from verl.workers.config import DistillationLossConfig
from verl.workers.utils.padding import left_right_2_no_padding

OPD_SOURCE = "LLM-Fusion/IF"


def _data(opd_mask: torch.Tensor, response_mask: torch.Tensor) -> TensorDict:
    return TensorDict(
        {"opd_mask": opd_mask, "response_mask": response_mask},
        batch_size=[opd_mask.shape[0]],
    )


class TestBuildOpdMask:
    """Which samples are routed to OPD."""

    def test_marks_only_the_listed_sources(self):
        sources = [OPD_SOURCE, "LLM-Fusion/Math", OPD_SOURCE, "LLM-Fusion/Code"]
        assert build_opd_mask(sources, [OPD_SOURCE]).tolist() == [1.0, 0.0, 1.0, 0.0]

    def test_matching_is_exact_not_prefix(self):
        """'LLM-Fusion/IF' must not capture 'LLM-Fusion/IFEval'.

        A substring test would silently route the eval-named source into distillation.
        """
        mask = build_opd_mask(["LLM-Fusion/IFEval", OPD_SOURCE], [OPD_SOURCE])
        assert mask.tolist() == [0.0, 1.0]

    def test_several_sources_can_be_routed(self):
        mask = build_opd_mask(["a", "b", "c"], ["a", "c"])
        assert mask.tolist() == [1.0, 0.0, 1.0]

    def test_numpy_zero_d_entries_are_still_strings(self):
        """data_source arrives as 0-d numpy objects; str() must recover the value."""
        sources = np.array([OPD_SOURCE, "LLM-Fusion/Math"], dtype=object)
        assert build_opd_mask(sources, [OPD_SOURCE]).tolist() == [1.0, 0.0]


class TestFuseAdvantagesPerDomain:
    """Each sample ends up on exactly one supervisor -- never a mixture of the two."""

    @pytest.fixture
    def tensors(self):
        bsz, resp_len = 4, 3
        # Distinct values so a sample that received the wrong supervisor cannot pass by
        # coincidence: RL advantages are large, teacher advantages small.
        rl = torch.full((bsz, resp_len), 10.0)
        teacher = torch.full((bsz, resp_len), 0.5)
        opd_mask = torch.tensor([1.0, 0.0, 1.0, 0.0])
        response_mask = torch.ones(bsz, resp_len)
        return rl, teacher, opd_mask, response_mask

    def test_opd_samples_get_the_teacher_signal_scaled_by_the_coefficient(self, tensors):
        rl, teacher, opd_mask, response_mask = tensors
        fused, _ = fuse_advantages_per_domain(rl, teacher, 2.0, _data(opd_mask, response_mask))
        # samples 0 and 2 are OPD: coef * teacher advantage, the RL value must not appear.
        assert torch.allclose(fused[0], torch.full((3,), 1.0))
        assert torch.allclose(fused[2], torch.full((3,), 1.0))

    def test_other_samples_keep_the_rl_advantage_untouched(self, tensors):
        rl, teacher, opd_mask, response_mask = tensors
        fused, _ = fuse_advantages_per_domain(rl, teacher, 2.0, _data(opd_mask, response_mask))
        # samples 1 and 3 are RLVR: byte-identical to the RL advantage.
        assert torch.equal(fused[1], rl[1])
        assert torch.equal(fused[3], rl[3])

    def test_coefficient_does_not_leak_into_the_rl_domains(self, tensors):
        """The point of the mode: strengthening OPD must not dilute RLVR."""
        rl, teacher, opd_mask, response_mask = tensors
        data = _data(opd_mask, response_mask)
        weak, _ = fuse_advantages_per_domain(rl, teacher, 1.0, data)
        strong, _ = fuse_advantages_per_domain(rl, teacher, 100.0, data)
        assert torch.equal(weak[1], strong[1]), "the RL sample moved when the OPD weight changed"
        assert not torch.allclose(weak[0], strong[0]), "the OPD sample did not move"

    def test_all_or_nothing_masks_reduce_to_the_pure_algorithms(self, tensors):
        rl, teacher, _, response_mask = tensors
        bsz = rl.shape[0]
        # Every sample on OPD -> plain OPD.
        fused, _ = fuse_advantages_per_domain(
            rl, teacher, 1.0, _data(torch.ones(bsz), response_mask)
        )
        assert torch.equal(fused, teacher)
        # No sample on OPD -> plain RL.
        fused, _ = fuse_advantages_per_domain(
            rl, teacher, 1.0, _data(torch.zeros(bsz), response_mask)
        )
        assert torch.equal(fused, rl)

    def test_reports_the_routed_fraction(self, tensors):
        """A run stuck at 0.0 (or 1.0) is not routing; make it visible."""
        rl, teacher, opd_mask, response_mask = tensors
        _, metrics = fuse_advantages_per_domain(rl, teacher, 1.0, _data(opd_mask, response_mask))
        assert metrics["distillation/opd_sample_frac"].aggregate() == pytest.approx(0.5)

    def test_reports_the_teacher_advantage_on_opd_tokens_only(self, tensors):
        rl, teacher, opd_mask, response_mask = tensors
        # Make the teacher advantage differ per sample; only the OPD ones may be averaged.
        teacher = torch.tensor([[0.2], [9.9], [0.4], [9.9]]).expand_as(rl)
        _, metrics = fuse_advantages_per_domain(rl, teacher, 1.0, _data(opd_mask, response_mask))
        # mean over OPD samples 0 and 2 -> (0.2 + 0.4) / 2, excluding the 9.9 padding-like rows.
        assert metrics["distillation/opd_teacher_advantage_mean"].aggregate() == pytest.approx(0.3)

    def test_a_batch_with_no_opd_sample_is_legal_but_still_reported(self, tensors):
        rl, teacher, _, response_mask = tensors
        _, metrics = fuse_advantages_per_domain(
            rl, teacher, 1.0, _data(torch.zeros(rl.shape[0]), response_mask)
        )
        assert metrics["distillation/opd_sample_frac"].aggregate() == pytest.approx(0.0)

    def test_metric_keys_are_identical_with_and_without_opd_samples(self, tensors):
        """Every micro batch must emit the same metric keys.

        DP aggregation requires all Metric instances to hold one value per micro batch. A metric
        emitted only when the batch contains an OPD sample collects fewer values than its
        siblings, and aggregate_dp raises "All Metric instances must have the same number of
        values for dp aggregation" -- which killed the first step of the first real run, because
        micro batches with no routed sample are the *common* case (IF is ~12% of the mixture).
        """
        rl, teacher, opd_mask, response_mask = tensors
        _, with_opd = fuse_advantages_per_domain(rl, teacher, 1.0, _data(opd_mask, response_mask))
        _, without_opd = fuse_advantages_per_domain(
            rl, teacher, 1.0, _data(torch.zeros(rl.shape[0]), response_mask)
        )
        assert set(with_opd) == set(without_opd), (
            f"metrics differ: only-with-OPD={set(with_opd) - set(without_opd)}, "
            f"only-without-OPD={set(without_opd) - set(with_opd)}"
        )

    def test_missing_mask_raises_instead_of_silently_distilling_nothing(self, tensors):
        rl, teacher, _, response_mask = tensors
        # A batch that never went through the routing step: no 'opd_mask' key at all.
        unrouted = TensorDict({"response_mask": response_mask}, batch_size=[rl.shape[0]])
        with pytest.raises(ValueError, match="opd_mask"):
            fuse_advantages_per_domain(rl, teacher, 1.0, unrouted)


class TestOpdMaskReachesTheLoss:
    """The mask has to survive the batch transforms between the trainer and the loss.

    If a transform dropped or reshaped it, the loss would raise (no 'opd_mask'), or worse, pick
    up a mask of the wrong length and route the wrong samples. These assert the two transforms
    on that path leave a per-sample rank-1 mask intact.
    """

    def test_survives_left_right_2_no_padding(self):
        """_update_actor runs this on the whole batch before update_actor.

        It converts only the per-token keys (input_ids, position_ids, loss_mask, teacher
        logprobs) to jagged; a per-sample key like opd_mask must pass through untouched.
        """
        bsz, seq_len, resp_len = 4, 8, 3
        td = TensorDict(
            {
                "input_ids": torch.randint(0, 100, (bsz, seq_len)),
                "attention_mask": torch.ones(bsz, seq_len, dtype=torch.long),
                "position_ids": torch.arange(seq_len).unsqueeze(0).expand(bsz, seq_len).contiguous(),
                "response_mask": torch.zeros(bsz, resp_len),
                "advantages": torch.randn(bsz, resp_len),
                "opd_mask": torch.tensor([1.0, 0.0, 1.0, 0.0]),
            },
            batch_size=[bsz],
        )
        out = left_right_2_no_padding(td)
        assert "opd_mask" in out.keys(), "opd_mask was dropped by the no-padding conversion"
        mask = out["opd_mask"]
        assert not mask.is_nested, "opd_mask must stay rank-1, not become a per-token tensor"
        assert mask.shape == (bsz,)
        assert mask.tolist() == [1.0, 0.0, 1.0, 0.0]

    def test_can_be_written_into_a_dataproto_batch(self):
        """_attach_opd_mask assigns a new key onto the live DataProto batch."""
        bsz = 4
        proto = DataProto(
            batch=TensorDict({"advantages": torch.randn(bsz, 3)}, batch_size=[bsz]),
            non_tensor_batch={"data_source": np.array(["a", "b", "a", "b"], dtype=object)},
        )
        proto.batch["opd_mask"] = torch.from_numpy(build_opd_mask(["a", "b", "a", "b"], ["a"]))
        assert proto.batch["opd_mask"].tolist() == [1.0, 0.0, 1.0, 0.0]


class TestPerDomainConfigValidation:
    """The inconsistent combinations must fail at config time, not silently no-op."""

    def _config(self, **kwargs) -> DistillationLossConfig:
        kwargs.setdefault("teacher_advantage_coef", 1.0)
        return DistillationLossConfig(
            loss_mode="k3",
            advantage_fusion="per_domain",
            **kwargs,
        )

    def test_accepts_a_valid_routing(self):
        cfg = self._config(opd_data_sources=[OPD_SOURCE])
        assert cfg.advantage_fusion == "per_domain"

    def test_empty_domain_list_is_rejected(self):
        """An empty list leaves every sample on RL and the teacher silently does nothing."""
        with pytest.raises(ValueError, match="opd_data_sources"):
            self._config(opd_data_sources=[])

    def test_missing_domain_list_is_rejected(self):
        with pytest.raises(ValueError, match="opd_data_sources"):
            self._config(opd_data_sources=None)

    def test_non_positive_coefficient_is_rejected(self):
        with pytest.raises(ValueError, match="teacher_advantage_coef"):
            self._config(opd_data_sources=[OPD_SOURCE], teacher_advantage_coef=0.0)

    def test_requires_an_estimator_loss_mode(self):
        """A top-k distributional loss has no per-token log-prob difference to route."""
        with pytest.raises(ValueError):
            DistillationLossConfig(
                loss_mode="forward_kl_topk",
                advantage_fusion="per_domain",
                teacher_advantage_coef=1.0,
                opd_data_sources=[OPD_SOURCE],
            )
