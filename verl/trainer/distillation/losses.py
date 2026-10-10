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

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch
from tensordict import TensorDict

from verl.base_config import BaseConfig
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils import tensordict_utils as tu
from verl.utils.metric import AggregationType, Metric
from verl.workers.config import ActorConfig, DistillationConfig, DistillationLossConfig
from verl.workers.utils.losses import ppo_loss
from verl.workers.utils.padding import no_padding_2_padding

DistillationLossFn = Callable[
    [
        ActorConfig,  # actor_config
        DistillationConfig,  # distillation_config
        dict,  # model_output
        TensorDict,  # micro batch input
    ],
    tuple[torch.Tensor, dict[str, Any]],
]


def is_distillation_enabled(config: Optional[DistillationConfig]) -> bool:
    """Check if distillation is enabled based on the provided configuration."""
    if config is None:
        return False
    return config.enabled


@dataclass
class DistillationLossSettings(BaseConfig):
    """
    Settings for a distillation loss function to be registered.

    Args:
        names (str | list[str]): Name(s) to register the distillation loss function under.
        use_topk (bool): Whether the loss function uses top-k log probabilities.
        use_estimator (bool): Whether the loss function uses single-sample KL estimators.
    """

    names: str | list[str] = field(default_factory=list)
    use_topk: bool = False
    use_estimator: bool = False

    _mutable_fields = {"names"}

    def __post_init__(self):
        self.names = [self.names] if isinstance(self.names, str) else self.names
        if sum([self.use_topk, self.use_estimator]) != 1:
            raise ValueError(
                f"Expected only one of use_estimator, use_topk, but got {self.use_estimator=}, {self.use_topk=}."
            )


DISTILLATION_LOSS_REGISTRY: dict[str, DistillationLossFn] = {}
DISTILLATION_SETTINGS_REGISTRY: dict[str, DistillationLossSettings] = {}


def register_distillation_loss(
    loss_settings: DistillationLossSettings,
) -> Callable[[DistillationLossFn], DistillationLossFn]:
    """Register a distillation loss function with the given name."""

    def decorator(func: DistillationLossFn) -> DistillationLossFn:
        for name in loss_settings.names:
            if name in DISTILLATION_LOSS_REGISTRY:
                raise ValueError(f"Distillation loss function with name '{name}' is already registered.")
            DISTILLATION_LOSS_REGISTRY[name] = func
            DISTILLATION_SETTINGS_REGISTRY[name] = loss_settings
        return func

    return decorator


def get_distillation_loss_fn(loss_name: str) -> DistillationLossFn:
    """Get the distillation loss function with a given name."""
    if loss_name not in DISTILLATION_LOSS_REGISTRY:
        raise ValueError(
            f"Unsupported loss mode: {loss_name}. Supported modes are: {list(DISTILLATION_LOSS_REGISTRY.keys())}"
        )
    return DISTILLATION_LOSS_REGISTRY[loss_name]


def get_distillation_loss_settings(loss_name: str) -> DistillationLossSettings:
    """Get the distillation loss settings with a given name."""
    if loss_name not in DISTILLATION_SETTINGS_REGISTRY:
        raise ValueError(
            f"Unsupported loss mode: {loss_name}. Supported modes are: {list(DISTILLATION_SETTINGS_REGISTRY.keys())}"
        )
    return DISTILLATION_SETTINGS_REGISTRY[loss_name]


def compute_distillation_loss_range(
    distillation_losses: torch.Tensor, response_mask: torch.Tensor
) -> dict[str, Metric]:
    """Compute min and max distillation loss over valid response tokens."""
    if response_mask.is_nested:
        distillation_losses_response = distillation_losses[response_mask.bool().to_padded_tensor(False)]
    else:
        distillation_losses_response = distillation_losses[response_mask.bool()]
    return {
        "distillation/loss_min": Metric(AggregationType.MIN, distillation_losses_response.min()),
        "distillation/loss_max": Metric(AggregationType.MAX, distillation_losses_response.max()),
    }


def compute_topk_loss(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    data: TensorDict,
    student_logits: torch.Tensor,
    data_format: str,
) -> torch.Tensor:
    """Compute the topk loss in logit processor.

    Returns:
    - distillation_losses: (bsz, seqlen/cp_size)
    - student_mass: (bsz, seqlen/cp_size)
    - teacher_mass: (bsz, seqlen/cp_size)
    """
    loss_mode = distillation_config.distillation_loss.loss_mode
    match config.strategy:
        # VeOmni uses FSDP2 internally, so its loss computation is identical to FSDP.
        case "fsdp" | "veomni":
            import verl.trainer.distillation.fsdp.losses as fsdp_losses

            distillation_loss_fn = (
                fsdp_losses.compute_argmax_ce
                if loss_mode == "argmax_ce"
                else fsdp_losses.compute_forward_kl_topk
            )
        case "megatron":
            if loss_mode == "argmax_ce":
                raise NotImplementedError(
                    "loss_mode='argmax_ce' is only implemented for the fsdp/veomni strategy. "
                    "Add compute_argmax_ce to verl.trainer.distillation.megatron.losses, or use "
                    "loss_mode='forward_kl_topk' with topk=1."
                )
            import verl.trainer.distillation.megatron.losses as megatron_losses

            distillation_loss_fn = megatron_losses.compute_forward_kl_topk
        case _:
            raise NotImplementedError(f"Unsupported strategy: {config.strategy=}")

    loss_kwargs = {}
    if loss_mode == "argmax_ce":
        # The actor's sampled-token loss and entropy must not read logits overwritten by CE.
        loss_kwargs["inplace_backward"] = not any(
            tu.get_non_tensor_data(data, key=key, default=default)
            for key, default in (
                ("log_probs_requires_grad", True),
                ("calculate_entropy", False),
                ("calculate_sum_pi_squared", False),
            )
        )

    outputs = distillation_loss_fn(
        student_logits=student_logits,
        teacher_topk_log_probs=data["teacher_logprobs"],
        teacher_topk_ids=data["teacher_ids"],
        config=distillation_config,
        data_format=data_format,
        **loss_kwargs,
    )

    expected_shape = student_logits.shape[:2]
    for k, v in outputs.items():
        assert v.shape == expected_shape, f"Expected shape {expected_shape}, but got {v.shape} for {k=}."

    return outputs


def distillation_ppo_loss(
    config: ActorConfig,
    distillation_config: Optional[DistillationConfig],
    model_output: dict = None,
    data: TensorDict = None,
    dp_group=None,
    student_logits: torch.Tensor = None,
    data_format: str = "thd",
):
    """Loss function used both for logit processor and final policy loss.
    - student_logits is not None, compute the topk loss in logit processor.
    - student_logits is None, compute final policy loss.

    [split sequence across sp/cp groups]
                   |
    [model forward and output logits: (bsz, seqlen/cp_size, vocab_size/tp_size)]
                   |
    [logits processor compute topk loss: (bsz, seqlen/cp_size)]
                   |
    [all gather topk loss across sp/cp groups: (bsz, seqlen)]
                   |
    [combine topk loss with policy loss]

    Args:
        config: Actor configuration.
        distillation_config: Distillation configuration.
        model_output: Model output, including log_probs, entropy.
        data: Micro input batch, contains
          - teacher_logprobs: (bsz, seqlen, topk)
          - teacher_ids: (bsz, seqlen, topk)
        student_logits: (bsz, seqlen/cp_size, vocab_size/tp_size).
        data_format: "thd" or "bshd", models not support THD format, e.g GPT-OSS, Qwen3.5

    Returns:
    - student_logits is not None, return the topk loss tensor (bsz, seqlen/cp_size).
    - student_logits is None, return the final policy loss scalar and metrics.
    """

    # Called as logits processor
    if student_logits is not None:
        return compute_topk_loss(config, distillation_config, data, student_logits, data_format)

    # Called as final policy loss
    loss_config = distillation_config.distillation_loss

    # "additive" (KDRL) and "per_domain" share the fused-advantage machinery: one PPO clip over a
    # per-token advantage that carries the teacher signal. They differ only in how that advantage
    # is composed -- summed everywhere, versus routed per sample.
    if loss_config.advantage_fusion in ("additive", "per_domain"):
        # KDRL (https://arxiv.org/abs/2506.02208): the teacher signal enters the *same* advantage
        # as the verifiable reward, so the two are fused before clipping:
        #
        #   A_t = A_hat + beta * (log pi_teacher - log pi_student)
        #
        # This is deliberately not the `else` branch with a different coefficient. Below, the RL
        # loss and the distillation loss are two separate PPO surrogates, each clipped on its own
        # advantage and then added. Summing two clipped terms is not the same as clipping their
        # sum: with a single fused advantage the clip decides once, on the combined direction,
        # which is what lets the teacher term flip the sign of an update that the verifier wanted.
        # That sign flip is the property the paper isolates -- it is why teacher-modulated
        # variants exist at all -- so a two-loss implementation is not a stand-in for it.
        teacher_advantage, teacher_metrics = compute_teacher_advantage(model_output, data)
        advantages = data["advantages"]
        if advantages.is_nested:
            advantages = advantages.to_padded_tensor(0.0)
        if loss_config.advantage_fusion == "additive":
            fused_advantages = advantages + loss_config.teacher_advantage_coef * teacher_advantage
            routing_metrics: dict[str, Any] = {}
        else:
            fused_advantages, routing_metrics = fuse_advantages_per_domain(
                advantages, teacher_advantage, loss_config.teacher_advantage_coef, data
            )
        policy_loss, policy_metrics = ppo_loss(
            config, model_output, data, dp_group, advantages=fused_advantages
        )
        policy_metrics.update(teacher_metrics)
        policy_metrics.update(routing_metrics)
        # The teacher no longer contributes its own loss term, so distillation/loss is absent by
        # design. Reading a run's teacher weight from `distillation/teacher_advantage_*` instead.
    else:
        distill_loss, distill_metrics = distillation_loss(config, distillation_config, model_output, data)
        policy_loss, policy_metrics = ppo_loss(config, model_output, data, dp_group)
        if not loss_config.use_task_rewards:
            policy_loss = 0.0

        # Combine distillation with policy loss
        policy_metrics.update(distill_metrics)
        distillation_loss_coef = (
            loss_config.distillation_loss_coef if loss_config.use_task_rewards else 1.0
        )
        policy_loss += distill_loss * distillation_loss_coef
        policy_metrics["distillation/loss"] = Metric(value=distill_loss, aggregation=AggregationType.SUM)

    # Representation distillation (OPRD), additive on top of whatever the output-space term is.
    # Separate from distillation_loss() above because it does not read log probs at all: it
    # compares hidden states, so it needs model_output's per-layer tensors rather than the
    # teacher's log probs, and it is the one term whose gradient does not pass through the LM
    # head. rep_coef == 0.0 skips it entirely rather than adding a multiplied-by-zero tensor.
    if loss_config.rep_coef > 0.0:
        rep_loss, rep_metrics = compute_representation_loss(loss_config, model_output, data)
        policy_loss = policy_loss + rep_loss * loss_config.rep_coef
        for name, value in rep_metrics.items():
            policy_metrics[name] = Metric(value=value, aggregation=AggregationType.MEAN)

    return policy_loss, policy_metrics


def compute_representation_loss(
    loss_config,
    model_output: dict,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Align the student's hidden states with the teacher's on the supervised suffix.

    The student's layers arrive in ``model_output`` as jagged tensors over prompt+response (the
    engine emits them next to log_probs); the teacher's arrive in ``data`` already sliced to the
    last-k window by the trainer, which is what keeps this affordable -- the unrestricted tensor
    is ~190 KB per token per layer.

    Both sides are put through the same padding helper and the same gather, so the one-token
    left shift and the per-sample window cannot drift between them.
    """
    from verl.trainer.distillation.representation import (
        gather_last_k,
        representation_distillation_loss,
        teacher_hidden_key,
    )
    from verl.workers.utils.padding import no_padding_2_padding

    layers = tuple(loss_config.rep_layers)
    response_mask = data["response_mask"]
    if response_mask.is_nested:
        response_mask = response_mask.to_padded_tensor(0)

    student_layers = []
    teacher_layers = []
    mask = None
    for layer in layers:
        student_key = f"rep_hidden_{layer}"
        if student_key not in model_output:
            raise ValueError(
                f"representation distillation needs {student_key!r} from the student forward but "
                "it is absent; check that rep_hidden_layers reached the engine."
            )
        teacher_key = teacher_hidden_key(layer)
        if teacher_key not in data.keys():
            raise ValueError(
                f"representation distillation needs {teacher_key!r} in the batch but it is "
                "absent; check that the in-process teacher was asked for this layer."
            )

        student_hidden = no_padding_2_padding(model_output[student_key], data)
        student_gathered, mask = gather_last_k(student_hidden, response_mask, loss_config.rep_last_k)
        student_layers.append(student_gathered)
        teacher_layers.append(data[teacher_key])

    student_stacked = torch.stack(student_layers, dim=1)
    teacher_stacked = torch.stack(teacher_layers, dim=1).to(student_stacked.device)

    return representation_distillation_loss(
        student_stacked,
        teacher_stacked,
        mask,
        normalize=loss_config.rep_normalize,
    )


def fuse_advantages_per_domain(
    advantages: torch.Tensor,
    teacher_advantage: torch.Tensor,
    teacher_advantage_coef: float,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Pick one supervisor per sample: OPD on the configured domains, the verifier elsewhere.

    This is the routed counterpart of ``advantage_fusion="additive"`` (KDRL). KDRL adds the two
    signals on every token; here they never mix -- a sample is either distilled or reinforced:

        data_source in opd_data_sources -> A_t = coef * (log pi_teacher - log pi_student)
        otherwise                       -> A_t = the estimator's RL advantage

    Keeping the selection in ``advantages`` rather than masking the two losses separately is
    deliberate: ``agg_loss`` normalises by the token count of the whole global batch, so two
    independently masked losses would each be scaled by a denominator from a different sample
    population, and the effective weight of each domain would drift with the batch's domain mix.

    Returns the fused advantage plus diagnostics that make a mis-routed run visible: the OPD
    sample fraction and, on OPD response tokens only, the mean teacher advantage. A run where
    the fraction is 0.0 (or 1.0) is not routing at all, and the mean teacher advantage should
    start positive and decay toward 0 as the student converges onto the teacher.
    """
    if "opd_mask" not in data.keys():
        raise ValueError(
            "advantage_fusion='per_domain' needs a per-sample 'opd_mask' in the batch, but it is "
            "absent. It is written by the trainer from distillation_loss.opd_data_sources; check "
            "that the training loop runs the routing step before update_actor."
        )

    opd_mask = data["opd_mask"]
    if opd_mask.is_nested:
        opd_mask = opd_mask.to_padded_tensor(0.0)
    # (bsz,) -> broadcast over the response dimension so torch.where can select per sample.
    sample_is_opd = opd_mask.to(device=advantages.device).bool()
    while sample_is_opd.dim() < advantages.dim():
        sample_is_opd = sample_is_opd.unsqueeze(-1)
    token_is_opd = sample_is_opd.expand_as(advantages)

    fused_advantages = torch.where(
        token_is_opd, teacher_advantage_coef * teacher_advantage, advantages
    )

    response_mask = data["response_mask"]
    if response_mask.is_nested:
        response_mask = response_mask.to_padded_tensor(False)
    response_mask = response_mask.bool()

    metrics: dict[str, Any] = {
        "distillation/opd_sample_frac": Metric(
            AggregationType.MEAN, sample_is_opd.float().mean()
        ),
    }
    # Every metric below must be emitted on *every* micro batch, including ones with no OPD
    # sample at all. DP aggregation (verl.utils.metric.utils.aggregate_dp) requires all Metric
    # instances to hold the same number of values -- one per micro batch -- so a metric that
    # appears only on some batches raises "All Metric instances must have the same number of
    # values for dp aggregation" and kills the step. Micro batches with no routed sample are
    # normal here (IF is ~12% of the mixture), so this is the common case, not an edge case.
    opd_tokens = response_mask & token_is_opd
    zero = torch.zeros((), device=advantages.device)
    if opd_tokens.any():
        opd_adv = teacher_advantage[opd_tokens]
        opd_adv_mean = opd_adv.mean()
        # Share of OPD response tokens the teacher still scores above the student. Decays toward
        # 0.5 as the student converges onto the teacher.
        opd_preferred = (opd_adv > 0).float().mean()
    else:
        # A whole run at 0.0 means opd_data_sources matched nothing; opd_sample_frac above
        # surfaces that. A single empty micro batch is expected and says nothing.
        opd_adv_mean, opd_preferred = zero, zero
    metrics["distillation/opd_teacher_advantage_mean"] = Metric(AggregationType.MEAN, opd_adv_mean)
    metrics["distillation/opd_teacher_preferred_frac"] = Metric(AggregationType.MEAN, opd_preferred)
    return fused_advantages, metrics


def compute_teacher_advantage(
    model_output: dict,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """KDRL's per-token teacher advantage, d_t = log pi_teacher(y_t | h_t) - log pi_student(y_t | h_t).

    This is the quantity that on-policy distillation minimises, used here as a dense token-level
    *reward* rather than as a loss: it is detached, so it reaches the model only through the
    advantage of the policy-gradient update, never through its own gradient. That is also why the
    padding positions are zeroed rather than left at whatever the jagged->padded conversion
    produced -- an advantage is only ever read under ``response_mask``, and a nonzero pad value
    would corrupt the reported statistics without showing up in the loss.

    Two layouts reach this point, exactly as in ``compute_distillation_loss_reverse_kl_estimator``:
    the vLLM teacher writes ``teacher_logprobs`` as a jagged tensor over prompt+response during
    rollout, while the in-process HF teacher writes a plain (bsz, resp_len) tensor already sliced
    to the response. Both are normalised to (bsz, resp_len) here.

    Returns:
    - teacher_advantage: (bsz, resp_len), zero outside the response mask.
    - metrics: diagnostics on the teacher advantage over response tokens only.
    """
    student_log_probs = no_padding_2_padding(model_output["log_probs"], data)
    teacher_log_probs = data["teacher_logprobs"]
    if teacher_log_probs.is_nested:
        teacher_log_probs = no_padding_2_padding(teacher_log_probs, data).squeeze(-1)
    response_mask = data["response_mask"]
    if response_mask.is_nested:
        response_mask = response_mask.to_padded_tensor(False)
    mask = response_mask.bool()
    assert teacher_log_probs.shape == student_log_probs.shape == mask.shape, (
        f"teacher {tuple(teacher_log_probs.shape)}, student {tuple(student_log_probs.shape)} and "
        f"mask {tuple(mask.shape)} must agree to form a per-token teacher advantage."
    )

    teacher_advantage = (teacher_log_probs - student_log_probs).detach()
    teacher_advantage = torch.where(mask, teacher_advantage, torch.zeros_like(teacher_advantage))

    if mask.any():
        valid = teacher_advantage[mask]
        metrics = {
            "distillation/teacher_advantage_mean": Metric(AggregationType.MEAN, valid.mean()),
            "distillation/teacher_advantage_std": Metric(AggregationType.MEAN, valid.std()),
            "distillation/teacher_advantage_max": Metric(AggregationType.MAX, valid.max()),
            "distillation/teacher_advantage_min": Metric(AggregationType.MIN, valid.min()),
            # Fraction of response tokens the teacher scores above the student. Near 1.0 the
            # teacher still dominates and beta is mostly shaping; near 0.5 the two have converged
            # and the term is close to symmetric noise.
            "distillation/teacher_preferred_frac": Metric(AggregationType.MEAN, (valid > 0).float().mean()),
        }
    else:
        metrics = {}
    return teacher_advantage, metrics


def distillation_loss(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output: dict,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the distillation loss and related metrics.

    Returns:
    - distillation_loss: Aggregated distillation loss scalar.
    - distillation_metrics: Dictionary of metrics.
    """
    assert distillation_config is not None
    loss_config: DistillationLossConfig = distillation_config.distillation_loss
    distillation_loss_fn = get_distillation_loss_fn(loss_config.loss_mode)
    distillation_losses, distillation_metrics = distillation_loss_fn(
        config=config,
        distillation_config=distillation_config,
        model_output=model_output,
        data=data,
    )
    response_mask = data["response_mask"]
    loss_agg_mode = config.loss_agg_mode

    distillation_metrics.update(
        compute_distillation_loss_range(distillation_losses=distillation_losses, response_mask=response_mask)
    )
    if loss_config.loss_max_clamp is not None:
        # clamping min is for k1 loss which can be negative
        distillation_losses = distillation_losses.clamp(min=-loss_config.loss_max_clamp, max=loss_config.loss_max_clamp)

    if loss_config.use_policy_gradient:
        # Use negative distillation loss as reward, as done by https://thinkingmachines.ai/blog/on-policy-distillation/.
        policy_loss_fn = get_policy_loss_fn(loss_config.policy_loss_mode)
        for k, v in config.global_batch_info.items():
            loss_config.global_batch_info[k] = v
        log_prob = no_padding_2_padding(model_output["log_probs"], data)
        old_log_prob = data["old_log_probs"]
        if old_log_prob.is_nested:
            old_log_prob = data["old_log_probs"].to_padded_tensor(0.0)
        if response_mask.is_nested:
            response_mask = response_mask.to_padded_tensor(False)
        rollout_is_weights = data.get("rollout_is_weights", None)
        distillation_loss, pg_metrics = policy_loss_fn(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=-distillation_losses.detach(),
            response_mask=response_mask,
            loss_agg_mode=loss_agg_mode,
            config=loss_config,
            rollout_is_weights=rollout_is_weights,
        )
        pg_metrics = {f"distillation/{k[len('actor/') :]}": v for k, v in pg_metrics.items()}
        distillation_metrics.update(pg_metrics)
    else:
        # Directly backpropagate distillation loss as a supervised loss, as in https://arxiv.org/abs/2306.13649.
        if response_mask.is_nested:
            response_mask = response_mask.to_padded_tensor(False)
        distillation_loss = agg_loss(
            loss_mat=distillation_losses,
            loss_mask=response_mask,
            loss_agg_mode=loss_agg_mode,
            **config.global_batch_info,
        )

    return distillation_loss, distillation_metrics


@register_distillation_loss(DistillationLossSettings(names=["forward_kl_topk"], use_topk=True))  # type: ignore[arg-type]
def compute_forward_kl_topk(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output: dict,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute forward KL distillation loss and related metrics using top-k log probabilities.

    Returns:
    - distillation_losses: (bsz, resp_len)
    - distillation_metrics: Dictionary of metrics.
    """
    # topk loss has been computed in logits processor
    distillation_losses = no_padding_2_padding(model_output["distillation_losses"], data)
    student_mass = no_padding_2_padding(model_output["student_mass"], data)
    teacher_mass = no_padding_2_padding(model_output["teacher_mass"], data)
    overlap_count = model_output.get("overlap_count")
    overlap_token_advantage = model_output.get("overlap_token_advantage")
    if overlap_count is not None and overlap_token_advantage is not None:
        overlap_count = no_padding_2_padding(overlap_count, data)
        overlap_token_advantage = no_padding_2_padding(overlap_token_advantage, data)
    if data["response_mask"].is_nested:
        response_mask_bool = data["response_mask"].bool().to_padded_tensor(False)
    else:
        response_mask_bool = data["response_mask"].bool()
    assert distillation_losses.shape == student_mass.shape == teacher_mass.shape == response_mask_bool.shape

    overlap_metrics = {}
    if overlap_count is not None and overlap_token_advantage is not None:
        assert overlap_count.shape == overlap_token_advantage.shape == response_mask_bool.shape
        valid_overlap_count = overlap_count[response_mask_bool]
        k = distillation_config.distillation_loss.topk
        assert k is not None
        # Diagnostics for tracking teacher/student top-k overlap in OPD, following
        # "Rethinking On-Policy Distillation of Large Language Models" (arXiv:2604.13016):
        # overlap ratio and average teacher-token KL contribution on overlapped tokens.
        overlap_metrics["distillation/overlap_ratio"] = (valid_overlap_count.float().mean() / k).item()
        overlap_position_mask = response_mask_bool & (overlap_count > 0)
        if overlap_position_mask.any():
            overlap_metrics["distillation/overlap_token_advantage"] = (
                overlap_token_advantage[overlap_position_mask].mean().item()
            )
        else:
            overlap_metrics["distillation/overlap_token_advantage"] = 0.0

    # Log amount of mass in the top-k log probabilities for both student and teacher.
    student_mass = student_mass[response_mask_bool]
    teacher_mass = teacher_mass[response_mask_bool]
    distillation_metrics = {
        "distillation/student_mass": student_mass.mean().item(),
        "distillation/student_mass_min": Metric(AggregationType.MIN, student_mass.min()),
        "distillation/student_mass_max": Metric(AggregationType.MAX, student_mass.max()),
        "distillation/teacher_mass": teacher_mass.mean().item(),
        "distillation/teacher_mass_min": Metric(AggregationType.MIN, teacher_mass.min()),
        "distillation/teacher_mass_max": Metric(AggregationType.MAX, teacher_mass.max()),
        **overlap_metrics,
    }

    # Due to use of top-k, student and teacher distributions don't sum to 1 -> divergences can be negative.
    distillation_losses = distillation_losses.clamp_min(0.0)

    return distillation_losses, distillation_metrics


@register_distillation_loss(DistillationLossSettings(names=["argmax_ce"], use_topk=True))  # type: ignore[arg-type]
def compute_argmax_ce(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output: dict,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Collect the cross-entropy against the teacher's argmax token, computed in the logits processor.

    Returns:
    - distillation_losses: (bsz, resp_len), the per-token CE from :func:`compute_topk_loss`.
    - distillation_metrics: Dictionary of metrics.

    Deliberately *not* sharing the body with :func:`compute_forward_kl_topk`, because the two
    differ in exactly the two places that decide what the objective converges to:

    * no ``clamp_min(0.0)`` here. That clamp is what stops topk=1 supervising a position once
      the student is more confident on the argmax than the teacher is; removing it is the whole
      point of this mode, and sharing code would make it look like an oversight.
    * ``overlap_ratio`` is normalised by 1 rather than by ``topk``: with a single supervised
      teacher token it is the argmax-agreement rate, not a coverage fraction.
    """
    distillation_losses = no_padding_2_padding(model_output["distillation_losses"], data)
    student_mass = no_padding_2_padding(model_output["student_mass"], data)
    teacher_mass = no_padding_2_padding(model_output["teacher_mass"], data)
    overlap_count = model_output.get("overlap_count")
    overlap_token_advantage = model_output.get("overlap_token_advantage")
    if overlap_count is not None and overlap_token_advantage is not None:
        overlap_count = no_padding_2_padding(overlap_count, data)
        overlap_token_advantage = no_padding_2_padding(overlap_token_advantage, data)
    if data["response_mask"].is_nested:
        response_mask_bool = data["response_mask"].bool().to_padded_tensor(False)
    else:
        response_mask_bool = data["response_mask"].bool()
    assert distillation_losses.shape == student_mass.shape == teacher_mass.shape == response_mask_bool.shape

    overlap_metrics = {}
    if overlap_count is not None and overlap_token_advantage is not None:
        assert overlap_count.shape == overlap_token_advantage.shape == response_mask_bool.shape
        valid_overlap_count = overlap_count[response_mask_bool]
        # Exactly one teacher token is supervised, so the ratio is the fraction of response
        # positions where the student's argmax already matches the teacher's. Near 1.0 the
        # student has converged onto the teacher's greedy path and this loss has little left
        # to say; near 0.0 it is doing all the work.
        overlap_metrics["distillation/overlap_ratio"] = valid_overlap_count.float().mean().item()
        overlap_position_mask = response_mask_bool & (overlap_count > 0)
        if overlap_position_mask.any():
            overlap_metrics["distillation/overlap_token_advantage"] = (
                overlap_token_advantage[overlap_position_mask].mean().item()
            )
        else:
            overlap_metrics["distillation/overlap_token_advantage"] = 0.0

    # Both masses are the probability on the argmax token, so student_mass / teacher_mass reads
    # as "how much of the teacher's confidence has the student absorbed". student_mass exceeding
    # teacher_mass is expected here -- that over-sharpening is what this mode does not prevent.
    student_mass = student_mass[response_mask_bool]
    teacher_mass = teacher_mass[response_mask_bool]
    distillation_metrics = {
        "distillation/student_mass": student_mass.mean().item(),
        "distillation/student_mass_min": Metric(AggregationType.MIN, student_mass.min()),
        "distillation/student_mass_max": Metric(AggregationType.MAX, student_mass.max()),
        "distillation/teacher_mass": teacher_mass.mean().item(),
        "distillation/teacher_mass_min": Metric(AggregationType.MIN, teacher_mass.min()),
        "distillation/teacher_mass_max": Metric(AggregationType.MAX, teacher_mass.max()),
        **overlap_metrics,
    }

    # CE is non-negative by construction, so unlike the top-k divergence there is nothing to
    # clamp at zero. An upper bound is still useful -- a token the student assigns ~0
    # probability would otherwise dominate the step -- and loss_max_clamp in distillation_loss()
    # already applies one when configured.
    return distillation_losses, distillation_metrics


@register_distillation_loss(
    DistillationLossSettings(names=["kl", "k1", "abs", "mse", "k2", "low_var_kl", "k3"], use_estimator=True)
)  # type: ignore[arg-type]
def compute_distillation_loss_reverse_kl_estimator(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the distillation loss and related metrics using single-sample KL estimators.

    Uses the kl_penalty function from core_algos which supports various KL divergence
    estimators: "kl", "k1", "abs", "mse", "k2", "low_var_kl", "k3".

    When distillation_loss.reward_extrapolation_lambda != 1.0, the G-OPD/ExOPD term
    (https://arxiv.org/abs/2602.12125) is subtracted; see below.

    Returns:
    - distillation_losses: (bsz, resp_len)
    - distillation_metrics: Dictionary of metrics.
    """
    student_log_probs = no_padding_2_padding(model_output["log_probs"], data)
    # The vLLM teacher (distillation.mode="vllm") produces log probs per sample during rollout as
    # a jagged tensor over prompt+response; the in-process HF teacher produces one plain
    # (bsz, resp_len) tensor per step, already sliced to the response exactly like ref_log_prob.
    # Discriminating on the layout rather than on a mode flag keeps this shape-polymorphic, and
    # the assert below validates both.
    teacher_log_probs = data["teacher_logprobs"]
    if teacher_log_probs.is_nested:
        teacher_log_probs = no_padding_2_padding(teacher_log_probs, data).squeeze(-1)
    if data["response_mask"].is_nested:
        response_mask_bool = data["response_mask"].bool().to_padded_tensor(False)
    else:
        response_mask_bool = data["response_mask"].bool()
    assert teacher_log_probs.shape == student_log_probs.shape == response_mask_bool.shape

    loss_config: DistillationLossConfig = distillation_config.distillation_loss
    distillation_losses = kl_penalty(
        logprob=student_log_probs, ref_logprob=teacher_log_probs, kl_penalty=loss_config.loss_mode
    )
    # Since k1 can be negative, log the mean absolute loss.
    metrics = {
        "distillation/abs_loss": Metric(AggregationType.MEAN, distillation_losses[response_mask_bool].abs().mean()),
    }

    # Reward extrapolation (ExOPD). With use_policy_gradient, the caller turns this into
    # `advantages = -distillation_losses`, so for k1 the advantage is currently
    # (log pi_teacher - log pi_student) -- exactly standard OPD. Subtracting
    # (lambda - 1) * (teacher - reference) here makes the advantage
    #
    #   (log pi_teacher - log pi_student) + (lambda - 1)(log pi_teacher - log pi_reference)
    #
    # which is the G-OPD advantage. At lambda == 1.0 the term is identically zero, so that
    # path is left untouched rather than computed and multiplied by 0.
    #
    # Subtracting before the caller's loss_max_clamp is deliberate: the clamp then also bounds
    # the extrapolation, which is the term that destabilises training at large lambda.
    extrapolation_lambda = loss_config.reward_extrapolation_lambda
    if extrapolation_lambda != 1.0:
        if "ref_log_prob" not in data.keys():
            # Silently falling back to standard OPD here would be indistinguishable from a
            # working ExOPD run in every log line and metric, so refuse instead.
            raise ValueError(
                f"reward_extrapolation_lambda={extrapolation_lambda} needs reference-policy log "
                "probs, but 'ref_log_prob' is not in the batch. The reference worker is only "
                "built when need_reference_policy() says so -- check that it accounts for "
                "reward extrapolation."
            )
        # ref_log_prob is written by _compute_ref_log_prob already sliced to the response and
        # is not touched by left_right_2_no_padding, so it is plain (bsz, resp_len) here --
        # the same shape the student and teacher tensors were just converted into.
        ref_log_probs = data["ref_log_prob"]
        if ref_log_probs.is_nested:
            ref_log_probs = ref_log_probs.to_padded_tensor(0.0)
        assert ref_log_probs.shape == student_log_probs.shape, (
            f"ref_log_prob {tuple(ref_log_probs.shape)} != student log probs "
            f"{tuple(student_log_probs.shape)}"
        )
        extrapolation = (extrapolation_lambda - 1.0) * (teacher_log_probs - ref_log_probs)
        distillation_losses = distillation_losses - extrapolation
        metrics["distillation/extrapolation_term"] = Metric(
            AggregationType.MEAN, extrapolation[response_mask_bool].mean()
        )
        # How far the teacher has moved from the reference. This is the implicit reward the
        # extrapolation scales up, so if it is ~0 there is nothing for lambda to amplify.
        metrics["distillation/teacher_ref_gap"] = Metric(
            AggregationType.MEAN, (teacher_log_probs - ref_log_probs)[response_mask_bool].abs().mean()
        )

    return distillation_losses, metrics
