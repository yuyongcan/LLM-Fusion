# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

from verl.base_config import BaseConfig
from verl.utils.config import omega_conf_to_dataclass

from .rollout import RolloutConfig

__all__ = ["DistillationLossConfig", "DistillationTeacherModelConfig", "DistillationConfig"]

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@dataclass
class DistillationLossConfig(BaseConfig):
    """Configuration for distillation loss settings.

    loss_mode (str):
        Distillation loss function to use. Three families:
        - ``forward_kl_topk``: forward KL over the teacher's top-`topk` tokens, computed in the
          logits processor. Mode-covering, and deterministic given the rollout (no dependence
          on the student's sampled token, so no sampling variance).
        - ``argmax_ce``: cross-entropy against the teacher's single argmax token (top-1 only,
          so `topk` must be 1). This is `forward_kl_topk` at topk=1 with two things removed:
          the `q = pi_teacher(argmax)` weighting, and the clamp that stops supervising a
          position once the student is more confident than the teacher. Its fixed point is the
          teacher's greedy decoding rather than the teacher's distribution, so it is not
          consistent and cannot converge to pi_teacher -- which is fine when evaluation is
          greedy and a liability when it is sampled.
        - ``k1``/``k3``/...: single-sample KL estimators at the student's sampled token. Reverse
          KL (mode-seeking), and high variance because they only read one sampled token.
    topk (int, optional):
        Number of top tokens to consider for top-k distillation losses. Must be 1 for
        loss_mode="argmax_ce".
    use_task_rewards (bool):
        Whether to include task rewards alongside distillation loss.
    distillation_loss_coef (float):
        Coefficient for distillation loss when combined with task rewards.
    loss_max_clamp (float, optional):
        Maximum value to clamp distillation loss. If None, no clamping is applied.
    log_prob_min_clamp (float, optional):
        Minimum value to clamp log probabilities for stability, e.g., log q - log p where p or q are
        very close to zero. If None, no clamping is applied.
    use_policy_gradient (bool):
        Whether to incorporate distillation loss as a reward, as done
        by https://thinkingmachines.ai/blog/on-policy-distillation/. Recommended to use loss_mode=k1.
        Otherwise, distillation loss is directly backpropagated as a supervised loss,
        as in https://arxiv.org/abs/2306.13649. Recommended to use loss_mode=k3 or forward_kl_topk.
    advantage_fusion (str):
        How the teacher signal enters the RL update. "none" (default) keeps it as a second
        policy loss added to the RL one -- each clipped independently. "additive" instead folds
        the per-token teacher advantage into the RL advantage and runs a single PPO-clipped
        surrogate, which is KDRL (https://arxiv.org/abs/2506.02208):
        A_t = A_hat + teacher_advantage_coef * (log pi_teacher - log pi_student).
        The distinction is not cosmetic: two independently clipped losses differ from one
        clipped loss on the summed advantage, and the additive form additionally lets the
        teacher term flip the sign of the update, which is exactly the failure mode that
        motivated the teacher-modulated alternatives.
    teacher_advantage_coef (float):
        beta of KDRL. Ignored unless advantage_fusion="additive". Reported values span two
        orders of magnitude across tasks, so this needs a per-task ablation.
    policy_loss_mode (str):
        Name of the policy loss to use when use_policy_gradient is true.
    clip_ratio (float):
        PPO clipping ratio for policy loss.
    clip_ratio_low (float):
        Lower bound for PPO clipping ratio.
    clip_ratio_high (float):
        Upper bound for PPO clipping ratio.
    reward_extrapolation_lambda (float):
        Reward scaling factor lambda of Generalized On-Policy Distillation
        (https://arxiv.org/abs/2602.12125). Standard OPD weights the implicit reward
        log(teacher/reference) and the KL regularization equally; lambda controls that ratio.
        1.0 reproduces standard OPD exactly. Values above 1.0 ("reward extrapolation", ExOPD)
        push the student past the teacher along the teacher-minus-reference direction, and the
        paper reports 1.25 as consistently better and 1.5 as unstable. Requires a reference
        policy, which is the frozen initial student; see need_reference_policy().
    rep_coef (float):
        Weight of the representation-distillation term (OPRD,
        https://arxiv.org/abs/2606.06021), which aligns student and teacher hidden states
        instead of only their output distributions. 0.0 (default) disables it and is
        bit-for-bit the behaviour from before this knob existed. Requires
        distillation.mode="in_process", since only the in-process teacher can return hidden
        states. Composes additively with the output-space loss: L = L_opd + rep_coef * L_rep.
        Note the scale interacts with rep_normalize -- a normalized MSE is much smaller than an
        unnormalized one, so a coefficient tuned for one is wrong for the other.
    rep_layers (tuple[int, ...]):
        Which layers to align, indexed into HF hidden_states, where 0 is the embedding output
        and k is the output of block k. Layer 0 is rejected: it carries no teacher computation,
        only the embedding table. The default samples four points up the stack rather than all
        of them -- all 36 layers of a 4B model over a 2000-token suffix is ~350 MB per sample,
        and the cost is linear in this list.
    rep_last_k (int):
        Align only the final k response tokens. OPRD's figure 8 measures where the two models
        actually disagree: the first k tokens of a response sit above 97% cosine similarity for
        every k, while the last k start at 91.65% and stay 4+ points lower. Supervising the
        prefix therefore spends memory on positions that carry no residual signal.
    rep_normalize (bool):
        L2-normalize both hidden states before the MSE, making the loss scale-invariant and
        purely directional. True (default) matches the reference implementation's shipped
        behaviour; False compares raw vectors, which keeps magnitude information (hidden-state
        norms vary several-fold across positions, and massive-activation dimensions dominate)
        but leaves the loss scale tied to the model's activation scale, so rep_coef must then
        be far smaller. The OPRD authors note their own paper numbers used the unnormalized
        form with a much larger coefficient.
    loss_settings (DistillationLossSettings, optional):
        Runtime-populated settings based on loss_mode. Not set by user.
    """

    loss_mode: str = "k3"
    topk: Optional[int] = 128
    use_task_rewards: bool = True
    distillation_loss_coef: float = 1.0
    loss_max_clamp: Optional[float] = 10.0
    log_prob_min_clamp: Optional[float] = -10.0

    use_policy_gradient: bool = True
    policy_loss_mode: str = "vanilla"
    clip_ratio: float = 0.2
    clip_ratio_low: float = 0.2
    clip_ratio_high: float = 0.2

    # KDRL (https://arxiv.org/abs/2506.02208): "none" keeps the teacher as a separate policy
    # loss, "additive" folds it into the RL advantage under a single PPO clip.
    advantage_fusion: str = "none"
    teacher_advantage_coef: float = 0.0

    # lambda of G-OPD/ExOPD. 1.0 is standard OPD; >1.0 is reward extrapolation.
    reward_extrapolation_lambda: float = 1.0

    # Representation distillation (OPRD, https://arxiv.org/abs/2606.06021): align hidden states
    # instead of, or in addition to, the output distribution. 0.0 leaves it off.
    rep_coef: float = 0.0
    rep_layers: tuple[int, ...] = (9, 18, 27, 36)
    rep_last_k: int = 2000
    rep_normalize: bool = True

    # Store global batch info for loss aggregation:
    # dp_size: data parallel size
    # batch_num_tokens: number of valid tokens in global batch
    # global_batch_size: global batch size
    global_batch_info: dict = field(default_factory=dict)

    # Store distillation loss settings for computing the specified loss_mode
    # Not set by user, populated at runtime
    loss_settings: Optional[dict] = None

    def __post_init__(self):
        self._mutable_fields.add("loss_settings")
        from verl.trainer.distillation.losses import DistillationLossSettings, get_distillation_loss_settings

        self.loss_settings: DistillationLossSettings = get_distillation_loss_settings(self.loss_mode)

        if self.loss_mode == "argmax_ce":
            # argmax_ce supervises exactly one teacher token, so a larger topk would only make
            # vLLM materialise (and the batch carry) log probs that are then discarded. Pinning
            # it also keeps distillation/overlap_ratio interpretable: it is normalised by 1, so
            # it is the argmax-agreement rate rather than a coverage fraction over k slots.
            if self.topk != 1:
                raise ValueError(
                    f"loss_mode='argmax_ce' supervises only the teacher's top-1 token, so topk "
                    f"must be 1, got {self.topk}. Use loss_mode='forward_kl_topk' to spread the "
                    "target over the teacher's top-k."
                )
            if self.use_policy_gradient:
                # Under policy gradient the loss becomes a per-token advantage A_t = -L_t and
                # the only gradient path is log pi_S(y_t) at the *sampled* token. For an
                # estimator loss that advantage is log pi_T(y_t) - log pi_S(y_t) -- how much the
                # teacher likes the token the student chose -- which is what on-policy
                # distillation means. For argmax_ce it degenerates to log pi_S(a): largest in
                # magnitude exactly where the student disagrees with the teacher, and ~0 where
                # it has already matched. The update then reinforces the student's own token
                # hardest at the positions where it is furthest from the teacher, so the sign of
                # the distillation is inverted and the reward curve still looks plausible.
                raise ValueError(
                    "loss_mode='argmax_ce' requires use_policy_gradient=False. Under policy "
                    "gradient the per-token advantage becomes log pi_student(argmax_teacher) "
                    "and the gradient flows only through the sampled token, so the update "
                    "reinforces the student's own token most strongly where it disagrees with "
                    "the teacher -- distillation with the sign flipped. Backpropagate the "
                    "cross-entropy directly instead."
                )

        if self.policy_loss_mode != "vanilla":
            raise NotImplementedError(
                f"Only vanilla policy loss is currently supported when use_policy_gradient is True, "
                f"but got {self.policy_loss_mode}."
            )

        if self.use_policy_gradient and self.loss_mode == "forward_kl_topk":
            print(
                "WARNING: forward_kl_topk is most effective as a supervised distillation loss "
                "(use_policy_gradient=False). With policy gradient, the update uses only the sampled"
                " token's logprob ∇logπ(a), so the top-k distributional signal (how non-sampled logits "
                "should move) is largely unused."
            )

        if not self.use_policy_gradient and self.loss_mode == "k1" and self.advantage_fusion != "additive":
            # The objection is about backpropagating a k1 *loss*. Under advantage_fusion="additive"
            # the teacher never becomes a loss: it enters as a detached per-token advantage and
            # reaches the model only through log pi_theta in the policy gradient, so the missing
            # teacher dependence is not a defect here. Blocking it would force KDRL users to set
            # use_policy_gradient=True, a knob this path never reads.
            raise ValueError(
                "Directly backpropagating k1 loss is incorrect since gradient of k1 loss"
                " wrt model weights does not depend on teacher log probabilities."
            )

        # Reward extrapolation adds (lambda - 1) * (teacher - reference) to the per-token
        # implicit reward, which is only defined where the loss IS a difference of log
        # probabilities. Fail here rather than eight minutes into a run.
        if self.reward_extrapolation_lambda != 1.0:
            if self.loss_settings.use_topk:
                raise ValueError(
                    f"reward_extrapolation_lambda={self.reward_extrapolation_lambda} requires a "
                    f"log-prob-difference loss, but loss_mode={self.loss_mode!r} is a top-k "
                    "distributional loss whose value is not log pi_student - log pi_teacher. "
                    "Use an estimator loss such as k1, or leave the lambda at 1.0."
                )
            if not self.use_policy_gradient:
                raise ValueError(
                    f"reward_extrapolation_lambda={self.reward_extrapolation_lambda} requires "
                    "use_policy_gradient=True. The extrapolation term is detached and only "
                    "reaches the model through the advantage, so with the supervised path it "
                    "would shift the loss without changing a single gradient."
                )
            if self.reward_extrapolation_lambda < 1.0:
                # Interpolation is a valid part of G-OPD, but it is not what MOPD wants, so say so.
                print(
                    f"WARNING: reward_extrapolation_lambda={self.reward_extrapolation_lambda} < 1.0 is "
                    "reward INTERPOLATION: the student is pulled toward a point between the reference "
                    "and the teacher, i.e. deliberately weaker than the teacher."
                )

        self._validate_advantage_fusion()
        self._validate_representation_distillation()

    def _validate_advantage_fusion(self) -> None:
        """Validate how the teacher signal enters the RL update (KDRL)."""
        if self.advantage_fusion not in ("none", "additive"):
            raise ValueError(
                f"advantage_fusion must be 'none' or 'additive', got {self.advantage_fusion!r}."
            )

        if self.advantage_fusion == "none":
            if self.teacher_advantage_coef != 0.0:
                # A beta with the fusion switched off is silently discarded, and every log
                # line would still claim the teacher was weighted at that value.
                raise ValueError(
                    f"teacher_advantage_coef={self.teacher_advantage_coef} has no effect unless "
                    "advantage_fusion='additive'. Set the fusion mode, or leave the coefficient at 0.0."
                )
            return

        # ---- advantage_fusion == "additive" (KDRL) ----
        if self.teacher_advantage_coef <= 0.0:
            raise ValueError(
                f"teacher_advantage_coef={self.teacher_advantage_coef} must be > 0 when "
                "advantage_fusion='additive'. A non-positive weight would push the student away "
                "from the teacher on every token."
            )
        if not self.use_task_rewards:
            # Without the RL advantage there is nothing to fuse into: the additive form would
            # collapse to plain OPD scaled by beta, which is a different algorithm wearing the
            # same name and would not show up as wrong in any metric.
            raise ValueError(
                "advantage_fusion='additive' (KDRL) requires use_task_rewards=True: the teacher "
                "advantage is added to the RL advantage, so with task rewards off there is no RL "
                "term to add it to."
            )
        if self.loss_settings.use_topk:
            raise ValueError(
                f"advantage_fusion='additive' needs a per-token log-prob difference "
                f"d_t = log pi_teacher - log pi_student, but loss_mode={self.loss_mode!r} is a "
                "top-k distributional loss. Use an estimator loss such as k1."
            )
        if self.reward_extrapolation_lambda != 1.0:
            # Mathematically these compose -- lambda reshapes d_t, beta weighs it -- but the
            # combination is untested and the two knobs interact in the same term, so a bad
            # pairing would be hard to attribute. Refuse rather than guess.
            raise ValueError(
                f"advantage_fusion='additive' and reward_extrapolation_lambda="
                f"{self.reward_extrapolation_lambda} are not validated together. Both act on the "
                "per-token teacher advantage; validate one at a time."
            )

    def _validate_representation_distillation(self) -> None:
        """Validate the representation-distillation term (OPRD)."""
        if self.rep_coef == 0.0:
            # Off: the term is never computed, so none of the other knobs can matter.
            return
        if self.rep_coef < 0.0:
            raise ValueError(
                f"rep_coef={self.rep_coef} must be >= 0. A negative coefficient would reward "
                "the student for moving its representations away from the teacher's."
            )
        if not self.rep_layers:
            raise ValueError("rep_coef > 0 requires a non-empty rep_layers.")
        if len(set(self.rep_layers)) != len(self.rep_layers):
            raise ValueError(f"rep_layers has duplicates: {list(self.rep_layers)}.")
        for layer in self.rep_layers:
            if layer < 1:
                # Index 0 of HF hidden_states is the embedding output. Aligning it distils the
                # embedding table rather than any computation the teacher performed, and for a
                # student that starts from the teacher's own base it is very nearly a no-op.
                # The reference implementation likewise starts its "all layers" range at 1.
                raise ValueError(
                    f"rep_layers contains {layer}, but layers are indexed into HF hidden_states "
                    "where 0 is the embedding output; supervising it distils the embedding "
                    "table, not teacher computation. Use 1..num_hidden_layers."
                )
        if self.rep_last_k < 1:
            raise ValueError(f"rep_last_k={self.rep_last_k} must be >= 1.")


@dataclass
class DistillationTeacherModelConfig(BaseConfig):
    """Configuration for on-policy distillation teacher.

    key (str, optional):
        Identifier to route examples to the teacher model in multi-teacher setting.
    model_path (str, optional):
        Model path for the teacher model. Can be a local path or a Hugging Face model
    inference (RolloutConfig):
        Rollout configuration for the teacher model inference during distillation.
    num_replicas (int):
        Number of inference replicas of this teacher to launch. Each replica occupies
        `per_replica_world_size` GPUs (= inference.data_parallel_size *
        inference.tensor_model_parallel_size * inference.pipeline_model_parallel_size),
        so the teacher's total GPU footprint is
        `num_replicas * per_replica_world_size`.
    """

    _mutable_fields = BaseConfig._mutable_fields | {"num_replicas", "key"}

    key: Optional[str] = None
    model_path: Optional[str] = None
    inference: RolloutConfig = field(default_factory=RolloutConfig)
    num_replicas: Optional[int] = 0

    @property
    def per_replica_world_size(self) -> int:
        return (
            self.inference.tensor_model_parallel_size
            * self.inference.data_parallel_size
            * self.inference.pipeline_model_parallel_size
        )

    @property
    def world_size(self) -> int:
        return self.num_replicas * self.per_replica_world_size

    def check_configured(self):
        if self.model_path is None:
            raise ValueError("model_path must be specified for distillation teacher model config.")
        if self.key is None:
            raise ValueError("key must be specified for distillation teacher model config.")
        if self.num_replicas is None:
            raise ValueError("num_replicas must be specified for distillation teacher model config.")

    def validate_and_prepare_for_distillation(self, use_topk: bool, topk: Optional[int]) -> None:
        # Prompt + Response from student are fed into teacher as context
        max_model_len = self.inference.max_model_len
        student_prompt_length = self.inference.prompt_length
        student_response_length = self.inference.response_length
        required_context_len = student_prompt_length + student_response_length + 1
        if max_model_len is not None and required_context_len > max_model_len:
            raise ValueError(
                "Distillation teacher inference requires room for the student prompt, the full student "
                f"response, and one generated token, but got {student_prompt_length=}, "
                f"{student_response_length=}, {required_context_len=}, {max_model_len=}."
            )
        self.inference.prompt_length = self.inference.prompt_length + self.inference.response_length
        self.inference.response_length = 1
        self._validate_topk_logprobs(use_topk=use_topk, topk=topk)

    def _validate_topk_logprobs(self, use_topk: bool, topk: Optional[int]) -> None:
        if not use_topk:
            return
        if topk is None:
            raise ValueError("topk must be specified when use_topk is True.")

        engine_name = self.inference.name
        engine_kwargs = self.inference.engine_kwargs
        match engine_name:
            case "vllm":
                vllm_engine_kwargs = dict(engine_kwargs.get("vllm", {}))
                max_logprobs = vllm_engine_kwargs.get("max_logprobs")
                if max_logprobs is None:
                    vllm_engine_kwargs["max_logprobs"] = topk
                    max_logprobs = topk
                if max_logprobs < topk:
                    raise ValueError(
                        f"VLLM max_logprobs ({max_logprobs}) must be >= distillation_loss topk "
                        f"({topk}) to enable distillation loss computation."
                    )
                engine_kwargs["vllm"] = vllm_engine_kwargs
            case "sglang":
                # SGLang's top_logprobs_num is a per-request parameter, so there is no
                # engine-boot cap to align (unlike vLLM's max_logprobs). The async
                # server translates sampling_params["prompt_logprobs"] into
                # return_logprob + logprob_start_len=0 + top_logprobs_num at call time.
                pass
            case _:
                raise NotImplementedError(
                    f"DistillationTeacherModelConfig does not support inference engine {engine_name}"
                )


@dataclass
class DistillationConfig(BaseConfig):
    """Configuration for on-policy distillation.

    enabled (bool):
        Whether on-policy distillation is enabled.
    mode (str):
        Where the teacher runs. "vllm" (default) serves the teacher from a dedicated vLLM
        resource pool and computes its log probs per sample during rollout. "in_process" loads
        the teacher as a frozen HF model fused into ActorRolloutRefWorker -- like the reference
        policy -- and computes log probs as one batched forward between ref_log_prob and
        update_actor, while vLLM is asleep. "in_process" claims no teacher pool (give those GPUs
        to trainer.nnodes), supports exactly one teacher, and requires an estimator loss_mode.
    n_gpus_per_node (int):
        Number of GPUs per node in the teacher resource pool. Must be 0 when mode="in_process".
    nnodes (int):
        Number of nodes in the teacher resource pool. Must be 0 when mode="in_process".
    teacher_models (dict[str, TeacherModelConfig]):
        Configurations for teacher models used for multi-teacher distillation.
    teacher_key (str):
        Key to route examples to the appropriate teacher model in multi-teacher setups. Should correspond to a field in
        the data proto, e.g., data_source.
    distillation_loss (DistillationLossConfig):
    Configuration for distillation loss settings.

    NOTE: The `teacher_model` entry is in the `teacher_models` dict by default.
    Since it is popped when other teacher entries are added, using `teacher_model` as
    one of several keys silently drops it. For example, the following CLI overrides result
    in ONLY `teacher_model2` being used:

    ```bash
    distillation.teacher_models.teacher_model.key=openai/gsm8k
    distillation.teacher_models.teacher_model.model_path=Qwen/Qwen3-4B
    +distillation.teacher_models.teacher_model2.key=hiyouga/geometry3k
    +distillation.teacher_models.teacher_model2.model_path=Qwen/Qwen3-VL-4B-Instruct
    ```
    Instead, give the first teacher a different name:

    ```bash
    +distillation.teacher_models.teacher_model1.key=openai/gsm8k
    +distillation.teacher_models.teacher_model1.model_path=Qwen/Qwen3-4B
    +distillation.teacher_models.teacher_model2.key=hiyouga/geometry3k
    +distillation.teacher_models.teacher_model2.model_path=Qwen/Qwen3-VL-4B-Instruct
    ```
    """

    _mutable_fields = BaseConfig._mutable_fields | {"teacher_models", "n_gpus_per_node", "nnodes"}

    enabled: bool = False
    mode: str = "vllm"
    n_gpus_per_node: int = 0
    nnodes: int = 0
    teacher_models: dict[str, DistillationTeacherModelConfig] = field(default_factory=dict)
    teacher_key: str = "data_source"
    distillation_loss: DistillationLossConfig = field(default_factory=DistillationLossConfig)

    def __post_init__(self):
        if not self.enabled:
            return

        if self.mode not in ("vllm", "in_process"):
            raise ValueError(f"distillation.mode must be 'vllm' or 'in_process', got {self.mode!r}.")

        if self.distillation_loss.rep_coef > 0.0 and self.mode != "in_process":
            # Only the in-process HF teacher can return hidden states. The vLLM teacher exposes
            # prompt_logprobs and nothing else, so this combination would train plain OPD while
            # every log line claimed representation distillation was on.
            raise ValueError(
                f"distillation_loss.rep_coef={self.distillation_loss.rep_coef} > 0 requires "
                f"distillation.mode='in_process', got {self.mode!r}: hidden states are only "
                "available from the in-process HF teacher."
            )

        if self.mode == "in_process":
            self._validate_in_process()
            return

        self.teacher_models = self._resolve_teacher_models()
        teacher_world_size_sum = 0
        for teacher_model in self.teacher_models.values():
            teacher_model.validate_and_prepare_for_distillation(
                use_topk=self.distillation_loss.loss_settings.use_topk,
                topk=self.distillation_loss.topk,
            )
            teacher_world_size_sum += teacher_model.world_size
        total_pool_size = self.n_gpus_per_node * self.nnodes
        if teacher_world_size_sum != total_pool_size:
            raise ValueError(
                f"Sum of teacher (num_replicas * per_replica_world_size) ({teacher_world_size_sum}) must match "
                f"the distillation resource pool size "
                f"({self.n_gpus_per_node=} * {self.nnodes=} = {total_pool_size})."
            )

    def _validate_in_process(self) -> None:
        """Validate the in-process (HF) teacher mode.

        This mode runs the teacher as a frozen HF model fused into ActorRolloutRefWorker, the
        same way the reference policy is fused, so it claims no resource pool of its own. Every
        check here is a config-time failure for something that would otherwise either strand
        GPUs or surface as a wrong number many minutes into a run.
        """
        if len(self.teacher_models) != 1:
            # Per-sample routing needs one live engine per teacher; a single in-process teacher
            # cannot serve several routing keys, and N resident teachers is not the design.
            raise ValueError(
                f"distillation.mode='in_process' supports exactly one teacher, got "
                f"{len(self.teacher_models)}: {sorted(self.teacher_models)}. Use mode='vllm' for "
                "multi-teacher routing."
            )
        if self.n_gpus_per_node * self.nnodes != 0:
            # The teacher pool would be created and then never used, silently stranding GPUs.
            raise ValueError(
                f"distillation.mode='in_process' must not request a teacher resource pool, but got "
                f"{self.n_gpus_per_node=} * {self.nnodes=}. Set both to 0 and give those GPUs to "
                "trainer.nnodes instead."
            )
        if self.distillation_loss.loss_settings.use_topk:
            # In-process teacher returns the sampled-token log prob, not a top-k slice.
            raise ValueError(
                f"distillation.mode='in_process' does not support top-k distillation "
                f"(loss_mode={self.distillation_loss.loss_mode!r}). Use an estimator loss_mode "
                "(k1, k3, ...) or mode='vllm'."
            )

        teacher_model = next(iter(self.teacher_models.values()))
        if not teacher_model.model_path:
            raise ValueError("distillation.mode='in_process' requires teacher_models.*.model_path to be set.")

    def _resolve_teacher_models(self) -> dict[str, DistillationTeacherModelConfig]:
        assert "teacher_model" in self.teacher_models
        if len(self.teacher_models) == 1:
            # Single teacher occupies the entire teacher resource pool.
            teacher_model = self.teacher_models["teacher_model"]
            inference = teacher_model.inference
            per_replica = (
                inference.tensor_model_parallel_size
                * inference.data_parallel_size
                * inference.pipeline_model_parallel_size
            )
            pool_size = self.n_gpus_per_node * self.nnodes
            if pool_size % per_replica != 0:
                raise ValueError(
                    f"Single teacher's per_replica_world_size ({per_replica}) must divide the distillation "
                    f"resource pool size ({self.n_gpus_per_node=} * {self.nnodes=} = {pool_size})."
                )
            teacher_model.num_replicas = pool_size // per_replica
            teacher_model.key = "default"
        else:
            # Multiple teachers: remove default single teacher config
            self.teacher_models.pop("teacher_model")

        # Teacher models dict is keyed by teacher_key instead of YAML entry name
        teacher_models = {}
        for teacher_config in self.teacher_models.values():
            teacher_config = omega_conf_to_dataclass(teacher_config, dataclass_type=DistillationTeacherModelConfig)
            teacher_config.check_configured()
            if teacher_config.key in teacher_models:
                raise ValueError(f"Duplicate teacher key {teacher_config.key} found in teacher models.")
            teacher_models[teacher_config.key] = teacher_config
        return teacher_models
