#!/usr/bin/env bash
# Single-teacher on-policy distillation | Qwen3-4B | FSDP | vLLM
#
# train_mopd.sh distils from the five per-domain experts, routed per sample by
# `data_source`. This variant distils from ONE already-merged model instead --
# by default the five-expert Task Arithmetic merge at lambda=1.0, which is the
# best uniform-coefficient merge in results.md (AVG8 66.00 vs 63.53 for the
# released lambda=0.6).
#
# The student is the same base model, the same Mix data, the same hyperparameters
# as train_mopd.sh. The ONLY difference is the teacher, so the two runs are
# directly comparable: does distilling a merged model into the base recover what
# the merge achieved, without having to keep five experts around at inference?
#
# ---------------------------------------------------------------------------------
# Why this is a separate script rather than CLI overrides on train_mopd.sh
#
# Hydra can add and change keys but cannot remove them. train_mopd.sh hardcodes
# five `+distillation.teacher_models.<name>.*` blocks; appending a sixth
# `teacher_model` entry would give six teachers, not one. The teacher block has
# to be replaced, so the script is copied.
# ---------------------------------------------------------------------------------
# Routing: there is none, and that is deliberate.
#
# `AsyncTeacherLLMServerManager._resolve_teacher_key()` short-circuits when only
# one teacher is configured ("route everything to the one teacher regardless of
# the sample's key"), so `distillation.teacher_key` is inert here. All five
# domains' samples go to the merged teacher. That is the intent -- the merged
# model is supposed to be competent in all five.
# ---------------------------------------------------------------------------------
# GPU layout: 24 student + 8 teacher = 32, nothing idle
#
#   training pool:  3 nodes x 8 = 24 GPUs  (trainer.nnodes x trainer.n_gpus_per_node)
#   teacher pool:   1 node  x 8 =  8 GPUs  (distillation.nnodes x distillation.n_gpus_per_node)
#
# The teacher pool is a SEPARATE Ray resource pool and is NOT part of
# NNODES x NGPUS_PER_NODE. train_mopd.sh leaves three GPUs idle on the teacher
# node (five teachers, tp1); here all eight are used as eight replicas of the
# same model.
#
# ⚠️ TEACHER_TP MUST BE 1, and must be passed explicitly.
#
# The single-teacher path derives the replica count from the pool size
# (verl/workers/config/distillation.py:288):
#
#     num_replicas = (n_gpus_per_node * nnodes) / (tp * dp * pp)
#
# so tp=1 gives 8 // 1 = 8 independent replicas, one whole model per GPU.
# But distillation.yaml defaults `tensor_model_parallel_size: 2`, which would
# silently give 8 // 2 = 4 replicas each sharded across two GPUs -- a different
# topology, no error. Hence the explicit override below.
#
# Requests are spread across the eight replicas by GlobalRequestLoadBalancer
# (verl/workers/rollout/llm_server.py): a single Ray actor shared by all
# AgentLoopWorkers, picking the least-in-flight replica. Teacher calls use a
# fresh uuid per request so the sticky-session path never fires, which is
# correct -- a teacher call is one prefill, not a conversation. Balancing by
# in-flight count rather than round-robin matters because teacher cost scales
# with sequence length, and Math responses run to 16k while IF responses are
# 1-2k.
#
# Multi-node: start the Ray cluster first, then run this on the head node only.

set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

source "${SCRIPT_DIR}/../paths.sh"
require_paths MODEL_PATH TRAIN_DATA_ROOT EVAL_DATA_ROOT OUTPUT_ROOT TEACHER_MODEL

TRAIN_FILE=${TRAIN_FILE:-${MIX_TRAIN}}

# ---- cluster (training pool only; the teacher pool is counted separately) ----
NNODES=${NNODES:-3}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

INFER_BACKEND=${INFER_BACKEND:-vllm}
if [ "${INFER_BACKEND}" = "vllm" ]; then
    export VLLM_USE_V1=1
fi

# ---- teacher pool ----
# TEACHER_TP=1 is load-bearing, see the header. TEACHER_WORLD_SIZE is the pool
# size; num_replicas is derived from it by verl, not set here.
TEACHER_TP=${TEACHER_TP:-1}
TEACHER_NNODES=${TEACHER_NNODES:-1}
TEACHER_WORLD_SIZE=${TEACHER_WORLD_SIZE:-8}
# Each replica owns a whole GPU (tp1), so a 4B model has all the room it needs:
# 0.8 x 95 GiB = 76 GiB. This is not the crowded-node case that would need 0.3.
TEACHER_GPU_MEM_UTIL=${TEACHER_GPU_MEM_UTIL:-0.8}

# ---- optimisation ----
# 264 = 24 x 11, divisible by the 3 x 8 training pool's data-parallel size.
# Unchanged from train_mopd.sh: the student pool is the same 24 GPUs.
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-264}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-264}
# 5120 covers 100% of the training data and 100% of the eval suite; the measured
# per-domain distribution, and why a lower value empties the Agent domain, are
# documented in train_mix_rl.sh. `truncation='error'` aborts on an overlong prompt
# rather than cutting it, so filter_overlong_prompts below drops the handful of
# outliers at load time instead. Note this also sets the teacher's inference
# prompt_length further down, so both sides stay aligned.
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-5120}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-16384}
MAX_NUM_TOKENS=${MAX_NUM_TOKENS:-$(( MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 1 ))}
ACTOR_LR=${ACTOR_LR:-1e-6}
NORM_ADV_BY_STD_IN_GRPO=${NORM_ADV_BY_STD_IN_GRPO:-FALSE}
# Give each response equal aggregation weight regardless of its length; see
# train_mix_rl.sh. The distillation term uses the same aggregation. Set
# LOSS_AGG_MODE=token-mean to recover the previous token-weighted aggregation.
LOSS_AGG_MODE=${LOSS_AGG_MODE:-seq-mean-token-mean}

# ---- rollout ----
ROLLOUT_TP=${ROLLOUT_TP:-1}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.9}
ROLLOUT_N=${ROLLOUT_N:-4}

MAX_ASSISTANT_TURNS=${MAX_ASSISTANT_TURNS:-16}
MAX_USER_TURNS=${MAX_USER_TURNS:-16}
MAX_PARALLEL_CALLS=${MAX_PARALLEL_CALLS:-1}
MAX_TOOL_RESPONSE_LENGTH=${MAX_TOOL_RESPONSE_LENGTH:-2048}
MULTITURN_FORMAT=${MULTITURN_FORMAT:-hermes}

# ---- distillation loss ----
DISTILLATION_LOSS_MODE=${DISTILLATION_LOSS_MODE:-k1}
USE_POLICY_GRADIENT=${USE_POLICY_GRADIENT:-True}
USE_TASK_REWARDS=${USE_TASK_REWARDS:-False}

# ---- reward extrapolation (ExOPD, https://arxiv.org/abs/2602.12125) ----
# Standard OPD is a dense KL-constrained RL whose implicit reward and KL regularization are
# forced to weigh the same, which is why an OPD student tracks its teacher but rarely passes
# it. This is that weight, lambda:
#
#   advantage = (log pi_teacher - log pi_student)
#             + (lambda - 1) * (log pi_teacher - log pi_reference)
#
# Default 1.0 = standard OPD, bit-for-bit what this script did before the knob existed.
# 1.25 is the paper's value across every setting it reports; 1.5 it reports as unstable.
#
# pi_reference is verl's own `ref` worker, which loads actor_rollout_ref.model.path -- the
# frozen initial student. That is exactly the reference this formula wants here, because the
# student starts from the same base the teacher was built from. Nothing extra to configure.
#
# ⚠️ The reference implementation at code/G-OPD names things the other way round
# (there `ref_log_prob` is the TEACHER and `base_log_prob` is the reference), because
# verl 0.6.1 had no teacher pool and had to repurpose the ref worker. Do not port its
# names across; this fork keeps the paper's meaning.
#
# ⚠️ lambda != 1.0 builds the reference model inside the training pool: one extra frozen 4B
# model per training GPU plus one extra forward pass per step. No extra GPUs are needed
# (verl fuses it into ActorRolloutRefWorker on the same pool), but expect a slower step.
# It also makes the REF array below actually do something -- at lambda=1.0 that worker is
# never built, so `param_offload` there has always been a no-op.
REWARD_EXTRAPOLATION_LAMBDA=${REWARD_EXTRAPOLATION_LAMBDA:-1.0}

# ---- bookkeeping ----
PROJECT_NAME=${PROJECT_NAME:-llm_fusion_repro}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-mopd_merged_teacher_qwen3_4b}
SAVE_FREQ=${SAVE_FREQ:-25}
TEST_FREQ=${TEST_FREQ:-25}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-10}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-400}

CKPT_DIR=${CKPT_DIR:-${OUTPUT_ROOT}/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}}
ROLLOUT_DATA_DIR=${ROLLOUT_DATA_DIR:-${OUTPUT_ROOT}/rollout_data/${EXPERIMENT_NAME}}
VALIDATION_DATA_DIR=${VALIDATION_DATA_DIR:-${OUTPUT_ROOT}/validation_data/${EXPERIMENT_NAME}}

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.norm_adv_by_std_in_grpo=${NORM_ADV_BY_STD_IN_GRPO}
    algorithm.use_kl_in_reward=False
    data.train_files="${TRAIN_FILE}"
    data.val_files="${EVAL_SUITE}"
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.shuffle=True
    data.filter_overlong_prompts=True
    data.truncation='error'
    data.return_raw_chat=True
    data.reward_fn_key=data_source
)

MODEL=(
    actor_rollout_ref.model.path=${MODEL_PATH}
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.loss_agg_mode=${LOSS_AGG_MODE}
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=${INFER_BACKEND}
    actor_rollout_ref.rollout.mode=async
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.enforce_eager=False
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    # temperature=1 is required, not just conventional: the teacher's inference
    # temperature interpolates from this one, and _get_teacher_sampling_params
    # raises NotImplementedError for anything but 1.0 because vLLM does not
    # support temperature with prompt_logprobs.
    actor_rollout_ref.rollout.temperature=1
    actor_rollout_ref.rollout.max_num_batched_tokens=32768
    actor_rollout_ref.rollout.max_model_len=${MAX_NUM_TOKENS}
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.n=4
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95
    # Mix rows all carry an explicit agent_name; this default only applies if
    # you swap in data that does not.
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent
    actor_rollout_ref.rollout.multi_turn.enable=True
    actor_rollout_ref.rollout.multi_turn.tool_config_path=${TOOL_CONFIG}
    actor_rollout_ref.rollout.multi_turn.format=${MULTITURN_FORMAT}
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${MAX_ASSISTANT_TURNS}
    actor_rollout_ref.rollout.multi_turn.max_user_turns=${MAX_USER_TURNS}
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=${MAX_PARALLEL_CALLS}
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=${MAX_TOOL_RESPONSE_LENGTH}
)

REF=(
    actor_rollout_ref.ref.fsdp_config.param_offload=True
)

TRAINER=(
    trainer.critic_warmup=0
    trainer.val_before_train=True
    trainer.logger='["console", "wandb"]'
    trainer.project_name=${PROJECT_NAME}
    trainer.experiment_name=${EXPERIMENT_NAME}
    trainer.default_local_dir=${CKPT_DIR}
    trainer.rollout_data_dir=${ROLLOUT_DATA_DIR}
    trainer.validation_data_dir=${VALIDATION_DATA_DIR}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.save_freq=${SAVE_FREQ}
    trainer.test_freq=${TEST_FREQ}
    trainer.total_epochs=${TOTAL_EPOCHS}
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS}
)

REWARD=(
    reward.sandbox_fusion.url=${SANDBOX_FUSION_URL}
    reward.sandbox_fusion.max_concurrent=${SANDBOX_MAX_CONCURRENT}
    reward.sandbox_fusion.memory_limit_mb=${SANDBOX_MEMORY_LIMIT_MB}
)

# One teacher, so this uses the `teacher_model` entry that already exists in
# distillation.yaml -- no `+` prefix. Adding entries under new names instead
# would pop `teacher_model` and put us back in the multi-teacher path.
#
# `key` is deliberately not set: the single-teacher path overwrites it with
# "default" (distillation.py:295) and routing is short-circuited anyway.
#
# `num_replicas` is deliberately not set either: it is derived as
# pool_size / (tp*dp*pp) = 8 / 1 = 8.
#
# DISTILLATION_MODE selects where the teacher runs:
#
#   vllm        (default) teacher served from its own pool, log probs per sample during
#               rollout. The inference.* keys below configure that engine.
#   in_process  teacher is a frozen HF model inside ActorRolloutRefWorker, one batched
#               forward per step. It claims no pool, so nnodes/n_gpus_per_node must be 0
#               (the config rejects a nonzero pool outright rather than stranding those
#               GPUs) and the inference.* keys are inert -- no engine is built from them.
#               Give the freed GPUs to the training pool via NNODES instead.
DISTILLATION_MODE=${DISTILLATION_MODE:-vllm}
if [ "${DISTILLATION_MODE}" = "in_process" ]; then
    TEACHER_WORLD_SIZE=0
    TEACHER_NNODES=0
fi

# Representation distillation (OPRD). REP_COEF=0 (default) leaves it off entirely; any
# positive value requires DISTILLATION_MODE=in_process, since hidden states are only
# available from the in-process teacher. REP_LAYERS is a hydra list literal.
REP_COEF=${REP_COEF:-0.0}
REP_LAYERS=${REP_LAYERS:-[9,18,27,36]}
REP_LAST_K=${REP_LAST_K:-2000}
REP_NORMALIZE=${REP_NORMALIZE:-True}

DISTILL=(
    distillation.enabled=True
    distillation.mode=${DISTILLATION_MODE}
    distillation.n_gpus_per_node=${TEACHER_WORLD_SIZE}
    distillation.nnodes=${TEACHER_NNODES}
    distillation.teacher_models.teacher_model.model_path=${TEACHER_MODEL}
    distillation.teacher_models.teacher_model.inference.name=${INFER_BACKEND}
    distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=${TEACHER_TP}
    distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=${TEACHER_GPU_MEM_UTIL}
    distillation.teacher_models.teacher_model.inference.prompt_length=${MAX_PROMPT_LENGTH}
    distillation.teacher_models.teacher_model.inference.response_length=${MAX_RESPONSE_LENGTH}
    distillation.teacher_models.teacher_model.inference.max_model_len=${MAX_NUM_TOKENS}
    distillation.distillation_loss.loss_mode=${DISTILLATION_LOSS_MODE}
    distillation.distillation_loss.use_task_rewards=${USE_TASK_REWARDS}
    distillation.distillation_loss.use_policy_gradient=${USE_POLICY_GRADIENT}
    distillation.distillation_loss.loss_max_clamp=10.0
    distillation.distillation_loss.log_prob_min_clamp=-10.0
    distillation.distillation_loss.reward_extrapolation_lambda=${REWARD_EXTRAPOLATION_LAMBDA}
    distillation.distillation_loss.rep_coef=${REP_COEF}
    distillation.distillation_loss.rep_layers=${REP_LAYERS}
    distillation.distillation_loss.rep_last_k=${REP_LAST_K}
    distillation.distillation_loss.rep_normalize=${REP_NORMALIZE}
)

python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "${REWARD[@]}" \
    "${DISTILL[@]}" \
    "$@"
