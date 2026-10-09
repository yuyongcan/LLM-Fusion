#!/usr/bin/env bash
# Evaluate one checkpoint on the eight benchmarks reported in the paper:
#
#   Math      AIME 2025, AIME 2026
#   Science   GPQA
#   Code      LiveCodeBench v5, LiveCodeBench v6
#   IF        IFEval, IFBench
#   Agent     BFCL v3 (multi-turn tool calling)
#
# Runs verl's trainer in val_only mode, which pays Ray and vLLM start-up once and
# then evaluates every set in a single job. Per-set accuracy is logged under
# `val-core/<data_source>/...`; with VAL_N > 1 both mean@n and best@n are
# reported, and the raw generations land in ${VALIDATION_DATA_DIR} as one JSONL
# per benchmark.
#
# MODEL_PATH points at whatever you want scored: the base model, a trained
# checkpoint exported to HuggingFace format, or a merged model.
#
# Usage:
#   bash examples/llm_fusion/eval/eval.sh
#   MODEL_PATH=/path/to/model VAL_N=16 bash examples/llm_fusion/eval/eval.sh

set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

source "${SCRIPT_DIR}/../paths.sh"
require_paths MODEL_PATH TRAIN_DATA_ROOT EVAL_DATA_ROOT OUTPUT_ROOT

# A train file is still required by the trainer config in val_only mode and is
# never read. It must hold at least TRAIN_BATCH_SIZE rows: the loader drops the
# last partial batch and then asserts the dataloader is non-empty, so a 30-row
# benchmark would fail. The Agent training set is the smallest one large enough.
TRAIN_FILE=${TRAIN_FILE:-${AGENT_TRAIN}}

# ---- cluster ----
NNODES=${NNODES:-4}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

INFER_BACKEND=${INFER_BACKEND:-vllm}
if [ "${INFER_BACKEND}" = "vllm" ]; then
    export VLLM_USE_V1=1
fi

# ---- sampling ----
VAL_N=${VAL_N:-16}
VAL_TEMPERATURE=${VAL_TEMPERATURE:-0.6}
VAL_TOP_P=${VAL_TOP_P:-0.95}
# Prompts per validation batch; each is sampled VAL_N times, so the engine sees
# VAL_BATCH_SIZE * VAL_N concurrent sequences. Left unset, verl puts the whole
# suite in one batch: ~1.6k prompts times VAL_N.
#
# 64 is sized against the KV cache, not against verl's max_num_seqs (1024 per
# engine, so 8 single-GPU engines admit far more than we ever submit). On a 96G
# card ROLLOUT_GPU_MEM_UTIL=0.7 leaves ~55 GiB of KV; Qwen3-4B needs 36 layers *
# 2 * 8 kv heads * 128 head_dim * 2 bytes = 144 KiB per token, so ~400k tokens
# per engine. 64 * 16 = 1024 sequences spread over 8 engines is 128 running per
# engine -- enough to keep decode saturated, and close to what the cache holds
# for typical response lengths. The previous default of 128 doubled that, and
# vLLM then preempted requests that had num_computed_tokens=0 and recomputed
# them: harmless (mixrl step400 logged 374 such warnings, up to 12 per request,
# and finished all eight sets) but it wastes throughput. Lower it further if
# you see num_preemptions climbing on a smaller-GPU or higher-VAL_N run.
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-64}

# 5120 covers 100% of the eval suite. The longest prompts are BFCL v3 at 4873
# tokens -- its rows carry 18-39 tool schemas, which dominate the text -- and one
# 2819-token GPQA item; the seven tool-free sets all have p99 under 1400. See the
# measured distribution in ../train/train_mix_rl.sh. `truncation='error'` aborts
# on an overlong prompt rather than cutting it, so filter_overlong_prompts below
# drops outliers at load.
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-5120}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-24576}

ROLLOUT_TP=${ROLLOUT_TP:-1}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.8}
# Inference dtype. bfloat16 matches training and is what every reported number
# uses. float32 is for precision controls only: a task vector theta - theta_base
# is small enough that a good fraction of its entries fall below one bfloat16
# rounding unit. Note a matched control found no score difference either way, so
# fp32 here is a diagnostic, not an accuracy fix. Weights double to ~16GB, so
# drop ROLLOUT_GPU_MEM_UTIL alongside it.
ROLLOUT_DTYPE=${ROLLOUT_DTYPE:-bfloat16}

# verl builds the actor worker even in val_only mode, so these still have to be
# set: FSDPActorConfig refuses to instantiate without either use_dynamic_bsz or
# an explicit micro batch size.
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-128}
ROLLOUT_N=${ROLLOUT_N:-16}
ACTOR_LR=${ACTOR_LR:-1e-6}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.001}
NORM_ADV_BY_STD_IN_GRPO=${NORM_ADV_BY_STD_IN_GRPO:-FALSE}

MAX_ASSISTANT_TURNS=${MAX_ASSISTANT_TURNS:-16}
MAX_USER_TURNS=${MAX_USER_TURNS:-16}
MAX_PARALLEL_CALLS=${MAX_PARALLEL_CALLS:-1}
MAX_TOOL_RESPONSE_LENGTH=${MAX_TOOL_RESPONSE_LENGTH:-2048}
MULTITURN_FORMAT=${MULTITURN_FORMAT:-hermes}

# ---- bookkeeping ----
PROJECT_NAME=${PROJECT_NAME:-llm_fusion_eval}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-eval_$(basename "${MODEL_PATH}")}
VALIDATION_DATA_DIR=${VALIDATION_DATA_DIR:-${OUTPUT_ROOT}/eval/${EXPERIMENT_NAME}}

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.norm_adv_by_std_in_grpo=${NORM_ADV_BY_STD_IN_GRPO}
    algorithm.use_kl_in_reward=False
    data.train_files="${TRAIN_FILE}"
    data.val_files="${EVAL_SUITE}"
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.val_batch_size=${VAL_BATCH_SIZE}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=True
    data.truncation='error'
    data.return_raw_chat=True
)

MODEL=(
    actor_rollout_ref.model.path=${MODEL_PATH}
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF}
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768
    actor_rollout_ref.actor.use_dynamic_bsz=True
)

REF=(
    actor_rollout_ref.ref.fsdp_config.param_offload=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=${INFER_BACKEND}
    actor_rollout_ref.rollout.mode=async
    actor_rollout_ref.rollout.dtype=${ROLLOUT_DTYPE}
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.enforce_eager=False
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.max_num_batched_tokens=32768
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    actor_rollout_ref.rollout.temperature=1
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.n=${VAL_N}
    actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMPERATURE}
    actor_rollout_ref.rollout.val_kwargs.top_p=${VAL_TOP_P}
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent
    actor_rollout_ref.rollout.multi_turn.enable=True
    actor_rollout_ref.rollout.multi_turn.tool_config_path=${TOOL_CONFIG}
    actor_rollout_ref.rollout.multi_turn.format=${MULTITURN_FORMAT}
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${MAX_ASSISTANT_TURNS}
    actor_rollout_ref.rollout.multi_turn.max_user_turns=${MAX_USER_TURNS}
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=${MAX_PARALLEL_CALLS}
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=${MAX_TOOL_RESPONSE_LENGTH}
)

# With float32 inference the KV cache is fp32 too, and vLLM preallocates against
# the model config's nominal max_position_embeddings (262144 for Qwen3-4B), which
# needs ~72 GiB -- more than gpu_memory_utilization can grant, so the engine
# refuses to start. The real ceiling here is MAX_PROMPT_LENGTH +
# MAX_RESPONSE_LENGTH, far below that, so cap the unused context when set. This
# only affects KV preallocation, not the outputs.
if [[ -n "${MAX_MODEL_LEN:-}" ]]; then
    ROLLOUT+=( actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN} )
fi

TRAINER=(
    trainer.critic_warmup=0
    trainer.val_only=True
    trainer.val_before_train=True
    # Load the HuggingFace weights at MODEL_PATH directly. Without this verl
    # looks for a training checkpoint to resume from and would score that
    # instead. It also makes validate() report step 0, so the dumps land under
    # ${VALIDATION_DATA_DIR}/0.
    trainer.resume_mode=disable
    trainer.save_freq=-1
    trainer.test_freq=1
    trainer.total_epochs=1
    trainer.logger='["console"]'
    trainer.project_name=${PROJECT_NAME}
    trainer.experiment_name=${EXPERIMENT_NAME}
    trainer.validation_data_dir=${VALIDATION_DATA_DIR}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
)

REWARD=(
    reward.sandbox_fusion.url=${SANDBOX_FUSION_URL}
    reward.sandbox_fusion.max_concurrent=${SANDBOX_MAX_CONCURRENT}
    reward.sandbox_fusion.memory_limit_mb=${SANDBOX_MEMORY_LIMIT_MB}
)

python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "${REWARD[@]}" \
    "$@"
