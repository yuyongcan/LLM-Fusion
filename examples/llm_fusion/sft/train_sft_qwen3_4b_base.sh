#!/usr/bin/env bash
# Phase 1 | full-data SFT | Qwen3-4B-Base | verl FSDP engine | torchrun
#
# Turns Qwen3-4B-*Base* into the chat/reasoning model the RL stages start from, on the
# ~12.2M-row Nemotron mixture. Sibling of train/train_expert_math.sh, but the launch
# mechanism differs -- see "torchrun, not Ray" below.
#
# Usage (single node, or one invocation per node with NODE_RANK/MASTER_ADDR set):
#   bash train_sft_qwen3_4b_base.sh <save_path> [hydra overrides...]
#
# On the cluster, run ../../../run_sft_multinode.sh instead: it derives the topology and
# fans this script out to every node. DRY_RUN=1 prints the command without running it.
#
# ---------------------------------------------------------------------------------------
# torchrun, not Ray -- and why that contradicts HANDOFF.md §3 only in appearance
#
# Every other training entry point here is Ray-based (verl.trainer.main_ppo), so the head
# node submits once and Ray schedules the whole cluster. HANDOFF.md §3 and overview.md §4c
# therefore say not to copy a pssh fan-out launcher. That advice is about **RL**.
#
# verl.trainer.sft_trainer is a plain torchrun program. `torchrun --nnodes=N` forks only
# the local ranks and then blocks in rendezvous, so every node needs its own invocation
# with its own --node_rank. Hence run_sft_multinode.sh does need the fan-out.
#
# verl/trainer/sft_trainer_ray.py exists and would preserve the head-only model, but it is
# unusable at this data scale: its _build_dataloader hardcodes dp_rank=0/dp_size=1
# (sft_trainer_ray.py:181-185), i.e. the whole dataset is built in the driver process.
# Measured cost is ~7.7GB per 110k rows, so 12.2M rows is ~855GB in one process, and it
# does not honour the rank_sharded contract below either.
# ---------------------------------------------------------------------------------------
#
# Optimizer: AdamW, not Adam
#
# The source paper says "Adam ... weight decay 0.1". Taken literally that is
# torch.optim.Adam, whose weight_decay is COUPLED L2 -- the penalty is added to the
# gradient and then divided by Adam's second-moment estimate, so its effective strength
# scales with 1/sqrt(v). 0.1 is a normal AdamW value and a very large Adam value.
#
# Measured, three runs, different batch sizes and LR schedules: loss bottoms out at step
# 40-115 and then climbs monotonically (+29% to +32%), grad_norm rising alongside. The
# AdamW control, identical in every other respect, descended for all 205 steps:
#
#   AdamW (20-step means)            Adam (same data, same LR)
#     steps   1-20   0.9653            min  0.8797 @ step 115
#     steps  81-100  0.7070            last 1.1639 @ step 363   => +32.3%
#     steps 181-200  0.6626  <- still falling
#
# Set OPTIMIZER=Adam to reproduce the literal reading; expect the rebound.
#
# ---------------------------------------------------------------------------------------
# ⚠️ optim.clip_grad is passed explicitly and must stay that way
#
# This repo lowered the shared default to 0.2 in verl/trainer/config/optim/fsdp.yaml for an
# RL collapse, where grad_norm runs around 0.02 (p90 0.05). SFT is a different regime
# entirely -- measured over the 205-step AdamW run:
#
#   mean 0.486   p50 0.362   p90 0.784   p99 1.767   max 1.825
#   clipped at 0.2 : 205/205 steps (100.0%)
#   clipped at 1.0 :  15/205 steps (7.3%)
#
# Inheriting 0.2 would rescale every single update and there would be no error to see it
# by. Do not "simplify" this line away, and do not change the YAML default -- that bound
# belongs to the RL runs.
# ---------------------------------------------------------------------------------------
#
# Why SFT_SHARD_BY_RANK=1 is mandatory
#
# sft_trainer.py has every rank build the whole dataset and then split it with a
# DistributedSampler. At ~7.7GB per 110k rows that is ~700GB per process for the full
# mixture; eight ranks per node would need ~5.6TB against 2.2TB of RAM. With sharding on,
# dataset.py gives each rank only row_groups[rank::world_size], and the trainer detects the
# pre-sharded dataset and uses a one-replica sampler
# (verl/trainer/sft_trainer.py, `rank_sharded`). Without that trainer-side half the sampler
# would shard the already-disjoint shard again and the run would silently train
# 1/world_size of the data.
#
# Before the first real run, gate on reasoning preservation -- losing the thinking traces
# does NOT show up in the loss curve, so nothing else will tell you:
#   pytest tests/utils/dataset/test_m2rl_sft_dataset_on_cpu.py -q
# ---------------------------------------------------------------------------------------

set -xeuo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: train_sft_qwen3_4b_base.sh <save_path> [hydra overrides...]" >&2
    exit 1
fi

save_path=$1
shift 1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

source "${SCRIPT_DIR}/../paths.sh"
require_paths SFT_BASE_MODEL SFT_DATA_ROOT

# ---- cluster ----
# Topology comes from the launcher (or from Taiji's own variables when running by hand).
NNODES=${NNODES:-${HOST_NUM:-1}}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-${HOST_GPU_NUM:-8}}
NODE_RANK=${NODE_RANK:-${INDEX:-0}}
MASTER_ADDR=${MASTER_ADDR:-${CHIEF_IP:-127.0.0.1}}
MASTER_PORT=${MASTER_PORT:-29500}

# ---- data ----
MAX_LENGTH=${MAX_LENGTH:-16384}
# Per-GPU token budget for one micro-batch, deliberately well above MAX_LENGTH so several
# sequences pack into each forward. With it equal to MAX_LENGTH the card runs ~1 sequence
# per forward and pays a full FSDP all-gather/reduce-scatter round for very little work
# (measured 51.7 s/step at 11.4GB allocated but 55.8GB reserved on a 96GB H20).
#
# ⚠️ 65536 is the CEILING here, not a conservative starting point. It ran 205 steps x 32
# ranks and 10 steps x 64 ranks with zero OOM (73-80GB reserved), but 98304 was measured
# NOT to work: 87GB and a single step that never finished in 9 minutes -- the allocator
# thrashes against the limit instead of failing cleanly (same signature as
# TRAIN_BATCH_SIZE=5120). Raising it needs more headroom than Liger frees. Re-measure
# before changing.
MAX_TOKEN_LEN_PER_GPU=${MAX_TOKEN_LEN_PER_GPU:-65536}
MAX_SAMPLES=${MAX_SAMPLES:--1}
# Each rank builds its own dataset shard; forked loader workers multiply the resident copy.
NUM_WORKERS=${NUM_WORKERS:-2}

# ---- optimisation ----
# batch 1280, lr 5e-5, wd 0.1, 10% warmup, cosine, 1 epoch -- the configuration measured
# healthy above.
#
# ⚠️ Do not raise TRAIN_BATCH_SIZE to "use the GPUs better". Two measured reasons:
#   1. It does not reduce communication. FSDP's all-gather is per-micro-batch, so at a
#      fixed token budget 1280 -> 5120 just turns 2 micro-batches per rank into 6. It
#      merges 4 steps into 1 and changes nothing per token.
#   2. It does not fit. 1280 already reserves 73-80GB of 95GB; 5120 hit 86-89GB and a
#      single step did not finish in 25 minutes (GPUs pinned at 100% inside backward --
#      allocator thrashing, not a hang).
# And because sft_loss is token-mean over the global batch, gradient magnitude is
# batch-independent: a bigger batch only cuts the update count (1280 -> 5120 takes 10000
# steps down to 2500), so LR must rise with it or the run simply undertrains.
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-1280}
LR=${LR:-5e-5}
OPTIMIZER=${OPTIMIZER:-AdamW}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.1}
WARMUP_RATIO=${WARMUP_RATIO:-0.1}
LR_SCHEDULER=${LR_SCHEDULER:-cosine}
# See the header. Explicit because the repo default (0.2) clips 100% of SFT steps.
CLIP_GRAD=${CLIP_GRAD:-1.0}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
# Leave null to let one epoch define the length. ⚠️ When set together with the cosine
# schedule, this is also the cosine period -- set it to the real epoch length, not to a
# shorter value borrowed from a comparison run, or the LR anneals to zero early.
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-null}
SP_SIZE=${SP_SIZE:-1}
# On by default: the only config knob that measurably helped. 64-rank, 10 steps, exclusive:
#   use_liger=0 -> MFU 0.360, 60-65 s/it, 79.7GB reserved
#   use_liger=1 -> MFU 0.383, 55-57 s/it, 73.7GB reserved
# Single GPU at T=16384 isolates it from communication: 8616 -> 7914 ms (+8.9%) and
# activation memory 38.6 -> 24.5GB (-37%).
#
# It fuses swiglu/rmsnorm/rope/cross_entropy but NOT the biggest single allocation (the
# [tokens, 151936] logits tensor): verl forces fused_linear_cross_entropy=False because it
# conflicts with verl's forward patching (transformer_impl.py:282). It does change numerics
# slightly, so set USE_LIGER=0 if you need to match an earlier run bit-for-bit.
USE_LIGER=${USE_LIGER:-1}

# ---- bookkeeping ----
PROJECT_NAME=${PROJECT_NAME:-llm_fusion_sft}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-sft_qwen3_4b_base_${NNODES}x${NGPUS_PER_NODE}}
SAVE_FREQ=${SAVE_FREQ:-200}
# Each checkpoint carries model + optimizer + extra + hf_model, ~56GB at 4B.
MAX_CKPT_TO_KEEP=${MAX_CKPT_TO_KEEP:-3}
# 'optimizer' and 'extra' are what make resume_mode=auto work, and a multi-day run must be
# resumable; 'hf_model' is what the vLLM eval can actually load. Dropping either turns a
# single crash into a restart from scratch, or a checkpoint nothing downstream can read.
CKPT_CONTENTS=${CKPT_CONTENTS:-'["model","optimizer","extra","hf_model"]'}
LOGGER=${LOGGER:-'["console"]'}

# See the header: without this each rank needs ~700GB of RAM for the full mixture.
export SFT_SHARD_BY_RANK=${SFT_SHARD_BY_RANK:-1}

# Hydra writes its run dir relative to CWD, and every rank on every node shares this CWD
# over the shared filesystem and starts within the same second. Measured on 4x8: one rank
# lost the race and died with FileExistsError on outputs/<ts>/.hydra/hydra.yaml, and
# torchrun then SIGTERM'd the other 7 ranks of that node before the first step. Nothing but
# .hydra/ and the trainer log goes there, so give each rank its own directory.
export HYDRA_RUN_DIR="${HYDRA_RUN_DIR:-${save_path}/hydra}"

DATA=(
    data.train_files="${SFT_TRAIN}"
    data.train_max_samples="${MAX_SAMPLES}"
    data.train_batch_size="${TRAIN_BATCH_SIZE}"
    data.max_length="${MAX_LENGTH}"
    data.max_token_len_per_gpu="${MAX_TOKEN_LEN_PER_GPU}"
    data.num_workers="${NUM_WORKERS}"
    data.messages_key=messages
    data.tools_key=tools
    data.pad_mode=no_padding
    data.use_dynamic_bsz=true
    # The data is pre-filtered to <=MAX_LENGTH, so a surviving overlong row means the
    # tokenizer, chat template or data version drifted. Fail instead of truncating quietly.
    data.truncation=error
    # Keep false. dataset.py::sanity_check exempts only its per-turn fallback rows; this
    # flag is global and would also disable the check on the prefix path, which is the only
    # remaining detector for real tokenization bugs.
    data.ignore_input_ids_mismatch=false
    # The reason this directory exists: preserve reasoning_content. See dataset.py.
    data.custom_cls.path="${SCRIPT_DIR}/dataset.py"
    data.custom_cls.name=M2RLSFTDataset
)

MODEL=(
    model.path="${SFT_BASE_MODEL}"
    model.use_remove_padding=true
    model.use_liger="${USE_LIGER}"
)

OPTIM=(
    optim.optimizer="${OPTIMIZER}"
    optim.lr="${LR}"
    optim.weight_decay="${WEIGHT_DECAY}"
    optim.lr_warmup_steps_ratio="${WARMUP_RATIO}"
    optim.lr_scheduler_type="${LR_SCHEDULER}"
    optim.clip_grad="${CLIP_GRAD}"
)

ENGINE=(
    engine=fsdp
    engine.ulysses_sequence_parallel_size="${SP_SIZE}"
)

TRAINER=(
    trainer.default_local_dir="${save_path}"
    trainer.project_name="${PROJECT_NAME}"
    trainer.experiment_name="${EXPERIMENT_NAME}"
    trainer.logger="${LOGGER}"
    trainer.total_epochs="${TOTAL_EPOCHS}"
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS}"
    trainer.save_freq="${SAVE_FREQ}"
    trainer.max_ckpt_to_keep="${MAX_CKPT_TO_KEEP}"
    trainer.nnodes="${NNODES}"
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}"
    "checkpoint.save_contents=${CKPT_CONTENTS}"
    # See HYDRA_RUN_DIR above.
    'hydra.run.dir=${oc.env:HYDRA_RUN_DIR}/rank${oc.env:RANK,0}'
)

cmd=(
    torchrun
    --nnodes="${NNODES}"
    --nproc_per_node="${NGPUS_PER_NODE}"
    --node_rank="${NODE_RANK}"
    --master_addr="${MASTER_ADDR}"
    --master_port="${MASTER_PORT}"
    -m verl.trainer.sft_trainer
    "${DATA[@]}"
    "${MODEL[@]}"
    "${OPTIM[@]}"
    "${ENGINE[@]}"
    "${TRAINER[@]}"
    "$@"
)

if [ "${DRY_RUN:-0}" = "1" ]; then
    set +x
    printf '%q ' "${cmd[@]}"
    echo
    exit 0
fi

exec "${cmd[@]}"
