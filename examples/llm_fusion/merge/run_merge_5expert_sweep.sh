#!/usr/bin/env bash
# Sweep the Task-Arithmetic scaling coefficient over all FIVE experts.
#
# Thin wrapper around run_merge.sh. run_merge.sh already supports a list of
# lambdas via SCALINGS, so this script only pins the two things that are easy
# to get wrong and that silently produce an unusable sweep:
#
#   1. MERGE_MODELS must be all five experts. A previous sweep in
#      outputs_llm_fusion/merge_4b_mcs merged only math, code and science, so
#      its numbers are not comparable to merge_4b's and its ifbench/bfcl scores
#      are just the base model's. This script builds the list itself and refuses
#      to run unless it holds exactly five models.
#   2. MERGE_OUT_ROOT must not point at merge_4b_mcs. Its per-coefficient
#      directories already exist (as three-expert merges), and run_merge.sh's
#      already_merged() would skip every coefficient, reporting success while
#      doing no work.
#
# The default output root is merge_4b, which holds the released five-expert
# lambda=0.6 merge. That one is skipped as already merged, so the sweep lands
# next to it and merge_4b ends up as the complete five-expert sweep.
#
# Prereq: convert_experts_to_hf.sh, so the five experts exist as fp32 HF models.
#
# MEMORY: merge_llms.py holds the base and all five experts in CPU RAM in fp32
# at once, roughly 90 GB at 4B. Merges run sequentially for that reason.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# ---- paths (override by exporting before launch) ---------------------------
WS_ROOT=${WS_ROOT:-/taijifs_sz34/share_304169095/hunyuan/yongcanyu}

export MODEL_PATH=${MODEL_PATH:-${WS_ROOT}/checkpoints/Qwen/Qwen3-4B-Instruct-2507}
export OUTPUT_ROOT=${OUTPUT_ROOT:-${WS_ROOT}/outputs_llm_fusion}
export HF_EXPERTS_ROOT=${HF_EXPERTS_ROOT:-${REPO_ROOT}/teacher_experts_fp32/hf_experts_fp32_4b}
export MERGE_OUT_ROOT=${MERGE_OUT_ROOT:-${OUTPUT_ROOT}/merge_4b}

# ---- the sweep -------------------------------------------------------------
# lambda = 0.6 already exists in merge_4b and is skipped by already_merged().
export SCALINGS=${SCALINGS:-"0.4 0.8 1.0 1.2"}
export SAVE_DTYPE=${SAVE_DTYPE:-bfloat16}
export WEIGHT_MASK_RATE=${WEIGHT_MASK_RATE:-0.2}
# Task Arithmetic only; the method comparison is not part of this sweep.
export RUN_ALL_METHODS=0

# ---- all five experts, built here rather than inherited -------------------
export MERGE_MODELS="${HF_EXPERTS_ROOT}/math,${HF_EXPERTS_ROOT}/code,${HF_EXPERTS_ROOT}/science,${HF_EXPERTS_ROOT}/if,${HF_EXPERTS_ROOT}/agent"

n_experts=$(awk -F, '{print NF}' <<<"${MERGE_MODELS}")
if [ "${n_experts}" -ne 5 ]; then
    echo "ERROR: MERGE_MODELS has ${n_experts} entries, expected 5." >&2
    exit 1
fi

if [ "${MERGE_OUT_ROOT}" = "${OUTPUT_ROOT}/merge_4b_mcs" ]; then
    echo "ERROR: refusing to write into merge_4b_mcs: its coefficient dirs" >&2
    echo "       already exist as three-expert merges and would be skipped." >&2
    exit 1
fi

echo "base    : ${MODEL_PATH}"
echo "experts : ${n_experts} from ${HF_EXPERTS_ROOT}"
echo "lambdas : ${SCALINGS}"
echo "out     : ${MERGE_OUT_ROOT}/merged_models"
echo ""

exec bash "${SCRIPT_DIR}/run_merge.sh"
