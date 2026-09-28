#!/usr/bin/env bash
# Merge the five fp32 experts with a *per-expert* scaling coefficient.
#
# Why: the uniform-lambda sweep in outputs_llm_fusion/merge_4b does not peak at
# the same lambda everywhere. ifbench keeps climbing to 1.2 (53.33) while
# aime25, gpqa and lcb_v5 peak at 1.0 and drop at 1.2. A single lambda has to
# trade those off; per-expert lambdas do not.
#
# run_merge.sh cannot express this: merge_llms.py takes one --scaling_coefficient
# and asserts it is a float. So this script calls
# merge_llms_per_coefficient.py, which computes
#     merged = base + sum_i lambda_i * (expert_i - base)
# with one lambda per expert. Everything else (fp32 compute, no masking, bf16
# on disk) is identical to the released Task Arithmetic merge.
#
# Defaults reproduce the config asked for on 2026-09-16: every expert at 1.0
# except `if` at 1.2.
#
# Prereq: convert_experts_to_hf.sh, so the experts exist as fp32 HF models.
#
# MEMORY: experts are loaded and freed one at a time, so peak RAM is the base
# plus one expert plus the accumulator, roughly 48 GB at 4B. No GPU needed.

set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

# ---- paths ----------------------------------------------------------------
# These are set BEFORE sourcing paths.sh, not after. paths.sh defaults
# HF_EXPERTS_ROOT and MERGE_OUT_ROOT to ${OUTPUT_ROOT}/merge/..., and
# `${VAR:-default}` keeps an already-set value -- so anything assigned after
# the source is silently ignored and the merge looks for experts that were
# never written. (Same trap as the SWANLAB_* variables in start.md §5.1.)
WS_ROOT=${WS_ROOT:-/taijifs_sz34/share_304169095/hunyuan/yongcanyu}
MODEL_PATH=${MODEL_PATH:-${WS_ROOT}/checkpoints/Qwen/Qwen3-4B-Instruct-2507}
OUTPUT_ROOT=${OUTPUT_ROOT:-${WS_ROOT}/outputs_llm_fusion}
HF_EXPERTS_ROOT=${HF_EXPERTS_ROOT:-${REPO_ROOT}/teacher_experts_fp32/hf_experts_fp32_4b}
MERGE_OUT_ROOT=${MERGE_OUT_ROOT:-${OUTPUT_ROOT}/merge_4b_percoef}
export MODEL_PATH OUTPUT_ROOT HF_EXPERTS_ROOT MERGE_OUT_ROOT

source "${SCRIPT_DIR}/../paths.sh"
require_paths MODEL_PATH OUTPUT_ROOT

PYTHON=${PYTHON:-python3}

# ---- per-expert lambdas ---------------------------------------------------
COEF_MATH=${COEF_MATH:-1.0}
COEF_CODE=${COEF_CODE:-1.0}
COEF_SCIENCE=${COEF_SCIENCE:-1.0}
COEF_IF=${COEF_IF:-1.2}
COEF_AGENT=${COEF_AGENT:-1.0}

SAVE_DTYPE=${SAVE_DTYPE:-float32}

# Order is math, code, science, if, agent -- the order run_merge.sh uses for the
# released checkpoints. The merge itself is symmetric, but the two lists below
# MUST stay in the same order: coefficient i applies to model i.
MERGE_MODELS="${HF_EXPERTS_ROOT}/math,${HF_EXPERTS_ROOT}/code,${HF_EXPERTS_ROOT}/science,${HF_EXPERTS_ROOT}/if,${HF_EXPERTS_ROOT}/agent"
SCALING_COEFFICIENTS="${COEF_MATH},${COEF_CODE},${COEF_SCIENCE},${COEF_IF},${COEF_AGENT}"

NAME=task_arithmetic_percoef_math${COEF_MATH}_code${COEF_CODE}_science${COEF_SCIENCE}_if${COEF_IF}_agent${COEF_AGENT}

for d in "${MODEL_PATH}" ${MERGE_MODELS//,/ }; do
    if [ ! -f "${d}/config.json" ]; then
        echo "ERROR: ${d} is not an HF model dir (no config.json)." >&2
        echo "Run examples/llm_fusion/merge/convert_experts_to_hf.sh first." >&2
        exit 1
    fi
done

echo "base    : ${MODEL_PATH}"
echo "lambdas : math=${COEF_MATH} code=${COEF_CODE} science=${COEF_SCIENCE} if=${COEF_IF} agent=${COEF_AGENT}"
echo "out     : ${MERGE_OUT_ROOT}/merged_models/${NAME}"

if [ -f "${MERGE_OUT_ROOT}/merged_models/${NAME}/config.json" ]; then
    echo "SKIP ${NAME} (already merged)"
    exit 0
fi

mkdir -p "${MERGE_OUT_ROOT}/merged_models" "${MERGE_OUT_ROOT}/save_merge_llm_logs"
export PYTHONPATH="${MERGELM}:${PYTHONPATH:-}"

${PYTHON} "${SCRIPT_DIR}/merge_llms_per_coefficient.py" \
    --pretrained_model_name "${MODEL_PATH}" \
    --models_to_merge "${MERGE_MODELS}" \
    --scaling_coefficients "${SCALING_COEFFICIENTS}" \
    --save_model_path "${MERGE_OUT_ROOT}/merged_models/${NAME}" \
    --save_dtype "${SAVE_DTYPE}" \
    2>&1 | tee "${MERGE_OUT_ROOT}/save_merge_llm_logs/${NAME}.log"

echo ""
echo "Merged model (${SAVE_DTYPE}) at ${MERGE_OUT_ROOT}/merged_models/${NAME}:"
ls -la "${MERGE_OUT_ROOT}/merged_models/${NAME}"
echo ""
echo "Evaluate with:"
echo "  MODEL_PATH=${MERGE_OUT_ROOT}/merged_models/${NAME} \\"
echo "  EXPERIMENT_NAME=eval_merge bash examples/llm_fusion/eval/eval.sh"
