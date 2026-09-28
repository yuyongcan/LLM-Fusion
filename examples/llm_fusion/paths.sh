# Shared paths for the training and evaluation scripts in this directory.
# Sourced by every run_*.sh; not meant to be executed directly.
#
# Fill in the four roots below, or export them before launching. Every script
# checks the roots it needs and exits with the missing variable names, so a
# forgotten path fails immediately instead of halfway into a run.

# Base policy to train or evaluate. The paper uses Qwen3-4B-Instruct-2507.
MODEL_PATH=${MODEL_PATH:-}

# Training data: the output directory of download_data.py, holding
# <Config>/train-*.parquet for Math, Science, Code, IF, Agent, and Mix.
TRAIN_DATA_ROOT=${TRAIN_DATA_ROOT:-}

# Evaluation data: the eight benchmark parquets (see EVAL SETS below).
EVAL_DATA_ROOT=${EVAL_DATA_ROOT:-}

# Where checkpoints, rollout dumps, and validation dumps are written. Needs
# room for several full-parameter checkpoints per run.
OUTPUT_ROOT=${OUTPUT_ROOT:-}

# train_mopd.sh only: directory holding the five domain teachers as HuggingFace
# exports, one subdirectory per domain (math, science, code, if, agent).
TEACHER_ROOT=${TEACHER_ROOT:-}

# train_mopd_single_teacher.sh only: one HuggingFace model directory to distil
# from, instead of TEACHER_ROOT's five. Typically a merged model produced by
# merge/run_merge.sh. bf16 is fine and preferred -- vLLM loads bf16 regardless,
# so fp32 only doubles disk and load time (unlike the merge itself, where fp32
# is required because the task vectors are small).
TEACHER_MODEL=${TEACHER_MODEL:-}

# ---- training sets --------------------------------------------------------
# A glob over each domain's parquet shards. Keep these quoted wherever they are
# used, or the shell expands them into several words before verl sees them.
MATH_TRAIN=${MATH_TRAIN:-${TRAIN_DATA_ROOT}/Math/train-*.parquet}
SCIENCE_TRAIN=${SCIENCE_TRAIN:-${TRAIN_DATA_ROOT}/Science/train-*.parquet}
CODE_TRAIN=${CODE_TRAIN:-${TRAIN_DATA_ROOT}/Code/train-*.parquet}
IF_TRAIN=${IF_TRAIN:-${TRAIN_DATA_ROOT}/IF/train-*.parquet}
AGENT_TRAIN=${AGENT_TRAIN:-${TRAIN_DATA_ROOT}/Agent/train-*.parquet}
MIX_TRAIN=${MIX_TRAIN:-${TRAIN_DATA_ROOT}/Mix/train-*.parquet}

# ---- SFT (phase 1: Base -> chat/reasoning model) --------------------------
# Deliberately separate from MODEL_PATH. MODEL_PATH is the *policy* the RL stages
# start from (the paper uses Qwen3-4B-Instruct-2507, already instruction-tuned);
# SFT has to start from Qwen3-4B-*Base*. Sharing one variable would silently
# fine-tune an already-instruct model and there would be nothing in the loss
# curve to tell you.
#
# ⚠️ Patch the eos contract on this directory before training -- Qwen3-4B-Base
# ships generation_config.eos_token_id = 151643 (<|endoftext|>) while its chat
# template ends assistant turns with <|im_end|> (151645). verl copies the base
# generation_config into every checkpoint verbatim, so without the patch each
# saved model advertises a stop token it never emits and vLLM generates to the
# token cap. run_sft_multinode.sh gates on this.
SFT_BASE_MODEL=${SFT_BASE_MODEL:-}

# Directory holding the SFT mixture parquet(s).
SFT_DATA_ROOT=${SFT_DATA_ROOT:-}

# The training file. Default is the mixture pre-filtered to <=16384 *training*
# tokens by sft/filter_by_token_length.py -- not the raw train.parquet. Filtering
# offline is what lets the run use truncation=error (a surviving overlong row then
# means tokenizer/template/data drift, rather than being silently cut).
SFT_TRAIN=${SFT_TRAIN:-${SFT_DATA_ROOT}/train.le16384.parquet}

# ---- EVAL SETS: the eight benchmarks reported in the paper ----------------
# Downloaded by download_data.py --target test. Override individually to point
# at your own copies.
AIME2025_TEST=${AIME2025_TEST:-${EVAL_DATA_ROOT}/AIME2025/test-*.parquet}
AIME2026_TEST=${AIME2026_TEST:-${EVAL_DATA_ROOT}/AIME2026/test-*.parquet}
GPQA_TEST=${GPQA_TEST:-${EVAL_DATA_ROOT}/GPQA/test-*.parquet}
LCB_V5_TEST=${LCB_V5_TEST:-${EVAL_DATA_ROOT}/LCB_v5/test-*.parquet}
LCB_V6_TEST=${LCB_V6_TEST:-${EVAL_DATA_ROOT}/LCB_v6/test-*.parquet}
IFEVAL_TEST=${IFEVAL_TEST:-${EVAL_DATA_ROOT}/IFEval/test-*.parquet}
IFBENCH_TEST=${IFBENCH_TEST:-${EVAL_DATA_ROOT}/IFBench/test-*.parquet}
BFCL_TEST=${BFCL_TEST:-${EVAL_DATA_ROOT}/BFCL_v3/test-*.parquet}

EVAL_SUITE=${EVAL_SUITE:-"[${AIME2025_TEST},${AIME2026_TEST},${GPQA_TEST},${LCB_V5_TEST},${LCB_V6_TEST},${IFEVAL_TEST},${IFBENCH_TEST},${BFCL_TEST}]"}

# ---- merge ----------------------------------------------------------------
# Used only by examples/llm_fusion/merge/. The five FSDP actor checkpoints to
# fuse, as written by the training scripts:
#   ${OUTPUT_ROOT}/<experiment_name>/global_step_<N>/actor
# There is no sensible default, so convert_experts_to_hf.sh names whichever of
# these is unset rather than guessing a step.
MATH_ACTOR=${MATH_ACTOR:-}
SCIENCE_ACTOR=${SCIENCE_ACTOR:-}
CODE_ACTOR=${CODE_ACTOR:-}
IF_ACTOR=${IF_ACTOR:-}
AGENT_ACTOR=${AGENT_ACTOR:-}

# fp32 HuggingFace exports of the five experts: written by
# convert_experts_to_hf.sh, read by run_merge.sh. fp32 rather than bf16 because
# the RL task vectors are small enough that the round-trip changes the merge --
# see the note in merge/convert_experts_to_hf.sh. Budget ~15 GB per expert at 4B.
HF_EXPERTS_ROOT=${HF_EXPERTS_ROOT:-${OUTPUT_ROOT}/merge/hf_experts_fp32}

# Merged models land in ${MERGE_OUT_ROOT}/merged_models/<method>, logs in
# ${MERGE_OUT_ROOT}/save_merge_llm_logs/<method>.
MERGE_OUT_ROOT=${MERGE_OUT_ROOT:-${OUTPUT_ROOT}/merge}

# The vendored MergeLM that produced every Merge result in the paper.
MERGELM=${MERGELM:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/third_party/MergeLM}

# ---- tool config ----------------------------------------------------------
# Tool schemas for the multi-turn agent rollout (WorkBench + BFCL v3 backends).
# Regenerate with verl/utils/bfcl_multiturn/generate_tool_config.py if you
# change the tool set.
TOOL_CONFIG=${TOOL_CONFIG:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/config/workbench_bfcl_tool_config.yaml}

# ---- code reward sandbox --------------------------------------------------
# The Code domain scores rollouts by executing unit tests in SandboxFusion
# (https://github.com/bytedance/SandboxFusion). Without a reachable server the
# reward silently falls back to a different scorer, so start one before running
# anything that touches Code, Mix, or the LiveCodeBench eval sets.
SANDBOX_FUSION_URL=${SANDBOX_FUSION_URL:-http://localhost:8080/run_code}
SANDBOX_MAX_CONCURRENT=${SANDBOX_MAX_CONCURRENT:-64}
SANDBOX_MEMORY_LIMIT_MB=${SANDBOX_MEMORY_LIMIT_MB:-4096}

# Exit with the names of any unset variables among the arguments. Tracing is
# suppressed so the message is readable under the callers' `set -x`.
require_paths() {
    { set +x; } 2>/dev/null
    local missing=()
    local name
    for name in "$@"; do
        [[ -n "${!name:-}" ]] || missing+=("${name}")
    done
    if ((${#missing[@]})); then
        echo "" >&2
        echo "Missing required paths. Set these in examples/llm_fusion/paths.sh," >&2
        echo "or export them before launching:" >&2
        printf '  %s\n' "${missing[@]}" >&2
        echo "" >&2
        exit 1
    fi
    set -x
}
