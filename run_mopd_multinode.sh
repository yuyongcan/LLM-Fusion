#!/usr/bin/env bash
# MOPD（5 教师 on-policy 蒸馏）4 机 32 卡启动入口。
#
# 与 run_train_multinode.sh 的关系：那个脚本只 case 了 agent|math|science|code|if|mix，
# 不认 mopd，且会把 EXPERIMENT_NAME/TRAIN_FILE 按 domain-RL 的形态覆盖掉。
# 所以这里单独做一份，只复用它那几条真正通用的结论（代理绕行 / NCCL / swanlab / 前置检查）。
#
#   bash run_mopd_multinode.sh [hydra overrides...]
#
# ---------------------------------------------------------------------------------
# 拓扑：train_mopd.sh 原生就是给 4 机设计的，不用改
#
#   训练池   NNODES=3 × 8 = 24 卡   （trainer.nnodes × trainer.n_gpus_per_node）
#   教师池   1 机 × 5 = 5 卡        （distillation.nnodes × distillation.n_gpus_per_node）
#   -------------------------------------------------
#   合计 29 / 32，教师节点上 3 张卡空转（脚本头注释里写明的预期）
#
# 教师池是**独立的 Ray resource pool**，不算在 NNODES 里。池大小必须等于
# sum(num_replicas × tp) = 5 × 1 × 1 = 5。
# ---------------------------------------------------------------------------------
set -uo pipefail

B=/taijifs_sz34/share_304169095/hunyuan/yongcanyu
REPO="${B}/code/LLM-Fusion"
cd "${REPO}"

: "${NODE_IP_LIST:?NODE_IP_LIST 未设置 —— 不在 Taiji 节点上？}"
mapfile -t NODE_IPS < <(echo "${NODE_IP_LIST}" | tr ',' '\n' | cut -d: -f1 | grep .)
HEAD_IP="${CHIEF_IP:-${NODE_IPS[0]}}"

# ---- 根路径 ----
# ⚠️ env_llm_fusion.sh 里 NNODES=1 / NGPUS_PER_NODE=8 是硬 export（不是 ${:-}），
#    会把 train_mopd.sh 的 NNODES=3 默认值打掉 → 训练池缩成 8 卡、
#    TRAIN_BATCH_SIZE=264 不再被 dp size 整除。所以 source 之后必须显式写回。
source env_llm_fusion.sh
export NNODES=${NNODES_OVERRIDE:-3}
export NGPUS_PER_NODE=8

# ---- 教师 ----
# env_llm_fusion.sh 指向 checkpoints/Siye01/experts_hf（bf16，五个域齐全）。
# 仓库里另有 teacher_experts_fp32/hf_experts_fp32_4b，是 fp32 导出（16G/个）。
# train_mopd.sh 头注释明确要 bf16：vLLM 无论如何都按 bf16 载入，fp32 只是白
# 占一倍磁盘和 host 内存，对教师打分完全无损。所以用前者。
export TEACHER_ROOT=${TEACHER_ROOT:-${B}/checkpoints/Siye01/experts_hf}

TRAIN_POOL=$(( NNODES * NGPUS_PER_NODE ))
TEACHER_POOL=5
WORLD_SIZE=$(( TRAIN_POOL + TEACHER_POOL ))

# ---- 代理绕行 ----
CLUSTER_NO_PROXY=$(printf '%s' "${NODE_IP_LIST}" | tr ',' '\n' | sed 's/:[0-9][0-9]*$//' \
                   | awk 'NF && !seen[$0]++' | paste -sd, -)
export no_proxy=".woa.com,.oa.com,mirrors.cloud.tencent.com,localhost,127.0.0.1,0.0.0.0,${CLUSTER_NO_PROXY}"
export NO_PROXY="${no_proxy}"
export RAY_ADDRESS=${RAY_ADDRESS:-${HEAD_IP}:6379}

# 每节点各自的 localhost sandbox（Mix 含 Code，reward 要执行单测；
# 只在 head 起会让落在 worker 的那一半静默 fallback，分数腰斩且不报错）
export SANDBOX_FUSION_URL=${SANDBOX_FUSION_URL:-http://localhost:8080/run_code}

# ---- NCCL ----
# 真正生效的那份由 setup_ray_cluster.sh 在 ray start 时注入（verl 的 runtime_env
# 是写死白名单，NCCL_* 不在其中）。这里 export 只为本 shell / driver。
export NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL:-0}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-bond1}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# ---- SwanLab ----
# Pod 注入的 SWANLAB_* 会打败脚本默认值（${VAR:-默认} 的语义），实际 host 是
# train-exp 而非 taiji-exp。两个必须覆盖的：
#   LOG_DIR : Pod 给的是共享盘，swanlab 的 TemporaryFile(dir=) 探测在共享盘上
#             抛 FileNotFoundError → 误报 FILESYSTEM may be slow or remote。
#   项目名   : llm_fusion 被别的 workspace 占了，会在 vLLM 起完后才抛 422。
export SWANLAB_LOG_DIR=/tmp/swanlog_llm_fusion
export SWANLAB_MODE="${SWANLAB_MODE:-cloud}"
mkdir -p "${SWANLAB_LOG_DIR}"

# ⚠️ train_mopd.sh 原版写的是 trainer.logger='["console","wandb"]'，
#    但本 Pod 没有 wandb 凭据 —— 而 verl 只在 val_only 路径把指标打到 console，
#    训练循环的 per-step 指标只走 backend。照原样跑 = 全程没有指标。
LOGGER=${LOGGER:-'["console","swanlab","tensorboard"]'}

# ---- 实验命名 ----
# CKPT_DIR 在 train_mopd.sh 里由 ${OUTPUT_ROOT}/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}
# 推出，PROJECT_NAME 一改就跟着走，所以这里不需要额外钉 default_local_dir。
export PROJECT_NAME=${PROJECT_NAME:-llm_fusion_repro}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-mopd_5teacher_qwen3_4b_${NNODES}x${NGPUS_PER_NODE}+${TEACHER_POOL}t}
CKPT_CONTENTS=${CKPT_CONTENTS:-'["model"]'}

LOG_DIR=${B}/logs/llm_fusion
mkdir -p "${LOG_DIR}"
LOG=${LOG_DIR}/train_mopd_${NNODES}x${NGPUS_PER_NODE}_$(date +%Y%m%d_%H%M%S).log

export PATH="/opt/conda/envs/torch-base/bin:${PATH}"

# ================== 前置检查 ==================
echo "[check] Ray 集群 GPU 数 ..."
TOTAL_GPU=$(ray status 2>/dev/null | grep -oE '[0-9.]+/[0-9.]+ GPU' | grep -oE '/[0-9.]+' | tr -d '/' | head -1)
if [ -z "${TOTAL_GPU}" ]; then
    echo "ERROR: 连不上 Ray (${RAY_ADDRESS})。先跑 setup_ray_cluster.sh。" >&2
    exit 1
fi
if [ "${TOTAL_GPU%.*}" -lt "${WORLD_SIZE}" ]; then
    echo "ERROR: 需要 ${WORLD_SIZE} 卡（训练 ${TRAIN_POOL} + 教师 ${TEACHER_POOL}），Ray 实际 ${TOTAL_GPU}。" >&2
    echo "       GPU 不足时 verl 会静默 pending 挂住，不报错。" >&2
    exit 1
fi

echo "[check] 五个教师目录 ..."
MISSING_T=()
for d in math science code if agent; do
    [ -f "${TEACHER_ROOT}/${d}/config.json" ] || MISSING_T+=("${d}")
done
if [ "${#MISSING_T[@]}" -gt 0 ]; then
    echo "ERROR: TEACHER_ROOT=${TEACHER_ROOT} 缺这些域的 HF 导出：${MISSING_T[*]}" >&2
    echo "       MOPD 要五个凑齐，用 examples/llm_fusion/merge/convert_experts_to_hf.sh 导出。" >&2
    exit 1
fi

echo "[check] 逐节点 sandbox（head 通 != 全集群通）..."
SANDBOX_MISSING=()
for ip in "${NODE_IPS[@]}"; do
    probe="curl -s -m 10 --noproxy '*' -X POST http://localhost:8080/run_code \
        -H 'Content-Type: application/json' -d '{\"code\":\"print(6*7)\",\"language\":\"python\"}'"
    if [ "${ip}" = "${HEAD_IP}" ]; then
        out=$(eval "${probe}" 2>/dev/null)
    else
        out=$(ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=10 "${ip}" "${probe}" 2>/dev/null)
    fi
    echo "${out}" | grep -q '42' || SANDBOX_MISSING+=("${ip}")
done
if [ "${#SANDBOX_MISSING[@]}" -gt 0 ]; then
    echo "ERROR: 这些节点的 sandbox 不可达：${SANDBOX_MISSING[*]}" >&2
    echo "       修复：在每个缺失节点上跑 bash ${REPO}/setup_sandbox.sh" >&2
    exit 1
fi

cat <<EOF

============ MOPD 启动配置 ============
  训练池     : ${NNODES} 机 × ${NGPUS_PER_NODE} 卡 = ${TRAIN_POOL}
  教师池     : 1 机 × ${TEACHER_POOL} 卡（5 教师 tp1 单副本，独立 resource pool）
  合计       : ${WORLD_SIZE} / ${TOTAL_GPU}  （教师节点 3 卡空转，设计如此）
  节点       : ${NODE_IPS[*]}   (head=${HEAD_IP})
  教师       : ${TEACHER_ROOT}  (bf16)
  实验名     : ${EXPERIMENT_NAME}
  项目名     : ${PROJECT_NAME}
  logger     : ${LOGGER}   (原脚本是 wandb，本 Pod 无凭据，已换)
  swanlab    : ${SWANLAB_WEB_HOST}  log_dir=${SWANLAB_LOG_DIR}
  ckpt       : ${OUTPUT_ROOT}/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}
  日志       : ${LOG}
=======================================

EOF

setsid nohup bash examples/llm_fusion/train/train_mopd.sh \
    "trainer.logger=${LOGGER}" \
    "actor_rollout_ref.actor.checkpoint.save_contents=${CKPT_CONTENTS}" \
    "$@" \
    > "${LOG}" 2>&1 < /dev/null & disown

echo "已启动。"
echo "  跟踪日志 : tail -f ${LOG}"
echo "  reward   : python summarize_rollout.py \$OUTPUT_ROOT/rollout_data/${EXPERIMENT_NAME}"
echo "  swanlab  : ${SWANLAB_WEB_HOST}"
