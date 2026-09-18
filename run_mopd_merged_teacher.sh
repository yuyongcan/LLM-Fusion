#!/usr/bin/env bash
# 单教师 MOPD（蒸馏已合并模型）4 机 32 卡启动入口。
#
# 与 run_mopd_multinode.sh 的关系：那个跑论文的 5 教师配置（按 data_source 路由）。
# 这个换成**一个已合并的模型**当教师，默认 merge5 Task Arithmetic λ=1.0
# （results.md 里均匀系数的最优点，AVG8 66.00，发布配置 λ=0.6 只有 63.53）。
#
#   bash run_mopd_merged_teacher.sh [hydra overrides...]
#
# 学生侧超参、拓扑、batch 与 run_mopd_multinode.sh **完全一致**，唯一变量是教师。
# 所以两个作业可以直接横比：把 merge 的成果蒸回 base，能不能拿到 merge 的水平，
# 同时推理侧只需要保留一个模型而不是五个专家。
#
# ---------------------------------------------------------------------------------
# 拓扑：24 学生 + 8 教师 = 32，零空转
#
#   训练池   NNODES=3 × 8 = 24 卡   （trainer.nnodes × trainer.n_gpus_per_node）
#   教师池   1 机 × 8 = 8 卡        （distillation.nnodes × distillation.n_gpus_per_node）
#
# 5 教师那版是教师节点 8 卡里只用 5 张、3 张空转；这版 8 张全用上，跑**同一个模型的
# 8 份副本**（tp=1，每卡一份完整 4B）。请求由 GlobalRequestLoadBalancer 按
# in-flight 最少分发，8 份对等副本，教师侧不会成为瓶颈。
#
# 教师池是**独立的 Ray resource pool**，不算在 NNODES 里。
# ⚠️ 副本数不用自己算也不该自己设：verl 由池大小推出 8/(tp·dp·pp)=8。
#    但 TEACHER_TP 必须显式传 1 —— distillation.yaml 默认 tp=2，会静默变成
#    「4 份 tp2 副本」而不是「8 份 tp1 副本」，不报错。详见 train_mopd_single_teacher.sh 头注释。
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
#    会把脚本的 NNODES=3 默认值打掉 → 训练池缩成 8 卡、
#    TRAIN_BATCH_SIZE=264 不再被 dp size 整除。所以 source 之后必须显式写回。
source env_llm_fusion.sh
export NNODES=${NNODES_OVERRIDE:-3}
export NGPUS_PER_NODE=8

# ---- 教师：一个已合并的模型 ----
# 默认 merge5 λ=1.0。另一个候选是 merge_4b_percoef 的逐专家系数版（AVG8 66.33），
# 换的话直接 export TEACHER_MODEL。
# bf16 即可：vLLM 无论如何按 bf16 载入，fp32 只是白占一倍磁盘和 host 内存。
# （注意这跟 merge **输入**的要求相反 —— 那里必须 fp32，因为 RL task vector 很小，
#   bf16 的量化误差和 delta 同量级。教师只做打分，不做差，无此问题。）
export TEACHER_MODEL=${TEACHER_MODEL:-${B}/outputs_llm_fusion/merge_4b/merged_models/task_arithmetic_scaling_coefficient_1.0}

TRAIN_POOL=$(( NNODES * NGPUS_PER_NODE ))
TEACHER_POOL=${TEACHER_POOL:-8}
export TEACHER_WORLD_SIZE=${TEACHER_POOL}
export TEACHER_TP=${TEACHER_TP:-1}
WORLD_SIZE=$(( TRAIN_POOL + TEACHER_POOL ))

# ---- 代理绕行 ----
CLUSTER_NO_PROXY=$(printf '%s' "${NODE_IP_LIST}" | tr ',' '\n' | sed 's/:[0-9][0-9]*$//' \
                   | awk 'NF && !seen[$0]++' | paste -sd, -)
export no_proxy=".woa.com,.oa.com,mirrors.cloud.tencent.com,localhost,127.0.0.1,0.0.0.0,${CLUSTER_NO_PROXY}"
export NO_PROXY="${no_proxy}"
export RAY_ADDRESS=${RAY_ADDRESS:-${HEAD_IP}:6379}

# 每节点各自的 localhost sandbox。⚠️ 虽然 use_task_rewards=False，_compute_score
# 照样跑（记指标），且 25 步一次的验证要评 LCB v5/v6 —— 缺 sandbox 会静默 fallback
# 到另一个 scorer，分数腰斩且日志无报错。
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

# ⚠️ train_mopd_single_teacher.sh 沿用上游的 trainer.logger='["console","wandb"]'，
#    但本 Pod 没有 wandb 凭据 —— 而 verl 只在 val_only 路径把指标打到 console，
#    训练循环的 per-step 指标只走 backend。照原样跑 = 全程没有指标。
LOGGER=${LOGGER:-'["console","swanlab","tensorboard"]'}

# ---- 实验命名 ----
# CKPT_DIR 在 train_mopd_single_teacher.sh 里由
# ${OUTPUT_ROOT}/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME} 推出，
# PROJECT_NAME 一改就跟着走，所以这里不需要额外钉 default_local_dir。
export PROJECT_NAME=${PROJECT_NAME:-llm_fusion_repro}
# 教师模型的标识进实验名，免得换了教师还对不上号。
#   task_arithmetic_scaling_coefficient_1.0        -> ta1.0
#   task_arithmetic_percoef_math1.0_..._agent1.0   -> percoef
TEACHER_TAG=${TEACHER_TAG:-$(basename "${TEACHER_MODEL}" \
    | sed -e 's/^task_arithmetic_scaling_coefficient_/ta/' \
          -e 's/^task_arithmetic_percoef_.*/percoef/' \
          -e 's/^task_arithmetic_/ta_/')}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-mopd_merged_${TEACHER_TAG}_qwen3_4b_${NNODES}x${NGPUS_PER_NODE}+${TEACHER_POOL}t}
CKPT_CONTENTS=${CKPT_CONTENTS:-'["model"]'}

LOG_DIR=${B}/logs/llm_fusion
mkdir -p "${LOG_DIR}"
# 可被 run_mopd_exopd.sh 覆盖，让 ExOPD 的日志与 λ=1 基线在 ls 里就能分开。
LOG=${LOG:-${LOG_DIR}/train_mopd_merged_${NNODES}x${NGPUS_PER_NODE}_$(date +%Y%m%d_%H%M%S).log}

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

echo "[check] 教师模型 ..."
if [ ! -f "${TEACHER_MODEL}/config.json" ]; then
    echo "ERROR: TEACHER_MODEL=${TEACHER_MODEL} 不是 HF 模型目录（没有 config.json）。" >&2
    echo "       可选的已合并模型：" >&2
    ls -d "${B}"/outputs_llm_fusion/*/merged_models/*/ 2>/dev/null | sed 's/^/         /' >&2
    exit 1
fi

# tp=1 是这套拓扑的前提。distillation.yaml 默认 tp=2，会静默给出 4 份 tp2 副本。
echo "[check] 教师副本数 ..."
if [ "$(( TEACHER_POOL % TEACHER_TP ))" -ne 0 ]; then
    echo "ERROR: TEACHER_POOL=${TEACHER_POOL} 不能被 TEACHER_TP=${TEACHER_TP} 整除。" >&2
    exit 1
fi
TEACHER_REPLICAS=$(( TEACHER_POOL / TEACHER_TP ))

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

======== 单教师 MOPD（已合并模型）启动配置 ========
  训练池     : ${NNODES} 机 × ${NGPUS_PER_NODE} 卡 = ${TRAIN_POOL}
  教师池     : 1 机 × ${TEACHER_POOL} 卡（独立 resource pool，不算在 NNODES 里）
  教师副本   : ${TEACHER_REPLICAS} 份 × tp${TEACHER_TP}（同一个模型，load balancer 按 in-flight 最少分发）
  合计       : ${WORLD_SIZE} / ${TOTAL_GPU}   （零空转）
  节点       : ${NODE_IPS[*]}   (head=${HEAD_IP})
  教师模型   : ${TEACHER_MODEL}
  路由       : 无（单教师路径短路 data_source，五个域全发这一个）
  实验名     : ${EXPERIMENT_NAME}
  项目名     : ${PROJECT_NAME}
  logger     : ${LOGGER}   (上游是 wandb，本 Pod 无凭据，已换)
  swanlab    : ${SWANLAB_WEB_HOST:-<unset>}  log_dir=${SWANLAB_LOG_DIR}
  ckpt       : ${OUTPUT_ROOT}/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}
  日志       : ${LOG}
==================================================

EOF

setsid nohup bash examples/llm_fusion/train/train_mopd_single_teacher.sh \
    "trainer.logger=${LOGGER}" \
    "actor_rollout_ref.actor.checkpoint.save_contents=${CKPT_CONTENTS}" \
    "$@" \
    > "${LOG}" 2>&1 < /dev/null & disown

echo "已启动。"
echo "  跟踪日志 : tail -f ${LOG}"
echo "  reward   : python summarize_rollout.py \$OUTPUT_ROOT/rollout_data/${EXPERIMENT_NAME}"
echo "  swanlab  : ${SWANLAB_WEB_HOST:-<unset>}"
echo
echo "启动约 10 min 后核对教师拓扑（这是最容易静默配错的地方）："
echo "  bash check_teacher_topology.sh ${LOG}"
