#!/usr/bin/env bash
# 多机多卡 domain-RL / Mix RL 完整训练入口（论文超参）。
#
# 仿 MOPD/mopd_sft/run_adamw_control.sh 的形态：一个薄的「实验配置 + 启动」脚本，
# 把拓扑推导、前置检查、环境注入都收在这里，真正的超参交给
# examples/llm_fusion/train/train_expert_math.sh（论文原版，不改）。
#
#   bash run_train_multinode.sh {agent|math|science|code|if|mix} [hydra overrides...]
#
# ---------------------------------------------------------------------------------
# 为什么不需要 MOPD 那种 fan-out 启动器
#
# MOPD 的 launch_multinode.sh 存在的原因是 **torchrun 不是 Ray**：torchrun 只 fork
# 本地 8 个 rank，必须在每台机器各起一份并分配不同的 --node_rank，所以要 pssh 分发。
#
# verl 是 Ray-based：chief 上提交一次，Ray 自己把 WorkerDict 调度到全集群。
# 所以这里只需要「先把 Ray 集群拉起来」（setup_ray_cluster.sh），然后在 head
# 单点提交。拓扑通过 trainer.nnodes × trainer.n_gpus_per_node 声明。
#
# 但 Ray 也带来两个 torchrun 没有的坑，见下面 NCCL 和 sandbox 两节。
# ---------------------------------------------------------------------------------
set -uo pipefail

WHICH=${1:-}
if [ -z "${WHICH}" ]; then
    echo "usage: $0 {agent|math|science|code|if|mix} [hydra overrides...]" >&2
    exit 1
fi
shift 1

B=/taijifs_sz34/share_304169095/hunyuan/yongcanyu
REPO="${B}/code/LLM-Fusion"
cd "${REPO}"

# ---- 拓扑：从平台变量推导，不硬编码 IP ----
# NODE_IP_LIST 是 "ip:slots,ip:slots,..."，权威来源。
# ⚠️ 不要用 NODE_NUM 当节点数：该平台上它是总 GPU 数（16），不是机器数（2）。
: "${NODE_IP_LIST:?NODE_IP_LIST 未设置 —— 不在 Taiji 节点上？}"
mapfile -t NODE_IPS < <(echo "${NODE_IP_LIST}" | tr ',' '\n' | cut -d: -f1 | grep .)
HEAD_IP="${CHIEF_IP:-${NODE_IPS[0]}}"
NNODES=${NNODES:-${HOST_NUM:-${#NODE_IPS[@]}}}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-${HOST_GPU_NUM:-8}}

if [ "${NODE_IPS[0]}" != "${HEAD_IP}" ]; then
    echo "ERROR: NODE_IP_LIST[0]=${NODE_IPS[0]} 但 CHIEF_IP=${HEAD_IP}；拒绝猜测 head。" >&2
    exit 1
fi

# ---- 根路径 ----
# ⚠️ env_llm_fusion.sh 里 NNODES/NGPUS_PER_NODE 是硬 export（不是 ${:-}），
#    所以先存下推导值，source 之后再覆盖回去。
_NNODES=${NNODES} _NGPUS=${NGPUS_PER_NODE}
source env_llm_fusion.sh
export NNODES=${_NNODES} NGPUS_PER_NODE=${_NGPUS}

WORLD_SIZE=$(( NNODES * NGPUS_PER_NODE ))

# ---- 代理绕行 ----
# 外网走代理，但 Ray/NCCL 的集群内流量和 localhost 的 sandbox 绝不能走代理。
CLUSTER_NO_PROXY=$(printf '%s' "${NODE_IP_LIST}" | tr ',' '\n' | sed 's/:[0-9][0-9]*$//' \
                   | awk 'NF && !seen[$0]++' | paste -sd, -)
export no_proxy=".woa.com,.oa.com,mirrors.cloud.tencent.com,localhost,127.0.0.1,0.0.0.0,${CLUSTER_NO_PROXY}"
export NO_PROXY="${no_proxy}"
export RAY_ADDRESS=${RAY_ADDRESS:-${HEAD_IP}:6379}

# ---- 每台机器路由到本机 sandbox ----
# Code/Mix 的 reward 与 LCB v5/v6 的打分都要执行单测。打分任务由 Ray 分散到所有
# worker，所以**每个节点都得有一个 sandbox，且各自访问自己的 localhost**。
#
# 用 localhost 而不是 head 的 IP，有两个好处：
#   1. 16 卡并发分摊到 N 个 sandbox，head 不会成为瓶颈；
#   2. 打分流量不跨机。
#
# ⚠️ 反面教材（2026-09-11 实测）：只在 head 起 sandbox 时，落在 worker 的那一半
#    打分全部 Connection refused，然后**静默 fallback 到另一个 scorer**，
#    LCB 分数正好腰斩（0.30 vs 论文 base 0.59）且多次验证纹丝不动。不会报错。
export SANDBOX_FUSION_URL=${SANDBOX_FUSION_URL:-http://localhost:8080/run_code}

# ---- NCCL ----
# 本机栈（NCCL 2.28.9 + CUDA 13.0 + driver 535.247.01 + mlx5_bond_1..8）在
# GPUDirect RDMA 注册 GPU 显存的路径上不兼容，跨机 FSDP init 会报
# `ibv_reg_mr_iova2 ... Invalid argument`。NCCL_NET_GDR_LEVEL=0 关直连、保留 IB。
#
# ⚠️ 这里 export 只是给本 shell 和 ray start（若由本脚本代起）用。
#    verl 给 Ray worker 下发的 runtime_env 是一份写死的白名单
#    （verl/trainer/constants_ppo.py），NCCL_NET_GDR_LEVEL **不在其中**，
#    所以真正生效的是 setup_ray_cluster.sh 在 `ray start` 时注入的那份。
#    详见 overview.md 5.7。
export NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL:-0}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-bond1}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# ---- SwanLab（自建实例 taiji-exp）----
# verl 的 tracking.py 只读 SWANLAB_API_KEY / SWANLAB_LOG_DIR / SWANLAB_MODE；
# project / experiment_name 来自 trainer.project_name / trainer.experiment_name；
# SWANLAB_WEB_HOST / SWANLAB_API_HOST 由 swanlab SDK 自己消费。
export SWANLAB_WEB_HOST="${SWANLAB_WEB_HOST:-http://taiji-exp.taiji.woa.com/}"
export SWANLAB_API_HOST="${SWANLAB_API_HOST:-http://taiji-exp.taiji.woa.com/api}"
export SWANLAB_API_KEY="${SWANLAB_API_KEY:-7f6uQlMFc3T21Ah5hpMwh}"
export SWANLAB_MODE="${SWANLAB_MODE:-cloud}"
# ⚠️ SWANLAB_LOG_DIR 必须是**节点本地**路径，不能放共享盘（2026-09-11 实测）：
#    swanlab 0.10.0 的 safe_mkdir 用 `tempfile.TemporaryFile(dir=...)` 探测可写性，
#    该调用「创建后立即 unlink 但保持打开」，这个共享盘不支持，抛 FileNotFoundError，
#    被当成可重试的延迟自旋到 5s，最后报成误导性的
#    `exists but is not writable / FILESYSTEM may be slow or remote`。
#    swanlog 只是 SDK 本地缓存（指标仍上传 taiji-exp），放 /tmp 不损失任何东西。
#    真正生效的那份由 setup_ray_cluster.sh 在 ray start 时注入（见 §5 白名单问题）。
export SWANLAB_LOG_DIR="${SWANLAB_LOG_DIR:-/tmp/swanlog_llm_fusion}"
mkdir -p "${SWANLAB_LOG_DIR}"
# ⚠️ SWANLAB_SAVE_DIR 是另一条，且更早生效（2026-09-22 实测）：它决定 `.netrc` 等
#    全局文件的位置，在 **import swanlab** 时就被读（env.py:183 set_default ->
#    get_save_dir），而 get_save_dir 只创建末级目录、不创建父目录 —— 父目录不存在
#    就抛 `FileNotFoundError: <父目录> not found`，连 import 都过不了。
#    本 Pod 注入的值指向一个不存在的 <instance> 目录，于是下面的门禁会误报
#    「没装 swanlab」。用 ${VAR:-} 不行（Pod 的值会赢），必须无条件覆盖。
export SWANLAB_SAVE_DIR=/tmp/swanlab_llm_fusion
mkdir -p "${SWANLAB_SAVE_DIR}"

# tensorboard 留作本地兜底（swanlab 挂了也还有指标）。
# ⚠️ TENSORBOARD_DIR 同样传不到 Ray actor，实际会落在 <repo>/tensorboard_log/，
#    这是 verl 的默认相对路径。已在 .gitignore 里排除。
LOGGER=${LOGGER:-'["console","swanlab","tensorboard"]'}

# ---- 论文超参（Table 3，per-domain RL / Mix RL）----
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-128}
export ROLLOUT_N=${ROLLOUT_N:-16}
# ⚠️ MAX_PROMPT_LENGTH 故意不在这里给默认值（2026-09-22）。
#    这里 export 一个值，会盖掉 train_*.sh 里 ${MAX_PROMPT_LENGTH:-N} 的默认 ——
#    `${VAR:-默认}` 的语义是「VAR 已设就用 VAR」，于是本文件的值永远赢，域脚本
#    自己调的长度静默失效。各域数据的 prompt 分布不同（train_mix_rl.sh 按实测
#    p99/max 定在 2048），该由域脚本决定；想临时覆盖就在启动 shell 里 export。
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-24576}
export ACTOR_LR=${ACTOR_LR:-1e-6}
export ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.9}
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-10}
export SAVE_FREQ=${SAVE_FREQ:-25}
export TEST_FREQ=${TEST_FREQ:-25}
export PROJECT_NAME=${PROJECT_NAME:-llm_fusion_repro}
CKPT_CONTENTS=${CKPT_CONTENTS:-'["model"]'}

# ---- 每个域的差异 ----
# 训练集 + 论文步数 + Agent 专属的 tool_agent 覆盖。
# 步数取自论文 Table 3：per-domain RL 300、Mix RL 800
# （脚本默认 325/1500 是 overshoot，Table 4 的成本按 300/800 测）。
EXTRA_OVERRIDES=()
case "${WHICH}" in
    agent)
        TRAIN_SUBDIR=Agent;  DEFAULT_STEPS=300
        # ⚠️ Agent 训练行没有 agent_name 列（实测 10229 行全无），不传这个会
        #    fallback 到 single_turn_agent —— 单轮、无工具、reward 直接塌，且不报错。
        #    判据：num_turns/mean 必须远大于 1（实测 7.8~10.5）。
        EXTRA_OVERRIDES+=(actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent)
        ;;
    math)    TRAIN_SUBDIR=Math;    DEFAULT_STEPS=300 ;;
    science) TRAIN_SUBDIR=Science; DEFAULT_STEPS=300 ;;
    code)    TRAIN_SUBDIR=Code;    DEFAULT_STEPS=300 ;;
    if)      TRAIN_SUBDIR=IF;      DEFAULT_STEPS=300 ;;
    mix)     TRAIN_SUBDIR=Mix;     DEFAULT_STEPS=800 ;;
    *) echo "usage: $0 {agent|math|science|code|if|mix} [overrides...]" >&2; exit 1 ;;
esac

export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-${DEFAULT_STEPS}}
export TRAIN_FILE="${TRAIN_DATA_ROOT}/${TRAIN_SUBDIR}/train-*.parquet"   # 保持引号
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-expert_${WHICH}_qwen3_4b_${NNODES}x${NGPUS_PER_NODE}}

# Mix 用论文自己的脚本（多域混采逻辑不同）
TRAIN_SH=examples/llm_fusion/train/train_expert_math.sh
[ "${WHICH}" = "mix" ] && TRAIN_SH=examples/llm_fusion/train/train_mix_rl.sh

LOG_DIR=${B}/logs/llm_fusion
mkdir -p "${LOG_DIR}"
LOG=${LOG_DIR}/train_${WHICH}_${NNODES}x${NGPUS_PER_NODE}_$(date +%Y%m%d_%H%M%S).log

# ================== 前置检查 ==================
# 这三项都是「不检查就会静默跑错」的，宁可启动前失败。
export PATH="/opt/conda/envs/torch-base/bin:${PATH}"

echo "[check] Ray 集群 GPU 数 ..."
TOTAL_GPU=$(ray status 2>/dev/null | grep -oE '[0-9.]+/[0-9.]+ GPU' | grep -oE '/[0-9.]+' | tr -d '/' | head -1)
if [ "${TOTAL_GPU%.*}" != "${WORLD_SIZE}" ]; then
    echo "ERROR: 期望 ${WORLD_SIZE} 卡，Ray 实际 ${TOTAL_GPU:-none}。先跑 setup_ray_cluster.sh。" >&2
    echo "       GPU 不足时 verl 会静默 pending 挂住，不报错。" >&2
    exit 1
fi

echo "[check] 逐节点 sandbox（head 通 != 全集群通）..."
SANDBOX_MISSING=()
for ip in "${NODE_IPS[@]}"; do
    if [ "${ip}" = "${HEAD_IP}" ]; then
        probe_cmd="curl -s -m 10 --noproxy '*' -X POST http://localhost:8080/run_code \
            -H 'Content-Type: application/json' -d '{\"code\":\"print(6*7)\",\"language\":\"python\"}'"
        out=$(eval "${probe_cmd}" 2>/dev/null)
    else
        out=$(ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=10 "${ip}" \
            "curl -s -m 10 --noproxy '*' -X POST http://localhost:8080/run_code \
             -H 'Content-Type: application/json' \
             -d '{\"code\":\"print(6*7)\",\"language\":\"python\"}'" 2>/dev/null)
    fi
    if ! echo "${out}" | grep -q '42'; then
        SANDBOX_MISSING+=("${ip}")
    fi
done

if [ "${#SANDBOX_MISSING[@]}" -gt 0 ]; then
    echo "ERROR: 这些节点的 sandbox 不可达：${SANDBOX_MISSING[*]}" >&2
    echo "       Agent 域的 reward 不需要它，但每 ${TEST_FREQ} 步的验证集含 LCB v5/v6；" >&2
    echo "       没有它打分会**静默退化**（分数腰斩且不报错）。" >&2
    echo "       修复：在每个缺失节点上跑 bash ${REPO}/setup_sandbox.sh" >&2
    exit 1
fi

echo "[check] swanlab ..."
if [ "${LOGGER}" != "${LOGGER/swanlab/}" ]; then
    for ip in "${NODE_IPS[@]}"; do
        # TaskRunner 可能被调度到任一节点，所以两边都要装
        # ⚠️ stderr 必须丢掉，不能 2>&1（2026-09-11 实测踩到）：swanlab 0.7.20 在
        #    import 时会打一条 `FutureWarning: The pynvml package is deprecated`，
        #    2>&1 会把它一起塞进 ${ok}，于是 ${ok} == "...警告...\nok" != "ok"，
        #    明明装着也判成没装。只看退出码即可。
        if [ "${ip}" = "${HEAD_IP}" ]; then
            ok=$(python -c "import swanlab" 2>/dev/null && echo ok)
        else
            # ssh 非交互不继承本 shell 的 export，SWANLAB_SAVE_DIR 要显式带上，
            # 否则 worker 上读到 Pod 注入的坏路径、import 失败，判成没装。
            ok=$(ssh -o BatchMode=yes -o StrictHostKeyChecking=no "${ip}" \
                "SWANLAB_SAVE_DIR=${SWANLAB_SAVE_DIR} \
                 /opt/conda/envs/torch-base/bin/python -c 'import swanlab'" 2>/dev/null && echo ok)
        fi
        if [ "${ok}" != "ok" ]; then
            echo "ERROR: ${ip} 上 \`import swanlab\` 失败。两种原因，先分清再动手：" >&2
            echo "  1) 真没装      -> pip install swanlab（每个节点都要，TaskRunner 可能落在任一节点）" >&2
            echo "  2) 装了但 import 就炸 -> 多半是 SWANLAB_SAVE_DIR 指向不存在的父目录。" >&2
            echo "     判据：SWANLAB_SAVE_DIR=/tmp/x python -c 'import swanlab' 能过，裸跑不能。" >&2
            echo "     本脚本已无条件覆盖成 ${SWANLAB_SAVE_DIR}；若还失败，看完整 traceback：" >&2
            echo "       ssh ${ip} '/opt/conda/envs/torch-base/bin/python -c \"import swanlab\"'" >&2
            exit 1
        fi
    done
fi

cat <<EOF

============ 启动配置 ============
  域         : ${WHICH}  (${TRAIN_SUBDIR})
  拓扑       : ${NNODES} 机 × ${NGPUS_PER_NODE} 卡 = ${WORLD_SIZE}
  节点        : ${NODE_IPS[*]}   (head=${HEAD_IP})
  步数       : ${TOTAL_TRAINING_STEPS}   batch ${TRAIN_BATCH_SIZE} × n=${ROLLOUT_N}
  实验名      : ${EXPERIMENT_NAME}
  sandbox    : ${SANDBOX_FUSION_URL}  (每节点各自的 localhost)
  logger     : ${LOGGER}
  swanlab    : ${SWANLAB_WEB_HOST}  (mode=${SWANLAB_MODE})
  ckpt       : ${OUTPUT_ROOT}/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}
               world_size=${WORLD_SIZE} 分片，与其它卡数的产物不通用
  日志       : ${LOG}
==================================

EOF

# setsid 彻底脱离终端：nohup 只挡 SIGHUP，后台任务仍在同一进程组，
# 工具/SSH 超时的 SIGTERM 会连坐整组（见 overview.md 5.3）。
setsid nohup bash "${TRAIN_SH}" \
    "${EXTRA_OVERRIDES[@]}" \
    "trainer.logger=${LOGGER}" \
    "actor_rollout_ref.actor.checkpoint.save_contents=${CKPT_CONTENTS}" \
    "$@" \
    > "${LOG}" 2>&1 < /dev/null & disown

echo "已启动。"
echo "  跟踪日志 : tail -f ${LOG}"
echo "  reward   : python summarize_rollout.py \$OUTPUT_ROOT/rollout_data/${EXPERIMENT_NAME}"
echo "  swanlab  : ${SWANLAB_WEB_HOST}"
