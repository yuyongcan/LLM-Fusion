#!/usr/bin/env bash
# 多机多卡 SFT 完整训练入口：Qwen3-4B-Base -> chat/reasoning 模型（phase 1）。
#
# 形态同 run_train_multinode.sh：拓扑推导、前置门禁、环境注入收在这里，超参交给
# examples/llm_fusion/sft/train_sft_qwen3_4b_base.sh。
#
#   bash run_sft_multinode.sh [hydra overrides...]
#
#   # 冒烟（先打通链路）
#   TOTAL_TRAINING_STEPS=20 SAVE_FREQ=10 MAX_CKPT_TO_KEEP=2 \
#   EXPERIMENT_NAME=sft_smoke_qwen3_4b_base bash run_sft_multinode.sh
#
#   # 完整 1 epoch —— 不要传 TOTAL_TRAINING_STEPS：留空(null)时由
#   # TOTAL_EPOCHS=1 x len(dataloader) 推导（sft_trainer.py），改 batch size 会自动跟着变。
#   # 写死步数等于把「一个 epoch」钉在某个特定 batch size 上，换 bs 就错。
#   SAVE_FREQ=200 bash run_sft_multinode.sh
#
# ---------------------------------------------------------------------------------
# ⚠️ 为什么这里**需要** pssh fan-out，而 run_train_multinode.sh 不需要
#
# HANDOFF.md §3 和 overview.md §4c 说「别照抄 MOPD 那套 fan-out 启动器」。那条规则
# 针对的是 **RL**：verl.trainer.main_ppo 是 Ray-based，chief 提交一次，Ray 自己把
# WorkerDict 调度到全集群，所以只要先 setup_ray_cluster.sh 把集群拉起来。
#
# 但 verl.trainer.sft_trainer 是**纯 torchrun**。`torchrun --nnodes=N` 只 fork 本地
# 8 个 rank，然后阻塞在 rendezvous 等其余节点的 rank 出现；不在每台机器各起一份并
# 分配不同的 --node_rank，就会一直挂到 MASTER_PORT 超时。而本平台
# TAIJI_BASIC_EXEC_START_IN_ALL_MPI_PODS=false，只有 chief 执行 start_cmd，平台也
# 不会替我们起。所以 fan-out 必须在这里做。
#
# verl/trainer/sft_trainer_ray.py 确实存在、也能保留 head 单点提交，但在本数据量下
# 不可用：它的 _build_dataloader 写死 dp_rank=0/dp_size=1（sft_trainer_ray.py:181-185），
# 即**在 driver 单进程加载全量数据集**（实测 7.7GB/110k 行 -> 12.2M 行约 855GB），
# 且不认 rank_sharded 契约。
#
# 与 RL 启动器的其它结构差异，全部源于 torchrun：
#   - 每台机器 --node_rank 不同，所以逐 host 下发而不是广播同一条命令。rank 用
#     NODE_IP_LIST 中的原始位置（平台的 INDEX 就是这么推的），**不能自行排序**。
#   - rank 之间是对等的，不是 head/worker，所以任一节点失败必须拆掉全部 —— 见
#     cleanup()，残留的 torchrun 会一直占着 8 张卡。
#   - 不需要 ray status 门禁，也不需要 sandbox（SFT 没有 code reward）。
# ---------------------------------------------------------------------------------
set -uo pipefail

B=/taijifs_sz34/share_304169095/hunyuan/yongcanyu
REPO="${B}/code/LLM-Fusion"
cd "${REPO}"

TRAIN_SH="${REPO}/examples/llm_fusion/sft/train_sft_qwen3_4b_base.sh"
[ -f "${TRAIN_SH}" ] || { echo "not found: ${TRAIN_SH}" >&2; exit 2; }

# ---- 拓扑：从平台变量推导，不硬编码 IP ----
# NODE_IP_LIST 是 "ip:slots,ip:slots,..."，权威来源。
# ⚠️ 不要用 NODE_NUM 当节点数：该平台上它是总 GPU 数，不是机器数。
: "${NODE_IP_LIST:?NODE_IP_LIST 未设置 —— 不在 Taiji 节点上？}"
mapfile -t NODE_IPS < <(echo "${NODE_IP_LIST}" | tr ',' '\n' | cut -d: -f1 | grep .)
CHIEF="${CHIEF_IP:-${NODE_IPS[0]}}"
NNODES=${NNODES:-${HOST_NUM:-${#NODE_IPS[@]}}}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-${HOST_GPU_NUM:-8}}
MASTER_PORT=${MASTER_PORT:-29500}
WORLD_SIZE=$(( NNODES * NGPUS_PER_NODE ))

# rank 0 必须是 chief：那是其余所有节点 rendezvous 时拨的地址。
if [ "${NODE_IPS[0]}" != "${CHIEF}" ]; then
    echo "ERROR: NODE_IP_LIST[0]=${NODE_IPS[0]} 但 CHIEF_IP=${CHIEF}；拒绝猜测 rank 映射。" >&2
    exit 1
fi

# ---- 根路径 ----
# ⚠️ env_llm_fusion.sh 里 NNODES/NGPUS_PER_NODE 是硬 export（不是 ${:-}），
#    所以先存下推导值，source 之后再覆盖回去。
_NNODES=${NNODES} _NGPUS=${NGPUS_PER_NODE}
source env_llm_fusion.sh
export NNODES=${_NNODES} NGPUS_PER_NODE=${_NGPUS}

# SFT 从 *Base* 起跑，不是 RL 用的 Instruct —— 见 paths.sh 里的说明。
export SFT_BASE_MODEL=${SFT_BASE_MODEL:-${B}/checkpoints/Qwen/Qwen3-4B-Base}
export SFT_DATA_ROOT=${SFT_DATA_ROOT:-${B}/data/M2RL/sft_train}
export SFT_TRAIN=${SFT_TRAIN:-${SFT_DATA_ROOT}/train.le16384.parquet}

# ---- 论文超参（见 train_sft_qwen3_4b_base.sh 头部的 AdamW 实测依据）----
# 5120：64 卡实测比 1280 更优（MFU 0.46 vs 0.383，一个 epoch 5.4 天 vs 6.5 天，
# 每 token 7.71us vs 9.33us，reserved 81.6GB 无 OOM）。use_dynamic_bsz=true 下峰值
# 显存由 MAX_TOKEN_LEN_PER_GPU 决定，不由总 batch 决定 —— 加大 batch 只是让
# micro-batch 顺序变多。
# ⚠️ 但 sft_loss 是 token-mean，梯度幅度与 batch 无关：1280->5120 把 update 数从
#    10000 砍到 2500，同 LR 下只走 1/4 的更新距离。本次刻意保持 lr=5e-5（与已验证
#    健康的那条曲线同 LR），代价是最终 loss 大概率高于 bs=1280 能达到的水平。
#    想补偿就设 LR=1e-4（sqrt 缩放）。
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-5120}
export LR=${LR:-5e-5}
export OPTIMIZER=${OPTIMIZER:-AdamW}
export MAX_LENGTH=${MAX_LENGTH:-16384}
export MAX_TOKEN_LEN_PER_GPU=${MAX_TOKEN_LEN_PER_GPU:-65536}
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
export SAVE_FREQ=${SAVE_FREQ:-200}
export PROJECT_NAME=${PROJECT_NAME:-llm_fusion_sft}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-sft_qwen3_4b_base_${NNODES}x${NGPUS_PER_NODE}}

SAVE_PATH=${SAVE_PATH:-${OUTPUT_ROOT}/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}}
mkdir -p "${SAVE_PATH}"
export HYDRA_RUN_DIR=${HYDRA_RUN_DIR:-${SAVE_PATH}/hydra}

LOG_DIR=${LOG_DIR:-${SAVE_PATH}/launch_logs}
mkdir -p "${LOG_DIR}"
START_TIME="$(date +%y%m%d-%H-%M-%S)"

# ---- 代理绕行 ----
# torchrun 的 rendezvous 和 NCCL 都是集群内流量，走 star-proxy 会直接坏掉。
CLUSTER_NO_PROXY=$(printf '%s' "${NODE_IP_LIST}" | tr ',' '\n' | sed 's/:[0-9][0-9]*$//' \
                   | awk 'NF && !seen[$0]++' | paste -sd, -)
EXPORT_NO_PROXY=".woa.com,.oa.com,mirrors.cloud.tencent.com,localhost,127.0.0.1,0.0.0.0,${CLUSTER_NO_PROXY}"
export no_proxy="${EXPORT_NO_PROXY}" NO_PROXY="${EXPORT_NO_PROXY}"

# ---- NCCL ----
# 本机栈（NCCL 2.28.9 + CUDA 13.0 + driver 535.247.01 + H20 + mlx5_bond_1..8）在
# GPUDirect RDMA 注册 GPU 显存的路径上不兼容，跨机 FSDP init 会报
# `ibv_reg_mr_iova2 ... Invalid argument`。NCCL_NET_GDR_LEVEL=0 关掉显存直连、保留
# IB（数据经 pinned host memory 中转）。
# ⚠️ 不要图省事设 NCCL_IB_DISABLE=1 —— 那会完全退回 TCP，更慢。
#
# 注：这里的 export 对 SFT 是**真正生效**的。RL 那边要靠 setup_ray_cluster.sh 在
# `ray start` 时注入，是因为 verl 给 Ray worker 下发的 runtime_env 是写死的白名单
# （verl/trainer/constants_ppo.py，NCCL_NET_GDR_LEVEL 不在其中）。torchrun 没有这层，
# 环境变量由下面的 pssh 显式带过去。
export NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL:-0}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-bond1}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
# expandable_segments 让 allocator 归还非连续空闲块；不开的话实测 reserved(55.8GB)
# 是 allocated(11.4GB) 的 ~5 倍，全是碎片。
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# ---- SwanLab ----
# ⚠️ SWANLAB_LOG_DIR / SWANLAB_SAVE_DIR 必须是节点本地路径，不能放共享盘 —— 详见
#    HANDOFF.md §4（swanlab 的 safe_mkdir 用 tempfile 探测可写性，共享盘不支持，
#    报成误导性的 "FILESYSTEM may be slow or remote"）。
export SWANLAB_WEB_HOST="${SWANLAB_WEB_HOST:-http://taiji-exp.taiji.woa.com/}"
export SWANLAB_API_HOST="${SWANLAB_API_HOST:-http://taiji-exp.taiji.woa.com/api}"
export SWANLAB_MODE="${SWANLAB_MODE:-cloud}"
export SWANLAB_LOG_DIR="${SWANLAB_LOG_DIR:-/tmp/swanlog_llm_fusion_sft}"
export SWANLAB_SAVE_DIR=/tmp/swanlab_llm_fusion_sft
mkdir -p "${SWANLAB_LOG_DIR}" "${SWANLAB_SAVE_DIR}"
# SFT 的 Tracking 只在 logging rank 构造，所以 swanlab 只需要 chief 装。
LOGGER=${LOGGER:-'["console","swanlab","tensorboard"]'}
export LOGGER

# ================== 前置门禁 ==================
# 都是「不检查就会静默跑错」的。
export PATH="/opt/conda/envs/torch-base/bin:${PATH}"

echo "[check] batch size 能被 world_size 整除 ..."
if [ $(( TRAIN_BATCH_SIZE % WORLD_SIZE )) -ne 0 ]; then
    echo "ERROR: TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} 不能被 world_size=${WORLD_SIZE} 整除。" >&2
    exit 1
fi

echo "[check] 数据文件与 row group 数 ..."
# dataset.py 按 row_groups[rank::world_size] 分片，row group 少于 rank 数时会有 rank
# 拿到 0 个（它会直接报错，但等到那时已经浪费了一次启动）。
if [ ! -s "${SFT_TRAIN}" ]; then
    echo "ERROR: 训练数据不存在：${SFT_TRAIN}" >&2
    echo "       用 examples/llm_fusion/sft/filter_by_token_length.py 生成。" >&2
    exit 1
fi
ROW_GROUPS=$(python -c "
import pyarrow.parquet as pq
print(pq.ParquetFile('${SFT_TRAIN}').num_row_groups)" 2>/dev/null)
if [ -z "${ROW_GROUPS}" ]; then
    echo "ERROR: 读不到 ${SFT_TRAIN} 的 parquet 元数据。" >&2
    exit 1
fi
if [ "${ROW_GROUPS}" -lt "${WORLD_SIZE}" ]; then
    echo "ERROR: row groups=${ROW_GROUPS} < world_size=${WORLD_SIZE}，会有 rank 拿到 0 个分片。" >&2
    exit 1
fi

echo "[check] base 模型的 eos 契约 ..."
# Qwen3-4B-Base 出厂 generation_config.eos_token_id=151643 (<|endoftext|>)，但它的
# chat template 用 <|im_end|> (151645) 收尾。verl 每次 save 都逐字拷贝 base 的
# generation_config，所以没打补丁的话每个 checkpoint 都宣称一个它永不发出的停止符 ——
# 实测 324 条 rollout 全部跑到 30000 token 上限。不报错，只是生成不停。
EOS_OK=$(python -c "
import json
cfg = json.load(open('${SFT_BASE_MODEL}/generation_config.json'))
eos = cfg.get('eos_token_id')
eos = eos if isinstance(eos, list) else [eos]
print('ok' if 151645 in eos else 'bad:%s' % eos)" 2>/dev/null)
if [ "${EOS_OK}" != "ok" ]; then
    echo "ERROR: ${SFT_BASE_MODEL} 的 generation_config.eos_token_id 不含 151645 (<|im_end|>)：${EOS_OK:-读取失败}" >&2
    echo "       产出的 checkpoint 会永不停止生成。先修 base 目录的 eos 契约" >&2
    echo "       （generation_config.json -> [151645, 151643]，tokenizer_config.json" >&2
    echo "        的 eos_token -> '<|im_end|>'，config.json -> 151645），一次修好则" >&2
    echo "       后续每个 checkpoint 都正确。" >&2
    exit 1
fi

echo "[check] worker 可达 ..."
UNREACHABLE=()
for ip in "${NODE_IPS[@]:1}"; do
    ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=10 "${ip}" true \
        >/dev/null 2>&1 || UNREACHABLE+=("${ip}")
done
if [ "${#UNREACHABLE[@]}" -gt 0 ]; then
    echo "ERROR: 这些节点 ssh 不通：${UNREACHABLE[*]}" >&2
    echo "       torchrun 会挂在 rendezvous 直到超时，且没有有用的报错。" >&2
    exit 1
fi

if [ "${LOGGER}" != "${LOGGER/swanlab/}" ]; then
    echo "[check] swanlab（只需 chief：SFT 的 Tracking 只在 logging rank 构造）..."
    # ⚠️ stderr 必须丢掉，不能 2>&1：swanlab import 时会打 FutureWarning，
    #    折进变量会让「装了」也判成「没装」。只看退出码。
    if ! python -c "import swanlab" >/dev/null 2>&1; then
        echo "ERROR: chief 上 \`import swanlab\` 失败。" >&2
        echo "  1) 真没装 -> pip install swanlab" >&2
        echo "  2) 装了但 import 就炸 -> 多半是 SWANLAB_SAVE_DIR 指向不存在的父目录；" >&2
        echo "     本脚本已无条件覆盖成 ${SWANLAB_SAVE_DIR}。" >&2
        echo "  或者：LOGGER='[\"console\",\"tensorboard\"]' 跳过。" >&2
        exit 1
    fi
fi

# ---- 拆除 ----
# 残留在 worker 上的 torchrun 会一直占着那台机的 8 张卡，下一次启动就会以 OOM 或
# MASTER_PORT 被占报错，而错误信息和真实原因毫无关系。
#
# "[v]erl" 的方括号是 load-bearing 的：没有它，pattern 会匹配到 ssh 为运行这条命令
# 自己 spawn 的 `bash -c` 包装进程（pattern 就在那个进程的 argv 里），于是 pkill 杀掉
# 自己的父 shell，ssh 挂住而什么都没拆掉。实测：180s 超时，worker 全部存活。
CLEANUP_DONE=0
cleanup() {
    local status="${1:-0}"
    [ "${CLEANUP_DONE}" = "1" ] && return
    CLEANUP_DONE=1
    trap - EXIT INT TERM HUP

    if [ "${#NODE_IPS[@]}" -gt 1 ]; then
        echo >&2
        echo "[launch] 拆除 workers (status=${status}) ..." >&2
        for ip in "${NODE_IPS[@]:1}"; do
            ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=10 "${ip}" \
                'pkill -f "[v]erl.trainer.sft_trainer" 2>/dev/null || true
                 sleep 2
                 pkill -9 -f "[v]erl.trainer.sft_trainer" 2>/dev/null || true' \
                >/dev/null 2>&1 &
        done
        wait
        echo "[launch] workers 已停止。" >&2
    fi
}
trap 'cleanup 130; exit 130' INT TERM HUP
trap 'cleanup $?' EXIT

# ---- 下发给 worker ----
# ⚠️ 每个训练开关都必须显式转发。MOPD 踩过：chief 用过滤后数据 + 16384，worker 静默
#    用脚本默认的 train.parquet + 32768，四节点参数不一致跑了很久才发现。
TRAIN_ENV_VARS=(
    SFT_BASE_MODEL SFT_DATA_ROOT SFT_TRAIN
    MAX_LENGTH MAX_TOKEN_LEN_PER_GPU MAX_SAMPLES NUM_WORKERS
    TRAIN_BATCH_SIZE LR OPTIMIZER WEIGHT_DECAY WARMUP_RATIO LR_SCHEDULER CLIP_GRAD
    TOTAL_EPOCHS TOTAL_TRAINING_STEPS SP_SIZE USE_LIGER
    SAVE_FREQ MAX_CKPT_TO_KEEP CKPT_CONTENTS
    PROJECT_NAME EXPERIMENT_NAME LOGGER HYDRA_RUN_DIR SFT_SHARD_BY_RANK
    MODEL_PATH TRAIN_DATA_ROOT EVAL_DATA_ROOT OUTPUT_ROOT
    NCCL_NET_GDR_LEVEL NCCL_SOCKET_IFNAME NCCL_IB_DISABLE NCCL_IB_GID_INDEX NCCL_DEBUG
    TORCH_NCCL_ASYNC_ERROR_HANDLING PYTORCH_CUDA_ALLOC_CONF
    SWANLAB_WEB_HOST SWANLAB_API_HOST SWANLAB_API_KEY SWANLAB_MODE
    SWANLAB_LOG_DIR SWANLAB_SAVE_DIR
    DRY_RUN
)
REMOTE_ENV=""
for name in "${TRAIN_ENV_VARS[@]}"; do
    if [ -n "${!name+x}" ]; then
        printf -v value_q '%q' "${!name}"
        REMOTE_ENV+="export ${name}=${value_q}; "
    fi
done
printf -v train_sh_q '%q' "${TRAIN_SH}"
printf -v save_path_q '%q' "${SAVE_PATH}"
printf -v repo_q '%q' "${REPO}"
REMOTE_ARGS=""
for arg in "$@"; do
    printf -v arg_q '%q' "${arg}"
    REMOTE_ARGS+=" ${arg_q}"
done

cat <<EOF

============ 启动配置 ============
  拓扑       : ${NNODES} 机 × ${NGPUS_PER_NODE} 卡 = ${WORLD_SIZE}
  节点        : ${NODE_IPS[*]}   (rank 0 = chief ${CHIEF}, 本终端前台)
  base 模型   : ${SFT_BASE_MODEL}   (eos 契约已校验)
  数据       : ${SFT_TRAIN}
               ${ROW_GROUPS} row groups / ${WORLD_SIZE} ranks
  优化器      : ${OPTIMIZER}  lr=${LR}  bs=${TRAIN_BATCH_SIZE}  clip_grad=${CLIP_GRAD:-1.0}
  步数       : ${TOTAL_TRAINING_STEPS:-null}  (null = 由 1 epoch 决定)
  实验名      : ${EXPERIMENT_NAME}
  logger     : ${LOGGER}
  ckpt       : ${SAVE_PATH}
               world_size=${WORLD_SIZE} 分片，与其它卡数的产物不通用
  worker 日志 : ${LOG_DIR}/rank<N>_${START_TIME}.log
==================================

EOF

for rank in "${!NODE_IPS[@]}"; do
    [ "${rank}" = "0" ] && continue
    ip="${NODE_IPS[$rank]}"
    log="${LOG_DIR}/rank${rank}_${START_TIME}.log"
    echo "[launch] 启动 rank ${rank} @ ${ip}"
    printf -v log_q '%q' "${log}"
    pssh -i -t 0 -H "${ip}" \
        "${REMOTE_ENV}export NNODES=${NNODES} NGPUS_PER_NODE=${NGPUS_PER_NODE} \
                NODE_RANK=${rank} MASTER_ADDR=${CHIEF} MASTER_PORT=${MASTER_PORT} \
                no_proxy='${EXPORT_NO_PROXY}' NO_PROXY='${EXPORT_NO_PROXY}'; \
         cd ${repo_q} && \
         nohup bash ${train_sh_q} ${save_path_q}${REMOTE_ARGS} > ${log_q} 2>&1" \
        >"${LOG_DIR}/pssh_rank${rank}_${START_TIME}.log" 2>&1 &
done

# ---- rank 0 在本终端前台跑 ----
# chief 留前台是 Ctrl-C 能用、训练日志能看见的原因；上面的 EXIT trap 再去收 worker。
echo "[launch] 启动 rank 0（本机，前台）"
echo
export NODE_RANK=0
export MASTER_ADDR="${CHIEF}"
export MASTER_PORT

set +e
# LOGGER 已经 export 了，训练脚本自己会读；不必再作为 override 传一遍。
bash "${TRAIN_SH}" "${SAVE_PATH}" "$@" \
    2>&1 | tee "${LOG_DIR}/rank0_${START_TIME}.log"
status=${PIPESTATUS[0]}
set -e

if [ "${status}" -ne 0 ]; then
    echo >&2
    echo "[launch] rank 0 退出码 ${status}。" >&2
    echo "[launch] worker 日志：${LOG_DIR}/rank*_${START_TIME}.log" >&2
fi
exit "${status}"
