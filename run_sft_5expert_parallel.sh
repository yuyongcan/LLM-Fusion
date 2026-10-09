#!/usr/bin/env bash
# 5 域 SFT **并行**训练：把 16 个节点切成 5 个互不相交的组，每组独立跑一个域的 SFT。
#
# 与 run_sft_5expert_serial.sh 的关系：两者是**同一件事的两种排布**，共用
# run_sft_multinode.sh 作为单组启动器。
#   串行 —— 每个域独占全部 16 机 128 卡，跑完一个再跑下一个。
#   并行 —— 按各域行数把节点切开，5 个 torchrun 同时跑。
#
#   setsid nohup bash run_sft_5expert_parallel.sh > <log> 2>&1 < /dev/null & disown
#
# ⚠️ 必须用 setsid：本脚本要连续跑 3~4 天，用普通后台会在终端断开/命令超时时被连坐杀掉
#    （2026-09-28 实测：前台长命令被中断时，那一组 32 卡的 Ray 集群整个没了）。
#
# ---------------------------------------------------------------------------------
# ⚠️ 并行为什么**不比串行快**（先读这段再决定用哪个）
#
# 在「每卡吞吐与 dp 规模无关」的假设下，并行永远不快于串行：
#     makespan_parallel = max_d(W_d / p_d)  >=  sum_d(W_d / P) = makespan_serial
# 等号只在 p_d 严格正比于 W_d 时成立，而节点只能整数个地切，所以必然有零头。
# 本数据上最优整数分配的理论损耗约 21%（串行 64.5h vs 并行 78h，见文件末尾时长对照）。
#
# 仍然值得选并行的三个理由：
#   1. 单组 8~32 卡落在**已实测健康**的区间（32 ranks × 205 steps 零 OOM，73-80GB
#      reserved）；128 卡单作业在本集群从未跑过，跨机 IB 的实际 MFU 是未知量。若 128 卡
#      FSDP 的 MFU 比 32 卡差 25% 以上，并行的劣势立刻反转。
#   2. 故障隔离：一个域挂掉不影响另外四个，且另外四个的进度不丢。串行跑到第三天挂掉，
#      后面两个域就完全没有产物。
#   3. 每组是独立的 chief + rendezvous，任一域失败可 DOMAINS= 单独重跑。
#
# 若你就要「最快拿齐 5 个专家」且愿意赌 128 卡扩展性 —— 用 run_sft_5expert_serial.sh。
# ---------------------------------------------------------------------------------
#
# 可调旋钮（全部环境变量）：
#   DOMAINS="agent math"        只跑部分域（续跑用）
#   PLAN_ONLY=1                 只算分配方案并打印，不过滤不训练
#   SKIP_FILTER=1               跳过长度过滤（会撞 truncation=error，不推荐）
#   TARGET_BS=5120              每组全局 batch 会被取整到该组 world_size 的整数倍
#   FILTER_WORLD_SIZE=128       过滤并行度（CPU only，不占卡）
#   MAX_CKPT_TO_KEEP=2          每组保留的 ckpt 数
# ---------------------------------------------------------------------------------
set -uo pipefail

B=/taijifs_sz34/share_304169095/hunyuan/yongcanyu
REPO="${B}/code/LLM-Fusion"
cd "${REPO}"

: "${NODE_IP_LIST:?NODE_IP_LIST 未设置 —— 不在 Taiji 节点上？}"
mapfile -t NODE_IPS < <(echo "${NODE_IP_LIST}" | tr ',' '\n' | cut -d: -f1 | grep .)
NNODES_TOTAL=${#NODE_IPS[@]}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-${HOST_GPU_NUM:-8}}

SFT_DATA_ROOT=${SFT_DATA_ROOT:-${B}/data/M2RL/sft_train}
SFT_BASE_MODEL=${SFT_BASE_MODEL:-${B}/checkpoints/Qwen/Qwen3-4B-Base}
MAX_LENGTH=${MAX_LENGTH:-16384}

# ---- SFT 超参（与 run_sft_multinode.sh 默认一致，显式写出便于审计）----
TARGET_BS=${TARGET_BS:-5120}
export LR=${LR:-5e-5}
export OPTIMIZER=${OPTIMIZER:-AdamW}
export CLIP_GRAD=${CLIP_GRAD:-1.0}
export MAX_TOKEN_LEN_PER_GPU=${MAX_TOKEN_LEN_PER_GPU:-65536}
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
export SAVE_FREQ=${SAVE_FREQ:-200}
# 5 组同时写盘，每组只留 2 个：5 × 2 × ~48GB ≈ 480GB。想省一半盘传 '["model"]'，
# 代价是中断后优化器状态（Adam 二阶矩）从头。
export CKPT_CONTENTS=${CKPT_CONTENTS:-'["model","optimizer"]'}
export MAX_CKPT_TO_KEEP=${MAX_CKPT_TO_KEEP:-2}
export PROJECT_NAME=${PROJECT_NAME:-llm_fusion_sft}

DRY_RUN=${DRY_RUN:-0}
PLAN_ONLY=${PLAN_ONLY:-0}
SKIP_FILTER=${SKIP_FILTER:-0}
DOMAINS=${DOMAINS:-"agent math science code if"}
FILTER_WORLD_SIZE=${FILTER_WORLD_SIZE:-$(( NNODES_TOTAL * NGPUS_PER_NODE ))}

LOG_DIR=${B}/logs/llm_fusion
STATE_DIR=${LOG_DIR}/.sft_parallel_$(date +%Y%m%d_%H%M%S)
mkdir -p "${LOG_DIR}" "${STATE_DIR}"

SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=15"
PYTHON=/opt/conda/envs/torch-base/bin/python

log() { echo "[$(date +%m-%d\ %H:%M:%S)] $*" | tee -a "${LOG_DIR}/sft_5expert_parallel_$(date +%Y%m%d_%H%M%S).log"; }

# ================== 门禁 1：所有卡的占用必须归零 ==================
wait_gpus_free() {
    local max_wait_min=${1:-360} waited=0
    while true; do
        local busy=""
        for ip in "${NODE_IPS[@]}"; do
            local n
            n=$(${SSH} "${ip}" "nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l" 2>/dev/null)
            [ "${n:-0}" -ne 0 ] && busy+=" ${ip}(${n})"
        done
        if [ -z "${busy}" ]; then
            log "所有 ${#NODE_IPS[@]} 个节点无占卡进程，可以开始。"
            return 0
        fi
        if [ "${waited}" -ge "${max_wait_min}" ]; then
            log "ERROR: 等待 ${max_wait_min} 分钟后仍有节点占卡：${busy}" >&2
            return 1
        fi
        log "等待卡释放（已等 ${waited} min）：${busy}"
        sleep 300
        waited=$(( waited + 5 ))
    done
}

# ================== 门禁 2：每个节点的 Python 环境 ==================
# verl 源码在共享盘，但「这个环境里装了 verl」记在节点本地 site-packages 的
# __editable__ .pth 里 —— Pod 换了就得重做 pip install -e . --no-deps（HANDOFF §1）。
check_env() {
    local bad=()
    for ip in "${NODE_IPS[@]}"; do
        if ! ${SSH} "${ip}" "PATH=/opt/conda/envs/torch-base/bin:\$PATH python -c 'import verl, torch, transformers, pyarrow' >/dev/null 2>&1"; then
            bad+=("${ip}")
        fi
    done
    if [ "${#bad[@]}" -gt 0 ]; then
        log "ERROR: 这些节点的 Python 环境不完整：${bad[*]}" >&2
        log "       换 Pod 后需要在每个节点跑一次：pip install -e . --no-deps" >&2
        return 1
    fi
    log "Python 环境：${#NODE_IPS[@]} 个节点 OK"
    return 0
}

# ================== 每域的数据准备 ==================
# 原始 Nemotron 文件里有超过 16384 token 的样本，不过滤会在训练时被截断
# （data.truncation=error 会直接报错）。过滤脚本用 M2RLSFTDataset 自己算长度。
prepare_data() {
    local dom="$1"; shift
    local srcs=("$@")
    local merged="${SFT_DATA_ROOT}/${dom}.parquet"
    local source="${merged}"

    if [ "${#srcs[@]}" -eq 1 ]; then
        source="${srcs[0]}"
    elif [ ! -f "${merged}" ]; then
        log "[${dom}] 合并 ${#srcs[@]} 个源文件 -> $(basename "${merged}")"
        ${PYTHON} - "${merged}" "${srcs[@]}" <<'PY'
import sys
import pyarrow.parquet as pq
out, files = sys.argv[1], sys.argv[2:]
writer = None
for f in files:
    pf = pq.ParquetFile(f)
    for batch in pf.iter_batches(batch_size=20000):
        if writer is None:
            writer = pq.ParquetWriter(out, batch.schema)
        writer.write_batch(batch)
    print("  merged", f, pf.metadata.num_rows)
writer.close()
PY
    fi

    local filtered="${SFT_DATA_ROOT}/${dom}.le${MAX_LENGTH}.parquet"
    if [ -f "${filtered}" ]; then
        log "[${dom}] 过滤数据已存在：$(basename "${filtered}")"
        echo "${filtered}"
        return 0
    fi
    if [ "${SKIP_FILTER}" = "1" ]; then
        log "[${dom}] SKIP_FILTER=1，直接用源数据（可能撞 truncation=error）"
        echo "${source}"
        return 0
    fi

    local parts="${SFT_DATA_ROOT}/.${dom}.le${MAX_LENGTH}.parts"
    log "[${dom}] 按真实 token 长度过滤（${FILTER_WORLD_SIZE} 进程并行）..."
    mkdir -p "${parts}"
    # 每个节点起 NGPUS_PER_NODE 个进程，rank 偏移量按节点在 NODE_IP_LIST 中的位置
    for ni in "${!NODE_IPS[@]}"; do
        local ip="${NODE_IPS[$ni]}"
        ${SSH} "${ip}" "cd ${REPO} && for r in \$(seq 0 $(( NGPUS_PER_NODE - 1 ))); do \
            RANK=\$(( ${ni} * ${NGPUS_PER_NODE} + r )); \
            nohup ${PYTHON} examples/llm_fusion/sft/filter_by_token_length.py filter \
                --source '${source}' --output-dir '${parts}' \
                --model-path '${SFT_BASE_MODEL}' \
                --rank \${RANK} --world-size ${FILTER_WORLD_SIZE} \
                --max-length ${MAX_LENGTH} \
                > '${parts}/rank\${RANK}.log' 2>&1 & done; wait" \
            > "${LOG_DIR}/filter_${dom}_node${ni}.log" 2>&1 &
    done
    wait
    log "[${dom}] 各 rank 过滤完成，合并分片..."
    ${PYTHON} examples/llm_fusion/sft/filter_by_token_length.py merge \
        --source "${source}" --parts-dir "${parts}" \
        --output "${filtered}" --world-size "${FILTER_WORLD_SIZE}" \
        2>&1 | tail -3 | tee -a "${LOG_DIR}/sft_5expert_parallel.log"
    rm -rf "${parts}"
    echo "${filtered}"
}

# ================== 分配规划 ==================
# 穷举 NNODES_TOTAL 拆成 |DOMAINS| 个正整数的一切组合（16 拆 5 只有 1365 种），
# 取 min max(rows_d / nodes_d)。贪心（每次给当前最重的域加一台）不是最优，这里直接穷举。
plan_alloc() {
    local -a doms=("$@")
    ${PYTHON} - "${NNODES_TOTAL}" "${doms[@]}" <<'PY'
import sys
total = int(sys.argv[1]); doms = sys.argv[2:]
rows = {}
for d in doms:
    p = __import__('os').environ.get('ROWS_%s' % d.upper())
    rows[d] = int(p)

def comps(n, k):
    if k == 1:
        yield (n,)
        return
    for i in range(1, n - k + 2):
        for rest in comps(n - i, k - 1):
            yield (i,) + rest

best = None
for c in comps(total, len(doms)):
    m = max(rows[d] / c[i] for i, d in enumerate(doms))
    if best is None or m < best[0]:
        best = (m, c)
m, c = best
ideal = sum(rows.values()) / total
print("PLAN overhead=%.3f  (max %.4fM rows/node vs ideal %.4fM, +%.1f%%)"
      % (m / ideal, m / 1e6, ideal / 1e6, (m / ideal - 1) * 100))
for i, d in enumerate(doms):
    print("PLAN %s %d %.6f %.3f" % (d, c[i], rows[d], rows[d] / c[i] / 1e6))
PY
}

# ================== 生成并下发每组的启动包装 ==================
# 组与组之间节点互不相交，所以每组必须有自己的 chief：run_sft_multinode.sh 里
# `CHIEF=${CHIEF_IP:-NODE_IPS[0]}` 且 rank0 在本地前台跑 —— 平台在所有节点上都
# 注入了 CHIEF_IP=28.45.3.115，不覆盖的话「NODE_IPS[0] != CHIEF」门禁会直接拒绝启动。
# 因此每组 ssh 到自己的第一台机器上跑，并 export 一组私有拓扑变量。
launch_group() {
    local dom="$1" port="$2" train_file="$3"; shift 3
    local ips=("$@")
    local n=${#ips[@]}
    local ws=$(( n * NGPUS_PER_NODE ))
    # batch 取整到 world_size 的倍数：门禁要求 TRAIN_BATCH_SIZE % WORLD_SIZE == 0，
    # 24 卡这类不能整除 5120 的组用就近值（5112）。
    local mult=$(( (TARGET_BS + ws / 2) / ws ))
    [ "${mult}" -lt 1 ] && mult=1
    local bs=$(( mult * ws ))
    local head=${ips[0]}
    local iplist; iplist=$(printf '%s:8,' "${ips[@]}"); iplist=${iplist%,}
    local exp="sft_${dom}_qwen3_4b_base_${n}x${NGPUS_PER_NODE}"
    local log="${LOG_DIR}/sft_par_${dom}_$(date +%Y%m%d_%H%M%S).log"
    local status="${STATE_DIR}/${dom}.status"
    local wrap="${STATE_DIR}/wrap_${dom}.sh"

    cat > "${wrap}" <<EOF
#!/usr/bin/env bash
# 自动生成 by run_sft_5expert_parallel.sh —— 域 ${dom} 的组启动器
cd ${REPO}
export NODE_IP_LIST='${iplist}'
export CHIEF_IP='${head}'
export NNODES=${n}
export NGPUS_PER_NODE=${NGPUS_PER_NODE}
export MASTER_PORT=${port}
export TRAIN_BATCH_SIZE=${bs}
export SFT_BASE_MODEL='${SFT_BASE_MODEL}'
export SFT_DATA_ROOT='${SFT_DATA_ROOT}'
export SFT_TRAIN='${train_file}'
export MAX_LENGTH=${MAX_LENGTH}
export MAX_TOKEN_LEN_PER_GPU='${MAX_TOKEN_LEN_PER_GPU}'
export LR='${LR}'
export OPTIMIZER='${OPTIMIZER}'
export CLIP_GRAD='${CLIP_GRAD}'
export TOTAL_EPOCHS=${TOTAL_EPOCHS}
export SAVE_FREQ=${SAVE_FREQ}
export CKPT_CONTENTS='${CKPT_CONTENTS}'
export MAX_CKPT_TO_KEEP=${MAX_CKPT_TO_KEEP}
export PROJECT_NAME=${PROJECT_NAME}
export EXPERIMENT_NAME=${exp}
export SWANLAB_LOG_DIR=/tmp/swanlog_sft_${dom}
export SWANLAB_SAVE_DIR=/tmp/swanlab_sft_${dom}
bash run_sft_multinode.sh > '${log}' 2>&1
echo \$? > '${status}'
EOF

    log "[${dom}] 启动 ${n} 机 × ${NGPUS_PER_NODE} 卡 = ${ws}  (bs=${bs}, chief=${head}:${port})"
    [ "${DRY_RUN}" = "1" ] && { echo "  DRY_RUN: ${wrap}"; return 0; }

    # 远程 detach：stdout/stderr 全部重定向后 ssh 通道才会立刻关闭，否则会挂住等输出。
    ${SSH} "${head}" "setsid nohup bash ${wrap} </dev/null >/dev/null 2>&1 &" \
        >/dev/null 2>&1 || log "[${dom}] WARNING: 下发到 ${head} 失败"
    sleep 2
}

# ================== 拆除 ==================
# 残留的 torchrun 会一直占着整机的 8 张卡；下次启动会以 OOM 或 MASTER_PORT 被占的形式
# 失败，而报错与真实原因毫无关系。"[v]erl" 的方括号是 load-bearing 的：没有它 pattern
# 会匹配到 ssh 自己 spawn 的 bash -c 包装进程，于是 pkill 杀掉自己的父 shell。
cleanup() {
    trap - EXIT INT TERM HUP
    log "拆除所有组的 torchrun ..."
    for ip in "${NODE_IPS[@]}"; do
        ${SSH} "${ip}" 'pkill -f "[v]erl.trainer.sft_trainer" 2>/dev/null || true
             sleep 2
             pkill -9 -f "[v]erl.trainer.sft_trainer" 2>/dev/null || true' \
            >/dev/null 2>&1 &
    done
    wait
}

# ---- 域 -> 源文件 映射 ----
declare -A SRC=(
    [math]="${SFT_DATA_ROOT}/Nemotron-Math-v2.parquet ${SFT_DATA_ROOT}/Nemotron-Math-Proofs-v1.parquet"
    [code]="${SFT_DATA_ROOT}/Nemotron-Competitive-Programming-v1.parquet"
    [science]="${SFT_DATA_ROOT}/Nemotron-Science-v1.parquet"
    [if]="${SFT_DATA_ROOT}/Nemotron-Instruction-Following-Chat-v1.parquet"
    [agent]="${SFT_DATA_ROOT}/Nemotron-Agentic-v1.parquet"
)

# shellcheck disable=SC2206
read -r -a DOM_ARR <<< "${DOMAINS}"
for d in "${DOM_ARR[@]}"; do
    [ -n "${SRC[$d]:-}" ] || { echo "未知域：${d}（可选：${!SRC[*]}）" >&2; exit 1; }
done

cat <<EOF

======== 5 域 SFT 并行训练 ========
  拓扑     : ${NNODES_TOTAL} 机 × ${NGPUS_PER_NODE} 卡 = $(( NNODES_TOTAL * NGPUS_PER_NODE ))
  base     : ${SFT_BASE_MODEL}
  数据根   : ${SFT_DATA_ROOT}
  域顺序   : ${DOMAINS}
  目标 batch: ${TARGET_BS}（每组取整到 world_size 的倍数）
  状态目录 : ${STATE_DIR}
====================================

EOF

# ---------------- 阶段 1：过滤全部域 ----------------
declare -A TRAIN_FILE=()
declare -A SRC_ROWS=()
if [ "${SKIP_FILTER}" = "1" ] || [ "${PLAN_ONLY}" = "1" ]; then
    log "SKIP_FILTER/PLAN_ONLY：跳过过滤阶段"
else
    wait_gpus_free 360 || exit 1
    check_env || exit 1
fi

for dom in "${DOM_ARR[@]}"; do
    # shellcheck disable=SC2206
    read -r -a files <<< "${SRC[$dom]}"
    SRC_ROWS[$dom]=$(${PYTHON} -c "
import sys, pyarrow.parquet as pq
print(sum(pq.ParquetFile(f).metadata.num_rows for f in sys.argv[1:]))" "${files[@]}" 2>/dev/null)
    f="${SFT_DATA_ROOT}/${dom}.le${MAX_LENGTH}.parquet"
    if [ -f "${f}" ]; then
        TRAIN_FILE[$dom]="${f}"
    elif [ "${PLAN_ONLY}" = "1" ]; then
        # 只做规划：不落盘，行数退回源文件总和（math 是两个文件，不能用第一个）
        TRAIN_FILE[$dom]=""
    elif [ "${SKIP_FILTER}" != "1" ]; then
        TRAIN_FILE[$dom]="$(prepare_data "${dom}" "${files[@]}" | tail -1)"
    else
        TRAIN_FILE[$dom]="${files[0]}"
    fi
done

# ---------------- 阶段 2：按真实行数分配节点 ----------------
# 优先用过滤后的行数；还没过滤（PLAN_ONLY）就退回源行数，并在输出里标明。
for dom in "${DOM_ARR[@]}"; do
    if [ -n "${TRAIN_FILE[$dom]:-}" ] && [ -f "${TRAIN_FILE[$dom]}" ]; then
        n=$(${PYTHON} -c "import pyarrow.parquet as pq; print(pq.ParquetFile('${TRAIN_FILE[$dom]}').metadata.num_rows)" 2>/dev/null)
        export "ROWS_${dom^^}=$n"
    else
        export "ROWS_${dom^^}=${SRC_ROWS[$dom]}"
    fi
done

PLAN=$(plan_alloc "${DOM_ARR[@]}")
echo "${PLAN}" | tee -a "${LOG_DIR}/sft_5expert_parallel.log"

if [ "${PLAN_ONLY}" = "1" ]; then
    echo
    echo "PLAN_ONLY=1：到此为止，不过滤不训练。"
    exit 0
fi

# 解析 PLAN 行：PLAN <dom> <nodes> <rows> <M rows/node>
declare -A NODES_FOR=()
while read -r tag dom n _rows _load; do
    [ "${tag}" = "PLAN" ] || continue
    case "${dom}" in overhead*) continue;; esac
    NODES_FOR[$dom]=$n
done <<< "$(echo "${PLAN}" | grep '^PLAN ')"

total_assigned=0
for dom in "${DOM_ARR[@]}"; do total_assigned=$(( total_assigned + ${NODES_FOR[$dom]:-0} )); done
if [ "${total_assigned}" -ne "${NNODES_TOTAL}" ]; then
    log "ERROR: 分配节点数 ${total_assigned} != ${NNODES_TOTAL}" >&2
    exit 1
fi

# ---------------- 阶段 3：切片并下发 ----------------
trap 'cleanup; exit 130' INT TERM HUP
offset=0
port_base=29500
for dom in "${DOM_ARR[@]}"; do
    n=${NODES_FOR[$dom]}
    group_ips=("${NODE_IPS[@]:${offset}:${n}}")
    offset=$(( offset + n ))

    # row group 数必须 >= world_size，否则会有 rank 分到 0 个分片
    rg=$(${PYTHON} -c "import pyarrow.parquet as pq; print(pq.ParquetFile('${TRAIN_FILE[$dom]}').num_row_groups)" 2>/dev/null)
    ws=$(( n * NGPUS_PER_NODE ))
    if [ -n "${rg}" ] && [ "${rg}" -lt "${ws}" ]; then
        log "ERROR: [${dom}] row groups=${rg} < world_size=${ws}" >&2
        exit 1
    fi
    log "[${dom}] 节点 ${group_ips[*]}  (row_groups=${rg} / ${ws} ranks)"
    launch_group "${dom}" "$(( port_base + offset ))" "${TRAIN_FILE[$dom]}" "${group_ips[@]}"
done

if [ "${DRY_RUN}" = "1" ]; then
    echo "DRY_RUN=1：包装脚本已生成于 ${STATE_DIR}/wrap_*.sh，未下发。"
    exit 0
fi

# ---------------- 阶段 4：等待 ----------------
log "5 个组已下发，开始轮询（每 5 分钟一次）..."
while true; do
    pending=0
    for dom in "${DOM_ARR[@]}"; do
        [ -f "${STATE_DIR}/${dom}.status" ] || pending=$(( pending + 1 ))
    done
    [ "${pending}" = "0" ] && break
    log "仍有 ${pending} 个域在跑：$(for d in "${DOM_ARR[@]}"; do [ -f "${STATE_DIR}/${d}.status" ] || printf '%s ' "$d"; done)"
    sleep 300
done

echo
log "===== 结果 ====="
rc=0
for dom in "${DOM_ARR[@]}"; do
    st=$(cat "${STATE_DIR}/${dom}.status" 2>/dev/null)
    if [ "${st}" = "0" ]; then
        log "  ${dom}: OK   ckpt -> ${B}/runs/llm_fusion/checkpoints/${PROJECT_NAME}/sft_${dom}_qwen3_4b_base_*"
    else
        log "  ${dom}: FAILED (退出码 ${st:-unknown})   日志 ${LOG_DIR}/sft_par_${dom}_*.log"
        rc=1
    fi
done
exit "${rc}"
