#!/usr/bin/env bash
# 串行训练 5 个域 SFT 专家（phase 1：Qwen3-4B-Base -> chat/reasoning）。
#
# 与 run_sft_multinode.sh 的关系：那个是**单个** SFT run 的站点启动器（torchrun +
# pssh fan-out）。本脚本是它上层的驱动器，按域依次做「准备数据 -> 跑一个 epoch -> 下一个」。
#
#   setsid nohup bash run_sft_5expert_serial.sh > <log> 2>&1 < /dev/null & disown
#
# ⚠️ 必须用 setsid：本脚本要连续跑 5~6 天，用普通后台会在终端断开/命令超时时被
#    连坐杀掉（2026-09-28 实测：前台长命令被中断时，那一组 32 卡的 Ray 集群整个没了）。
#
# ---------------------------------------------------------------------------------
# 时长预算（bs=5120 / 64 卡 / 1 epoch，按行数比例从 train.le16384 的 5.4 天推出）
#
#   agent  0.34M  ~3h      science 2.26M  ~21h     math 3.29M  ~30h
#   code   3.93M  ~36h     if      4.31M  ~39h     合计 ~5.4 天 + 数据过滤 ~3h
#
# 顺序是**先跑最短的 agent 当链路冒烟**（3h 就能看出数据/环境/存盘有没有问题），
# 再从小到大推进。想改顺序就覆盖 DOMAINS。
#
# 可调旋钮（全部环境变量）：
#   DOMAINS="agent math"       只跑部分域
#   SKIP_FILTER=1              直接用源数据不过滤（会撞 truncation，不推荐）
#   TRAIN_BATCH_SIZE=1280      HANDOFF §9 注：5120 已被实测为更优（MFU 0.46 vs 0.383）
#   CKPT_CONTENTS='["model"]'  只要权重，省一半磁盘（代价：中断后优化器状态从头）
#   MAX_CKPT_TO_KEEP=3         默认 2，只留最近两个
#   DRY_RUN=1                  只打印每一步要做什么，不真跑
# ---------------------------------------------------------------------------------
set -uo pipefail

B=/taijifs_sz34/share_304169095/hunyuan/yongcanyu
REPO="${B}/code/LLM-Fusion"
cd "${REPO}"

: "${NODE_IP_LIST:?NODE_IP_LIST 未设置 —— 不在 Taiji 节点上？}"
mapfile -t NODE_IPS < <(echo "${NODE_IP_LIST}" | tr ',' '\n' | cut -d: -f1 | grep .)
NNODES=${#NODE_IPS[@]}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-${HOST_GPU_NUM:-8}}
WORLD_SIZE=$(( NNODES * NGPUS_PER_NODE ))
CHIEF="${CHIEF_IP:-${NODE_IPS[0]}}"

SFT_DATA_ROOT=${SFT_DATA_ROOT:-${B}/data/M2RL/sft_train}
SFT_BASE_MODEL=${SFT_BASE_MODEL:-${B}/checkpoints/Qwen/Qwen3-4B-Base}
MAX_LENGTH=${MAX_LENGTH:-16384}
FILTER_WORLD_SIZE=${FILTER_WORLD_SIZE:-${WORLD_SIZE}}

# ---- SFT 超参（与 run_sft_multinode.sh 的默认值一致，这里显式写出来便于审计）----
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-5120}
export LR=${LR:-5e-5}
export OPTIMIZER=${OPTIMIZER:-AdamW}
export CLIP_GRAD=${CLIP_GRAD:-1.0}
export MAX_TOKEN_LEN_PER_GPU=${MAX_TOKEN_LEN_PER_GPU:-65536}
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
export SAVE_FREQ=${SAVE_FREQ:-200}
# ⚠️ 5 天级别的长跑，中断后从最近 ckpt 恢复时**优化器状态**很关键（Adam 的二阶矩），
#    所以默认连 optimizer 一起存。磁盘代价约 2 倍（48G/次 × ≤3 次/域 ≈ 700G，余量充足）。
#    只要权重就传 CKPT_CONTENTS='["model"]'。
export CKPT_CONTENTS=${CKPT_CONTENTS:-'["model","optimizer"]'}
export MAX_CKPT_TO_KEEP=${MAX_CKPT_TO_KEEP:-2}
export PROJECT_NAME=${PROJECT_NAME:-llm_fusion_sft}
export SFT_BASE_MODEL SFT_DATA_ROOT MAX_LENGTH
export SWANLAB_LOG_DIR=${SWANLAB_LOG_DIR:-/tmp/swanlog_llm_fusion_sft}
export SWANLAB_SAVE_DIR=${SWANLAB_SAVE_DIR:-/tmp/swanlab_llm_fusion_sft}

DRY_RUN=${DRY_RUN:-0}
SKIP_FILTER=${SKIP_FILTER:-0}
DOMAINS=${DOMAINS:-"agent math science code if"}

LOG_DIR=${B}/logs/llm_fusion
mkdir -p "${LOG_DIR}"
SUMMARY=${LOG_DIR}/sft_5expert_serial_$(date +%Y%m%d_%H%M%S).log
STATE=${LOG_DIR}/sft_5expert_serial_state.txt

log() { echo "[$(date +%m-%d\ %H:%M:%S)] $*" | tee -a "${SUMMARY}"; }

# merge_parts 要求每个 rank 的 part-*.parquet 与 part-*.json 都在，缺一个就整体
# FileNotFoundError。跑完先自查，别让 merge 来报错 —— 那时已经分不清是没跑完还是崩了。
list_missing_ranks() {
    local parts="$1" ws="$2" r=0 p j missing=""
    for (( r = 0; r < ws; r++ )); do
        printf -v p '%s/part-%05d.parquet' "${parts}" "${r}"
        printf -v j '%s/part-%05d.json' "${parts}" "${r}"
        [ -f "${p}" ] && [ -f "${j}" ] || missing+=" ${r}"
    done
    printf '%s' "${missing# }"
}

# ⚠️ ServerAliveInterval 是必需的，不是保险：过滤阶段一台机上 8 个 python 跑 20+ 分钟
# 不往 ssh 通道写一个字节，中间设备会掐掉这条 23 分钟静默的连接。ssh 一死，远端的
# `wait` 随之消失，驱动以为跑完了直接去做 merge，而 nohup 的 python 还在跑 —— 表现为
# 「部分 rank 没有产物」（2026-10-04 math 域：17 个 rank 缺失，34 个文件）。
SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=15 \
     -o TCPKeepAlive=yes -o ServerAliveInterval=60 -o ServerAliveCountMax=10"
PYTHON=/opt/conda/envs/torch-base/bin/python

cat <<EOF | tee -a "${SUMMARY}"

======== 5 域 SFT 串行训练 ========
  拓扑     : ${NNODES} 机 × ${NGPUS_PER_NODE} 卡 = ${WORLD_SIZE}   (chief=${CHIEF})
  base     : ${SFT_BASE_MODEL}
  数据根   : ${SFT_DATA_ROOT}
  域顺序   : ${DOMAINS}
  超参     : bs=${TRAIN_BATCH_SIZE} lr=${LR} ${OPTIMIZER} clip=${CLIP_GRAD} epochs=${TOTAL_EPOCHS}
  ckpt     : ${CKPT_CONTENTS}  保留 ${MAX_CKPT_TO_KEEP} 个，每 ${SAVE_FREQ} 步
  汇总日志 : ${SUMMARY}
====================================
EOF

# ================== 门禁 1：所有卡的占用必须归零 ==================
# KDRL 400 步那两个作业预计 10-01 凌晨才结束。SFT 要吃满 64 卡，带残留启动会以
# OOM 或 MASTER_PORT 被占的形式失败，而报错与真实原因毫无关系。
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
        log "ERROR: 这些节点的 Python 环境不完整（import verl/torch 失败）：${bad[*]}" >&2
        log "       换 Pod 后需要在每个节点跑一次：pip install -e . --no-deps（--no-deps 不能省）" >&2
        return 1
    fi
    log "Python 环境：${#NODE_IPS[@]} 个节点 OK"
    return 0
}

# ================== 每域的数据准备 ==================
# 原始 Nemotron 文件里有超过 16384 token 的样本，不过滤的话它们会在训练时被截断
# （data.truncation=error 甚至会直接报错）。过滤脚本用 M2RLSFTDataset 自己算长度，
# 不能用字符数估算。产物带 .le16384 后缀，与已有的 train.le16384.parquet 同规格。
prepare_data() {
    local dom="$1"; shift
    local srcs=("$@")
    local merged="${SFT_DATA_ROOT}/${dom}.parquet"
    local source="${merged}"

    if [ "${#srcs[@]}" -eq 1 ]; then
        source="${srcs[0]}"
    elif [ ! -f "${merged}" ]; then
        log "[${dom}] 合并 ${#srcs[@]} 个源文件 -> $(basename "${merged}")"
        if [ "${DRY_RUN}" = "1" ]; then echo "  DRY_RUN: 合并 ${srcs[*]}"; else
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
    if [ "${DRY_RUN}" = "1" ]; then echo "  DRY_RUN: filter -> ${filtered}"; else
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

        # ---- 校验 + 补跑 ----
        # 2026-10-04 math 域：rank 20~36（17 个）没产出，merge 抛 FileNotFoundError。
        # 那两个候选原因（ssh 长连接被掐 / rank 崩溃）都可以用「只重跑缺失的 rank」兜住，
        # 而且换一台机器重跑还能顺带区分：换了就好 = 机器问题，换了还崩 = 数据/代码问题。
        local attempt=0 missing=""
        while [ "${attempt}" -lt 3 ]; do
            missing=$(list_missing_ranks "${parts}" "${FILTER_WORLD_SIZE}")
            [ -z "${missing}" ] && break
            attempt=$(( attempt + 1 ))
            log "[${dom}] 缺失 ${missing} ；第 ${attempt}/3 轮补跑（换节点）..."
            local j=0 r
            for r in ${missing}; do
                local tip="${NODE_IPS[$(( (j + attempt) % ${#NODE_IPS[@]} ))]}"
                j=$(( j + 1 ))
                ${SSH} "${tip}" "cd ${REPO} && ${PYTHON} examples/llm_fusion/sft/filter_by_token_length.py filter \
                    --source '${source}' --output-dir '${parts}' \
                    --model-path '${SFT_BASE_MODEL}' \
                    --rank ${r} --world-size ${FILTER_WORLD_SIZE} \
                    --max-length ${MAX_LENGTH} \
                    > '${parts}/retry${attempt}_rank${r}.log' 2>&1" &
            done
            wait
        done
        missing=$(list_missing_ranks "${parts}" "${FILTER_WORLD_SIZE}")
        if [ -n "${missing}" ]; then
            log "[${dom}] ERROR: 补跑 3 轮后仍缺失 rank：${missing}" >&2
            log "[${dom}]       排查：${parts}/retry*_rank*.log（目录已保留，不再无条件删除）" >&2
            echo "${filtered}"
            return 1
        fi

        log "[${dom}] 各 rank 过滤完成（${FILTER_WORLD_SIZE}/${FILTER_WORLD_SIZE}），合并分片..."
        ${PYTHON} examples/llm_fusion/sft/filter_by_token_length.py merge \
            --source "${source}" --parts-dir "${parts}" \
            --output "${filtered}" --world-size "${FILTER_WORLD_SIZE}" \
            > "${LOG_DIR}/merge_${dom}.log" 2>&1
        local mrc=$?
        tail -3 "${LOG_DIR}/merge_${dom}.log" | tee -a "${SUMMARY}"
        if [ "${mrc}" -ne 0 ]; then
            log "[${dom}] ERROR: 合并失败（退出码 ${mrc}），保留 ${parts} 便于排查" >&2
            echo "${filtered}"
            return 1
        fi
        # 只有走到这一步才清：清早了会把 rank 日志一起埋掉，事后根本查不出谁没跑完。
        rm -rf "${parts}"
    fi
    echo "${filtered}"
}

run_domain() {
    local dom="$1"; shift
    local srcs=("$@")
    local t0=$(date +%s)

    local train_file
    train_file=$(prepare_data "${dom}" "${srcs[@]}" | tail -1)
    if [ "${DRY_RUN}" != "1" ] && [ ! -f "${train_file}" ]; then
        log "[${dom}] ERROR: 数据没准备好：${train_file}" >&2
        return 1
    fi

    # row group 数必须 >= world_size，否则会有 rank 分到 0 个分片（dataset.py 会报错，
    # 但那时已经浪费了一次启动）。
    local rg
    rg=$(${PYTHON} -c "import pyarrow.parquet as pq; print(pq.ParquetFile('${train_file}').num_row_groups)" 2>/dev/null)
    if [ -n "${rg}" ] && [ "${rg}" -lt "${WORLD_SIZE}" ]; then
        log "[${dom}] ERROR: row groups=${rg} < world_size=${WORLD_SIZE}" >&2
        return 1
    fi

    log "[${dom}] 开始训练：$(basename "${train_file}")  row_groups=${rg}"
    echo "RUNNING ${dom} $(date +%m-%d_%H:%M)" > "${STATE}"

    export SFT_TRAIN="${train_file}"
    export EXPERIMENT_NAME="sft_${dom}_qwen3_4b_base_${NNODES}x${NGPUS_PER_NODE}"
    if [ "${DRY_RUN}" != "1" ]; then
        bash run_sft_multinode.sh 2>&1 | tee -a "${LOG_DIR}/sft_${dom}_$(date +%Y%m%d_%H%M%S).log"
        local st=${PIPESTATUS[0]}
    else
        echo "  DRY_RUN: bash run_sft_multinode.sh  (SFT_TRAIN=${train_file})"
        local st=0
    fi

    local dt=$(( ($(date +%s) - t0) / 3600 ))
    if [ "${st}" -ne 0 ]; then
        log "[${dom}] 失败（退出码 ${st}，已耗时 ${dt}h）—— 后续域不再启动，修完可用 DOMAINS= 续跑。" >&2
        echo "FAILED ${dom} $(date +%m-%d_%H:%M)" > "${STATE}"
        return 1
    fi
    log "[${dom}] 完成，耗时 ${dt}h  ckpt: ${B}/runs/llm_fusion/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}"
    echo "DONE ${dom} $(date +%m-%d_%H:%M)" >> "${STATE}"
    return 0
}

# ---- 域 -> 源文件 映射 ----
# 五个文件拼起来正好是 train.parquet 的 14.12M 行；math 域是 Math-v2 + Math-Proofs 两个。
declare -A SRC=(
    [math]="${SFT_DATA_ROOT}/Nemotron-Math-v2.parquet ${SFT_DATA_ROOT}/Nemotron-Math-Proofs-v1.parquet"
    [code]="${SFT_DATA_ROOT}/Nemotron-Competitive-Programming-v1.parquet"
    [science]="${SFT_DATA_ROOT}/Nemotron-Science-v1.parquet"
    [if]="${SFT_DATA_ROOT}/Nemotron-Instruction-Following-Chat-v1.parquet"
    [agent]="${SFT_DATA_ROOT}/Nemotron-Agentic-v1.parquet"
)

if [ "${DRY_RUN}" = "1" ]; then
    log "DRY_RUN=1：跳过等卡与环境门禁"
else
    wait_gpus_free 360 || exit 1
    check_env || exit 1
fi

for dom in ${DOMAINS}; do
    [ -n "${SRC[$dom]:-}" ] || { log "未知域：${dom}（可选：${!SRC[*]}）" >&2; continue; }
    # shellcheck disable=SC2206
    read -r -a files <<< "${SRC[$dom]}"
    run_domain "${dom}" "${files[@]}" || exit 1
done

log "全部域完成：${DOMAINS}"
echo "ALL_DONE $(date +%m-%d_%H:%M)" >> "${STATE}"
