#!/usr/bin/env bash
# 核对单教师 MOPD 的教师拓扑是否如预期：8 份 tp1 副本，独占一个节点。
#
#   bash check_teacher_topology.sh [日志路径]        # 默认取最新的 mopd_merged 日志
#
# 为什么值得单独做一个脚本：这里有两个**配错了也不报错**的地方。
#
#   1. tp：distillation.yaml 默认 tensor_model_parallel_size=2。没显式覆盖成 1，
#      num_replicas 就是 8//2=4，得到「4 份 tp2」而不是「8 份 tp1」。训练照跑，
#      只是副本数少一半、每份跨两卡。
#   2. 教师是否真的独占一个节点：Ray 按 placement group 分配，日志里没有一行
#      直接写「教师在哪台机」。判据是「Ray 日志里不带 ip= 的 actor 就在 head 上」。
#
# 期望输出（8 份副本、教师全在同一节点、载的是同一个模型）：
#
#   tp            : 1                      期望 1
#   num_replicas  : 8                      期望 8
#   教师 engine   : 8 个，全在 head        期望 8
#   学生 engine   : 8 + 8 + 8（3 个节点）
#   教师模型      : 1 个不同的 model_path  期望 1（单教师）

set -uo pipefail

B=/taijifs_sz34/share_304169095/hunyuan/yongcanyu
LOG=${1:-$(ls -t "${B}"/logs/llm_fusion/train_mopd_merged_*.log 2>/dev/null | head -1)}

if [ -z "${LOG}" ] || [ ! -f "${LOG}" ]; then
    echo "找不到日志。用法: bash $0 <日志路径>" >&2
    exit 1
fi
echo "日志: ${LOG}"
echo "大小: $(du -h "${LOG}" | cut -f1)   最后修改: $(date -r "${LOG}" '+%m-%d %H:%M')"
echo

fail=0

# ---- 1. 教师的 tp（从命令行，不是从 dump 的 config）--------------------
# ⚠️ 两个坑，都踩过：
#  a) 不能用 `grep 'tensor_model_parallel_size' | tail -1` —— 这个键在日志里出现
#     多次（学生 rollout、reward_model、教师），最后一次是 reward_model 的默认 2，
#     跟教师无关。必须锚定 `teacher_models.teacher_model.` 前缀。
#  b) verl dump 的那份 config 是**解析前**的原始 Hydra 树，里面 num_replicas=0、
#     key=None 都是 yaml 默认值 —— DistillationConfig.__post_init__ 之后才变成
#     8 和 "default"，而那个结果不进日志。所以 num_replicas 不能从日志读，
#     只能由「实际起了几个 engine」反推（见下一节，那才是真凭据）。
echo "---- 教师并行度（命令行实参）----"
tp=$(grep -oE "teacher_models\.teacher_model\.inference\.tensor_model_parallel_size=[0-9]+" "${LOG}" \
     | head -1 | grep -oE '[0-9]+$')
pool=$(grep -oE "distillation\.n_gpus_per_node=[0-9]+" "${LOG}" | head -1 | grep -oE '[0-9]+$')
printf "  tp            : %-6s 期望 1\n" "${tp:-<未找到>}"
printf "  教师池        : %-6s 期望 8\n" "${pool:-<未找到>}"
if [ -n "${tp:-}" ] && [ -n "${pool:-}" ]; then
    printf "  => num_replicas: %-6s （由 pool/tp 推出，verl 不打印这个值）\n" "$(( pool / tp ))"
fi
[ "${tp:-0}" = "1" ]   || { echo "  ✗ tp != 1：教师会被切片，不是 8 份独立副本" >&2; fail=1; }
[ "${pool:-0}" = "8" ] || { echo "  ✗ 教师池 != 8" >&2; fail=1; }

# ---- 2. 实际显存占用（唯一可靠的凭据）----------------------------------
# ⚠️ 不要用「数日志里的 vLLMHttpServer 行数」当判据 —— Ray 的 log dedup 会把重复行
#    折成 `[repeated Nx across cluster]`，部分 engine 的启动行根本不单独出现。
#    2026-09-17 实测：日志只数出 21 个，而 32 张卡全部在用。数日志会误报。
#    显存分布才是硬凭据：教师节点每卡跑一份完整 4B（0.8×95≈76G，填了 KV cache 后
#    ~91G），学生节点每卡是 vLLM + FSDP actor 共存（~55G）。
echo "---- 实际显存占用（每节点 8 卡）----"
if [ -n "${NODE_IP_LIST:-}" ]; then
    head_min=999999
    for ip in $(echo "${NODE_IP_LIST}" | tr ',' '\n' | cut -d: -f1); do
        mem=$(ssh -o BatchMode=yes -o ConnectTimeout=10 "${ip}" \
              "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' '" 2>/dev/null)
        [ -z "${mem}" ] && { printf "  %-15s: <取不到>\n" "${ip}"; continue; }
        n_busy=$(echo "${mem}" | tr ' ' '\n' | awk 'NF && $1>1000' | wc -l)
        role="学生"
        [ "${ip}" = "${HEAD_IP:-}" ] && role="教师"
        printf "  %-15s %s: %s 张在用   %s\n" "${ip}" "${role}" "${n_busy}" \
               "$(echo "${mem}" | tr ' ' '\n' | awk 'NF{printf "%.0fG ", $1/1024}')"
        [ "${n_busy}" -eq 8 ] || { echo "    ✗ 不是 8 张全用" >&2; fail=1; }
    done
else
    echo "  <NODE_IP_LIST 未设置，跳过>"
fi

# ---- 3. engine 分布（仅供参考，会被 Ray log dedup 少数）-----------------
echo
echo "---- vLLM engine 启动行（会被 Ray dedup 少数，仅参考）----"
mapfile -t engines < <(grep -oE "vLLMHttpServer pid=[0-9]+(, ip=[0-9.]+)?" "${LOG}" | sort -u)
if [ "${#engines[@]}" -eq 0 ]; then
    echo "  还没有 engine 起来（vLLM 启动约 8 min）。稍后再看。"
    exit 0
fi
head_n=0
declare -A per_ip=()
for e in "${engines[@]}"; do
    if [[ "${e}" == *", ip="* ]]; then
        ip=${e##*, ip=}
        per_ip[$ip]=$(( ${per_ip[$ip]:-0} + 1 ))
    else
        head_n=$(( head_n + 1 ))
    fi
done
printf "  head（无 ip=）  : %-3s 个   ← 教师应该全在这里\n" "${head_n}"
for ip in "${!per_ip[@]}"; do
    printf "  %-15s: %-3s 个\n" "${ip}" "${per_ip[$ip]}"
done
printf "  合计            : %s 个（≤32，dedup 会少数，不作判据）\n" "${#engines[@]}"

# ---- 3. 学生 worker 不该出现在教师节点上 --------------------------------
# WorkerDict 是学生的 FSDP worker。一个都不该落在 head（教师节点）上。
echo
echo "---- 学生 FSDP worker ----"
student_on_head=$(grep -cE "WorkerDict pid=[0-9]+\)" "${LOG}" || true)
printf "  head 上的 WorkerDict : %-3s   期望 0（教师节点不跑学生）\n" "${student_on_head}"
[ "${student_on_head}" = "0" ] || {
    echo "  ✗ 学生 worker 落到了教师节点上" >&2
    fail=1
}

# ---- 4. 只有一个教师模型 ------------------------------------------------
echo
echo "---- 教师模型 ----"
mapfile -t paths < <(grep -oE "merged_models/[a-zA-Z0-9_.]+|experts_hf/[a-z]+" "${LOG}" | sort -u)
if [ "${#paths[@]}" -eq 0 ]; then
    echo "  <日志里还没出现模型路径>"
else
    printf '  %s\n' "${paths[@]}"
fi
if printf '%s\n' "${paths[@]}" | grep -q "experts_hf/"; then
    echo "  ✗ 出现了 experts_hf/ 路径 —— 这是 5 教师配置，不是单教师" >&2
    fail=1
fi

# ---- 5. 训练是否在动 ----------------------------------------------------
echo
echo "---- 进度 ----"
grep -oE "Training Progress: +[0-9]+%\|[^|]*\| [0-9]+/[0-9]+ \[[^]]*\]" "${LOG}" | tail -1 || echo "  <还没到训练循环>"
echo "  最近的错误（前 5 条，Failed to decode tool call 属正常）:"
grep -iE "Traceback|CUDA out of memory|ncclSystemError|EADDRINUSE|ApiError|ValueError" "${LOG}" \
    | grep -v "Failed to decode tool call" | tail -5 | sed 's/^/    /' || echo "    无"

echo
if [ "${fail}" = "0" ]; then
    echo "OK: 教师拓扑符合预期（8 份 tp1 副本，独占一节点）"
else
    echo "FAIL: 见上面的 ✗。tp 配错的话必须重启，跑到一半改不了。" >&2
    exit 1
fi
