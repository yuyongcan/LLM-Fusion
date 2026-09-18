#!/usr/bin/env bash
# 4 机并行评测 merge_4b 的五专家系数扫描: lambda = 0.4 / 0.8 / 1.0 / 1.2.
# lambda = 0.6 已经评过（结果与权重放在同一目录），这里不重复跑。
#
# 与 run_eval_experts.sh 的差别：那个脚本在单节点上串行跑 6 个模型；这里一台
# 机器跑一个模型，4 个系数同时开。
#
# ---- 前提（本脚本只做门禁检查，不负责搭建）----
# 1. 每个节点一个**独立的单机 Ray**（8 卡），不是 4 机 32 卡的共享集群。
#    verl 不起集群，靠 ray.init() 自动发现。若 4 个作业挂到同一个 32 卡集群上：
#      a) 4 × 8 卡正好吃满，但调度器会把某个作业的 worker 摊到多机上；
#      b) reward 的 sandbox 地址是 localhost:8080，worker 跑到别的节点就取不到，
#         LCB v5/v6 分数静默腰斩，日志里没有任何报错。
#    所以必须先拆掉多机集群、逐节点 `ray start --head --num-gpus 8`。
# 2. 每个节点 sandbox 就绪（bash setup_sandbox.sh），否则同上，LCB 腰斩。
#
# 用法:
#   bash run_eval_merge_sweep_5expert.sh              # 门禁 + 启动 4 个评测
#   COEFS="0.8" bash run_eval_merge_sweep_5expert.sh  # 只跑一个系数
#   VAL_N=4 bash run_eval_merge_sweep_5expert.sh      # 降采样试水
#
# 断点续跑：某系数输出目录里 8 个 benchmark 的 jsonl 齐了就跳过。

set -uo pipefail

B=/taijifs_sz34/share_304169095/hunyuan/yongcanyu
cd "${B}/code/LLM-Fusion"

: "${NODE_IP_LIST:?NODE_IP_LIST 未设置 —— 不在 Taiji 节点上？}"
mapfile -t NODE_IPS < <(echo "${NODE_IP_LIST}" | tr ',' '\n' | cut -d: -f1 | grep .)

COEFS=${COEFS:-"0.4 0.8 1.0 1.2"}
read -r -a COEF_ARR <<<"${COEFS}"

if [ "${#COEF_ARR[@]}" -gt "${#NODE_IPS[@]}" ]; then
    echo "ERROR: ${#COEF_ARR[@]} 个系数 > ${#NODE_IPS[@]} 个节点，一台机器只能跑一个。" >&2
    exit 1
fi

MERGED="${B}/outputs_llm_fusion/merge_4b/merged_models"
LOG_DIR="${B}/logs/llm_fusion/eval_merge_sweep"
SCRIPT_DIR="${LOG_DIR}/scripts"
mkdir -p "${LOG_DIR}" "${SCRIPT_DIR}"

export VAL_N=${VAL_N:-16}
export ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.7}

SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=15"
CONDA=/opt/conda/envs/torch-base/bin

echo "节点: ${#NODE_IPS[@]} 台   系数: ${COEFS}   VAL_N=${VAL_N}"
echo

# ---- 门禁：每节点 8 卡独立 Ray + sandbox + 权重就位 ----------------------
fail=0
for i in "${!COEF_ARR[@]}"; do
    coef=${COEF_ARR[$i]}
    ip=${NODE_IPS[$i]}
    MP="${MERGED}/task_arithmetic_scaling_coefficient_${coef}"

    gpu=$(${SSH} "${ip}" "PATH=${CONDA}:\$PATH RAY_ADDRESS=${ip}:6379 ray status 2>/dev/null \
        | grep -oE '[0-9.]+/[0-9.]+ GPU' | head -1" 2>/dev/null)
    total=${gpu#*/}; total=${total% GPU}
    sbx=$(${SSH} "${ip}" "curl -s -m 10 --noproxy '*' -X POST http://localhost:8080/run_code \
        -H 'Content-Type: application/json' \
        -d '{\"code\":\"print(2+2)\",\"language\":\"python\"}' 2>/dev/null \
        | grep -q Success && echo UP || echo DOWN" 2>/dev/null)
    cfg=$([ -f "${MP}/config.json" ] && echo yes || echo MISSING)

    printf "  %-14s ip=%-14s ray_gpu=%-6s sandbox=%-5s weights=%s\n" \
        "c${coef}" "${ip}" "${total:-NA}" "${sbx:-NA}" "${cfg}"

    [ "${total%.*}" = "8" ] || { echo "    !! ray 卡数不是 8（多机集群没拆干净？）" >&2; fail=1; }
    [ "${sbx}" = "UP" ]     || { echo "    !! sandbox 未就绪，LCB 分数会腰斩" >&2; fail=1; }
    [ "${cfg}" = "yes" ]    || { echo "    !! ${MP} 无 config.json" >&2; fail=1; }
done

if [ "${fail}" -ne 0 ]; then
    echo "" >&2
    echo "门禁未通过，未启动任何评测。修好后重跑本脚本即可。" >&2
    exit 1
fi
echo "门禁通过"
echo

# ---- 逐节点启动 ----------------------------------------------------------
# 远端脚本落在共享盘上，省掉层层引号转义；每个节点一份，互不覆盖。
for i in "${!COEF_ARR[@]}"; do
    coef=${COEF_ARR[$i]}
    ip=${NODE_IPS[$i]}
    MP="${MERGED}/task_arithmetic_scaling_coefficient_${coef}"
    OUT="${MP}/eval_n${VAL_N}"
    EXP="eval_merge5_c${coef}_n${VAL_N}"

    if [ "$(ls "${OUT}/0"/*.jsonl 2>/dev/null | wc -l)" -ge 8 ]; then
        echo "== c${coef} (${ip}): 已完成，跳过 (${OUT})"
        continue
    fi

    RS="${SCRIPT_DIR}/eval_c${coef}.sh"
    cat > "${RS}" <<EOS
#!/usr/bin/env bash
set -uo pipefail
cd ${B}/code/LLM-Fusion
source ./env_llm_fusion.sh

# start.md §0：Pod 注入的 SWANLAB_LOG_DIR 指向共享盘，swanlab 探活会失败。
# eval.sh 用 trainer.logger='["console"]'，指标直接进 console，不依赖 swanlab。
export SWANLAB_LOG_DIR=/tmp/swanlog_llm_fusion
export VAL_N=${VAL_N}
export NNODES=1
export NGPUS_PER_NODE=8
# 单机 8 卡：vLLM 与 FSDP actor 同卡共存，0.8 会 OOM（start.md §0）
export ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL}
# 关键：钉死本节点自己的集群，别让 ray.init() 捞到别的机器的 GCS
export RAY_ADDRESS=${ip}:6379

# TMPDIR 太深会撞 AF_UNIX 108 字符限制（code 判题用 multiprocessing.Manager）
export TMPDIR=/tmp
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

mkdir -p ${OUT}
ln -sfn ${OUT}/eval.log ${LOG_DIR}/${EXP}.log 2>/dev/null

cd ${B}/code/LLM-Fusion
setsid nohup env \\
    MODEL_PATH="${MP}" \\
    EXPERIMENT_NAME="${EXP}" \\
    VALIDATION_DATA_DIR="${OUT}" \\
    PROJECT_NAME=llm_fusion_eval_repro \\
    bash examples/llm_fusion/eval/eval.sh > "${OUT}/eval.log" 2>&1 < /dev/null &
echo "launched pid \$!"
EOS

    pid=$(${SSH} "${ip}" "bash ${RS}" 2>&1 | tail -1)
    echo "== c${coef} -> ${ip}  (${pid})"
    echo "     model: ${MP}"
    echo "     log:   ${OUT}/eval.log  (软链 ${LOG_DIR}/${EXP}.log)"
done

echo
echo "4 个评测已在各自节点后台启动 $(date '+%F %T')"
echo
echo "看进度:"
echo "  tail -f ${LOG_DIR}/eval_merge5_c<系数>_n${VAL_N}.log"
echo "查完成（每个目录应有 8 个 jsonl）:"
echo "  for c in ${COEFS}; do echo -n \"c\$c: \"; ls ${MERGED}/task_arithmetic_scaling_coefficient_\$c/eval_n${VAL_N}/0/*.jsonl 2>/dev/null | wc -l; done"
echo "汇总: python summarize_eval.py"
