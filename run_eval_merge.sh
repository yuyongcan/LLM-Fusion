#!/usr/bin/env bash
# 串行评测 merge 产物，8 基准全量，VAL_N=16。
#
# run_eval_experts.sh 的 merge 版：同样的 verl val_only 一趟一模型、同样的环境约束，
# 只换模型清单。本机这趟跑三个：
#   merge5_ta0.6    5 专家 (math,code,science,if,agent) TA lambda=0.6，即论文 released 配置
#   merge3_ta0.4    3 专家 (math,code,science)          TA lambda=0.4
#   merge3_ta0.6    3 专家 (math,code,science)          TA lambda=0.6
#
# 用法:
#   bash run_eval_merge.sh                          # 全部 3 个
#   MODELS="merge3_ta0.4" bash run_eval_merge.sh     # 只跑一个
#   VAL_N=4 bash run_eval_merge.sh                   # 降采样
#
# 断点续跑：8 个 dump 齐了的模型自动跳过，重跑删掉对应 eval_n<N> 目录即可。

set -uo pipefail

B=/taijifs_sz34/share_304169095/hunyuan/yongcanyu
cd "${B}/code/LLM-Fusion"
source ./env_llm_fusion.sh

# start.md §0：Pod 注入的 SWANLAB_LOG_DIR 指向共享盘，swanlab 探活会失败
export SWANLAB_LOG_DIR=/tmp/swanlog_llm_fusion
# eval.sh 内部 trainer.logger='["console"]'，val_only 路径指标直接进 console，无需 swanlab

export VAL_N=${VAL_N:-16}
export NNODES=1
export NGPUS_PER_NODE=8
# 单节点 8 卡：vLLM 与 FSDP actor 同卡共存，0.8 会 OOM（start.md §0）
export ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.7}

# TMPDIR 太深会撞 AF_UNIX 108 字符限制（code 判题用 multiprocessing.Manager）
export TMPDIR=/tmp
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

M5="${B}/outputs_llm_fusion/merge_4b/merged_models"
M3="${B}/outputs_llm_fusion/merge_4b_mcs/merged_models"
# MOPD 训练产物：FSDP 分片先 `python -m verl.model_merger merge` 导成 HF 再进来
HF="${B}/outputs_llm_fusion/hf_exports"
LOG_DIR="${B}/logs/llm_fusion/eval_merge"
mkdir -p "${LOG_DIR}"

MODELS=${MODELS:-"merge5_ta0.6 merge3_ta0.4 merge3_ta0.6"}

# LCB v5/v6 + code reward 要 sandbox，否则分数静默腰斩（HANDOFF §73）
if ! curl -s -m 10 --noproxy '*' -X POST http://localhost:8080/run_code \
        -H 'Content-Type: application/json' \
        -d '{"code":"print(2+2)","language":"python"}' 2>/dev/null | grep -q Success; then
    echo "FATAL: sandbox 未就绪，LCB 分数会静默腰斩。先跑 bash setup_sandbox.sh" >&2
    exit 1
fi
echo "sandbox OK"

for d in ${MODELS}; do
    case "$d" in
        merge5_ta0.6) MP="${M5}/task_arithmetic_scaling_coefficient_0.6" ;;
        merge3_ta0.4) MP="${M3}/task_arithmetic_scaling_coefficient_0.4" ;;
        merge3_ta0.6) MP="${M3}/task_arithmetic_scaling_coefficient_0.6" ;;
        merge3_ta0.8) MP="${M3}/task_arithmetic_scaling_coefficient_0.8" ;;
        merge3_ta1.0) MP="${M3}/task_arithmetic_scaling_coefficient_1.0" ;;
        merge3_ta1.2) MP="${M3}/task_arithmetic_scaling_coefficient_1.2" ;;
        mopd_5t_4b_step400) MP="${HF}/mopd_5t_4b_step400" ;;
        *) echo "!! 未知模型名 ${d}，跳过" >&2; continue ;;
    esac
    EXP="eval_${d}_n${VAL_N}"
    # 结果直接落在模型自己的目录下，跟权重放一起，免得对不上号
    OUT="${MP}/eval_n${VAL_N}"

    if [ ! -f "${MP}/config.json" ]; then
        echo "!! ${d}: ${MP} 无 config.json，跳过" >&2; continue
    fi
    # 8 个 benchmark 的 dump 齐了就算跑完
    if [ "$(ls "${OUT}/0"/*.jsonl 2>/dev/null | wc -l)" -ge 8 ]; then
        echo "== ${d}: 已完成，跳过 (${OUT})"; continue
    fi

    L="${OUT}/eval.log"
    mkdir -p "${OUT}"
    # 另在 logs/ 留个软链，方便集中 tail
    ln -sfn "${L}" "${LOG_DIR}/${EXP}.log" 2>/dev/null
    echo "===== ${d}  VAL_N=${VAL_N}  $(date '+%F %T') ====="
    echo "      model: ${MP}"
    echo "      log:   ${L}"

    MODEL_PATH="${MP}" \
    EXPERIMENT_NAME="${EXP}" \
    VALIDATION_DATA_DIR="${OUT}" \
    PROJECT_NAME=llm_fusion_eval_repro \
        bash examples/llm_fusion/eval/eval.sh > "${L}" 2>&1
    rc=$?

    if [ $rc -ne 0 ]; then
        echo "  !! ${d} 退出码 ${rc}，末尾日志："
        grep -iE "Error|Traceback|OutOfMemory|EADDRINUSE|assert" "${L}" | tail -5
    else
        echo "  ok ${d} $(date '+%F %T')"
    fi
    # 给 Ray/vLLM 完全释放显存的时间，否则下一趟起不来
    sleep 60
done

echo
echo "全部结束 $(date '+%F %T')。"
