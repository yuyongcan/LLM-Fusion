#!/usr/bin/env bash
# SFT 串行训练守护：每 30 min 检查 run_sft_5expert_serial.sh 是否还活着，
# 进程没了就按「未完成的域」清理残留并重启。只在 head 上跑一个实例。
#
#   setsid nohup bash watchdog_sft_serial.sh >/dev/null 2>&1 </dev/null & disown
#
# 与 IDE 侧巡检（automation id=sft，每小时）的分工：
#   本脚本 —— 30 min 粒度，机械地「死了就拉起」，不依赖 IDE 是否开着。
#   AI 巡检 —— 做根因分析（读日志判断是过滤缺 rank / OOM / NCCL / rendezvous 超时 …），
#              以及本脚本覆盖不到的「进程活着但挂死」（步数不涨）情形。
# 现场一律保留在 logs/llm_fusion/ 下，供事后归因。
#
# 可调：CHECK_INTERVAL=1800（秒）
set -uo pipefail

B=/taijifs_sz34/share_304169095/hunyuan/yongcanyu
REPO="${B}/code/LLM-Fusion"
LOG_DIR="${B}/logs/llm_fusion"
STATE="${LOG_DIR}/sft_5expert_serial_state.txt"
ALL_DOMAINS="agent math science code if"
CHECK_INTERVAL=${CHECK_INTERVAL:-1800}

: "${NODE_IP_LIST:?NODE_IP_LIST 未设置 —— 不在 Taiji 节点上？}"
mapfile -t NODE_IPS < <(echo "${NODE_IP_LIST}" | tr ',' '\n' | cut -d: -f1 | grep .)
SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=15"

WD_LOG="${LOG_DIR}/watchdog_sft_$(date +%Y%m%d_%H%M%S).log"
log() { echo "[$(date +%m-%d\ %H:%M:%S)] $*" >> "${WD_LOG}"; }

log "守护启动：间隔 ${CHECK_INTERVAL}s，${#NODE_IPS[@]} 个节点，守护日志 ${WD_LOG}"

while true; do
    if pgrep -f "run_sft_5expert_serial.sh" >/dev/null 2>&1; then
        log "驱动存活。"
    else
        # state.txt 里 DONE 行是追加的，跨多次启动累积，所以它是「已完成域」的权威来源。
        done_doms=$(grep -oE '^DONE [a-z]+' "${STATE}" 2>/dev/null | awk '{print $2}' | sort -u)
        remain=""
        for d in ${ALL_DOMAINS}; do
            case " ${done_doms} " in
                *" ${d} "*) ;;
                *) remain="${remain} ${d}" ;;
            esac
        done
        remain=$(echo "${remain}" | xargs 2>/dev/null || true)

        if [ -z "${remain}" ]; then
            log "全部域已完成（$(echo "${done_doms}" | tr '\n' ' ')），守护退出。"
            break
        fi

        log "驱动进程已消失！未完成域：${remain} —— 清理残留 torchrun 后重启。"
        # 残留的 torchrun 会一直占着整机的 8 张卡，下次启动会以 OOM 或「MASTER_PORT 被占」
        # 的形式失败，报错与真实原因毫无关系。"[v]erl" 的方括号是 load-bearing 的：
        # 没有它 pattern 会匹配到 ssh 自己 spawn 的 bash -c 包装进程，于是杀掉自己的父 shell。
        for ip in "${NODE_IPS[@]}"; do
            ${SSH} "${ip}" 'pkill -f "[v]erl.trainer.sft_trainer" 2>/dev/null || true
                 sleep 2
                 pkill -9 -f "[v]erl.trainer.sft_trainer" 2>/dev/null || true' \
                >/dev/null 2>&1 &
        done
        wait

        TS=$(date +%Y%m%d_%H%M%S)
        (
            cd "${REPO}" || exit 1
            DOMAINS="${remain}" setsid nohup bash run_sft_5expert_serial.sh \
                > "${LOG_DIR}/serial_restart_${TS}.log" 2>&1 < /dev/null &
        )
        log "已重启：DOMAINS='${remain}' -> ${LOG_DIR}/serial_restart_${TS}.log"
    fi
    sleep "${CHECK_INTERVAL}"
done
