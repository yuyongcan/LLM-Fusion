#!/usr/bin/env bash
# 把 Ray 集群重建成 <NNODES> 机 × 8 卡。
#
# 背景：verl 自己不起集群（examples/llm_fusion/train/*.sh 里零个 ray start），
# 它靠 ray.init() 自动发现已有集群。若集群的 GPU 数少于 trainer.nnodes ×
# trainer.n_gpus_per_node，任务会静默 pending 挂住而不报错 —— 所以本脚本最后
# 有一道卡数门禁。
#
# 另注（overview 5.4 的教训）：`ray stop --force` 杀不掉训练脚本内嵌起的集群，
# 必须补 pkill。2026-09-10 首次上多机时 head 的 raylet 就死在
# runtime_env_agent fate-share 上，而 gcs_server 残留，导致 ray status
# 看着有 2 节点、实际只有 8 卡。
#
# 用法: bash setup_ray_cluster.sh
set -uo pipefail

# ---- 拓扑：从平台变量推导，不硬编码 IP（换机器不用改脚本）----
# NODE_IP_LIST 是 "ip:slots,ip:slots,..."，是权威来源。
# ⚠️ 不要用 NODE_NUM 当节点数：本平台上它是总 GPU 数，不是机器数。
: "${NODE_IP_LIST:?NODE_IP_LIST 未设置 —— 不在 Taiji 节点上？}"
mapfile -t NODE_IPS < <(echo "${NODE_IP_LIST}" | tr ',' '\n' | cut -d: -f1 | grep .)
HEAD_IP="${CHIEF_IP:-${NODE_IPS[0]}}"
NGPU=${NGPUS_PER_NODE:-${HOST_GPU_NUM:-8}}
PORT=${RAY_PORT:-6379}
EXPECT_GPU=$(( ${#NODE_IPS[@]} * NGPU ))

if [ "${NODE_IPS[0]}" != "${HEAD_IP}" ]; then
    echo "ERROR: NODE_IP_LIST[0]=${NODE_IPS[0]} 但 CHIEF_IP=${HEAD_IP}；拒绝猜测 head。" >&2
    exit 1
fi

CONDA_BIN=${CONDA_BIN:-/opt/conda/envs/torch-base/bin}
SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=10"

# rendezvous 和 NCCL 流量绝不能走 HTTP 代理；所有集群 IP 必须进 no_proxy
CLUSTER_IPS=$(printf '%s' "${NODE_IP_LIST}" | tr ',' '\n' | sed 's/:[0-9][0-9]*$//' \
              | awk 'NF && !seen[$0]++' | paste -sd, -)
NOPROXY=".woa.com,localhost,127.0.0.1,.oa.com,${CLUSTER_IPS}"

# NCCL 基线（经验记录-通用.md 2026-09-07 第五节 + 2026-09-10 第六节）。
# 在 `ray start` 的 shell 里 export，raylet 及其拉起的所有 worker 进程都会继承 ——
# 这比只在训练脚本里 export 可靠：verl 只把一份写死的白名单
# (verl/trainer/constants_ppo.py) 作为 runtime_env 下发给 worker，
# NCCL_NET_GDR_LEVEL 不在白名单里，光在训练 shell 里 export 会静默失效。
#
# ⚠️ NCCL_SOCKET_IFNAME 会按 head IP 实际所在的网卡自动探测，不写死 bond1。
IFNAME=${NCCL_SOCKET_IFNAME:-$(ip -o -4 addr show 2>/dev/null | awk -v ip="${HEAD_IP}" '$4 ~ "^"ip"/" {print $2; exit}')}
IFNAME=${IFNAME:-bond1}

NCCL_ENV=(
    NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL:-0}   # 本机栈 GDR 注册不兼容，关直连、保留 IB
    NCCL_SOCKET_IFNAME=${IFNAME}
    NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
    NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3}
    NCCL_DEBUG=${NCCL_DEBUG:-WARN}
    TORCH_NCCL_ASYNC_ERROR_HANDLING=1
    # 又一个白名单受害者（2026-09-11 实测）：PYTORCH_CUDA_ALLOC_CONF 不在
    # PPO_RAY_RUNTIME_ENV 里，run_train_multinode.sh 里的 export 传不到 Ray worker，
    # 于是 overview 5.4 给 OOM 开的两味药只生效了 ROLLOUT_GPU_MEM_UTIL 一味。
    # PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
    # ⚠️ 不要设 VLLM_PORT（2026-09-11 试过，反而更糟）。
    # 启动时偶发（4 次里 2 次，都在 worker 上）：
    #   torch.distributed.DistNetworkError: ... port: 44643 ... EADDRINUSE
    # 机制是 TOCTOU：vllm/utils/network_utils.py 的 get_open_port() 走 bind(("",0))
    # 让内核从 ip_local_port_range（本机 32768-60999）分配，探测到「空闲」与真正
    # bind 之间有时间窗；每节点 8 个 engine 并发 init，期间内核可能把同一端口分给
    # 某条出向连接。
    # 但设 VLLM_PORT=61000 后 _get_open_port() 改成「从固定基址往上扫」
    # （network_utils.py:196-205），8 个 engine 各自独立扫、探测完都把 socket 关掉，
    # 于是**全部选中 61000**，只有第一个 TCPStore bind 成功 —— 偶发变必然。
    # 结论：保持 vLLM 默认的随机端口，启动失败就重跑（见 overview「启动偶发」一节）。
    # SwanLab 凭据：同一个白名单问题（2026-09-11 实测）。
    # SWANLAB_API_KEY 不在 PPO_RAY_RUNTIME_ENV 里，只在训练脚本里 export 传不到
    # TaskRunner actor —— swanlab.init() 于是退化成交互式登录并直接抛错：
    #   RuntimeError: Failed to initialize SwanLab in online mode: no TTY is available
    # 在 ray start 的 shell 里注入，raylet 拉起的所有 worker 才拿得到。
    # （另一条路是在每个节点 swanlab login 写 ~/.swanlab，但那是节点本地状态，
    #   换机器又得重做一次；注入到集群更符合「拓扑自动推导」的思路。）
    SWANLAB_API_KEY=${SWANLAB_API_KEY:-7f6uQlMFc3T21Ah5hpMwh}
    SWANLAB_API_HOST=${SWANLAB_API_HOST:-http://taiji-exp.taiji.woa.com/api}
    SWANLAB_WEB_HOST=${SWANLAB_WEB_HOST:-http://taiji-exp.taiji.woa.com/}
    SWANLAB_MODE=${SWANLAB_MODE:-cloud}
    # SWANLAB_LOG_DIR 必须是**节点本地**路径，不能放共享盘（2026-09-11 实测）。
    # tracking.py:132 的默认值 "swanlog" 是相对路径（传不到 actor 时按 actor cwd 解析
    # 成 <repo>/swanlog）；但换成共享盘绝对路径也会挂 —— swanlab 0.10.0 的
    # safe_mkdir 探测二用 `tempfile.TemporaryFile(dir=p)`，该调用会「创建后立即
    # unlink 但保持打开」，这个共享盘不支持这种语义，直接抛 FileNotFoundError。
    # 它是 OSError 子类，被 safe_mkdir 当成「可重试的文件系统延迟」自旋到 5s 超时，
    # 最后报成 `exists but is not writable / FILESYSTEM may be slow or remote` ——
    # 极具误导性：该目录 touch/rm 只要 20ms，权限也没问题。
    #   验证：python -c "import tempfile; tempfile.TemporaryFile(dir='<共享盘目录>')"
    # swanlog 只是 SDK 的本地缓存（指标照样上传到 taiji-exp），不是训练产物，
    # 所以放 /tmp 不损失任何东西；Pod 重启丢掉也无所谓。
    SWANLAB_LOG_DIR=${SWANLAB_LOG_DIR:-/tmp/swanlog_llm_fusion}
    # SWANLAB_SAVE_DIR 也必须节点本地，而且这一条比 LOG_DIR 更致命（2026-09-22 实测）：
    # 它决定 ~/.swanlab 等全局文件（.netrc 等）的位置，在 **import swanlab** 时就被读
    # （env.py:183 SwanLabEnv.set_default -> get_save_dir），不是等到 swanlab.init()。
    # get_save_dir() 出于安全只创建末级目录、不创建父目录，父目录不存在就抛
    #   FileNotFoundError: <父目录> not found
    # 本 Pod 注入的是 ${B}/basic_train_<task>/<instance>/swanlab，那个 <instance> 父目录
    # **并不存在**，于是 head 和全部 worker 上 `import swanlab` 直接失败 ——
    # run_train_multinode.sh 的 swanlab 门禁会报「没装 swanlab」，极具误导性（装着呢）。
    #   验证：SWANLAB_SAVE_DIR=/tmp/x python -c "import swanlab"   # ok
    #         python -c "import swanlab"                            # FileNotFoundError
    # 同一个白名单问题：它不在 PPO_RAY_RUNTIME_ENV 里，只能在 ray start 时注入。
    SWANLAB_SAVE_DIR=${SWANLAB_SAVE_DIR_OVERRIDE:-/tmp/swanlab_llm_fusion}
)

echo "拓扑: ${#NODE_IPS[@]} 机 × ${NGPU} 卡 = ${EXPECT_GPU}   head=${HEAD_IP}   iface=${IFNAME}"

echo "=== 1. 所有节点清理残留 ==="
for ip in "${NODE_IPS[@]}"; do
    if [ "${ip}" = "${HEAD_IP}" ]; then
        PATH="${CONDA_BIN}:${PATH}" ray stop --force > /dev/null 2>&1
        pkill -9 -f "ray::" > /dev/null 2>&1
        pkill -9 -f "raylet" > /dev/null 2>&1
        pkill -9 -f "gcs_server" > /dev/null 2>&1
    else
        # SSH 非交互不加载 bashrc，显式给 PATH
        ${SSH} "${ip}" "PATH=${CONDA_BIN}:\$PATH ray stop --force > /dev/null 2>&1; \
            pkill -9 -f 'ray::' > /dev/null 2>&1; \
            pkill -9 -f raylet > /dev/null 2>&1; \
            pkill -9 -f gcs_server > /dev/null 2>&1; true"
    fi
done
sleep 5
echo "清理完成"

echo "=== 2. 确认所有卡无残留进程 ==="
for ip in "${NODE_IPS[@]}"; do
    if [ "${ip}" = "${HEAD_IP}" ]; then
        n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
    else
        n=$(${SSH} "${ip}" "nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l")
    fi
    echo "  ${ip}: ${n} 个占卡进程"
    [ "${n}" -ne 0 ] && echo "  WARNING: ${ip} 仍有进程占 GPU，可能导致 vLLM OOM" >&2
done

echo "=== 3. 启动 head (${HEAD_IP}) ==="
env "${NCCL_ENV[@]}" \
    PATH="${CONDA_BIN}:${PATH}" no_proxy="${NOPROXY}" NO_PROXY="${NOPROXY}" \
    ray start --head \
        --node-ip-address=${HEAD_IP} \
        --port=${PORT} \
        --num-gpus ${NGPU} \
        --dashboard-host=0.0.0.0 2>&1 | tail -3
sleep 8

echo "=== 4. 启动 worker ==="
for ip in "${NODE_IPS[@]}"; do
    [ "${ip}" = "${HEAD_IP}" ] && continue
    echo "  -> ${ip}"
    ${SSH} "${ip}" "env ${NCCL_ENV[*]} \
        PATH=${CONDA_BIN}:\$PATH \
        no_proxy='${NOPROXY}' NO_PROXY='${NOPROXY}' \
        ray start --address=${HEAD_IP}:${PORT} --num-gpus ${NGPU}" 2>&1 | tail -2
done
sleep 10

echo "=== 5. 门禁：必须是 ${EXPECT_GPU} 卡 ==="
export PATH="${CONDA_BIN}:${PATH}"
export RAY_ADDRESS=${HEAD_IP}:${PORT}
ray status 2>&1 | grep -E 'GPU' | head -3

TOTAL_GPU=$(ray status 2>/dev/null | grep -oE '[0-9.]+/[0-9.]+ GPU' | grep -oE '/[0-9.]+' | tr -d '/' | head -1)
echo ""
echo "集群总 GPU: ${TOTAL_GPU:-unknown}"
if [ "${TOTAL_GPU%.*}" != "${EXPECT_GPU}" ]; then
    echo "ERROR: 期望 ${EXPECT_GPU} 卡，实际 ${TOTAL_GPU}。不要启动训练（会静默 pending 挂住）。" >&2
    exit 1
fi

# 验证 NCCL 环境真的到了每个 worker 进程里 —— 光看 ray status 不够
# SWANLAB_API_KEY 一起查：它缺了的话训练会在 vLLM 起完之后才炸（约 7 min），
# 与其等那么久，不如在这里就失败。
echo "=== 6. 抽查 worker 的 NCCL / SwanLab 环境 ==="
timeout 90 python -c "
import ray
ray.init(address='auto', log_to_driver=False)
@ray.remote(num_gpus=1)
def probe():
    import os, socket, tempfile
    # 直接复刻 swanlab safe_mkdir 的探测二：共享盘上 TemporaryFile 会抛
    # FileNotFoundError（不支持 unlink-while-open），只查 startswith('/') 查不出来。
    def _swanlog_ok():
        d = os.environ.get('SWANLAB_LOG_DIR') or ''
        if not d.startswith('/'):
            return 'BAD(relative)'
        try:
            os.makedirs(d, exist_ok=True)
            with tempfile.TemporaryFile(dir=d, prefix='.probe_') as f:
                f.write(b'0')
            return 'ok'
        except Exception as e:
            return f'BAD({type(e).__name__})'
    # 光查环境变量抓不住 SWANLAB_SAVE_DIR 那个坑（父目录不存在 -> import 就炸），
    # 所以这里真的 import 一次。放在最后，失败也只影响本探针。
    def _import_ok():
        try:
            import swanlab  # noqa: F401
            return 'ok'
        except Exception as e:
            return f'BAD({type(e).__name__})'
    return (socket.gethostbyname(socket.gethostname()), os.environ.get('NCCL_NET_GDR_LEVEL'),
            os.environ.get('NCCL_SOCKET_IFNAME'), bool(os.environ.get('SWANLAB_API_KEY')),
            _swanlog_ok(), _import_ok())
seen = {}
for ip, gdr, ifn, sw, swdir, imp in ray.get([probe.remote() for _ in range($EXPECT_GPU)]):
    seen[ip] = (gdr, ifn, sw, swdir, imp)
bad = [k for k, v in seen.items() if v[0] != '0' or not v[2] or v[3] != 'ok' or v[4] != 'ok']
for ip, v in sorted(seen.items()):
    print(f'  {ip}: NCCL_NET_GDR_LEVEL={v[0]} IFNAME={v[1]} SWANLAB_API_KEY={\"set\" if v[2] else \"MISSING\"} SWANLAB_LOG_DIR={v[3]} import_swanlab={v[4]}')
raise SystemExit(1 if bad else 0)
" 2>&1 | grep -E '^  [0-9]' || { echo "ERROR: NCCL/SwanLab 环境没传到 worker（跨机 FSDP 会失败 / swanlab.init 会要求交互登录或写不了 log_dir）" >&2; exit 1; }

echo "OK: ${#NODE_IPS[@]} 机 ${EXPECT_GPU} 卡就绪"
