#!/usr/bin/env bash
# 重建 SandboxFusion（Code reward + LCB v5/v6 打分需要）。
#
# 本机无 docker/podman，用静态 docker 二进制 + /dev/shm 当 data-root：
# 容器根目录是 overlayfs，不能再叠 overlay2；tmpfs 可以。
#
# 代价：产物在内存盘，Pod 重启即丢。所以本脚本起来后会 docker save 到共享盘，
# 下次重启直接 load，不必再走代理拉镜像。
#
# 用法: bash setup_sandbox.sh
set -uo pipefail

B=/taijifs_sz34/share_304169095/hunyuan/yongcanyu
D=${B}/docker_build/docker-static-29.7.2
SOCK=unix:///tmp/docker-build.sock
IMAGE=vemlp-cn-beijing.cr.volces.com/preset-images/code-sandbox:server-20250609
IMAGE_TAR=${B}/docker_build/code-sandbox-server-20250609.tar
LOG_DIR=${B}/logs/llm_fusion
mkdir -p "${LOG_DIR}"

DOCKER="${D}/docker -H ${SOCK}"

probe() {
    curl -s -m 10 --noproxy '*' -X POST http://localhost:8080/run_code \
        -H 'Content-Type: application/json' \
        -d '{"code":"print(2+2)","language":"python"}' 2>/dev/null | grep -q Success
}

if probe; then
    echo "sandbox 已在运行，无需重建"
    exit 0
fi

# ---- 1. dockerd ----
# 关键：dockerd 自己要带代理，否则 pull 超时（拉镜像是守护进程发起的，不是客户端）。
if ! pgrep -f "dockerd -H ${SOCK}" > /dev/null 2>&1; then
    echo "[1/4] 启动 dockerd ..."
    mkdir -p /dev/shm/docker-data
    # setsid 防止前台命令超时时 SIGTERM 连坐整个进程组（nohup 只挡 SIGHUP）
    env http_proxy="http://star-proxy.oa.com:3128" \
        https_proxy="http://star-proxy.oa.com:3128" \
        HTTP_PROXY="http://star-proxy.oa.com:3128" \
        HTTPS_PROXY="http://star-proxy.oa.com:3128" \
        no_proxy=".woa.com,localhost,127.0.0.1,.oa.com" \
        PATH="${D}:${PATH}" \
        setsid nohup "${D}/dockerd" -H "${SOCK}" \
            --data-root=/dev/shm/docker-data --storage-driver=overlay2 \
            --iptables=false --bridge=none --userland-proxy=false \
            > "${LOG_DIR}/dockerd.log" 2>&1 < /dev/null & disown

    for i in $(seq 1 30); do
        sleep 2
        ${DOCKER} info > /dev/null 2>&1 && break
    done
fi

if ! ${DOCKER} info > /dev/null 2>&1; then
    echo "ERROR: dockerd 起不来，看 ${LOG_DIR}/dockerd.log" >&2
    exit 1
fi
echo "dockerd OK"

# ---- 2. 镜像：优先从共享盘 load，没有再走代理 pull ----
if ! ${DOCKER} image inspect "${IMAGE}" > /dev/null 2>&1; then
    if [ -f "${IMAGE_TAR}" ]; then
        echo "[2/4] 从共享盘 load 镜像（${IMAGE_TAR}）..."
        ${DOCKER} load -i "${IMAGE_TAR}" > "${LOG_DIR}/sandbox_load.log" 2>&1
    else
        echo "[2/4] 共享盘无备份，走代理 pull（约 5~10 min）..."
        setsid nohup ${DOCKER} pull "${IMAGE}" \
            > "${LOG_DIR}/pull_sandbox.log" 2>&1 < /dev/null & disown
        PULL_PID=$!
        while kill -0 ${PULL_PID} 2>/dev/null; do sleep 10; done
    fi
fi

if ! ${DOCKER} image inspect "${IMAGE}" > /dev/null 2>&1; then
    echo "ERROR: 镜像没拿到，看 ${LOG_DIR}/pull_sandbox.log" >&2
    exit 1
fi
echo "镜像 OK"

# ---- 3. 起容器 ----
echo "[3/4] 启动 sandbox 容器 ..."
${DOCKER} rm -f sandbox > /dev/null 2>&1
${DOCKER} run -d --name sandbox --network host --shm-size=2g "${IMAGE}" \
    > "${LOG_DIR}/sandbox_run.log" 2>&1

for i in $(seq 1 60); do
    sleep 3
    probe && break
done

if ! probe; then
    echo "ERROR: sandbox 探活失败。容器日志：" >&2
    ${DOCKER} logs --tail 30 sandbox >&2 2>&1
    exit 1
fi
echo "sandbox 探活通过 (localhost:8080)"

# ---- 4. 备份镜像到共享盘，规避 Pod 重启丢失 ----
if [ ! -f "${IMAGE_TAR}" ]; then
    echo "[4/4] docker save 备份到共享盘（后台，约数 G）..."
    setsid nohup ${DOCKER} save -o "${IMAGE_TAR}" "${IMAGE}" \
        > "${LOG_DIR}/sandbox_save.log" 2>&1 < /dev/null & disown
    echo "     备份在后台进行：${IMAGE_TAR}"
else
    echo "[4/4] 共享盘已有备份，跳过"
fi

echo "DONE"
