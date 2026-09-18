# start.md — 启动实验的最小步骤

详细原理见 `HANDOFF.md` / `overview.md`。本文只列「做什么」。
以下步骤 2026-09-11 在 4 机 32 卡上实跑过一遍（math 域）。

```bash
B=/taijifs_sz34/share_304169095/hunyuan/yongcanyu
cd $B/code/LLM-Fusion
```

## 0. 两个必设的环境变量

```bash
export SWANLAB_LOG_DIR=/tmp/swanlog_llm_fusion   # Pod 预置的是共享盘路径，必须显式覆盖
export PROJECT_NAME=llm_fusion_repro             # llm_fusion 这名字 422
# 仅 1 机 8 卡时才要： export ROLLOUT_GPU_MEM_UTIL=0.7
```

## 1. 确认 Python 环境（每节点）

```bash
for ip in $(echo "$NODE_IP_LIST" | tr ',' '\n' | cut -d: -f1); do
  echo -n "$ip: "; ssh -o BatchMode=yes $ip \
    "/opt/conda/envs/torch-base/bin/python -c 'import verl,swanlab,vllm' && echo ok || echo MISSING"
done
```

> 2026-09-11 这台 Pod 的 4 个节点**开箱就是 ok**（env 随镜像带），没动过安装。
> 若真出现 `MISSING`，安装步骤照 `overview.md`（HANDOFF §10 提示原「新节点环境搭建」
> 一节已被删，只剩 overview 里那份）—— 未在本 Pod 上验证过。

## 2. sandbox（每节点都要，Pod 重启即丢）

4 节点并行，约 3~5 min：

```bash
R=$PWD
for ip in $(echo "$NODE_IP_LIST" | tr ',' '\n' | cut -d: -f1); do
  ssh -o BatchMode=yes $ip "cd $R && bash $R/setup_sandbox.sh" > /tmp/sbx_$ip.log 2>&1 &
done; wait
```

## 3. Ray 集群（只在 head 跑）

```bash
bash setup_ray_cluster.sh     # 期望末行 "OK: N 机 M 卡就绪"
```

⚠️ 第 6 步抽查可能打印**少于节点数**的 IP（32 个任务随机采样，可能漏掉某节点；
还可能打出无关网卡 IP 如 `28.41.172.73`），不代表缺节点。自己确认：

```bash
RAY_ADDRESS=$CHIEF_IP:6379 ray status | grep GPU
```

## 4. 启动训练

```bash
bash run_train_multinode.sh math \
  "trainer.default_local_dir=${B}/runs/llm_fusion/checkpoints/llm_fusion/expert_math_qwen3_4b_4x8"
```

- 域可选 `agent|math|science|code|if|mix`；实验名默认带 `${NNODES}x${NGPUS}` 后缀。
- **`default_local_dir` override 必须带**（理由见 §5 云端 swanlab 第 4 条）。
- ⚠️ 启动 banner 打的 ckpt 路径**不含**你的 override，别被它误导。

## 5. 云端 SwanLab（本项目唯一的指标现场，坑最多）

**1) Pod 会注入一整套 `SWANLAB_*`，并且会打败脚本默认值。**
两个脚本写的都是 `${SWANLAB_LOG_DIR:-/tmp/...}`，而 `${VAR:-默认}` 的语义是
「VAR 已设就用 VAR」，于是 Pod 的共享盘路径直接赢。所以 §0 那行 export 不能省。
同理，实际用的 host 是 Pod 的 **`train-exp`**，不是脚本默认、也不是 HANDOFF §7 写的
`taiji-exp` —— 看指标别找错实例。凭据（API_KEY）用 Pod 自带的即可。

**2) `SWANLAB_LOG_DIR` 必须节点本地，不能放共享盘。**
swanlab 用 `tempfile.TemporaryFile(dir=p)` 探测可写性（创建后立即 unlink 但保持打开），
共享盘不支持这语义 → `FileNotFoundError` → 被当成可重试延迟自旋到超时 →
报成误导性的 `FILESYSTEM may be slow or remote`。一行判断任意目录能不能用：

```bash
python -c "import tempfile; tempfile.TemporaryFile(dir='<目录>')"
```

swanlog 只是 SDK 本地缓存，**指标照样上传云端**，放 /tmp 不损失任何东西。

> **「放 /tmp 别的机器不就看不到了？」** 不会，两个原因（2026-09-11 实测）：
> 一个作业只有**一次** `swanlab.init()`（在 TaskRunner/driver 里），不是每节点一次 ——
> 实测 head 上有 run 目录，另外 3 个节点是**空目录**（只是启动脚本 `mkdir -p` 建的），
> 31 个 worker 根本不写日志，指标回传给 driver。跨机可见性靠的是云端那个 run URL，
> 不是文件系统。
> 真正的代价只有一条：**上传失败时**可重放的本地缓存在 head 的 /tmp 上，Pod 重启即丢；
> 但放共享盘换不来这份容错 —— `swanlab.init()` 会直接失败，连 run 都建不起来。
>
> ⚠️ 反向情形：`MOPD/mopd_sft/run_qwen3_4b_m2rl_sft_full.sh` 注释写「必须放共享盘」，
> 那对 **torchrun** 成立（每个 rank 各自解析路径、各写日志，放 /tmp 会打散数据）。
> verl 是 Ray-based 单 driver，不适用。**别照抄。**

**3) 换实例/账号后先 20 秒探项目名，别赔 8 分钟。**
`swanlab.init()` 在 vLLM 起完之后（约 8 min）才会抛 `POST /api/project 422` ——
该名字被别的 workspace 占用，当前账号建不了也写不进去，带 `workspace=` 无效。
本次实测 `llm_fusion` 仍 422、`llm_fusion_repro` 正常
（落在 `https://train-exp.taiji.woa.com/@yb_mm_edu/llm_fusion_repro`）：

```bash
python -c "
import swanlab, os
swanlab.login(api_key=os.environ['SWANLAB_API_KEY'], host=os.environ['SWANLAB_WEB_HOST'], save=False)
swanlab.init(project='<名字>', experiment_name='probe', mode='cloud'); swanlab.finish()"
```

> 探测会在项目里留一条 `probe` run，无害，可在 UI 删。

**4) 改了 `PROJECT_NAME` 就必须钉回 checkpoint 目录。**
`PROJECT_NAME` 同时决定 `trainer.default_local_dir`
（`${OUTPUT_ROOT}/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}`），
不 override 会让这个 expert 和另外四个**分家**，而 Merge 要五个凑齐。

**5) 凭据必须在 `ray start` 的 shell 里注入**（`setup_ray_cluster.sh` 已做）。
verl 下发给 Ray worker 的 `runtime_env` 是写死白名单（`verl/trainer/constants_ppo.py`），
`SWANLAB_API_KEY` 不在其中 —— 只在训练脚本里 export 传不到 TaskRunner，
`swanlab.init()` 会退化成交互式登录并抛 `no TTY is available`。
同一个根因也坑了 `NCCL_NET_GDR_LEVEL` / `PYTORCH_CUDA_ALLOC_CONF` / `TENSORBOARD_DIR`。

**6) `trainer.logger` 至少留一个真 backend。**
verl 只在 `val_only` 路径把指标打到 console；训练循环的 per-step 指标**只走 backend**。
只留 `["console"]` 日志里除了进度条什么都没有。默认已是
`["console","swanlab","tensorboard"]`（没有 wandb 凭据，别用默认值）。

## 6. 启动后核对（约 8 min 后）

```bash
L=$(ls -t $B/logs/llm_fusion/train_*.log | head -1)
grep -o "'default_local_dir': '[^']*'" $L | head -1        # 确认 override 生效
grep -iE "EADDRINUSE|Traceback|ncclSystemError|ApiError|CUDA out of memory" $L | head
```

- `EADDRINUSE`：约一半概率，**直接重跑第 4 步**，不要设 `VLLM_PORT`。
- reward 看 `python summarize_rollout.py $OUTPUT_ROOT/rollout_data/<实验名>`。
- Agent 域额外判据：`num_turns/mean` 必须远大于 1。
