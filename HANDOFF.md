# HANDOFF — LLM-Fusion 复现，换机器接手指南

> 面向「在一台新的 Taiji 机器上把这套东西跑起来」。
> 详细的踩坑分析见 `overview.md`；本文档只讲**接手要做什么、哪些东西不能照抄**。
> 写于 2026-09-11。

---

## 0. TL;DR

共享盘上的东西（代码、数据、权重、镜像 tar）**全部可以直接复用**，不用重新下载。
每台新机器需要自己做的只有三件事：**装 Python 环境 → 起 sandbox → 起 Ray 集群**。

```bash
B=/taijifs_sz34/share_304169095/hunyuan/yongcanyu
cd $B/code/LLM-Fusion

# 1. Python 环境：每个节点各做一次，约 10 min。照 §2 抄命令（没有一键脚本，
#    因为要走代理且 --no-deps 这类细节漏一个就会坏环境）。

# 2. sandbox：⚠️ 每个节点都要，从共享盘 load 约 3~5 min
bash setup_sandbox.sh                                    # head
ssh <worker_ip> "cd $PWD && bash $PWD/setup_sandbox.sh"  # 每个 worker

# 3. Ray 集群：只在 head 跑，会自动 ssh 到所有 worker
bash setup_ray_cluster.sh

# 4. 训练
bash run_train_multinode.sh agent     # 或 math/science/code/if/mix
```

第 3、4 步的脚本**已改成从 `NODE_IP_LIST` 自动推导拓扑**，换机器不用改代码。

> 第 4 步首次启动约一半概率会炸在 vLLM 的 `EADDRINUSE`（端口竞争）——
> **直接重跑同一条命令即可**，别去改端口配置。见 §5 末。

> ⚠️ **2026-09-11 补：新 Pod 上第 4 步之前还要先加两行**，否则必炸（都不报在点子上）：
>
> ```bash
> export SWANLAB_LOG_DIR=/tmp/swanlog_llm_fusion   # Pod 会预置一个共享盘路径，见 §4
> export ROLLOUT_GPU_MEM_UTIL=0.7                  # 仅单机 8 卡时；0.9 会在 step 3 OOM，见 §5
> ```
>
> 另外若 `swanlab.init()` 报 `422`，是项目名被占用，换 `PROJECT_NAME` 并同时用
> `trainer.default_local_dir` override 把 checkpoint 钉回原目录（见 §4）。


---

## 1. 什么在共享盘（可复用），什么是节点本地（要重做）

这是接手时最容易搞错的地方。

| 东西 | 位置 | 换机器要重做吗 |
| :--- | :--- | :--- |
| 代码（含 verl 源码） | `$B/code/LLM-Fusion` | ❌ 不用 |
| 训练/测试数据（3.5G） | `$B/data/LLM-Fusion/{train,test}` | ❌ 不用 |
| 权重（backbone + 3 融合模型 + 5 expert） | `$B/checkpoints/` | ❌ 不用 |
| SandboxFusion 镜像 tar（25G） | `$B/docker_build/code-sandbox-server-20250609.tar` | ❌ 不用（直接 load） |
| 训练产物 / checkpoint | `$B/runs/llm_fusion/` | ❌ 不用 |
| **conda 环境 + pip 包** | `/opt/conda/envs/torch-base`（**节点本地**） | ✅ **每节点都要装** |
| **verl 的 editable 注册**（`.pth`） | `/opt/conda/.../site-packages/`（**节点本地**） | ✅ **每节点都要装** |
| **NLTK 语料** | `/root/nltk_data`（**节点本地**） | ✅ **每节点都要下** |
| **运行中的 sandbox 容器** | `/dev/shm/docker-data`（**内存盘**） | ✅ **每节点都要起，Pod 重启即丢** |

> **关键点**：verl 的**源码**在共享盘，但**「这个环境里装了 verl」这件事**记在节点本地的
> `site-packages/__editable__.verl-*.pth` 里。所以新节点上 `import verl` 会失败，
> 必须重跑一次 `pip install -e . --no-deps`（它只写一个 .pth，不复制代码，很快）。

---
## 2. sandbox：每个节点都要起一个

Code 域的 reward、以及 LCB v5/v6 的评测，都要在 SandboxFusion 里跑单测。

```bash
bash setup_sandbox.sh                                        # head
ssh <worker_ip> "cd <repo> && bash <repo>/setup_sandbox.sh"  # 每个 worker
```

脚本会自动「已在跑就跳过 → 共享盘有 tar 就 `docker load` → 没有才走代理 pull」，
并在 pull 后自动 `docker save` 补备份。**有 tar 时约 3~5 min，不碰网络。**

### ⚠️ 这是本项目最隐蔽的坑（2026-09-11 实际踩到）

`localhost:8080` **只对起了容器的那个节点有效**。打分任务由 Ray 分散到所有 worker，
落在没有 sandbox 的节点上全部 `Connection refused`，然后**静默 fallback 到另一个
scorer** —— 不报错、不中断，只是分数错。

**症状**：LCB v5/v6 稳定停在 **0.30 左右**（论文 base 0.59），其余六个基准都正常；
且多次验证**纹丝不动**（0.3085 → 0.3084 → 0.3012）。

**判据**（可推广）：**「分数正好腰斩」+「多次测量几乎不变」= 一半 worker 走了另一条
代码路径**，而不是模型能力退化（那会随训练单调移动）。

**所以验证必须逐节点做，`head` 通不代表全集群通**：

```bash
for ip in $(echo "$NODE_IP_LIST" | tr ',' '\n' | cut -d: -f1); do
  echo -n "$ip: "
  ssh -o BatchMode=yes $ip "curl -s -m 8 --noproxy '*' -X POST http://localhost:8080/run_code \
    -H 'Content-Type: application/json' -d '{\"code\":\"print(6*7)\",\"language\":\"python\"}'" \
    | grep -o '42' || echo FAIL
done
```

`run_train_multinode.sh` 启动前会自动做这个检查并在缺失时拒绝启动。

> 另注：容器跑在 `/dev/shm`（内存盘），**Pod 重启即丢**，重启后要重跑本节。
> 镜像 tar 在共享盘，所以重建不需要重新 pull。

---

## 3. Ray 集群

**verl 自己不起集群**（`examples/llm_fusion/train/*.sh` 里零个 `ray start`），
它靠 `ray.init()` 发现已有集群。

```bash
bash setup_ray_cluster.sh     # 只在 head 跑，会自动 ssh 到所有 worker
```

脚本做六件事：清理所有节点残留 → 确认无进程占卡 → 起 head → 起所有 worker →
**卡数门禁** → **抽查 worker 的 NCCL + SwanLab 环境**。

第 6 步会在每个节点的 Ray worker 里实际打印 `os.environ`，校验四项：
`NCCL_NET_GDR_LEVEL=0`、`NCCL_SOCKET_IFNAME`、`SWANLAB_API_KEY` 已设、
`SWANLAB_LOG_DIR` 可写（**直接复刻 swanlab 的 `TemporaryFile` 探测** —— 只检查
是不是绝对路径查不出 §5 那个共享盘坑）。缺任何一项都在 30 秒内失败，
而不是等 vLLM 起完 8 分钟才炸。期望输出：

```
  28.45.5.209: NCCL_NET_GDR_LEVEL=0 IFNAME=bond1 SWANLAB_API_KEY=set SWANLAB_LOG_DIR=ok
  28.45.6.151: NCCL_NET_GDR_LEVEL=0 IFNAME=bond1 SWANLAB_API_KEY=set SWANLAB_LOG_DIR=ok
OK: 2 机 16 卡就绪
```


### ⚠️ 为什么不需要 pssh fan-out（别照抄 MOPD 那套）

`MOPD/mopd_sft/launch_multinode.sh` 用 pssh 把训练脚本分发到每台机器，**因为
torchrun 不是 Ray**：torchrun 只 fork 本地 8 个 rank，必须每台各起一份、各带不同
`--node_rank`。

verl 是 Ray-based：**head 单点提交一次，Ray 自己把 WorkerDict 调度到全集群**。
照搬 fan-out 会起两份互相抢卡的作业。（那个脚本自己的注释也写明了这个区别。）

### ⚠️ 卡数不够时 verl 会静默 pending 挂住

不报错、不超时，就是不动。所以脚本有硬门禁：
`ray status` 的总 GPU 必须等于 `节点数 × 每节点卡数`，否则拒绝继续。

接手时如果看到「`ray status` 显示 N 个节点但总 GPU 偏少」，
很可能是某个节点的 raylet 死了而 gcs_server 残留（`runtime_env_agent failed and
raylet fate-shares`）。`ray stop --force` **杀不掉训练脚本内嵌起的集群**，
必须补 `pkill -9 -f "ray::|raylet|gcs_server"`。脚本已经这么做了。

---

## 4. ⚠️ NCCL：最容易卡住换机器的一关

两机以上首次启动时，跨机 FSDP 初始化可能直接失败：

```
NCCL WARN Call to ibv_reg_mr_iova2 failed with error Invalid argument
torch.distributed.DistBackendError: ... ncclSystemError
```

**原因**：当前栈（NCCL 2.28.9 + CUDA 13.0 + driver 535.247.01 + `mlx5_bond_*`）在
**GPUDirect RDMA 注册 GPU 显存**这条路径上不兼容。网络本身是好的。

**解法**：`NCCL_NET_GDR_LEVEL=0` —— 关掉 GPU 显存直连 RDMA、**保留 IB/RoCE**，
跨机数据改走 pinned host memory 中转。不要一上来就 `NCCL_IB_DISABLE=1`（退回 TCP，慢得多）。

### 但这里有个 verl 专属的陷阱

**在训练脚本里 `export NCCL_NET_GDR_LEVEL=0` 是无效的。**
verl 给 Ray worker 下发的 `runtime_env` 是一份**写死的白名单**
（`verl/trainer/constants_ppo.py` 的 `PPO_RAY_RUNTIME_ENV`），`NCCL_NET_GDR_LEVEL`
不在其中。真正干活的 `WorkerDict` 进程由 raylet 拉起，**不继承你提交作业那个 shell
的环境** —— export 了等于没设，而且不报任何错。

**正确做法：在 `ray start` 的 shell 里设**（`setup_ray_cluster.sh` 已经这么做）。
脚本第 6 步会抽查每个 worker 的 `os.environ` 确认生效，不生效就报错退出。

> 同一个根因也解释了 `TENSORBOARD_DIR` 为什么不生效（指标会落到 verl 的默认相对路径
> `<repo>/tensorboard_log/`）。**凡是「由 daemon 拉起的进程」（raylet、dockerd、systemd）
> 都不继承你的 shell**；排查方法一律是「在真正干活的那个进程里打印 `os.environ`」。

### ⚠️ `SWANLAB_LOG_DIR` 不能放共享盘（2026-09-11 补，报错极具误导性）

凭据修好之后紧接着会遇到这个：

```
TimeoutError: Directory [.../runs/llm_fusion/swanlog] exists but is not writable
within 5.0s, FILESYSTEM may be slow or remote.
```

**这个报错是错的** —— 那个目录权限没问题，`touch` + `rm` 只要 **20ms**。

真实原因：swanlab 0.10.0 的 `safe_mkdir`（`swanlab/sdk/internal/pkg/fs/dir.py`）
用 `tempfile.TemporaryFile(dir=p)` 探测可写性，而该调用会「创建后立即 unlink 但保持
打开」。**这个共享盘不支持 unlink-while-open**，直接抛 `FileNotFoundError`；它是
`OSError` 的子类，被 `safe_mkdir` 的 `except OSError` 当成「可重试的文件系统延迟」
自旋到 5s 超时，最后报成上面那句关于「慢/远程文件系统」的话。

一行验证（这也是判断任意目录能不能给 swanlab 用的通用方法）：

```bash
python -c "import tempfile; tempfile.TemporaryFile(dir='<共享盘目录>')"  # FileNotFoundError
python -c "import tempfile; tempfile.TemporaryFile(dir='/tmp')"         # 正常
```

**解法**：`SWANLAB_LOG_DIR=/tmp/swanlog_llm_fusion`（节点本地）。两个脚本都已改。
swanlog 只是 SDK 的本地缓存，**指标照样上传到 taiji-exp**，不是训练产物，
放本地不损失任何东西，Pod 重启丢掉也无所谓。

> 注意这跟 MOPD 的经验相反 —— `MOPD/mopd_sft/run_qwen3_4b_m2rl_sft_full.sh` 的注释
> 写着「必须放共享盘」。那条对 torchrun 成立（每个节点各自解析这个路径），且当时的
> swanlab 版本还没有这个探测。别照抄。

### ⚠️ Pod 自身 export 的 `SWANLAB_*` 会打败脚本默认值（2026-09-11 补）

上一节的解法在**新 Pod 上会失效**。两个脚本写的都是
`${SWANLAB_LOG_DIR:-/tmp/swanlog_llm_fusion}`，而 Taiji 的「模型开发任务」会自动往
环境里注入一整套自己的 SwanLab 配置：

```
SWANLAB_LOG_DIR=/taijifs_sz34/.../basic_train_<任务id>/<实例id>/log   ← 共享盘！
SWANLAB_API_HOST=https://train-exp.taiji.woa.com/api                  ← 不是 taiji-exp
SWANLAB_API_KEY / SWANLAB_PROJ_NAME / SWANLAB_EXP_NAME / SWANLAB_WEBHOOK ...
```

`${VAR:-默认}` 的语义是「**VAR 已设就用 VAR**」，于是 Pod 那个共享盘路径直接赢过
`/tmp`，上一节的坑照样触发。**接手时必须显式覆盖**（只覆盖这一个，凭据用 Pod 自带的）：

```bash
export SWANLAB_LOG_DIR=/tmp/swanlog_llm_fusion
```

> 幸好 §3 第 6 步的门禁把它拦下来了（报 `SWANLAB_LOG_DIR=BAD(FileNotFoundError)`），
> 没等到 8 分钟后 vLLM 起完才炸。这道门禁值回票价 —— 别因为「看着啰嗦」把它删了。
>
> **可推广的教训**：凡是平台可能预置的变量，`${VAR:-默认}` 只是「建议值」不是「保证值」。

### ⚠️ SwanLab 项目名 `llm_fusion` 可能 422（2026-09-11 补）

换实例/换账号后，`swanlab.init()` 会在 vLLM 起完之后（约 8 min）才抛：

```
swanlab.error.ApiError: ('Trace id: ...', 'POST https://<host>/api/project',
                         '422 Unprocessable Entity')
```

二分结论：**key 有效、建新项目正常、只有 `llm_fusion` 这个名字 422** —— 该名字在实例上
已被别的 workspace 占用，当前账号（`yb_mm_edu`）建不了也写不进去；带 `workspace=` 参数
无效。**taiji-exp 和 train-exp 两个实例表现一致，所以别在 host 上浪费时间。**

判据（20 秒，别再赔 8 分钟）：

```bash
python -c "
import swanlab, os
swanlab.login(api_key=os.environ['SWANLAB_API_KEY'], host=os.environ['SWANLAB_WEB_HOST'], save=False)
swanlab.init(project='<换个没人用过的名字>', experiment_name='probe', mode='cloud'); swanlab.finish()"
```

**解法**：换 `PROJECT_NAME`，但必须把 checkpoint 目录钉回原位 —— `PROJECT_NAME` 同时
决定 `trainer.default_local_dir`（`${OUTPUT_ROOT}/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}`），
直接改会让这个 expert 和另外四个**分家**，而 Merge 要五个凑齐：

```bash
export PROJECT_NAME=llm_fusion_repro
bash run_train_multinode.sh if \
  "trainer.default_local_dir=${B}/runs/llm_fusion/checkpoints/llm_fusion/expert_if_qwen3_4b_1x8"
```

脚本把 `"$@"` 追加在固定 override **之后**，hydra 后者胜，所以盖得掉
（同理可证：`trainer.logger` 在命令行里本来就出现两次）。
⚠️ 启动 banner 打的是脚本自己算的路径，**不含你追加的 override**，别被它误导；
要确认实际生效值，grep 日志里的 `'default_local_dir': '...'`。

### ⚠️ `PYTORCH_CUDA_ALLOC_CONF` 也是白名单受害者（2026-09-11 补）

`overview.md` 5.4 给 vLLM OOM 开的是两味药：`ROLLOUT_GPU_MEM_UTIL` 下调 **+**
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。但后者**实际一直没生效** ——
它和 `NCCL_NET_GDR_LEVEL` 一样不在 `PPO_RAY_RUNTIME_ENV` 白名单里（可自行通读
`verl/trainer/constants_ppo.py`，全表就十来项），所以
`run_train_multinode.sh` 里那句 export 对 Ray worker 是空操作。

已按同样办法加进 `setup_ray_cluster.sh` 的 `NCCL_ENV` 数组。验证：

```bash
python -c "
import ray; ray.init(address='auto', log_to_driver=False)
@ray.remote(num_gpus=1)
def f():
    import os; return os.environ.get('PYTORCH_CUDA_ALLOC_CONF','<unset>')
print(ray.get(f.remote()))"          # 期望 expandable_segments:True
```

### ⚠️ 启动偶发 `EADDRINUSE`：重跑就行，别去设 `VLLM_PORT`

首次启动约一半概率会炸在（2026-09-11 实测 4 次中 2 次，两次都在 worker）：

```
torch.distributed.DistNetworkError: The server socket has failed to listen on any
local network address. port: 44643, ... EADDRINUSE
```

机制是 TOCTOU：vLLM 的 `get_open_port()` 走 `bind(("",0))` 让内核从
`ip_local_port_range`（本机 32768–60999）分配，**探测到「空闲」与真正 bind 之间有
时间窗**，而每节点 8 个 engine 并发 init，期间内核可能把同一端口分给某条出向连接。

**解法：直接重跑 `run_train_multinode.sh`。** 这是随机竞争，重试就过。

⚠️ **不要试图用 `VLLM_PORT` 固定端口** —— 试过，更糟，已回滚。设了之后
`_get_open_port()` 改走「从固定基址往上扫」（`vllm/utils/network_utils.py:196-205`），
8 个 engine 各自独立扫、探测完都把 socket 关掉，于是**全部选中同一个基址端口**，
只有第一个 `TCPStore` bind 成功 —— 偶发变必然。脚本里已写明这条，别重犯。

---

## 5. 启动训练

```bash
bash run_train_multinode.sh agent      # 300 步（论文 Table 3）
bash run_train_multinode.sh math       # science / code / if 同理
bash run_train_multinode.sh mix        # 800 步，自动切到 train_mix_rl.sh
bash run_train_multinode.sh code trainer.val_before_train=True   # 可追加 hydra override
```

拓扑自动推导，超参默认全按论文（batch 128 / mini 128 / n=16 / resp 16384 /
prompt 5120 / lr 1e-6 / GRPO + `norm_adv_by_std_in_grpo=FALSE` / KL loss
`low_var_kl` coef 0.001 / temp 1.0）。所有 `XXX=${XXX:-默认}` 都可以用环境变量覆盖。

### ⚠️ Agent 域必须传 `default_agent_loop=tool_agent`

Agent 的训练行**没有 `agent_name` 列**（实测 10229 行全无），不传这个会 fallback 到
`single_turn_agent` —— **单轮、无工具、reward 直接塌，且不报任何错**。
脚本的 `agent` 分支已经带上了；自己手写命令时别漏。

**生效判据：`num_turns/mean` 必须远大于 1**（实测 7.8~10.5，且随训练上升）。
等于 1 就是退化了，立刻停。

### ⚠️ `env_llm_fusion.sh` 里的 NNODES 是硬 export

它写的是 `export NNODES=1`（不是 `${NNODES:-1}`），所以**拓扑必须在 `source` 之后
覆盖**，否则永远是单机 8 卡。两个多机脚本都已处理。

### ⚠️ 单机 8 卡必须下调 `ROLLOUT_GPU_MEM_UTIL`（2026-09-11 实测，step 3 必炸）

论文那套 `gpu_memory_utilization=0.9` 是按 **2 机 16 卡**调的。换成 **1 机 8 卡**后，
训练会正常跑完 step 1~2，然后在 **step 3** 炸 CUDA OOM：

```
GPU 0: 95.00 GiB 总量，剩 478 MiB
  PID 422328 →  7.84 GiB   ← verl 自己的 actor FSDP worker
  PID 440432 → 86.68 GiB   ← vLLM（0.9 × 95 = 85.5）
```

**注意这跟 `overview.md` 5.4 不是一回事**：那次是外部 `gpu_hold.py` 占位，本次报错里
**两个 PID 都是本作业自己的**。根因是 actor 按 `world_size` 做 FSDP 分片，**卡数减半
→ 每卡权重/优化器状态翻倍**（16 卡约 3.9G，8 卡 7.84G）；而 `gpu_memory_utilization`
是占**总**显存的比例、不会随之让路，于是 85.5 + 7.84 = 93.3G，只剩 1.7G 余量，
一个 550 MiB 的分配就够炸。

**解法**：

```bash
export ROLLOUT_GPU_MEM_UTIL=0.7      # 0.7 × 95 = 66.5G，给 actor 留 28G
```

实测峰值显存 94.5G → 77.3G（余量 17G），而步时 288 → 297 s/it，**吞吐几乎没损失**
（KV cache 变少但本来就不是瓶颈）。

> **可推广的判据**：「前两步正常、第 3 步 OOM」几乎一定是**显存预算问题而非泄漏** ——
> 稳态峰值要到几步之后才出现。真泄漏会单调爬升。
> 换拓扑时要重新审视的不只是 `nnodes`，还有**所有按「占总量比例」写死的参数**。

### ⚠️ swanlab 预检的 `2>&1` bug（2026-09-11 已修）

`run_train_multinode.sh` 原来写的是 `ok=$(python -c "import swanlab" 2>&1 && echo ok)`，
而 swanlab 0.7.20 在 import 时会往 stderr 打一条
`FutureWarning: The pynvml package is deprecated`，被 `2>&1` 一起塞进 `${ok}`，于是
`${ok}` == `"...警告...\nok"` != `"ok"` —— **明明装着也判成没装，直接拒绝启动**。
已改成 `2>/dev/null` 只看退出码。同类写法（把 stderr 折进变量再做字符串比较）别再犯。

---

## 7. 看指标

```bash
# SwanLab（自建实例，推荐）
http://taiji-exp.taiji.woa.com/

# tensorboard 兜底
tensorboard --logdir <repo>/tensorboard_log

# reward 快查（无人值守时的唯一现场）
python summarize_rollout.py $OUTPUT_ROOT/rollout_data/<experiment_name>
```

**没有 wandb 凭据**，别用 `trainer.logger` 的默认值（含 wandb）。
默认已改成 `["console","swanlab","tensorboard"]`。

⚠️ verl 只在 `val_only` 路径把聚合指标打到 console；**训练循环的 per-step 指标只走
logger backend**。所以如果只留 `["console"]`，日志里除了进度条什么都没有 ——
必须至少留一个真正的 backend，或者靠 `trainer.rollout_data_dir` 的 jsonl。

---

## 8. 当前进度（交接时点）

⚠️ **同时有两台机器在跑，别把日志看混了**（2026-09-11 踩到：共享盘上的 Agent 日志
仍在被另一台机器写入，在本机看「进程不在、GPU 空」会误判成作业崩了）。
判断某个作业是不是**本机**的：看日志名里的拓扑后缀，或 `pgrep -f main_ppo`
配合 `nvidia-smi`；跨机唯一可靠的现场是 SwanLab 和 `rollout_data/`。

### 8a. Agent 域（**另一台机器**，2 机 16 卡）

| 项 | 值 |
| :--- | :--- |
| 进度 | **84/300**，每步中位数 **482s** |
| reward | step1 `0.245` → step84 `0.830`（峰值 0.934） |
| `num_turns/mean` | 9.6（`tool_agent` 确认生效） |
| 实验名 | `expert_agent_qwen3_4b` |
| 产物 | `$B/runs/llm_fusion/checkpoints/llm_fusion/expert_agent_qwen3_4b/global_step_{25,50,75,...}` |
| 日志 | `$B/logs/llm_fusion/train_agent_full_20260910_224545.log` |

**⚠️ 这个作业用的是旧脚本 `run_train_agent_full.sh`，且当时 sandbox 只在 head 上**，
所以 **step 25/50/75 三次中途验证的 LCB v5/v6 数字作废**（§3 那个坑，已于 step 81
前修复）。其余六个基准与本域 bfcl_v3（0.70 → 0.7175）都是有效的。

### 8b. Science 域（**本机**，2 机 16 卡，2026-09-11 12:44 启动）

| 项 | 值 |
| :--- | :--- |
| 进度 | step 3/300，**497 s/it** → 全程约 **41 小时** |
| reward | step1 `0.3154` → step2 `0.2935` → step3 `0.3477`（nonzero ~30%，resp_len ~9.5k） |
| 实验名 | `expert_science_qwen3_4b_2x8` |
| 节点 | head `28.45.6.151` + worker `28.45.5.209` |
| 日志 | `$B/logs/llm_fusion/train_science_2x8_20260911_124422.log` |

起步 reward ~0.32 对 Science 合理（GPQA 类多选，base 本就有一定正确率），
不是 §6 那种 reward 塌掉。**本机 16 卡在这 41 小时内被占满，起不了别的域。**

### 8c. IF 域（**第三台机器**，1 机 8 卡，2026-09-11 14:17 启动）

| 项 | 值 |
| :--- | :--- |
| 进度 | **75/300**（截至 2026-09-11 21:49），**269 s/it** → 全程约 **22 小时**，预计 09-12 12:30 前后完成 |
| reward | step1 `0.3887` → step75 `0.5015`（**稳定上行**；nonzero 35% → 50%，resp_len 1.0~2.4k） |
| 峰值显存 | 79.5G / 95G（`ROLLOUT_GPU_MEM_UTIL=0.7`，余量 15G） |
| 实验名 | `expert_if_qwen3_4b_1x8` |
| 节点 | 单机 `28.45.4.206`（`NODE_IP_LIST=28.45.4.206:8`） |
| 日志 | `$B/logs/llm_fusion/train_if_1x8_20260911_141753.log` |
| SwanLab | `train-exp` / 项目 `llm_fusion_repro`（`llm_fusion` 这个名字 422，见 §4） |
| 产物 | `$B/runs/llm_fusion/checkpoints/llm_fusion/expert_if_qwen3_4b_1x8/global_step_{25,50,75,...}` ← 已用 override 钉回 |

> reward 轨迹：`0.389`（step1）→ `0.344`（step5）→ … → `0.53/0.41/0.53/0.43/0.50`（step 71~75）。
> **前 5 步横在 0.35 是正常的**，不是 §5 那种 reward 塌掉（那会是全 0）：IF 的 base 本就有
> 三成多通过率，`nonzero 35%` 与之自洽。step 70 以后在 0.41~0.53 之间抖动是 GRPO 的正常方差，
> 看趋势不要看单点。

⚠️ 与前两个域的三处**不同**，接手时别照抄 8a/8b 的命令：

1. **`ROLLOUT_GPU_MEM_UTIL=0.7`**（1×8 下 0.9 会在 step 3 OOM，见 §5）；
2. **`PROJECT_NAME=llm_fusion_repro` + `trainer.default_local_dir` override**（见 §4）；
3. **checkpoint 是 `world_size=8` 分片**，与 8a/8b 那两批 16 卡产物**不通用**（见下）。

完整启动命令：

```bash
export SWANLAB_LOG_DIR=/tmp/swanlog_llm_fusion
export PROJECT_NAME=llm_fusion_repro
export ROLLOUT_GPU_MEM_UTIL=0.7
bash run_train_multinode.sh if \
  "trainer.default_local_dir=${B}/runs/llm_fusion/checkpoints/llm_fusion/expert_if_qwen3_4b_1x8"
```

reward 前四步横在 0.35 是**正常**的，不是 §5 那种 reward 塌掉（那会是全 0）：
IF 的 base 本就有三成多通过率，`nonzero 35%` 与之自洽。要判好坏得看 25 步以后有没有上行。

⚠️ 这台机器上**没有 `step:` 指标行进 console 日志**（8a/8b 有）—— 是 `print()` 的块
缓冲，tqdm 走 stderr 所以进度条照常刷。别据此判断作业异常；
reward 以 `summarize_rollout.py` 读 `rollout_data/` 的 jsonl 为准。

### checkpoint 的 world_size 陷阱

```
global_step_25/actor/model_world_size_16_rank_{0..15}.pt
```

**分片文件名绑定 `world_size`**。这批是 16 卡的产物，**与 8 卡的不通用**。
换机器后如果卡数不同，不能直接读旧 checkpoint 续训，也不能直接拿去 Merge ——
`convert_experts_to_hf.py` 必须按存盘时的卡数布局读。

⚠️ **本项目现在已经同时存在两种布局**：8a/8b 是 `world_size_16`，8c（IF）是
`world_size_8`。Merge 前要把五个 expert 各自按**存盘时的卡数**导出，
不能用同一套参数批量跑。

每个 checkpoint 约 **17G**（只存 model 分片），12 次共约 204G。
⚠️ 只存 model 的代价：resume 只恢复权重，**优化器动量和 RNG 从头开始**。
要长跑容错就把 `CKPT_CONTENTS` 改成 `'["model","optimizer","extra"]'`（约 48G/次）。

---

## 9. 待做

- [ ] 训完 Agent → 用 `run_repro.sh code` **重评 LCB v5/v6**（§8a 那三个点作废）
- [x] `science` expert：本机 2026-09-11 12:44 起跑，300 步约 41 小时（§8b）
- [x] `if` expert：第三台机 2026-09-11 14:17 起跑，1×8，300 步约 25 小时（§8c）
- [ ] 还差两个 expert：`math` / `code`（Merge 需要五个都有）
- [ ] Mix RL（800 步）
- [ ] Merge 端到端：`convert_experts_to_hf.sh` → `run_merge.sh`
      （Task Arithmetic λ=0.6 复现已发布的 Merge 模型）
      ⚠️ **Hub 上那五个 bf16 expert 不能做 merge 输入** —— RL 的 task vector
      `θ_expert − θ_base` 很小，bf16 只有 8 位尾数，round-trip 量化误差和 delta 同量级。
      必须自己训 + 用 `convert_experts_to_hf.py` 导 fp32（它与上游唯一的差别就是
      `tensor._local_tensor.float()` 那一行）。
- [ ] 把 tensorboard 目录从仓库里挪到共享盘（现在因 §5 的环境变量问题落在
      `<repo>/tensorboard_log/`，已加进 `.gitignore`）

---

## 10. 排障速查

| 症状 | 大概率原因 | 看哪 |
| :--- | :--- | :--- |
| 启动后卡住不动、不报错 | Ray 卡数 < 声明的拓扑 | `ray status`，§3 |
| `ibv_reg_mr_iova2 ... Invalid argument` | GDR 不兼容 + 环境变量没传到 worker | §4 |
| `EADDRINUSE` / `DistNetworkError` on port | vLLM 端口 TOCTOU 竞争，**重跑即可** | §4 末 |
| `no TTY is available for interactive login` | SwanLab 凭据没进 `ray start` 的 shell | §4 |
| `FILESYSTEM may be slow or remote` | `SWANLAB_LOG_DIR` 在共享盘（报错是误导的） | §4 |
| **`SWANLAB_LOG_DIR=BAD(...)` 被门禁拦下** | **Pod 预置的 `SWANLAB_*` 打败了 `${VAR:-默认}`** | **§4** |
| **`POST /api/project` 返 `422`** | **SwanLab 项目名被别的 workspace 占用，换 `PROJECT_NAME`** | **§4** |
| **step 1~2 正常、step 3 CUDA OOM** | **1×8 下 actor 每卡占 7.84G，`0.9` 装不下** | **§5** |
| **`ERROR: <ip> 上没有 swanlab`（但明明装了）** | **预检把 stderr 的 `FutureWarning` 折进了变量，已修** | **§5** |
| LCB 分数腰斩且多次不变 | 某些节点没有 sandbox | §2 |
| reward 全 0 / `num_turns/mean == 1` | 漏了 `default_agent_loop=tool_agent` | §5 |
| `import verl` 失败 | 新节点没做 editable 注册（`pip install -e . --no-deps`） | ⚠️ 见下 |
| numpy/torch/vllm 突然坏掉 | `pip install -e .` 漏了 `--no-deps`，把 numpy 降级了 | ⚠️ 见下 |
| NLTK 语料一个都下不下来 | 缺 `NLTK_ALLOW_PROXIED_URLOPEN=1` | ⚠️ 见下 |
| 日志里只有进度条，没有指标 | `trainer.logger` 只剩 console；或 `print()` 块缓冲（§8c） | §7 |
| 后台任务莫名被杀 | `nohup` 只挡 SIGHUP，要用 `setsid` | overview 5.3 |
| vLLM OOM 且报**陌生** PID | 别的进程占卡；`gpu_memory_utilization` 是占**总**显存比例 | overview 5.4 |
| `docker pull` 超时 | dockerd 不继承 shell 代理（拉镜像是守护进程发起的） | overview 5.2 |

> ⚠️ **原「§2. 新节点的环境搭建」整节已于 2026-09-11 14:15 被删**，上表最后三条
> 「见下」原本指向它。那一节讲的是每个新节点必做的 Python 环境（`pip install -e .
> --no-deps` 的 `--no-deps` 不能省、reward verifier 依赖、`NLTK_ALLOW_PROXIED_URLOPEN=1`
> 才能过代理下语料、以及 swanlab 要装在每个节点）。**接手前请先确认它是不是被误删了**；
> 相应内容在 `overview.md` 里还有。章节编号也因此断了（§5 之后直接跳到 §7）。

更详细的分析全部在 **`overview.md`**（§4b 完整训练、§4c 通用多机入口、§5 踩过的坑、
§6 checkpoint 精度链路），通用问题另见共享盘 `$B/经验记录-通用.md`。
