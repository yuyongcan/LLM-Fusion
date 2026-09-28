# LLM-Fusion 复现记录

> 环境搭建、数据/权重下载、四种训练范式小规模跑通的完整记录。
> 机器：单机 8×H20 (95G)，TencentOS 3.2，共享盘 `/taijifs_sz34/share_304169095`。
> 日期：2026-09-10

论文 *Consolidating RLVR Capabilities Across Domains: A Deep Dive into Fusion Paradigms*
把「整合 RLVR 领域专家」的三条路线按复用的产物分类：**Merge**（复用 task vector）、
**Mix RL**（复用数据集）、**MOPD**（两者都用），在同一批 expert / 数据 / benchmark 上对比。

---

## 1. 快速上手

```bash
cd /taijifs_sz34/share_304169095/hunyuan/yongcanyu/code/LLM-Fusion
source env_llm_fusion.sh          # 代理 + conda 环境 + 四个根路径

# 小规模训练（验证链路，各 6 步，单机 8 卡）
bash run_train_small.sh agent     # domain-RL / Agent
bash run_train_small.sh code      # domain-RL / Code
bash run_train_small.sh mix       # Mix RL
bash run_train_small.sh mopd      # MOPD 五教师蒸馏

# 完整训练（2 机 16 卡 × 300 步，论文超参）—— 见第 4b 节
bash setup_sandbox.sh             # ⚠️ 每个节点都要跑一次
bash setup_ray_cluster.sh         # 两节点重建 Ray，带 16 卡门禁
bash run_train_agent_full.sh      # Agent expert，约 41 小时

# SFT（phase 1：Qwen3-4B-Base → chat/reasoning 模型）—— 见第 5b 节
# ⚠️ 不需要 Ray、不需要 sandbox；它是 torchrun + pssh，本仓库唯一这么起的入口
TOTAL_TRAINING_STEPS=20 SAVE_FREQ=10 EXPERIMENT_NAME=sft_smoke_qwen3_4b_base \
    bash run_sft_multinode.sh     # 冒烟 20 步，约 25 min（8 机 64 卡）
bash run_sft_multinode.sh         # 完整 1 epoch，10000 步，约 6.5 天（Liger 默认开）

# 评测（4 个模型 × 6 个不需要 sandbox 的基准）
bash run_repro.sh nosandbox
bash run_repro.sh code            # LCB v5/v6，需要 sandbox 在 8080

# 看指标
tensorboard --logdir tensorboard_log                              # 完整训练用这个
python summarize_rollout.py $OUTPUT_ROOT/rollout_data/small_expert_agent   # 小跑用这个
```

本次新增的辅助文件（都在仓库根目录）：

| 文件 | 作用 |
| :--- | :--- |
| `env_llm_fusion.sh` | 代理、`CODEBUDDY_SAFE_DELETE_ENABLED`、conda PATH、四个根路径 |
| `run_train_small.sh` | 四种范式的小规模训练入口，含 GPU 切分与显存修正 |
| `run_train_agent_full.sh` | **2 机 16 卡 Agent 完整训练**，论文超参 × 300 步 |
| `run_train_multinode.sh` | **通用多机入口**（六个域任选），拓扑自动推导 + 三道前置门禁 + SwanLab |
| `setup_ray_cluster.sh` | **两节点清理 + `ray start` + 16 卡门禁**，并注入 NCCL 环境 |
| `setup_sandbox.sh` | **重建 sandbox**，自动「有 tar 就 load、没有才 pull」并备份 |
| `run_repro.sh` | 评测驱动，分 nosandbox / code 两批，可中断续跑 |
| `summarize_rollout.py` | 从 rollout dump 统计每步 reward |
| `check_rewards.py` | 五个域 reward scorer 的依赖自检 |

---

## 2. 环境

装在 **`/opt/conda/envs/torch-base`**（镜像自带环境，Python 3.13.13）：

| 组件 | 版本 |
| :--- | :--- |
| torch | 2.11.0+cu130 |
| vllm | 0.24.0 |
| transformers | 5.10.4 |
| verl (本仓库) | 0.9.0.dev（`pip install -e .`） |
| flash-attn-4 | 0.0.1.dev1409 |

**关键：装 verl 必须用 `--no-deps`。**

```bash
pip install -e . --no-deps
```

仓库的 `install_requires` 钉了 `numpy<2.0.0`，而该环境是 numpy 2.4.6。直接
`pip install -e .` 会把 numpy 降级，连带 torch 2.11 / vllm 0.24 一起坏掉。
这个环境本来就已满足除 numpy 外的全部依赖，所以跳过依赖解析、只做 editable 注册。

补装的 reward verifier 依赖：

```bash
pip install nltk langdetect immutabledict emoji syllapy 'math-verify[antlr4_9_3]'

# NLTK 语料：新版有 SSRF 防护，走代理时必须显式 opt-in，否则一个都下不下来
export NLTK_ALLOW_PROXIED_URLOPEN=1
python -c "import nltk; [nltk.download(p) for p in ['punkt','punkt_tab','stopwords','averaged_perceptron_tagger_eng']]"
```

自检（`python check_rewards.py`）：

```
aime2025     gt='42'   -> 1.0      # 对答案
aime2025     gt='42'   -> 0.0      # 错答案
gpqa         gt='A'    -> 1.0
IF deps      -> OK (nltk corpora present)
antlr4       -> OK
```

### SandboxFusion（Code 域 + LCB 打分）

本机无 docker/podman。按共享盘既有经验，用**静态 docker 二进制 + `/dev/shm` 当 data-root**
（容器根目录都是 overlayfs，不能再叠 overlay2；tmpfs 可以）：

```bash
D=/taijifs_sz34/share_304169095/hunyuan/yongcanyu/docker_build/docker-static-29.7.2
# dockerd 自己要带代理，否则 pull 超时（见第 5 节）
export http_proxy=... https_proxy=... HTTP_PROXY=... HTTPS_PROXY=...
PATH="$D:$PATH" nohup $D/dockerd -H unix:///tmp/docker-build.sock \
  --data-root=/dev/shm/docker-data --storage-driver=overlay2 \
  --iptables=false --bridge=none --userland-proxy=false &

$D/docker -H unix:///tmp/docker-build.sock run -d --name sandbox --network host \
  --shm-size=2g vemlp-cn-beijing.cr.volces.com/preset-images/code-sandbox:server-20250609
```

探活（注意 `--noproxy '*'`，否则 localhost 请求会被代理吃掉）：

```bash
curl -s --noproxy '*' -X POST http://localhost:8080/run_code \
  -H 'Content-Type: application/json' -d '{"code":"print(2+2)","language":"python"}'
# {"status":"Success",...,"stdout":"4\n",...}
```

**注意产物在内存盘，Pod 重启即丢**，需要长期用要 `docker save` 到共享盘。

> **2026-09-10 已应验**：Pod 重启后 `/dev/shm/docker-data` 整个消失、dockerd 也没了，
> 而当时没做备份，只能重新走代理 pull 一次（约 8 min）。现已一次性解决：

镜像备份在 **`$B/docker_build/code-sandbox-server-20250609.tar`**
（`B=/taijifs_sz34/share_304169095/hunyuan/yongcanyu`，约 数 G，与既有的
`conda_yyc_v1.tar`、`docker-static-29.7.2` 放在一起）。

**不要放进 `code/LLM-Fusion/`**：仓库 `.gitignore` 只忽略 `**/*.tar.gz`，不忽略 `*.tar`，
镜像 tar 放进去会一直挂在 `git status`，有被 `git add -A` 误提交进这个 upstream fork 的风险；
且镜像是机器级依赖（`run_repro.sh code`、别的项目都要用），不属于某一个仓库。

重建一律用 **`bash setup_sandbox.sh`**，不用记路径：它会自动
「已在跑就跳过 → 有 tar 就 `docker load` → 没有才走代理 pull」，
并在 pull 之后自动 `docker save` 补上备份。

### ⚠️ 多机时每个节点都要起一个 sandbox（2026-09-11 踩到）

**`localhost:8080` 只对起了容器的那个节点有效。** 打分任务由 Ray 分散到所有 worker，
落在没有 sandbox 的节点上全部 `Connection refused`，然后**静默退化成另一个 scorer**
（`reward_score/__init__.py`：`sandbox_fusion_url` 不可用就 fallback 到 `prime_code`）。

2 机 16 卡跑 Agent 训练时的现象：LCB v5/v6 稳定停在 **0.30 左右**，而论文 base 是 0.59；
其余六个基准都与 base 吻合。日志里 138 次 `Max retries exceeded with url: /run_code
(Connection refused)`，全部集中在验证那几步。

**判据**：分数几乎正好腰斩（一半 worker 正常、一半 fallback），且**三次验证纹丝不动**
（0.3085 → 0.3084 → 0.3012）。真的能力退化会随训练单调变化，不会卡在同一个值。

```bash
# 每个节点都要起；worker 上直接从共享盘 load，不必走代理
R=/taijifs_sz34/share_304169095/hunyuan/yongcanyu/code/LLM-Fusion
ssh <worker_ip> "cd $R && setsid nohup bash $R/setup_sandbox.sh > /tmp/sandbox.log 2>&1 < /dev/null & disown"

# 验证必须逐节点做，head 通 ≠ 全集群通
for ip in <head_ip> <worker_ip>; do
  ssh $ip "curl -s --noproxy '*' -X POST http://localhost:8080/run_code \
    -H 'Content-Type: application/json' -d '{\"code\":\"print(6*7)\",\"language\":\"python\"}'"
done   # 两个都要返回 42
```

> 这就是 `paths.sh` 警告的「没有可达 sandbox 会静默退化」在多机下的变形：
> **不要用 head 上的探活代表整个集群。**

---

## 3. 数据与权重

全部下载完成，行数与 `download_data.py` 的期望值 **14/14 完全一致**。

**训练集** `Siye01/LLM-Fusion-Train` → `data/LLM-Fusion/train`（3.0G）

| Config | 行数 | shards | Config | 行数 | shards |
| :--- | ---: | ---: | :--- | ---: | ---: |
| Math | 38,131 | 1 | IF | 16,575 | 1 |
| Science | 50,000 | 1 | Agent | 10,229 | 1 |
| Code | 19,169 | 15 | Mix | 87,699 | 16 |

**测试集** `Siye01/LLM-Fusion-Test` → `data/LLM-Fusion/test`（485M）：
AIME2025(30)、AIME2026(30)、GPQA(198)、LCB_v5(167)、LCB_v6(175)、
IFEval(541)、IFBench(300)、BFCL_v3(200)。Code 两个集合占了 483M。

**权重**（每个 7.6G）：

| 路径 | 用途 |
| :--- | :--- |
| `checkpoints/Qwen/Qwen3-4B-Instruct-2507` | backbone / task-vector 参考点 |
| `checkpoints/Siye01/Qwen3-4B-Inst-{Merge,Mix,MOPD}` | 三个已发布融合模型，用于评测 |
| `checkpoints/Siye01/experts_hf/{math,science,code,if,agent}` | 五个域 expert，MOPD teacher |

---

## 4. 四种训练范式小规模跑通

单机 8 卡各跑 6 步。**目的是验证链路通、reward 有效，不是复现精度**
（论文是 4 机 32 卡 × 300~800 步；正式训练见第 4b 节）。

与论文配置的差异：

| 项 | 论文 | 这里 | 原因 |
| :--- | :--- | :--- | :--- |
| `TRAIN_BATCH_SIZE` | 128 / 264 | 8（MOPD 6） | 单机；且必须能被 DP 数整除 |
| `ROLLOUT_N` | 16 | 4 | 时长 |
| `MAX_RESPONSE_LENGTH` | 16384 | 4096 | 显存 + 时长 |
| `TOTAL_TRAINING_STEPS` | 300 / 800 | 6 | 只验证链路 |
| `ROLLOUT_GPU_MEM_UTIL` | 0.9 | 0.7 | 卡上有外部占位进程，见 5.4 节 |
| logger | console + wandb | 仅 console | 无 wandb 凭据（完整训练改用 tensorboard） |

### 结果：四种范式全部跑通

每步 reward（从 `rollout_data/<exp>/<step>.jsonl` 统计）：

| step | Agent | Code | Mix | MOPD |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.156 | 0.006 | 0.313 | 0.333 |
| 2 | 0.406 | 0.144 | 0.406 | 0.250 |
| 3 | 0.188 | 0.028 | 0.188 | 0.417 |
| 4 | 0.406 | 0.206 | 0.338 | 0.083 |
| 5 | 0.125 | 0.159 | 0.303 | 0.425 |
| 6 | 0.344 | 0.188 | 0.156 | 0.458 |

rows/step：Agent/Code/Mix 各 32（8 prompt × 4），MOPD 24（6 × 4）。

几个可以确认的点：

- **reward 打分器真的在工作**，非零率 3%~46%，梯度有信号。
- **Code 的 reward 是分数值**（0.006 / 0.144 / 0.028…）而不是 0/1，
  说明 SandboxFusion 在逐条跑单测按通过率打分。这正是 `paths.sh` 里
  警告的「没有可达 sandbox 会静默退化成另一个 scorer」要验证的关键点。
- **6 步看不出趋势**，波动完全是小 batch 的噪声，不要解读为「学到了/没学到」。
- Agent 域日志里有 `Failed to decode tool call` —— base 模型输出的 tool call
  格式不规范，属正常现象（这恰恰是 Agent RL 要优化的目标）。

### MOPD 的 GPU 切分

teacher pool 是**独立的 Ray 资源池，不算在 `NNODES × NGPUS_PER_NODE` 里**，
通过 `distillation.nnodes × distillation.n_gpus_per_node` 申请。论文默认
24(训练) + 5(teacher) = 29 卡；单机 8 卡改成：

```bash
NNODES=1 NGPUS_PER_NODE=3                          # 训练池 3 卡
TEACHER_NNODES=1 TEACHER_TP=1 TEACHER_NUM_REPLICAS=1   # -> teacher 池 5 卡，见下
TRAIN_BATCH_SIZE=6 PPO_MINI_BATCH_SIZE=6           # 必须能被 DP 数 3 整除
```

> 上面三个 teacher 变量都是 1，但池子是 **5 卡**，不是笔误。`train_mopd.sh:66` 有
> `TEACHER_WORLD_SIZE=$(( NUM_TEACHERS * TEACHER_NUM_REPLICAS * TEACHER_TP ))`
> = 5 × 1 × 1 = 5 —— 那三个变量是**每个 teacher** 的配置，5 是 teacher 个数带来的。
> 脚本注释也写明「pool size must equal sum(num_replicas × tensor_parallel_size) over all teachers」。

实测显存分布，确认切分生效：

```
卡 0-2: 66.0G   训练池 (FSDP actor)
卡 3-7: 73.7G   五个 4B teacher，各一张
```

---

## 4b. Agent 域完整训练（2 机 16 卡，2026-09-11）

第 4 节是「链路验证」，这一节是**出成果的正式训练** —— 产出真正的 Agent expert
（论文 Table 1 的 per-domain RL 行，也是后续 Merge 的五个输入之一）。

```bash
bash setup_sandbox.sh                        # 每个节点都要跑，见 2 节的多机警告
bash setup_ray_cluster.sh                    # 两节点清理 + 重建，带 16 卡门禁
bash run_train_agent_full.sh                 # 论文超参 × 300 步
```

### 先纠正第 4 节引用的两个数字

| 项 | 之前写的 | 论文实际 | 出处 |
| :--- | :--- | :--- | :--- |
| per-domain RL 步数 | 325 / 400 | **300** | Table 3（325 只是脚本默认值，400 是 MOPD 的） |
| Mix RL 步数 | 1500（脚本默认） | **800** | Table 3 |
| Agent 域成本 | — | **700 GPU-h**（4B） | Table 4，「measured through step 300」 |

700 GPU-h / 16 卡 ≈ 44h；**实测每步中位数 476s**，300 步约 **39.6h**，
加 12 次八基准验证（每次约 565s）约 1.9h → **合计约 41 小时**。论文用 32 卡所以是一半时间。

### 与论文配置的差异：只有卡数

超参**完全按论文**（batch 128 / mini 128 / n=16 / resp 16384 / prompt 5120 / lr 1e-6 /
GRPO + `norm_adv_by_std_in_grpo=FALSE` / KL loss `low_var_kl` coef 0.001 / temp 1.0），
只有 `NNODES=2 NGPUS_PER_NODE=8`（论文 4×8）。所以**结果可与论文直接比较**，
不像第 4 节的小跑那样只能看趋势。

显存回到论文的 `ROLLOUT_GPU_MEM_UTIL=0.9` —— 5.4 节那个每卡占 3.5G 的 `gpu_hold.py` 已经没了。

### 必须显式传的两个覆盖

```bash
# 1. Agent 训练行没有 agent_name 列（实测 10229 行全无），不传这个会 fallback 到
#    single_turn_agent —— 单轮、无工具、reward 直接塌，且不报任何错。
actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent

# 2. env_llm_fusion.sh 里 NNODES/NGPUS_PER_NODE 是硬 export（不是 ${:-}），
#    必须在 source 之后覆盖，否则永远是单机 8 卡。
export NNODES=2 NGPUS_PER_NODE=8
```

**`tool_agent` 生效的判据：`num_turns/mean` 必须远大于 1。** 实测 7.8~10.5，
且随训练上升（模型学会多用工具）。若等于 1 就是退化了，立刻停。

### 实测训练曲线（前 81 步）

| step | 1 | 20 | 40 | 60 | 81 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `critic/score/mean` | 0.245 | 0.683 | ~0.80 | ~0.87 | **0.934** |
| `num_turns/mean` | 7.84 | 10.27 | ~10 | ~9.6 | 9.40 |

reward 涨到 **3.7 倍**，`prompt_length/clip_ratio` 与 `response_length/clip_ratio` 全程 0.0
（5120/16384 都够用），`aborted_ratio` 0.0，`grad_norm` 稳定在 0.16~0.18。

Agent prompt 含工具 schema 后约 4.1k token，**远低于 5120**，不会触发 `data.truncation='error'`。

### checkpoint：world_size 变成 16 了

`SAVE_FREQ=25` → 300 步存 12 次，实测**每次 17G**（只存 model 分片），共约 204G。

```
global_step_25/actor/model_world_size_16_rank_{0..15}.pt    # 16 个，不是 8 个
```

⚠️ **分片文件名绑定 world_size**：这批 16 卡的产物与第 4 节 8 卡的产物**不通用**，
后续 `convert_experts_to_hf.py` 做 Merge 必须按 16 卡布局读。

### 指标：改用 tensorboard（无 wandb 凭据）
5.5 节的问题（关掉 wandb 后 per-step 指标无处可看）这次用 tensorboard 后端解决 ——
verl 支持 `trainer.logger='["console","tensorboard"]'`，比只靠 rollout dump 好用得多。

```bash
tensorboard --logdir code/LLM-Fusion/tensorboard_log
```

⚠️ **但 `TENSORBOARD_DIR` 环境变量传不到 Ray actor**（和 5.7 节的 NCCL 是同一个根因），
所以实际落在 verl 的默认相对路径 `<repo>/tensorboard_log/<project>/<experiment>/`，
而不是我指定的共享盘目录。数据完整、不影响训练，但它在 git 仓库里
（`.gitignore` 没覆盖 `tensorboard_log/`），要留意别误提交。

---

## 4c. 通用多机入口 `run_train_multinode.sh`

`run_train_agent_full.sh` 是 Agent 专用、IP 硬编码的。要训另外四个 expert（Merge 需要五个都有）
或换机器数时用这个：

```bash
bash run_train_multinode.sh agent      # 300 步（论文 Table 3）
bash run_train_multinode.sh math       # 其余四个域同理
bash run_train_multinode.sh mix        # 800 步，自动切到 train_mix_rl.sh
# 后面可以直接追加 hydra override，原样透传
bash run_train_multinode.sh code trainer.val_before_train=True
```

### 为什么不是 MOPD 那种 fan-out 启动器

`MOPD/mopd_sft/launch_multinode.sh` 要用 pssh 把训练脚本分发到每台机器，
**因为 torchrun 不是 Ray** —— torchrun 只 fork 本地 8 个 rank，必须每台各起一份、
各自带不同的 `--node_rank`。

verl 是 Ray-based：**head 单点提交一次，Ray 自己把 WorkerDict 调度到全集群**，
拓扑只通过 `trainer.nnodes × trainer.n_gpus_per_node` 声明。所以这里不需要分发，
只需要先 `setup_ray_cluster.sh` 把集群拉起来。（那个脚本自己的注释也写明了这个区别。）

代价是 Ray 带来两个 torchrun 没有的坑，即 5.7（环境变量白名单）和 5.8（每节点 sandbox）。

### 拓扑自动推导，不硬编码 IP

沿用 MOPD 的做法，以 `NODE_IP_LIST` 为权威来源：

```bash
mapfile -t NODE_IPS < <(echo "${NODE_IP_LIST}" | tr ',' '\n' | cut -d: -f1 | grep .)
HEAD_IP="${CHIEF_IP:-${NODE_IPS[0]}}"
NNODES=${NNODES:-${HOST_NUM:-${#NODE_IPS[@]}}}
```

⚠️ **不要用 `NODE_NUM` 当节点数** —— 本平台上它是**总 GPU 数**（实测 16），不是机器数（2）。
用 `HOST_NUM`，或数 `NODE_IP_LIST` 的条目。

### 每台机器路由到本机 sandbox

```bash
export SANDBOX_FUSION_URL=http://localhost:8080/run_code
```

保持 `localhost` 而不是指向 head 的 IP，好处是 16 卡并发分摊到 N 个 sandbox、
打分流量不跨机。**前提是每个节点都真的起了一个** —— 所以脚本有逐节点探活门禁（见下）。

### 三道前置门禁（都是「不检查就会静默跑错」的）

| 门禁 | 不检查的后果 |
| :--- | :--- |
| Ray 集群 GPU 数 == `NNODES × NGPUS` | verl 静默 pending 挂住，不报错 |
| **逐节点** sandbox 探活返回 42 | 一半打分静默 fallback，LCB 腰斩（5.8 节） |
| 每节点都装了 swanlab | `TaskRunner` 可能落在任一节点，缺了就 ImportError |

已做过负向测试：把 `NODE_IP_LIST` 里塞一个不存在的 IP，门禁会准确报出
`sandbox 不可达：28.45.99.99` 并拒绝启动。

### SwanLab（自建实例 taiji-exp）

```bash
LOGGER='["console","swanlab","tensorboard"]'      # 默认值
```

verl 的 `tracking.py` 只读 `SWANLAB_API_KEY` / `SWANLAB_LOG_DIR` / `SWANLAB_MODE`；
`project` / `experiment_name` 来自 `trainer.project_name` / `trainer.experiment_name`；
`SWANLAB_WEB_HOST` / `SWANLAB_API_HOST` 由 swanlab SDK 自己消费。面板：
<http://taiji-exp.taiji.woa.com/>

几个注意点：

- **两个节点都要 `pip install swanlab`**（本次已装 0.10.0）。`Tracking()` 只在
  logging rank 构造，所以 `swanlab.init()` 全作业只调一次、多机下不用做 rank 保护；
  但 `TaskRunner` 可能被 Ray 调度到任一节点，所以装必须装两边。
- **`SWANLAB_LOG_DIR` 必须放共享盘**（这里是 `$OUTPUT_ROOT/swanlog`）。多机下每个
  节点都会解析这个路径；MOPD 踩过「默认值指向别人的 workspace 且本地不存在」的坑。
- `EXPERIMENT_NAME` 自动带上拓扑后缀（`expert_agent_qwen3_4b_2x8`），
  否则 1 机跑出来的实验在面板上会和 2 机的混在一起。
- 同时保留 tensorboard 作兜底，swanlab 连不上也还有指标。
- `tensorboard_log/`、`swanlog/`、`*.tar` 已加进 `.gitignore`
  （原本只忽略 `*.tar.gz`，27G 的 sandbox 镜像 tar 极易被 `git add -A` 误提交）。

---

## 5. 踩过的坑

5.1~5.4 与 5.7、5.8 已同步到共享盘的 `经验记录-通用.md`（通用问题），这里只留摘要。

### 5.1 NLTK 新版走代理被 SSRF 防护拦住

`nltk.download()` 报 `SSRF protection cannot be enforced (CWE-918)`，语料一个都下不下来。
新版 NLTK 要先解析域名、校验 IP 再按该 IP 发请求；走代理时出口由代理完成，
它钉不住目标 IP 就**主动拒绝**。解决：`export NLTK_ALLOW_PROXIED_URLOPEN=1`。

> 这类「库的安全校验与强制代理环境冲突」的报错通常不提代理二字，
> 看到 SSRF / IP pin / hostname validation 就先怀疑代理。

### 5.2 dockerd 不继承 shell 的代理变量

shell 里 export 了代理，`docker pull` 仍然超时。因为**拉镜像是 dockerd 守护进程发起的**，
不是 docker 客户端；dockerd 是早先启动的独立进程，环境完全隔离。
解决：让启动 dockerd 的那个 shell 先 export 代理。验证 `docker info | grep -i proxy`。

> `docker build --network host` 时构建容器内的网络走 Dockerfile 的 ENV，与此无关；
> **唯独 pull/push 走 dockerd 自身的环境**。

### 5.3 `nohup` 挡不住工具超时的 SIGTERM

`nohup ... &` 起的后台 pull，在前台命令被 120s 超时打断后一起死了
（dockerd 日志 `Not continuing with pull after error: context canceled`）。
`nohup` 只挡 SIGHUP，后台任务仍在同一进程组里，超时时整组收 SIGTERM。
解决：`setsid nohup <cmd> > log 2>&1 < /dev/null & disown`。

> 另外轮询进度时不要把 `sleep` 和探活写在同一条会超时的命令里。

### 5.4 共享 GPU 上必须下调 `gpu_memory_utilization`

vLLM 报 OOM，且报错里列出多个陌生 PID 各占 3.5G。
`gpu_memory_utilization` 是**占总显存的比例**，不是占剩余显存：
95G × 0.9 = 85.5G，加上本机 `gpu_hold.py` 每卡 3.5G 的占位就超了。
解决：`ROLLOUT_GPU_MEM_UTIL=0.7` + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。

**另一个要点**：`ray stop --force` **杀不掉训练脚本自己内嵌起的 Ray 集群**
（它不是 `ray start` 起的）。必须 `kill -9` 主进程 PID，再
`pkill -9 -f "ray::|raylet|gcs_server"`，否则下一个任务必然 OOM。

### 5.5 关掉 wandb 后训练指标无处可看

把 `trainer.logger` 改成 `["console"]` 后，日志里只剩进度条，
grep 不到任何 `step:` / `critic/score/mean`。verl 只在 `val_only` 验证路径
把聚合指标打到 console，训练循环的 per-step metrics 只走 logger backend。

解决：靠 `trainer.rollout_data_dir`，每步落一个 `<step>.jsonl`，含每条 rollout 的
`score` / `data_source` / 输出文本 —— 就是 `summarize_rollout.py` 读的东西。

> 无人值守跑 RL 务必设 `rollout_data_dir`：它既是 reward 是否生效的唯一凭据
> （能看出打分是 0/1 还是分数值），也是训练崩了之后唯一的现场。

### 5.6 `${1:?usage: ... {a|b}}` 会被 shell 截断

`WHICH=${1:?usage: $0 {agent|code|mix|mopd}}` 里的第一个 `}` 提前闭合了参数展开，
结果 `WHICH` 拿到的值是 `agent}`，case 永远不匹配。改成显式 `if [ -z ... ]` 判断。

### 5.7 ⚠️ verl 场景下 NCCL 的 export 传不到 Ray worker（多机必踩）

2 机 16 卡首次启动，跨机 FSDP 初始化直接失败：

```
NCCL WARN Call to ibv_reg_mr_iova2 failed with error Invalid argument
torch.distributed.DistBackendError: ... ncclSystemError
```

本机栈（NCCL 2.28.9 + CUDA 13.0 + driver 535.247.01 + `mlx5_bond_1..8`）与
`经验记录-通用.md` 2026-09-07 那次 GDR 注册不兼容**完全一致**，解法也一样是
`NCCL_NET_GDR_LEVEL=0`（关 GPU 显存直连 RDMA、保留 IB，走 pinned host memory 中转）。

**但笔记里的 `export` 写法在 verl 下无效。** verl 给 Ray worker 下发的 `runtime_env`
是一份**写死的白名单**（`verl/trainer/constants_ppo.py` 的 `PPO_RAY_RUNTIME_ENV`，
只含 `TOKENIZERS_PARALLELISM` / `NCCL_DEBUG` / `VLLM_*` / `CUDA_DEVICE_MAX_CONNECTIONS` 等），
`NCCL_NET_GDR_LEVEL` 不在其中。真正干活的是 raylet 拉起的 `WorkerDict` 进程，
**不继承你提交作业那个 shell 的环境** —— 于是 export 了等于没设，且不报任何错。

**正确做法：在 `ray start` 的 shell 里设**（已写进 `setup_ray_cluster.sh` 的 `NCCL_ENV`），
raylet 及其所有 worker 都会继承。**必须逐 worker 验证，不要假设**：

```python
import ray; ray.init(address='auto')
@ray.remote(num_gpus=1)
def probe():
    import os, socket
    return socket.gethostbyname(socket.gethostname()), os.environ.get('NCCL_NET_GDR_LEVEL')
print(ray.get([probe.remote() for _ in range(16)]))   # 两个节点都要出现 '0'
```

走 Hydra 下发（`ray_kwargs.ray_init.runtime_env.env_vars.XXX`）也能进 runtime_env，
但有两个连环坑，不推荐：

1. `runtime_env` 不在 `ppo_trainer.yaml` 的 schema 里（struct 模式），直接覆盖报
   `Key 'runtime_env' is not in struct`，必须加 `+` 前缀。
2. 加了 `+` 之后 Hydra 把 `=0` 解析成 **int**，而 Ray 要求 `env_vars` 是
   `Dict[str, str]`，报 `TypeError: ... the value 0 is of type int`。得写成 `'"0"'`。

> **同一个根因还解释了 `TENSORBOARD_DIR` 为什么不生效**（见 4b 节）：
> 凡是「由 daemon 拉起的进程」（raylet、dockerd、systemd）都不继承你的 shell。
> 和 5.2 的 dockerd 代理问题是一类。排查方法都一样：**在真正干活的那个进程里打印 `os.environ`**。

### 5.8 多机上 sandbox 只起在 head 会让 LCB 分数静默腰斩

详见第 2 节的多机警告。**判据值得单独记**：LCB v5/v6 停在 **0.30 左右**
（论文 base 0.59），而其余六个基准都与 base 吻合；且**三次验证纹丝不动**
（0.3085 → 0.3084 → 0.3012）。

**「正好腰斩」+「多次测量几乎不变」= 一半 worker 走了另一条代码路径**，
而不是模型能力退化（那会随训练单调移动）。日志里对应 138 次
`Max retries exceeded with url: /run_code (Connection refused)`，
目标 `host='localhost', port=8080`。

---

## 5b. SFT 链路（phase 1：Qwen3-4B-Base → chat/reasoning 模型，2026-09-27）

本仓库原先没有 project 级的 SFT，所有 RL 都从 HF 的 `Qwen3-4B-Instruct-2507` 起跑。
这一节记的是从 `../MOPD/mopd_sft/` 整合进来的 SFT 流水线。

```bash
# 冒烟（20 步，先打通链路）
TOTAL_TRAINING_STEPS=20 SAVE_FREQ=10 MAX_CKPT_TO_KEEP=2 \
EXPERIMENT_NAME=sft_smoke_qwen3_4b_base bash run_sft_multinode.sh

# 完整 1 epoch
TOTAL_TRAINING_STEPS=10000 SAVE_FREQ=200 bash run_sft_multinode.sh
```

| 文件 | 作用 |
| :--- | :--- |
| `run_sft_multinode.sh` | 站点启动器：拓扑推导、5 道门禁、pssh fan-out、`setsid nohup` |
| `examples/llm_fusion/sft/train_sft_qwen3_4b_base.sh` | 超参与 `torchrun` 命令 |
| `examples/llm_fusion/sft/dataset.py` | 保留 `reasoning_content` 的 `M2RLSFTDataset` |
| `examples/llm_fusion/sft/filter_by_token_length.py` | 按真实训练 token 长度离线过滤 |
| `tests/utils/dataset/test_m2rl_sft_dataset_on_cpu.py` | 19 个 CPU 断言（合成数据，可进 CI） |

### 5b.1 ⚠️ SFT 会继承 RL 的 `clip_grad=0.2`，裁掉 100% 的步

`verl/trainer/config/optim/fsdp.yaml` 里 `clip_grad` 被本仓库从上游的 `1.0` 改成了
`0.2`（为了 §5 那次 RL 崩溃，RL 的 grad_norm 中位数 0.02 / p90 0.05）。这个 group 是
**SFT 和 RL 共用**的，而 SFT 的 grad_norm 完全是另一个量级 —— 实测 205 步：

```text
mean 0.486   p50 0.362   p90 0.784   p99 1.767   max 1.825
clip_grad=0.2 会裁剪：205/205 步 (100.0%)
clip_grad=1.0 会裁剪： 15/205 步 (7.3%)
```

所以 `train_sft_qwen3_4b_base.sh` **显式传 `optim.clip_grad=1.0`**。不要改 YAML 默认值
（那是 RL 的稳定性边界），也不要把这行「简化」掉 —— 继承 0.2 会缩放每一次更新，且没有
任何报错能看出来。本次冒烟 step 4 实测 grad_norm 冲到 **7.25**，确实需要 1.0 来裁。

### 5b.2 ⚠️ `Adam` + `weight_decay=0.1` 会让 loss 在 step ~100 后单调上升

论文 Appendix A 写的是「Adam ... weight decay 0.1」。照字面用 `torch.optim.Adam`，它的
`weight_decay` 是**耦合 L2** —— 惩罚项加到梯度上，再被 Adam 的二阶矩估计除，有效强度按
`1/sqrt(v)` 缩放。0.1 对 AdamW 正常，对 Adam 极大。

实测三个 run（不同 batch size、不同 LR schedule）全部复现同一个失败特征：

```text
Adam  (bs=1280, cosine)    min 0.8797 @ step 115  →  1.1639 @ step 363   +32.3%
Adam  (bs=5120, constant)  min 0.8932 @ step  38  →  1.1519 @ step 270   +29.0%
AdamW (bs=1280, cosine)    205 步单调下降到 0.6626，无反弹
```

grad_norm 与 loss 同步上升（0.64 → 1.99）。现代论文写「Adam + weight decay」几乎都是指
AdamW，所以默认 `OPTIMIZER=AdamW`；`OPTIMIZER=Adam` 可复现字面读法。

### 5b.3 ⚠️ 这里**需要** pssh fan-out —— 与 §4c 和 HANDOFF §3 不矛盾

§4c 和 `HANDOFF.md` §3 说「别照抄 MOPD 那套 fan-out 启动器」。那条规则针对的是 **RL**：
`verl.trainer.main_ppo` 是 Ray-based，chief 提交一次，Ray 调度全集群。

但 `verl.trainer.sft_trainer` 是**纯 torchrun**，`--nnodes=N` 只 fork 本地 rank 然后阻塞在
rendezvous，必须每台机器各起一份并给不同的 `--node_rank`；本平台又是
`TAIJI_BASIC_EXEC_START_IN_ALL_MPI_PODS=false`，只有 chief 执行 `start_cmd`。所以 SFT 是
本仓库**唯一**需要 fan-out 的入口。

`verl/trainer/sft_trainer_ray.py` 确实存在、也能保留 head 单点提交，但本数据量下不可用：
它的 `_build_dataloader` 写死 `dp_rank=0 / dp_size=1`（`sft_trainer_ray.py:181-185`），即
**在 driver 单进程加载全量数据集**（实测 7.7 GB/110k 行 → 12.2M 行约 855 GB），且不认下面
的 `rank_sharded` 契约。

### 5b.4 按 rank 预分片，以及配套的 trainer 补丁

`sft_trainer.py` 让**每个** rank 都构建全量数据集再用 `DistributedSampler` 切分。实测一个
进程每 110k 行占 ~7.7 GB（在 110k/220k/440k 行处验证过，线性），全量 12.2M 行约 700 GB，
每机 8 个 rank 要 ~5.6 TB 对 2.2 TB 内存 —— 光加载就 OOM。

`SFT_SHARD_BY_RANK=1` 让每个 rank 只读 `row_groups[rank::world_size]`。但这需要 trainer
配合：`verl/trainer/sft_trainer.py` 已加上 `rank_sharded` 检测，对预分片数据集改用
`DistributedSampler(num_replicas=1, rank=0)`。**缺了这一半就会二次分片**，一个 epoch 静默
只训 `1/world_size` 的数据（MOPD 侧实测 27582 步塌成 861 步），不报错。

本次 8×8 实测确认契约成立 —— 各 rank 分片不等长，但 sampler 长度完全一致：

```text
dataset len: 200000 (rank 6/64 shard, sampler len: 200000)
dataset len: 191724 (rank 3/64 shard, sampler len: 200000)
distinct sampler len 值：只有 1 个
```

短分片在 `__getitem__` 里循环重复本地尾部少量样本，保证所有 rank 执行相同数量的
collective。代价是 padding overhead **4.74%**（min 181403 / max 200000 行），也正因为按
最大分片对齐，一个 epoch 是 **10000 步**而不是 `12220841/1280 = 9548` 步。

### 5b.5 MFU ≈ 0.36 是怎么来的：三件事叠在一起，其中两件不是问题

先说一个**取样陷阱**：`tail -1` 抓 MFU 会落在预热区。实测 step 1–5 是 0.19~0.22，
step 6 起才稳定在 0.30~0.40（平均 **0.360**）。判断 MFU 必须掐掉前 5 步。

**(1) verl 的 MFU 分母系统性低估 1.33 倍。**
`verl/utils/flops_counter.py:110` 用 `6*N*tokens`（前向 2 + 反向 4），但
`model.enable_gradient_checkpointing=True` 是开着的 —— 反向要重算一遍前向，实际是
`8*N*tokens`。所以**任何开了梯度重算的 verl run，MFU 都被低估 8/6**：

| | verl 口径 | 真实算力占用 |
| :--- | ---: | ---: |
| 单卡 T=16384（零通信） | 0.497 | 0.663 |
| 4 机 32 卡 | 0.453 | 0.604 |
| 8 机 64 卡 | 0.360 | 0.480 |

**(2) 单卡天花板本来就只有 0.50，所以 32 卡的 0.45 已经接近极限。**
在单卡上跑同样的 fwd+bwd（**完全没有任何通信**）实测：

```text
gc=ON  T= 8192:  verl_MFU=0.510   attention 平方项占记账 flops 的 23.1%
gc=ON  T=16384:  verl_MFU=0.497   attention 平方项占 37.5%
gc=ON  T=32768:  verl_MFU=0.475   attention 平方项占 54.6%
```

- **32 卡 / 单卡 = 91%** → FSDP 通信只吃掉 9%，**32 卡压根不是通信瓶颈**
- 64 卡 / 单卡 = 72% → 只有这里有 28% 的真损失

随 T 增大 MFU 下降是 **attention 的 O(T²)**，这部分算力是真花了，不是浪费。
单卡纯 bf16 GEMM 实测 139 TFLOPS（标称 148 的 94%），硬件本身没问题。

**(3) 梯度重算的代价是 30%，但它是必需的。**
同卡、唯一变量：`gc=ON` 3406 ms vs `gc=OFF` 2618 ms（T=8192）。但关掉后 T=16384
直接 OOM（T=8192 就要 70.8 GB）。这 30% 是买显存的价钱。

> ⚠️ **本节曾写成「64 卡是通信瓶颈」，那是错的**（基于 `tail -1` 抓到的预热值 0.203，
> 且没有单卡对照）。真实可优化量只有 64 卡那 28%：0.360 → 最多 ~0.49，而不是
> 「有一倍空间」。实际能拿到的只有 Liger 的 +6%（见 §5b.7）。

**一个完整 epoch 约 6.5 天**（10000 步 × 56 s，开 Liger）。取样口径见 §5b.7a。

### 5b.6 ⚠️ 放大 batch size 不能提 MFU，而且 bs=5120 会贴爆显存

直觉上「64 卡每卡只 20 条序列，加大 batch 就能摊薄通信」—— 实测是错的，两个原因：

**FSDP 的 all-gather 是 per-micro-batch 的。** token 预算 65536 不变时，
bs=1280→5120 只是把每卡 2 个 micro-batch 变成 6 个，通信次数同比例增加。它只是把
4 步合成 1 步，**对每 token 的通信量没有任何改善**。

**而且装不下。** bs=1280 时 reserved 已是 79.7 / 95 GB，余量仅 15 GB。bs=5120 实测
占到 **86~89 GB**，`backward` 严重变慢 —— 单步跑了 25 分钟还没出来（GPU 100%、
py-spy 反复采到 `_engine_run_backward`，不是挂死，是 allocator 在上限反复扩缩）。

所以**要放大 batch，必须先腾出显存**（见 §5b.7 的 Liger）。另外按 §5b.2，
`sft_loss` 是 token-mean，梯度幅度与 batch 无关 —— bs=1280→5120 会把 update 次数从
10000 砍到 2500，LR 不同步提高（sqrt 缩放约 1e-4）就是**欠训**。

### 5b.7 真正有效的加速项：只有 Liger

逐个标定过 `engine/fsdp.yaml` 里的开关。**64 卡集群 10 步实测**（独占，取 step≥6）：

| 配置 | MFU | s/it（稳定期瞬时） | reserved | 结论 |
| :--- | ---: | ---: | ---: | :--- |
| `use_liger=0` budget 65536 | 0.360 | 60~65 | 79.7 GB | 基线 |
| **`use_liger=1` budget 65536** | **0.383** | **55~57** | **73.7 GB** | ✅ **用这个** |
| `use_liger=1` budget 98304 | — | 单步 >9 min 未完成 | 87 GB 贴顶 | ❌ 装不下 |

单卡对照（T=16384，独占，`gc=ON`）：`liger=0` 8616 ms / 38.6 GB →
`liger=1` 7914 ms / **24.5 GB**，即 +8.9% 速度、激活显存 −37%。集群 +6%，方向一致。

其余开关都不值得动：

| 选项 | 实测/判断 | 结论 |
| :--- | :--- | :--- |
| `engine.use_torch_compile` | 只包 `entropy_from_logits`（PPO 路径）；SFT 的 `sft_loss` 不碰它 | ❌ **no-op** |
| `engine.reshard_after_forward=false` | 即 `SHARD_GRAD_OP`，省掉反向 all-gather，但参数常驻，显存反而更紧 | ❌ 没余量 |
| `ulysses_sequence_parallel_size>1` | 只在单条序列长到装不下时有用；这里 seq 均长 ~1.9k、已过滤 ≤16k | ❌ 不适用 |
| `param_offload` / `optimizer_offload` | 拿速度换显存，方向相反 | ❌ |

**为什么提高 token 预算没用（反直觉）。** Liger 省下的显存本想用来把预算从 65536 提到
98304，让每卡 micro-batch 从 2 个降到 1 个 —— 那才是真减少通信次数。但实测 98304 直接
贴到 87 GB，单步 9 分钟都跑不完（同 §5b.6 bs=5120 的失败签名）。省下的 ~6 GB
（79.7→73.7）远不够多装半个 micro-batch，**这条路被显存堵死了**。

注意 Liger 只 fuse swiglu / rmsnorm / rope / cross_entropy：
`fused_linear_cross_entropy` 被 verl 强制关掉（`transformer_impl.py:282`，与它的
forward patching 冲突），所以那个 `[T, 151936]` 的 logits 张量还在（T=65536 时 18.5 GB）。
上表测的就是 verl 的真实口径。

### 5b.7a ⚠️ 两个量测陷阱（都踩过，都会得出错误结论）

**1. `tail -1` 抓 MFU 会落在预热区。** step 1–5 是 0.19~0.22，step 6 起才稳定
0.30~0.40。我据此误判过「64 卡 MFU 只有 0.2」。判断 MFU 必须掐掉前 5 步。

**2. tqdm 的 `s/it` 是累计均值，不是瞬时步时。** 进度条上的 `109 s/it` 里还混着启动的
慢步；同一个 run 的稳定期瞬时值是 60~65 s/it。我据此把 Liger 的收益算成了 +95%，
实际只有 +6%~12%。要瞬时值就看相邻两行进度条的**时间差**，或直接用 MFU。

**结论：一个 epoch 约 6.5 天**（10000 步 × 56 s），不是早先估计的 12.6 天 —— 那个数字
同时踩了上面两个陷阱。

### 5b.8 其它几个已确认的点

- **`MAX_TOKEN_LEN_PER_GPU=65536`**（= 4×`max_length`）是有意的：等于 `max_length` 时每次
  forward 只有 ~1 条序列，FSDP 一整轮通信换很少的计算（实测 51.7 s/step、11.4 GB
  allocated 但 55.8 GB reserved）。65536 在 205 步 × 32 ranks 下**一次 OOM 都没有**，所以
  MOPD 那套 adaptive token budget（OOM 时缩预算重试）**没有移植** —— 承载它的
  `transformer_impl.py` 在两仓库已分化 657 行，收益不抵风险。数据已离线过滤到 ≤16384，
  长尾 OOM 的风险源在离线阶段就消除了。
- **`data.truncation=error`**：数据已过滤，再出现超长样本就该失败，以暴露 tokenizer /
  chat template / 数据版本漂移，而不是静默截断。
- **`data.ignore_input_ids_mismatch` 必须保持 `false`**：`dataset.py::sanity_check` 只豁免
  逐轮 fallback 的行；这个开关是全局的，会连 prefix 路径一起关掉，而那是真实
  tokenization bug 的唯一探测器。
- **`Prefix render is not monotonic` 警告可以忽略**：Qwen3 模板会重写尾轮 assistant 的空
  `<think>` 占位符，使前缀不再单调；dataset 检测到后只对该行降级为逐轮 tokenization，不
  终止训练、不跳过样本、不丢 reasoning、不让 user token 参与 loss。
- **eos 契约**：Qwen3-4B-Base 出厂 `generation_config.eos_token_id=151643`
  (`<|endoftext|>`)，但 chat template 用 `<|im_end|>` (151645) 收尾。verl 每次 save 都逐字
  拷贝 base 的 `generation_config`，所以没修的话每个 checkpoint 都宣称一个它永不发出的停止
  符（MOPD 实测 324 条 rollout 跑到 30000 token 上限，不报错）。base 目录已修成
  `[151645, 151643]`，`run_sft_multinode.sh` 启动前会门禁校验。
- **checkpoint 存 `["model","optimizer","extra","hf_model"]`**，不是 MOPD 的只存
  `hf_model`：多天的 run 必须能 `resume_mode=auto`（需要 optimizer + extra），只存 hf_model
  的话一次崩溃就从头再来。代价约 56 GB/ckpt。
- **中途评测**：MOPD 的 `eval_watcher.py`（轮询 checkpoint 在另一批卡上跑小片 benchmark）
  **没有移植** —— 8 台机器全被训练占满，没有空闲卡。改为用现成的 `run_eval_experts.sh`
  对已落盘的 `global_step_N/huggingface` 做离线评测。

---

## 6. Checkpoint 的存法（Merge 的前置条件）

### 存的是什么

默认 `save_contents: ['model','optimizer','extra']`，**FSDP 分片格式，不是 HF 格式**：

```
${OUTPUT_ROOT}/checkpoints/llm_fusion/<exp>/
├── latest_checkpointed_iteration.txt
└── global_step_<N>/
    ├── actor/                                   # ← Merge 要指向这里
    │   ├── model_world_size_<W>_rank_<R>.pt     # 每 rank 一个，共 W 个
    │   ├── optim_world_size_<W>_rank_<R>.pt
    │   ├── extra_state_world_size_<W>_rank_<R>.pt   # lr_scheduler + RNG
    │   ├── fsdp_config.json                     # {FSDP_version, world_size}
    │   └── huggingface/                         # 只有 config + tokenizer，无权重
    └── data.pt                                  # dataloader 状态
```

分片文件名绑定 `world_size`，所以 32 卡存的要按 32 卡布局读。

### 精度：磁盘上是 fp32（实测确认）

读共享盘上既有的 FSDP 分片：

```
param dtypes: {'torch.float32': 399}
model.embed_tokens.weight: dtype=torch.float32 shape=(4748, 2560) dtensor=True
```

399 个 tensor 全是 fp32，且是 DTensor（shape 是分片后的局部形状）。
原因是混合精度训练下计算走 bf16、但优化器维护的 **master weights 是 fp32**，
`model.state_dict()` 拿到的就是 master weights。

**但导出环节会掉精度**，这是 Merge 的关键：

```
训练 master weights (fp32)
   └─ FSDP shard 落盘 (fp32) ✅ 实测确认
      ├─ verl model_merger        → bf16 ❌  base_model_merger.py:394 写死
      ├─ save_contents 'hf_model' → bf16 ❌  fsdp_checkpoint_manager 写死
      └─ convert_experts_to_hf.py → fp32 ✅  Merge 必须走这条
```

`convert_experts_to_hf.py` 与上游唯一的区别就是那一行：

```python
state_dict[key].append(tensor._local_tensor.float())   # keep fp32, the ONLY change
```

因为 RL 的 task vector `θ_expert − θ_base` 很小，bf16 只有 8 位尾数，
round-trip 的量化误差和 delta 本身同量级，相减后噪声占比被放大。

> **推论**：Hub 上那五个 bf16 expert **不能用作 merge 输入**（README 也这么说）。
> 它们只能做 MOPD teacher 或评测 —— 那两种场景 vLLM 反正按 bf16 推理，不涉及做差。
> 要完整复现 Merge，必须自己训五个 expert 存 FSDP checkpoint，再 fp32 导出
> （每个约 15G，五个 75G），且 `merge_llms.py` 会把 base + 5 experts 全部
> fp32 载入 CPU 内存，4B 约需 90G RAM（本机 2.2T，够）。

### 只存 model 分片

按需求去掉 optimizer 和 extra_state：

```bash
actor_rollout_ref.actor.checkpoint.save_contents='["model"]'
```

已写进 `run_train_small.sh`（变量 `CKPT_CONTENTS`），四个范式共用。

体积账（4B 模型，每个 save step）：

| 内容 | 大小 | 说明 |
| :--- | ---: | :--- |
| model (fp32) | ~16G | task vector 需要，**必留** |
| optimizer (Adam 两个 moment, fp32) | ~32G | 去掉省 2/3 |
| extra_state | 很小 | lr_scheduler + RNG |

**代价**：`load_contents` 默认镜像 `save_contents`，所以 resume 不会去找不存在的文件、
不会报错；但断点续训只能恢复权重，**优化器动量和随机流从头开始**。
对「训完就导出做 Merge」的一次性训练无所谓；长跑要容错就别关 optimizer。

---

## 7. 当前状态与后续

**已完成**
- 环境、数据、权重、SandboxFusion 全部就绪并自检通过
- 四种训练范式（Agent / Code domain-RL、Mix RL、MOPD）小规模跑通
- checkpoint 精度链路查清，已改为只存 model 分片
- 机器扩到 **2 机 16 卡**；Ray 多机重建、NCCL GDR 绕行、每节点 sandbox 全部打通

**进行中**
- **Agent 域完整训练（2 机 16 卡 × 300 步，论文超参）**，见第 4b 节。
  截至 step 81：reward 0.245 → 0.934，`num_turns/mean` 9.4，无致命错误。
  每步约 476s，预计总计约 41 小时。
  产物：`$OUTPUT_ROOT/checkpoints/llm_fusion/expert_agent_qwen3_4b/global_step_{25,50,...,300}`

**待做**
- 4 个模型 × 6 个 nosandbox 基准的评测（`run_repro.sh nosandbox`，约 74 min/模型）
- LCB v5/v6 那批（`run_repro.sh code`）。
  ⚠️ **step 25/50/75 这三次中途验证的 LCB 数字作废**（5.8 节的 sandbox 问题，
  已于 step 81 前修复），需要训完重评
- 另外四个域的 expert（Math / Science / Code / IF），Merge 需要五个都有
- Merge 链路端到端验证：`convert_experts_to_hf.sh` → `run_merge.sh`
  （Task Arithmetic λ=0.6 复现已发布 Merge）。
  ⚠️ 注意 16 卡产物的 `world_size=16`，与 8 卡的分片布局不通用

**评测口径提醒**：`run_repro.sh` 用 `VAL_N=4` 而非论文的 16，为的是把
4 模型 × 8 基准压进单机可接受的时长。mean@4 的方差比 mean@16 大，
**趋势可比，绝对值有 1~2 点噪声**，不能直接和论文表格逐位对齐。
要严格对齐就设 `VAL_N=16`（时长约 ×4）。

训练中途的验证同样是 `val_kwargs.n=4`（脚本默认），口径与此一致。

一次跑通的 AIME2025 冒烟结果（base，`VAL_N=4`、`MAX_RESPONSE_LENGTH=4096`）：
`mean@4 = 0.317`。注意该次 response 均长 3468 已接近 4096 上限，
**被截断压低了分数**，正式评测用默认 24576。
