# E7 —— 联合多域 RL 的难度分层诊断

给操作这台 4 节点（4×8 GPU）机器的 agent 的交接文档。写于 2026-09-22。

你要**并行跑四个 GRPO 训练分支，每个节点一个**。每个分支都是一次五域联合 RL 运行，彼此之间**只有**训练混合数据被过滤或重采样的方式不同。目标不是训出更好的模型 —— 而是要确定五个域中有两个学不动的**原因**，以及这两种失败是否有各自独立的成因。

在动手之前请先读[背景](#背景)：整个实验都系于一个特定的指标，而那个最显而易见的指标是具有误导性的。

---

## 背景

一次五域联合 RL 运行（`train_mix_8x8_20260919_162504`，Qwen3-4B，GRPO，bs=128）在 step 1–270 上做了逐域分析。速率按每步计；参考对象是单域的 IF 专家运行，它达到了 dWrong/dt = −0.159。

| domain | prompt share | eff. groups/step | dWrong/dt | vs expert | group std | 判定 |
|---|---:|---:|---:|---:|---|---|
| Science | 19.5% | 15.8 | −0.0067 | 0.04× | 0.216 → **0.182** | 健康 |
| Agent | 14.8% | — | — | — | 0.500 → **0.426** | 健康 |
| Math | 23.4% | 19.5 | +0.0078 | −0.05× | 0.243 → 0.233 | **饱和** |
| IF | 18.0% | 7.7 | −0.0267 | 0.17× | 0.120 → 0.124 | **饥饿** |
| **Code** | **24.2%** | 10.4 | **−0.0011** | **0.01×** | 0.085 → 0.083 | **冻结** |

两种不同的病理，我们认为它们有不同的成因和不同的修法：

- **IF 是饥饿。** 它是 reward 最稀疏的域：68% 的 GRPO group 的 reward 方差为零，因此产生零 advantage、没有梯度。占 18% 的 prompt，最终只拿到约 15% 携带梯度的 group。一次 bs=512 的独立运行把 IF 的有效 group 提到专家等价水平的 0.82×，并把学习恢复到 0.51–0.66× —— 这是**部分**恢复，与「稀释是主因」一致。
- **Code 是冻结，而饥饿解释不了它。** Code 握有*最大*的 prompt 份额（24.2%），每步有效 group 也比 IF 多（10.4 对 7.7），然而它的全错 group 占比在 275 步中纹丝不动（57.6% → 57.4%），全对占比从未超过 3%。梯度供给是充足的，而它仍然学不动。假设是：该任务**超出了模型当前的能力**，于是大多数 prompt 永久处于全错状态。**若这个假设成立，那么更多的 batch 或更多的数据都修不了 Code。**

### 你要检验的假设

| | 梯度供给充足 | 梯度供给饥饿 |
|---|---|---|
| **任务在能力范围内** | 健康（Science、Agent） | **饥饿**（IF）—— 靠重新配平修复 |
| **任务超出能力范围** | **难度冻结**（Code）—— 重新配平无用 | 两者同时发生 |

预测是：**每种失败都有各自的修法，且这些修法不能互相替代。** 分支 1 应当解冻 Code 且不惊动 IF；分支 2 应当恢复 IF 而让 Code 保持冻结。如果某一个干预把两个都修好了，那么双因子的解释就崩溃了，而我们需要知道这一点。

### 指标陷阱 —— 请仔细阅读

**不要用 group reward std 的均值作为健康信号，也不要单独使用 `zero_var_frac`。** 两者都是双向的：

- 一个*还没起步*的域，离散度低、零方差高，因为它的 group 一致地全错。
- 一个*已经完成*的域，同样离散度低、零方差高，因为它的 group 一致地全对。

单域的 IF 专家正好演示了这一点：随着它学得很好，group std **下降** 0.120 → 0.087 → 0.066，而零方差**上升** 68% → 81%，因为全错 group 正在转化为全对 group（wrong 42.7% → 18.0%，right 25.3% → 62.9%）。本分析的一个早期版本把方差下降读成了恶化，事实恰恰相反。

**请改用这个分解：**

- `all_wrong_frac` ↓ = 该域正在解锁它原先解不出的问题（「课程解锁速率」，**dWrong/dt** —— 主要信号）
- `all_right_frac` ↑ = 该域正在巩固解法（**dRight/dt**）
- `effective_groups` = 这一步实际贡献梯度的 group 数

这些现在由 trainer 逐域记录（2026-09-22 加入，见[指标](#指标)），所以你不需要再对 rollout dump 做后处理。

---

## 四个分支

所有分支：Qwen3-4B-Instruct-2507、GRPO、五个域、**超参完全相同**。唯一的差别是训练混合数据。各占一个节点，并行运行。

| 分支 | 名称 | 干预 | 若 2×2 成立则预测 |
|---|---|---|---|
| 1 | `code_filtered` | 只保留通过率 ∈ (0,1) 的 Code prompt | Code 解冻；IF 不变 |
| 2 | `if_upsampled` | 把 IF prompt 复制 2.5×；Code 不动 | IF 恢复；Code 仍冻结 |
| 3 | `both` | 分支 1 与 2 同时施加 | 两者都恢复 —— 两个修复彼此独立 |
| 4 | `code_downsampled` | 只保留 10% 的 Code prompt | **其他域**在释放出的算力上变好 |

分支 4 是那个有实践回报的：如果丢掉一个过难的域能让其他域变好，那么在多域 RLVR 中，一个过难的域不只是自己学不会，它还会**挤掉那些本来学得会的域**。单纯的稀释故事预测不出这一点。

### 第 1 步 —— 测量 Code 的逐 prompt 通过率

分支 1 和 3 需要知道哪些 Code prompt 是根本可解的。**不要**从 prompt 文本猜测难度代理量。要去测量它：

使用已完成的那次基线运行留下的 rollout dump，位于
`runs/llm_fusion/rollout_data/mix_rl_qwen3_4b_8x8/`（step 1–382；**只使用 step ≤ 275**，见[数据卫生](#数据卫生)）。每行是一条 rollout，含 `input`、`score`、`gts`。按 `input` 分组，计算每个 prompt 的平均 score，并分类：

- `pass_rate == 0` → 从未被解出；这些正是分支 1 要移除的
- `0 < pass_rate < 1` → 可学习的信号；分支 1 保留这些
- `pass_rate == 1` → 已经解出

`research/group_stats.py` 和 `research/variance_trajectory.py` 里已经有域标注的逻辑 —— **请从 `research/variance_trajectory.py` 导入 `domain()`，而不是重写它。** 标注是微妙的：只有 IF 带显式的 `rm_type`；Code 靠 `gts` 中的 stdin/stdout 测试向量识别，Agent 靠期望的工具调用列表，Science 靠单个答案字母，Math 靠裸表达式。**Agent 必须在 Math 之前检查** —— 工具调用列表同样「不是字母」，而一个早期分析曾把每一条 Agent rollout 都误归为 Math。

在继续之前，用已知的混合比例校验你的标注：按 prompt 计数为 Math 25.1% / Science 22.4% / Code 21.9% / IF 18.9% / Agent 11.7%。

报告落入每个桶的 Code prompt 各有多少。如果全错桶**不是**占绝大多数（我们预期约 57%），就停下来说明 —— 说明前提就是错的。

### 第 2 步 —— 构建四个混合集

源数据：`data/LLM-Fusion/train/Mix/train-*.parquet`（16 个分片，87,699 行）。
Schema：`data_source`、`prompt`、`ability`、`reward_model`、`extra_info`、
`agent_name`。域就是 `data_source`（例如 `LLM-Fusion/Code`）—— **构建**混合集时直接用这一列；上面那个 `domain()` 启发式只在处理 rollout dump 时才需要，因为 dump 里没有这一列。

把每个分支的混合集写到 `data/LLM-Fusion/train/Mix_e7_<arm>/` 下各自的目录。**不要修改 `Mix/`** —— 基线和正在跑的对比都依赖它。

约束条件：

- 保留每一列，包括 `agent_name`（Agent 行带 `agent_name=tool_agent`，会被路由到多轮工具循环；丢了它会静默地破坏 Agent）。
- 用与第 1 步分组时相同的 key，把 Code prompt 匹配到它们的通过率桶。dump 里存的是*渲染后*的 prompt，而 parquet 里存的是结构化的 `prompt`；在信任 join 之前先验证它真的匹配上了，并报告匹配率。匹配率低意味着 join 写错了，而不是数据有问题。
- 分支 2 的上采样是**仅**对 `LLM-Fusion/IF` 行做行复制。
- 分片大小与原始保持大致可比（约 5.5k 行/分片）。

在报告里记录四个分支各自的逐域行数。

### 第 3 步 —— 启动

启动脚本：`examples/llm_fusion/train/train_mix_rl.sh`，由 `TRAIN_FILE` 驱动。
可参考 `run_train_multinode.sh` 了解本仓库是如何串起一次运行的；它负责 Ray/sandbox/环境的前置检查。

每节点设置（**每节点一个 8 卡作业 —— 不要跨节点**）：

```bash
export NNODES=1
export NGPUS_PER_NODE=8
export TRAIN_BATCH_SIZE=512          # 与 bs512 对比运行一致
export ACTOR_LR=2e-6                 # 也与它一致；见下面的警示
export TOTAL_TRAINING_STEPS=250
export SAVE_FREQ=50
export TEST_FREQ=25
export TRAIN_FILE="/taijifs_sz34/.../data/LLM-Fusion/train/Mix_e7_<arm>/train-*.parquet"
export EXPERIMENT_NAME=e7_<arm>_qwen3_4b
bash examples/llm_fusion/train/train_mix_rl.sh
```

每节点的前置条件：代码 sandbox 必须在 `localhost:8080` 上就绪（`bash setup_sandbox.sh`），否则 **Code 的 reward 会静默地趋近于零 —— 那会伪造出我们要检验的那个结果本身。** 启动前用 `run_eval_experts.sh` 里的 curl 探活验证，并在报告中说明你做了。

**加上梯度范数裁剪。** 基线运行在 step 300 发散了（见[数据卫生](#数据卫生)）。检查启动脚本已经设了什么，并确保有一个有限的裁剪值在生效。

分析时要带上两条警示：

1. bs=512 配 lr=2e-6，相对 bs=128/1e-6 的基线**同时**改变了 batch 和学习率，所以这两个效应是纠缠的。四个分支共享该设置，因此**分支之间的对比是干净的**；与 bs=128 基线的对比则不干净。
2. bs=512 跑 250 步的样本量约为 bs=128 跑 250 步的 4 倍。分支之间互相比，并与下面描述的 bs=512 运行比 —— 而不是与 bs=128 的步数比。

### 第 4 步 —— 分析

主要信号，逐分支、逐域：

```bash
python research/parse_train_log.py <log> \
    --grep 'grpo/group_by_domain/.*(all_wrong_frac|all_right_frac|effective_groups)' \
    --every 10
```

在一个远离预热区的窗口（例如 step 50–250）上拟合 **dWrong/dt**，并跨分支比较。参考点：expert_if −0.159（健康的单域）、Science −0.007（混合中健康）、Code −0.001（冻结）、IF −0.027（饥饿）。

**判定规则，现在就固定 —— 看到结果之后不得调整。**

| 观测 | 结论 |
|---|---|
| 分支 1 把 Code 的 dWrong/dt 提到其基线的 ≥10×（−0.001 → ≤−0.01），且 IF 保持在噪声范围内 | 难度冻结被确认为一种独立模式 |
| 分支 2 把 IF 提到 ≥0.8× 专家水平，且 Code 保持冻结 | 饥饿被确认为一种独立模式 |
| 分支 3 ≈ 分支 1 的 Code 效应 + 分支 2 的 IF 效应 | 两个修复彼此独立 —— 2×2 成立 |
| 分支 2 *也*解冻了 Code | **2×2 被证伪**；只有一个因子，请如实报告 |
| 分支 4 使非 Code 域的改善超出分支间噪声 | 过难的域会挤掉可学的域 |
| 没有任何分支使任何东西移动 | 某处配置错了 —— 先检查 sandbox |

同时报告逐分支的 `val-core/<benchmark>/acc/mean@4` 趋势，但要把训练动态当作主要依据：250 步不太可能让 benchmark 移动多少，而机制存在于 group 统计之中。

---

## 指标

2026-09-22 加入到 `verl/trainer/ppo/metric_utils.py`（由 `tests/trainer/ppo/test_metric_utils_on_cpu.py` 中的测试覆盖）。逐域、逐步：

| `grpo/group_by_domain/<domain>/` 下的键 | 含义 |
|---|---|
| `all_wrong_frac` | 一致得 0 分的 group 占比 —— **主要信号** |
| `all_right_frac` | 一致得非零分的 group 占比 |
| `zero_var_frac` | 退化 group（= 上面两者之和） |
| `reward_std_mean` | group 内 reward std 的均值 —— advantage 的幅度 |
| `num_groups` | 该域这一步的 group 数 |
| `effective_groups` | 非退化 group —— 实际可用的梯度 |

`<domain>` 是 `data_source` 的小写尾部：`math`、`science`、`code`、`if`、`agent`。注意 `reward_std_mean` 是**group 内** std 在 group 上的平均（正是 GRPO 的 advantage 所缩放的量），不是整个 batch 所有样本的 std。

对于部分给分的验证器，`all_right_frac` 的含义是「一致地非零」，而不是「完美」。

---

## 数据卫生

**基线运行在 step 300 之后发散了。** grad_norm 从 0.039 飙到 0.507，回答长度膨胀到 15,093 token（92% 被截断），熵塌缩到 0.009，aime25/aime26/gpqa/ifbench 全部归零。**只使用** `mix_rl_qwen3_4b_8x8` 中 step ≤ 275 的数据；`global_step_275` 是最后一个健康的 checkpoint。IF 与 Code 的发现不受影响 —— 两者从 step 0 就可见 —— 但不要把速率拟合穿过崩溃区间。

**bs=512 的对比运行停在 step 78**（`train_mix_8x8_20260921_164602`，日志自 11:41 起无更新，无存活进程）。它的数字（IF 恢复到 0.51–0.66×）来自 78 步，噪声较大。不要把它们当作已定论；若想有更扎实的对比，那次运行需要续跑 —— 续跑前请先协调，那个节点不是你的。

---

## 交付物

把结果写到 `research/experiments/20260922_difficulty_stratified/README.md`，
遵循 `../20260920_mopd_extrapolation/README.md` 的格式 —— 先写问题、方法、事先声明的判定规则，再写结果。

需包含：

1. 第 1 步得到的 Code 通过率分布，以及 parquet join 的匹配率。
2. 四个混合集各自的逐域行数。
3. 逐分支、逐域的 dWrong/dt 与 dRight/dt，并说明拟合窗口。
4. 哪一条判定规则被触发了 —— 包括触发的是证伪性的那一条。
5. 任何与上述背景相矛盾的东西。**一个干净的阴性结果在这里也是结果** —— 2×2 是一个假设，不是一个待确认的结论。

报告这些运行实际做了什么。如果某个分支崩了、配错了，或你跳过了什么，请明确说出来，而不是把部分数字当作完整结果上报。

## 背景资料

完整的研究计划与合并侧的结果：
`research/ideas/parallel_then_merge_vs_joint_rl.md`。
背景一节背后的全部测量数据：
`research/results/20260920_session_summary.md`。

应当复用而不是重新实现的工具：

| 工具 | 用途 |
|---|---|
| `research/parse_train_log.py` | verl console 日志 → 指标时间序列 |
| `research/variance_trajectory.py` | 从 dump 计算逐域 group 构成动态；也是 `domain()` 的来源 |
| `research/group_stats.py` | 单个窗口上的逐域 GRPO group 统计 |
