# 强 RL 专家合并基线

本实验在五个本地 Qwen3-4B RLVR 专家上运行官方实现。

## RAM+

- 来源：`research/baselines/RAM`，ACL 2026 官方仓库。
- 模式：`arm-r-v2`。
- 阈值：`1e-5`。
- 重缩放因子：`1.2`，即官方 README 中 RAM+ 的设置。
- 仅为兼容所做的本地补丁：官方加载器会尝试把 tokenizer 回写进每个输入 checkpoint。由于共享的 base checkpoint 是只读的，该写入被禁用；合并算法本身未改动。

## ResMerge

- 来源：`research/baselines/ResMerge`，官方仓库。
- Rank-1 谱头。
- SRC-A beta 范围 `0.7–1.0`。
- Head ratio budget `0.2`。

两者都使用与 Task Arithmetic 相同的 fp32 base 与专家，并保存 bf16 推理 checkpoint。评测遵循仓库的八基准协议；初筛可用 `n=4`，而最终主张需要全新的更高样本量确认。

## 推理筛选结果（`n=4`）

| 方法 | 自然范数 | 范数匹配到 TA |
|---|---:|---:|
| RAM+ | 58.61 | 61.98 |
| ResMerge | 56.05 | 62.16 |
| Soft-Overlap | — | 60.05 |
| Task Arithmetic, `lambda=1` | 64.50 | 64.50 |

这些数字是 AIME25、AIME26、GPQA 以及两个 LiveCodeBench 版本上未加权的平均。范数匹配挽回了 RAM+/ResMerge 的大部分损失，但两者的点估计仍低于 TA。它们自然部署时的 update norm 分别是 `0.714` 和 `0.271`，而 TA 是 `2.228`；对应的范数匹配乘数为 `3.121` 和 `8.234`。这些巨大的乘数清楚地表明：对 RLVR 合并而言，**方向和半径两者都重要**。

相对 TA 的配对 prompt-bootstrap 差距：scaled RAM+ 为 −2.52 pp [−4.71, −0.42]，scaled ResMerge 为 −2.34 pp [−4.77, +0.05]。ResMerge 的残余差距尚无结论；两者都比 TA 生成更少的 token、触顶也更少。在经校准的缩放与可比的生成开销下，把它们保留为竞争者。这些 n=4 推理结果**不**衡量 IF/Agent 的保留性。

缩放作用于此前已导出的 bf16 checkpoint。未来的对比必须在最终导出之后重新测量范数；仅凭所要求的缩放因子并不能证明部署范数精确匹配。完整的重估见[当前发现](../../results/current_findings.md)。
