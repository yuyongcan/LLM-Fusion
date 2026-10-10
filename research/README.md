# LLM-Fusion 合并方法研究

本目录研究如何合并五个 Qwen3-4B RLVR 专家：Math、
Science、Code、指令遵循（IF）和 Agent。

简短的中文总览见 [SUMMARY.zh-CN.md](SUMMARY.zh-CN.md)。

## 当前结论

- 目前还没有任何候选方法在所有五个域上相对最强基线表现出**已确认**的提升。
- Task Arithmetic（TA）仍然是锚点。在已保存的推理筛选集上，`lambda=1` 得分 64.50 AVG5。
- RAM+ 与 ResMerge 在做完 update-norm 匹配后能挽回大部分损失，但在当前筛选集上都没有超过 TA。
- 已测试过的 Soft-Overlap 合并不如 TA，不再是主攻方向。
- 表面上看到的 IF-Agent 交互效应、以及功能几何（functional geometry）的优势，在统计上都不稳定。
- 五专家系数校准现在只是诊断/对照手段，不是预期的学术贡献点。
- 当前活跃的假设是 **VICP（Verifier Influence Cone Projection，验证器影响锥投影）**：估计各参数分量的跨域验证器带符号影响力，再投影到一个经不确定性调整的鲁棒改进锥中。
- VICP 必须能预**先指定**的留出混合扰动做出预测，并且要胜过配平过的五标量校准对照组。否则即便某个系数组合碰巧分数更高，也判定其作为新合并方法不成立。

公开的基准跑测结果以及现有的校准划分属于**开发证据**，不是未被触碰过的测试数据。

## 建议阅读顺序

1. [SUMMARY.zh-CN.md](SUMMARY.zh-CN.md)：精简的项目总览。
2. [results/current_findings.md](results/current_findings.md)：证据与决策。
3. [notes/research_plan.md](notes/research_plan.md)：验收标准与后续实验。
4. [state.md](state.md)：已完成的工作与待办任务。
5. [results/20260917_reassessment/tables.md](results/20260917_reassessment/tables.md)：重算得到的数值表格。

`notes/`、`ideas/` 下的文件以及 `results/phase1_findings.md` 保留了历史上的推理过程。它们对追溯来源有用，但**不是**当前的决策依据。

## 目录结构

- `experiments/`：分析、构建 checkpoint、评测与汇总的代码。
- `results/`：精简的生成产物与解读。
- `baselines/`：vendored 的 RAM、ResMerge 和 SVC 实现。
- `papers/`：论文原始归档与摘录文本。
- `notes/`：实验协议、文献地图与早期假设。
- `ideas/`：候选方法，包括已被否决或尚未验证的提案。

体积较大的生成 checkpoint、日志、PID 文件和 Python 缓存均已排除。
原始评测 JSONL 予以保留，因为重估脚本要用它们复现分数、开销、来源哈希和 bootstrap 区间。

## 复现总览

```bash
python3 research/experiments/reassess_existing_results.py
```

重建模型或发起评测需要各个实验 README 中描述的路径与运行时。
硬件启动脚本里含有历史上的集群地址，复用前必须逐一审阅。
