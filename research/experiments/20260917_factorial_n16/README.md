# IF × Agent 析因确认

本实验在 `n=16` 下重复此前探索性的四格对比，基准为 AIME 2025、AIME 2026 和 GPQA —— 也就是早先 `n=4` 时交互效应出现的那三个基准。

所有模型都使用 Task Arithmetic 系数 `1.0`：

- `mcs_lambda1`：Math + Science + Code。
- `mcsi_lambda1`：MCS + IF。
- `mcsa_lambda1`：MCS + Agent。
- `five_lambda1`：MCS + IF + Agent。

交互对比量为

`five - mcsi - mcsa + mcs`。

这是在**已有公开 prompt** 上做的更高样本量诊断，**不是**独立确认。这三个基准是在 n=4 筛选之后才选定的。不要用它们的结果来选择最终的方法系数。

统一重分析给出交互项 +1.06 pp，95% CI [−1.26, +3.54]。
All5-MCS 为 +1.64 pp，CI [−0.62, +4.02]；两者都不是已确认的增益。
参见[重算证据](../../results/20260917_reassessment/tables.md)。
