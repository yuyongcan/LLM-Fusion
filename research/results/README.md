# 结果

生成的汇总与图表存放在此。报告准确率时要同时给出回答长度、截断率、样本量和不确定性；不要在同一份数据上既选系数又报分数，除非明确标注这是开发集上的结果。

- [current_findings.md](current_findings.md)：当前的解读与方法决策。
- [20260917_reassessment/tables.md](20260917_reassessment/tables.md)：生成的对比与不确定性。
- [20260917_reassessment/summary.json](20260917_reassessment/summary.json)：来源哈希与数值结果。
- [phase1_findings.md](phase1_findings.md)：早期实验叙述，含已修正的结论。
- [functional_geometry_4b/README.md](functional_geometry_4b/README.md)：该功能诊断的适用范围与局限。

公开集 AVG8、推理 AVG5、析因 AVG3 与校准五域宏平均是**互不相同的指标**。
在已用过的公开 prompt 上重复解码，并不能把它们变成独立的确认数据。
