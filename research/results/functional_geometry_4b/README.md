# 功能几何诊断

在基座模型激活上完成的 `Delta W x` 试点分析。记录的设置为：每域 8 条 prompt、64 个 token，最大 prompt 长度 1024；252 个 transformer 线性模块。JSON 中**没有**记录 prompt offset。该指标省略了 embedding 与归一化的扰动，也不把每个扰动传播到最终的 logit 或 verifier reward。

`functional_vs_transfer.csv` 把目标域的功能余弦与公开评测数据上既有的有向专家迁移量连接起来。在 20 个有向对上，Pearson r 为 0.5231，而全局权重余弦是 0.3887。去掉 IF→Agent 会反转该排序（0.4973 对 0.5701）；去掉所有 IF 配对得 0.3671 对 0.5855。各配对共享专家与域，并非独立样本。该诊断**没有**确立预测上的优越性，也没有给出一个有用的合并方向。

用 `python3 research/experiments/reassess_existing_results.py` 复现敏感性检查；参见
[current findings](../current_findings.md)。在依据该指标挑选分量之前，请在不相交的 prompt 批次上重复，并在预留的域/配对上检验预测。更多的几何工作应当绑定到一个具体的方法决策上。
