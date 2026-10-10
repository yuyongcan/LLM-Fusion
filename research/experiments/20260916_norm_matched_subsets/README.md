# 范数匹配的专家子集对照

## 研究问题

把全局余弦较低的 IF 和 Agent 专家加入 Math/Science/Code（MCS）合并时，推理能力是真的提升了，还是说表面上的增益只是由「更大的总 update norm」造成的？

## 固定参考点

目标范数是五专家 Task Arithmetic 在全局 `lambda=1.0` 下的更新范数。
系数由全局 task-vector Gram 矩阵精确计算得出，该矩阵位于
`research/results/geometry_4b/geometry.json`。

| 专家子集 | 统一系数 | 用途 |
|---|---:|---|
| Math, Science, Code | 1.3755469137 | 与五专家 λ=1.0 的更新范数相同 |
| Math, Science, Code, IF | 1.0800244928 | 隔离 Agent 的贡献 |
| Math, Science, Code, Agent | 1.2203741254 | 隔离 IF 的贡献 |

这些是**因果对照组**，不是提出的最终方法。用 `n=4` 做筛选；任何实质性的效应都要用全新的 `n=16+` 样本来确认。
