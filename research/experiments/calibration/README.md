# 合并校准划分

`manifest.json` 是一个确定性的、仅含索引的采样，每域 128 条训练行，用于试点的合并校准。它是在专家训练**之后**采样的，**可能包含**专家见过的样本。从训练文件中采样并不能确立与公开基准在内容上互不相交；没有记录过此类去重审计。源数据未作改动。

lambda=0.8/1.0/1.2 的统一系数运行已在 n=2 下完成。它们的五域均值是 58.68/58.91/57.71；逐域分数、开销与不确定性见[重算表格](../../results/20260917_reassessment/tables.md)。Agent 在 128 条来源行中有 125 条去重输入；做不确定性估计时，相同的输入必须留在同一个聚类中。

默认划分用以下命令生成：

```bash
python research/experiments/sample_calibration_split.py \
  --train-root /taijifs_sz34/share_304169095/hunyuan/yongcanyu/data/LLM-Fusion/train \
  --output research/experiments/calibration/manifest.json \
  --materialize-dir research/experiments/calibration/data \
  --per-domain 128
```

任何后续的校准运行都必须记录 manifest 哈希、候选系数网格、解码种子、验证器版本、平均回答长度和截断率。现有的公开基准已经影响过选择过程，属于开发数据。[当前研究计划](../../notes/research_plan.md)要求使用新预留的验证与确认 prompt。

用 `run_one_eval.sh` 在五个已物化的域文件上给 checkpoint 打分。默认是 `n=2`，用于低成本筛选；决赛候选应当用更大的 `n` 或独立的解码种子重跑。

## 下一步设计（已准备，未运行）

`next_design.json` 规定了 TA 锚点、十个单专家扰动，以及四个用于预测检验的混合设置。五个域都必须打分。已完成的三个统一设置只能识别一个一维响应；它们无法拟合五个各自独立的专家效应，也无法拟合完整的二次型。轴向设计可以识别主效应与对角曲率；检验预测时要把混合设置排除在拟合之外。若不再增加测量，它依然不足以支撑一个完整的 21 参数二次型。
