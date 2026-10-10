# 重算的证据表格

由 `research/experiments/reassess_existing_results.py` 从已保存的 JSONL 评测结果生成。
所有平均都对基准/域等权。分数使用平均准确率/reward，而非 best-of-n。
区间在模型之间按 prompt 配对、在每个基准内部重采样，并以已保存的生成为条件。
相同的输入构成一个聚类（在有 ground truth 时会做校验）；重复出现的来源行保留原始权重。
它们属于探索性的，未针对选择/多重比较做校正，也不确立种子鲁棒性。
长度触顶以 response_length >= 16384 判定；这是一个代理量，不是记录的结束原因。

## reasoning_n4

| Model | Score | Mean tokens | Cap hits (%) |
| --- | --- | --- | --- |
| ta_lambda1 | 64.50 | 6407 | 2.65 |
| ta_lambda1_other_run | 62.67 | 6476 | 2.75 |
| ram_plus | 58.61 | 4738 | 2.29 |
| ram_plus_normmatch | 61.98 | 5131 | 0.55 |
| resmerge | 56.05 | 4828 | 4.61 |
| resmerge_normmatch | 62.16 | 5895 | 0.76 |
| soft_overlap_normmatch | 60.05 | 5115 | 0.55 |
| if_embedding_1.2 | 63.09 | 6458 | 2.63 |

配对对比（除析因对比外，均为候选减参考）：

| Contrast | Delta (pp) | 95% CI (pp) | Tokens delta |
| --- | --- | --- | --- |
| ta_lambda1_other_run | -1.83 | [-4.17, +0.38] | +69 |
| ram_plus | -5.89 | [-8.58, -3.31] | -1668 |
| ram_plus_normmatch | -2.52 | [-4.71, -0.42] | -1276 |
| resmerge | -8.46 | [-11.17, -5.88] | -1579 |
| resmerge_normmatch | -2.34 | [-4.77, +0.05] | -512 |
| soft_overlap_normmatch | -4.46 | [-7.08, -1.89] | -1291 |
| if_embedding_1.2 | -1.42 | [-4.08, +1.08] | +52 |
| ram_plus_norm_gain | +3.37 | [+1.11, +5.74] | +392 |
| resmerge_norm_gain | +6.12 | [+3.69, +8.55] | +1067 |

## calibration_n2

| Model | Score | Mean tokens | Cap hits (%) |
| --- | --- | --- | --- |
| ta_lambda0.8 | 58.68 | 2529 | 1.17 |
| ta_lambda1 | 58.91 | 2792 | 2.27 |
| ta_lambda1.2 | 57.71 | 3121 | 3.20 |

配对对比（除析因对比外，均为候选减参考）：

| Contrast | Delta (pp) | 95% CI (pp) | Tokens delta |
| --- | --- | --- | --- |
| ta_lambda0.8 | -0.23 | [-1.97, +1.49] | -263 |
| ta_lambda1.2 | -1.21 | [-3.13, +0.72] | +329 |

## factorial_n16

| Model | Score | Mean tokens | Cap hits (%) |
| --- | --- | --- | --- |
| mcs_lambda1 | 59.69 | 6377 | 1.03 |
| mcsi_lambda1 | 59.63 | 6363 | 0.81 |
| mcsa_lambda1 | 60.33 | 6830 | 1.48 |
| five_lambda1 | 61.33 | 6755 | 0.75 |

配对对比（除析因对比外，均为候选减参考）：

| Contrast | Delta (pp) | 95% CI (pp) | Tokens delta |
| --- | --- | --- | --- |
| if_x_agent | +1.06 | [-1.26, +3.54] | -61 |
| all5_minus_mcs | +1.64 | [-0.62, +4.02] | +378 |
| if_given_mcs | -0.07 | [-2.03, +1.89] | -15 |
| agent_given_mcs | +0.64 | [-1.54, +2.76] | +453 |

## 校准集逐域分数

| Model | math | science | code | if | agent |
| --- | --- | --- | --- | --- | --- |
| ta_lambda0.8 | 51.17 | 53.91 | 21.93 | 69.14 | 97.27 |
| ta_lambda1 | 48.83 | 55.08 | 24.26 | 71.09 | 95.31 |
| ta_lambda1.2 | 47.66 | 56.25 | 21.74 | 73.44 | 89.45 |

校准 prompt 身份审计：

| Domain | Source rows | Unique inputs | Duplicate source rows |
| --- | --- | --- | --- |
| LLM-Fusion_Math | 128 | 128 | 0 |
| LLM-Fusion_Science | 128 | 128 | 0 |
| LLM-Fusion_Code | 128 | 128 | 0 |
| LLM-Fusion_IF | 128 | 128 | 0 |
| LLM-Fusion_Agent | 128 | 125 | 3 |

## 几何敏感性

| Subset | Pairs | Weight r | Layer 16 r | Functional r |
| --- | --- | --- | --- | --- |
| all_directed_pairs | 20 | 0.3887 | 0.4116 | 0.5231 |
| without_if_to_agent | 19 | 0.5701 | 0.5637 | 0.4973 |
| without_expert_math | 12 | 0.4251 | 0.5774 | 0.4698 |
| without_expert_science | 12 | 0.3237 | 0.3624 | 0.4937 |
| without_expert_code | 12 | 0.3892 | 0.4156 | 0.5913 |
| without_expert_if | 12 | 0.5855 | 0.5749 | 0.3671 |
| without_expert_agent | 12 | 0.6056 | 0.5927 | 0.5551 |

有向配对之间共享专家与目标；这些是描述性的相关性，不是独立样本。
功能指标测量的是基座激活上局部线性模块的输出位移，不是最终 logit 的变化。

## 统一扫描的可识别性

三个试点设置同时变动了全部五个专家系数。
线性设计秩：2/6；完整二次型秩：3/21。
它们无法识别各专家独立的效应，也无法识别成对的系数交互。
