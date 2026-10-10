# 实验

本目录包含可复现的分析、构建 checkpoint、评测与汇总代码。这里刻意**不**保存生成出来的 checkpoint 和日志；请用各实验自带的脚本重建它们。

## 当前活跃路径

- `calibration/next_design.json`：已准备好的五系数实验设计。
- `calibration/run_one_eval.sh`：对单个 checkpoint 跑全部五个域。
- `merge_group_coefficients.py`：可选的模块组系数校正。
- `reassess_existing_results.py`：从保留下来的原始 JSONL 输出复现精简的证据表格。

## 诊断性或历史性工作

- `analyze_*geometry.py` 与 `analyze_task_vector_support.py`：几何诊断；对提出假设有用，但不用于当前的系数选择。
- `soft_overlap_merge.py`：已否决的测试配置，为追溯来源而保留。
- 带日期的子目录：已完成筛选的确切启动/构建定义。

`reassess_existing_results.py` 在 CPU 上运行，会记录来源哈希、对重复 prompt 做聚类，并写入 `results/20260917_reassessment/`。

硬件启动脚本中包含历史上的绝对路径、节点地址和运行时假设。复用前请逐一审阅。
