# 仓库规范

## 项目结构与模块组织

`verl/` 是主 Python 包，包含 trainer、worker、模型后端、工具以及 Hydra YAML 配置。论文专属的工作流位于 `examples/llm_fusion/`：`sft/` 从 Qwen3-4B-Base 构建 phase-1 聊天模型；`train/` 启动 expert、Mix RL 和 MOPD 任务；`eval/` 评测 checkpoint；`merge/` 转换并合并 expert；`paths.sh` 集中管理数据与输出根路径。通用维护脚本在 `scripts/`，容器定义在 `docker/`，图片在 `assets/`，vendored 的合并实现在 `third_party/MergeLM/`。测试按组件分组放在 `tests/` 下，其中硬件相关的测试套件位于 `special_distributed`、`special_e2e` 和 `special_npu`。

## 构建、测试与开发命令

```bash
pip install -e ".[test]"                         # editable 安装并附带测试工具
pre-commit run --all-files                       # Ruff、mypy、生成的配置、健全性检查
pytest tests/tools/test_function_tool_on_cpu.py -q  # 跑一个针对性的 CPU 测试
python examples/llm_fusion/download_data.py --output-dir /path/to/data
bash examples/llm_fusion/train/train_expert_math.sh
MODEL_PATH=/path/to/model bash examples/llm_fusion/eval/eval.sh
```

请在环境变量或 `examples/llm_fusion/paths.sh` 中设置 `MODEL_PATH`、`TRAIN_DATA_ROOT`、`EVAL_DATA_ROOT` 和 `OUTPUT_ROOT`。GPU、NPU、Ray 以及端到端测试需要对应的运行时，不应当作普通的 CPU 检查来对待。

## 代码风格与命名约定

新写的 Python 接口使用四空格缩进并带类型注解。Ruff 负责格式化与 lint，行宽目标 120 字符，并把 `verl` 当作 first-party 导入排在前面。函数与文件名用 `snake_case`，类名用 `PascalCase`，常量用 `UPPER_CASE`。测试命名为 `test_*.py`；若某个测试明确只能在 CPU 上跑，沿用既有的 `_on_cpu.py` 后缀。提交前请运行 pre-commit。

## 测试规范

Pytest、`pytest-asyncio` 与重跑支持通过测试依赖配置。请在受影响的组件旁边补充针对性测试，并同时覆盖正常路径与失败路径。本地优先跑最小的相关测试文件；未运行的硬件相关套件要清楚说明。本仓库不强制要求全仓库覆盖率指标。

## 提交与 PR 规范

近期的提交使用简短描述性的标题，例如 `Release the Merge scripts`。提交要保持聚焦、使用祈使语气。PR 标题需符合 `[module] type: description` 格式，例如 `[trainer] fix: validate teacher paths`；允许的 type 包括 `feat`、`fix`、`refactor`、`chore` 和 `test`。请填写 PR 描述、关联相关 issue、列出验证命令；涉及训练/评测的工作还要说明硬件、配置和结果的变化。

## 安全与生成产物

不要把 API key、与机器相关的地址、数据集、checkpoint、rollout dump 和日志提交进仓库。Code 域的 reward 以及 LiveCodeBench 评测需要通过 `SANDBOX_FUSION_URL` 访问一个可达的 SandboxFusion 服务。
