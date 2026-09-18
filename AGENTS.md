# Repository Guidelines

## Project Structure & Module Organization

`verl/` is the main Python package, with trainers, workers, model backends, utilities, and Hydra YAML configuration. The paper-specific workflow lives in `examples/llm_fusion/`: `train/` launches expert, Mix RL, and MOPD runs; `eval/` evaluates checkpoints; `merge/` converts and combines experts; and `paths.sh` centralizes data and output roots. General maintenance utilities are in `scripts/`, container definitions in `docker/`, images in `assets/`, and the vendored merge implementation in `third_party/MergeLM/`. Tests are grouped by component under `tests/`, with hardware-heavy suites under `special_distributed`, `special_e2e`, and `special_npu`.

## Build, Test, and Development Commands

```bash
pip install -e ".[test]"                         # editable install plus test tools
pre-commit run --all-files                       # Ruff, mypy, generated configs, sanity checks
pytest tests/tools/test_function_tool_on_cpu.py -q  # run a focused CPU test
python examples/llm_fusion/download_data.py --output-dir /path/to/data
bash examples/llm_fusion/train/train_expert_math.sh
MODEL_PATH=/path/to/model bash examples/llm_fusion/eval/eval.sh
```

Set `MODEL_PATH`, `TRAIN_DATA_ROOT`, `EVAL_DATA_ROOT`, and `OUTPUT_ROOT` in the environment or `examples/llm_fusion/paths.sh`. GPU, NPU, Ray, and end-to-end tests require the corresponding runtime and should not be treated as ordinary CPU checks.

## Coding Style & Naming Conventions

Use four-space indentation and type annotations for new Python interfaces. Ruff formats and lints with a 120-character line target and sorts `verl` as first-party imports. Use `snake_case` for functions and files, `PascalCase` for classes, and `UPPER_CASE` for constants. Name tests `test_*.py`; use the established `_on_cpu.py` suffix when a test is explicitly CPU-only. Run pre-commit before submitting.

## Testing Guidelines

Pytest, `pytest-asyncio`, and rerun support are configured through the test requirements. Add focused tests beside the affected component and cover failure paths as well as normal behavior. Prefer the smallest relevant test file locally; clearly report any hardware-dependent suite not run. No repository-wide coverage percentage is mandated.

## Commit & Pull Request Guidelines

Recent commits use short, descriptive subjects such as `Release the Merge scripts`. Keep commits focused and imperative. PR titles are validated as `[module] type: description`, for example `[trainer] fix: validate teacher paths`; accepted types include `feat`, `fix`, `refactor`, `chore`, and `test`. Fill in the PR description, link relevant issues, list verification commands, and include hardware, configuration, or result changes for training/evaluation work.

## Security & Generated Artifacts

Keep API keys, host-specific addresses, datasets, checkpoints, rollout dumps, and logs out of commits. Code-domain rewards and LiveCodeBench evaluation require a reachable SandboxFusion service via `SANDBOX_FUSION_URL`.
