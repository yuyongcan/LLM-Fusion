# Experiments

This directory contains reproducible analysis, checkpoint-building, evaluation,
and summarization code. Generated checkpoints and logs are intentionally not
kept here; rebuild them with the experiment-specific scripts.

## Active path

- `calibration/next_design.json`: prepared five-coefficient design.
- `calibration/run_one_eval.sh`: evaluate one checkpoint on all five domains.
- `merge_group_coefficients.py`: optional module-group coefficient correction.
- `reassess_existing_results.py`: reproduce the compact evidence tables from
  retained raw JSONL outputs.

## Diagnostic or historical work

- `analyze_*geometry.py` and `analyze_task_vector_support.py`: geometry
  diagnostics; useful for hypotheses, not current coefficient selection.
- `soft_overlap_merge.py`: rejected tested configuration, retained for
  provenance.
- Dated subdirectories: exact launch/build definitions for completed screens.

`reassess_existing_results.py` runs on CPU, records source hashes, clusters
duplicate prompts, and writes to `results/20260917_reassessment/`.

Hardware launch scripts include historical absolute paths, node addresses, and
runtime assumptions. Review them before reuse.
