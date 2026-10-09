# LLM-Fusion Merge Research

This directory studies how to merge five Qwen3-4B RLVR experts: Math,
Science, Code, instruction following (IF), and Agent.

For a short Chinese overview, read [SUMMARY.zh-CN.md](SUMMARY.zh-CN.md).

## Current conclusion

- No candidate has yet shown a confirmed improvement over the strongest
  baseline across all five domains.
- Task Arithmetic (TA) remains the anchor. On the saved reasoning screen,
  `lambda=1` scores 64.50 AVG5.
- RAM+ and ResMerge recover much of their loss after update-norm matching, but
  neither beats TA in the current screen.
- The tested Soft-Overlap merge is worse than TA and is no longer the primary
  direction.
- The apparent IF-Agent interaction and functional-geometry advantage are not
  statistically stable.
- Five-expert coefficient calibration is now a diagnostic/control, not the
  intended academic contribution.
- The active hypothesis is Verifier Influence Cone Projection (VICP): estimate
  signed cross-domain verifier influence of parameter components, then project
  into an uncertainty-adjusted robust improvement cone.
- VICP must predict prespecified held-out mixed perturbations and outperform a
  matched five-scalar calibration control. Otherwise it is rejected as a new
  merge method, even if one coefficient setting happens to score better.

The public benchmark runs and the existing calibration split are development
evidence, not untouched test data.

## Read in this order

1. [SUMMARY.zh-CN.md](SUMMARY.zh-CN.md): compact project summary.
2. [results/current_findings.md](results/current_findings.md): evidence and
   decisions.
3. [notes/research_plan.md](notes/research_plan.md): acceptance criteria and
   next experiments.
4. [state.md](state.md): completed work and pending tasks.
5. [results/20260917_reassessment/tables.md](results/20260917_reassessment/tables.md):
   recomputed numerical tables.

The files under `notes/`, `ideas/`, and `results/phase1_findings.md` preserve
historical reasoning. They are useful for provenance but are not the current
decision source.

## Directory map

- `experiments/`: analysis, checkpoint-building, evaluation, and summary code.
- `results/`: compact generated outputs and interpretation.
- `baselines/`: vendored RAM, ResMerge, and SVC implementations.
- `papers/`: paper source archives and selected text.
- `notes/`: protocol, literature map, and earlier hypotheses.
- `ideas/`: candidate methods, including rejected or unvalidated proposals.

Large generated checkpoints, logs, PID files, and Python caches are excluded.
Raw evaluation JSONL files are retained because the reassessment script uses
them to reproduce scores, costs, source hashes, and bootstrap intervals.

## Reproduce the summary

```bash
python3 research/experiments/reassess_existing_results.py
```

Rebuilding models or launching evaluations requires the paths and runtime
described by each experiment README. Hardware launch scripts contain historical
cluster addresses and must be reviewed before reuse.
