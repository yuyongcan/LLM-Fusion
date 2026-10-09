# Merge Calibration Split

`manifest.json` is a deterministic, index-only sample of 128 training rows per
domain, used for pilot merge calibration. It was sampled after expert training
and may include examples seen by the experts. Sampling from training files
does not establish content disjointness from public benchmarks; no such
deduplication audit is recorded. Source data are unchanged.

The uniform lambda=0.8/1.0/1.2 runs are complete at n=2. Their five-domain means
are 58.68/58.91/57.71; see
[recomputed tables](../../results/20260917_reassessment/tables.md) for domain
scores, costs, and uncertainty. Agent has 125 unique inputs across 128 source
rows; identical inputs must stay in one cluster for uncertainty estimation.

The default split is generated with:

```bash
python research/experiments/sample_calibration_split.py \
  --train-root /taijifs_sz34/share_304169095/hunyuan/yongcanyu/data/LLM-Fusion/train \
  --output research/experiments/calibration/manifest.json \
  --materialize-dir research/experiments/calibration/data \
  --per-domain 128
```

Any later calibration run must record the manifest hash, candidate coefficient
grid, decoding seed, verifier version, mean response length, and truncation
rate. Existing public benchmarks have already influenced selection and are
development data. Newly reserved validation and confirmation prompts are
required by the [current research plan](../../notes/research_plan.md).

Use `run_one_eval.sh` to score a checkpoint on the five materialized domain
files. The default is `n=2` for inexpensive screening; finalists should be
repeated with larger `n` or independent decoding seeds.

## Next design (prepared, not run)

`next_design.json` specifies the TA anchor, ten one-expert perturbations, and
four mixed settings for prediction checks. All five domains must be scored.
The three completed uniform settings identify only a one-dimensional response;
they cannot fit five separate expert effects or a full quadratic. The axial
design identifies main effects and diagonal curvature; hold the mixed settings
out of fitting when checking predictions. It still cannot justify a full
21-parameter quadratic without additional measurements.
