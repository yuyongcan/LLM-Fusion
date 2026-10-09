# Research State

Updated: 2026-09-17

## Goal and status

Develop a reproducibly better and academically novel static five-expert model
merge. See
[the research plan](notes/research_plan.md) for prospective acceptance criteria
and [current findings](results/current_findings.md) for the evidence. No method
has yet demonstrated a confirmed improvement over the strongest eligible
baseline across all five domains.

The active hypothesis is VICP. Per-expert coefficient calibration is retained
matrix; it is not sufficient for a novelty claim.
only as a control and as the cheapest estimator of a first verifier-influence

## Resource guard (historical record, not a live status)

- Cluster: 4 nodes × 8 H20 GPUs.
- Last recorded holder node: `28.45.1.61` when it was not running an evaluation.
- Command: `gpu_hold.py --gpus 0,1,2,3,4,5,6,7 --util 100 --mem 2048`.
- Parent PID file: `/tmp/llm_fusion_gpu_hold.pid` on that node.
- Log: `/tmp/llm_fusion_gpu_hold.log` on that node.
- Historical stop command; recheck process identity before using it:
  `ssh 28.45.1.61 'kill "$(cat /tmp/llm_fusion_gpu_hold.pid)"'`

The previous record lists PID `402651` and 100% utilization on all eight GPUs.
The current documentation/CPU-analysis revision has not queried the cluster or
verified that PID. Check live resources before scheduling or stopping jobs.

## Established baseline

- The reproduced 4B base, diagonal per-domain experts, and five-expert Task Arithmetic at `lambda=0.6` match the paper within about 0.2 AVG8 points.
- The observed best uniform coefficient on the current evaluation set is `lambda=1.0` (66.00 AVG8), but this is a development-set selection and increases generation length.
- Raising only the IF coefficient from 1.0 to 1.2 improves IFBench by 2.69 points; the overall +0.33 AVG8 change is not yet statistically secure.

## Completed evidence

- Recomputed 15 saved model/run entries in three suites from raw JSONL, with
  source hashes and paired prompt-bootstrap intervals:
  `results/20260917_reassessment/`.
- RAM+/ResMerge norm-matched controls are complete: 61.98/62.16 reasoning AVG5
  at n=4 versus TA 64.50. ResMerge's -2.34 pp interval includes zero. Retain
  both scaled methods in the comparison set.
- Soft-Overlap reaches 60.05; its current configuration is not the primary
  method. Another TA run gives 62.67, emphasizing the need for repeated anchors.
- Functional geometry is complete for 8 prompts and 64 tokens/domain. Its
  transfer correlation advantage reverses in several sensitivity checks.
- Uniform calibration at lambda=0.8/1.0/1.2 is complete: five-domain means
  58.68/58.91/57.71. The lower coefficient is cheaper with an uncertain mean
  score difference; the higher one improves IF but reduces Agent performance.
- The manifest is frozen (SHA-256 prefix `b5f916f0`), with 128 source rows per
  domain. Agent has 125 unique inputs. The split may include expert-training
  examples and is pilot data, not a clean holdout.
- n=16 factorial reanalysis: interaction +1.06, CI [-1.26, +3.54]; All5-MCS
  +1.64, CI [-0.62, +4.02]. Neither is a confirmed positive effect.
- IF embedding correction is screened on reasoning only (63.09 AVG5); its
  IF/Agent outcome and the attention/MLP comparisons remain unmeasured here.

## Next implementation milestones

1. Reserve new calibration validation and final confirmation prompts with
   content/provenance checks. Existing public tests are development data.
2. Execute the prepared 11 axial and 4 mixed coefficient settings in
   `experiments/calibration/next_design.json`, recording per-domain reward,
   generation cost, seeds, and post-export norms. These settings are not run.
3. Implement a regularized, constrained scalar selector and validate its
   actual merged checkpoint against calibrated uniform TA. The three existing
   uniform points cannot identify independent expert effects.
4. Add at most two module residuals only if they beat the scalar control at
   equal calibration cost; evaluate IF and Agent directly.
5. Freeze the best candidate and baselines, then perform independent
   confirmation and repeated decoding. No independent confirmation is complete.
