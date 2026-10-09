# Strong RL-Expert Merging Baselines

This experiment runs official implementations on the five local Qwen3-4B RLVR experts.

## RAM+

- Source: `research/baselines/RAM`, official ACL 2026 repository.
- Mode: `arm-r-v2`.
- Threshold: `1e-5`.
- Rescale factor: `1.2`, the official README setting for RAM+.
- Compatibility-only local patch: the official loader attempted to save the
  tokenizer back into each input checkpoint. This write was disabled because
  the shared base checkpoint is read-only; the merge algorithm is unchanged.

## ResMerge

- Source: `research/baselines/ResMerge`, official repository.
- Rank-1 spectral head.
- SRC-A beta range `0.7–1.0`.
- Head ratio budget `0.2`.

Both use the same fp32 base and experts as Task Arithmetic and save a bf16 inference checkpoint. Evaluation follows the repository's eight-benchmark protocol; initial screening may use `n=4`, while final claims require fresh higher-sample confirmation.

## Reasoning-screen results (`n=4`)

| Method | Natural norm | Norm matched to TA |
|---|---:|---:|
| RAM+ | 58.61 | 61.98 |
| ResMerge | 56.05 | 62.16 |
| Soft-Overlap | — | 60.05 |
| Task Arithmetic, `lambda=1` | 64.50 | 64.50 |

The figures are the unweighted mean across AIME25, AIME26, GPQA, and two
LiveCodeBench versions. Norm matching recovers much of RAM+/ResMerge's loss,
but both point estimates remain below TA. Their natural deployed update norms are `0.714` and
`0.271`, versus `2.228` for TA; the corresponding norm-matching multipliers are
`3.121` and `8.234`. These large multipliers make clear that both direction and
radius matter for RLVR merging.

Paired prompt-bootstrap gaps versus TA are -2.52 pp [-4.71, -0.42] for scaled
RAM+ and -2.34 pp [-4.77, +0.05] for scaled ResMerge. ResMerge's residual gap
is unresolved, and both generate fewer tokens with fewer cap hits than TA.
Retain them as competitors under calibrated scale and comparable generation
cost. These n=4 reasoning results do not measure IF/Agent retention.

Scaling operates on the previously exported bf16 checkpoint. Future comparisons
must measure norms again after the final export; the requested scaling factor
alone does not certify an exact deployed norm match. See
[current findings](../../results/current_findings.md) for the full reassessment.
