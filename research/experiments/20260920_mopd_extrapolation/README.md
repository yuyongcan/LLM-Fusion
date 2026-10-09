# MOPD Extrapolation and Projection (2026-09-20)

## Questions

**H1 — extrapolation.** Distillation lands a shorter task vector than merging
(‖τ_MOPD‖ = 1.6356 vs ‖τ_merge‖ = 2.3058) while scoring about the same (65.45 vs
66.00 AVG8). Is there unrealized gain further along that direction, i.e. does
θ_base + α·τ_MOPD improve for α > 1?

Prior: no. The merge's own coefficient sweep already stops paying past c = 1.0
(pass@16 80.28 → 80.11, mean@16 66.00 → 65.08 from c=1.0 to c=1.2), and τ_MOPD
is a single blended direction that cannot be rescaled per domain the way
τ_merge can.

**H2 — redundancy.** Is most of τ_merge functionally inert? Split it against
τ_MOPD:

    c        = <τ_merge, τ_MOPD> / ‖τ_MOPD‖²  = 0.6449
    parallel = c · τ_MOPD                       ‖·‖ = 1.0560
    perp     = τ_merge − c · τ_MOPD             ‖·‖ = 2.0497

If the parallel part retains most of the merge's reasoning ability, then
distillation found a shorter route to the same function and the rest of τ_merge
is redundant — which would also explain why MOPD_merge1.0 beats the three-expert
merge on lcb_v5 (67.44 vs 67.10) and gpqa (64.49 vs 63.60).

The parallel component is necessarily shorter, so every comparison needs the
`*_normmatched` variants, which rescale each component back to ‖τ_merge‖. Without
that control a drop cannot be attributed to direction rather than update size.
Same discipline as `../20260916_norm_matched_subsets`.

## Precision: read this before reusing any of these scripts

A task vector is small relative to the weights (‖τ‖/‖θ_base‖ ≈ 0.1%), so a large
share of its entries fall below one bfloat16 rounding unit. Three separate places
destroyed task-vector arithmetic before the geometry gate caught them:

1. **`verl/model_merger` exported in bf16.** It cast the fp32 FSDP master weights
   with `.bfloat16()` while reassembling shards. Re-exporting in fp32 moves
   ‖τ_MOPD‖ from 1.4259 to 1.6356, and the two differ in direction by
   cos = 0.679. In one representative tensor, 70.7% of τ's entries were exactly
   zero in the bf16 export and none were in fp32. Fixed: `--save_dtype`, default
   float32.

2. **`from_config` silently re-rounds.** `AutoModelForCausalLM.from_config`
   honours `config.torch_dtype`, which is bfloat16 for the base model, so
   `load_state_dict` rounded the fp32 state into bf16 parameters and a later
   `.to(float32)` only widened the already-rounded values — producing a 15 GB
   "fp32" checkpoint holding bf16 data. Detect with
   `(t.to(torch.bfloat16).to(t.dtype) == t).all()`. Both scripts here now pin
   `config.torch_dtype` before constructing the model.

3. **Double rounding.** Building θ_base + α·τ from an already-bf16 export and
   storing it in bf16 collapses α = 1.15 to an effective 1.02. Build from fp32
   endpoints and store fp32.

**Inference precision is a separate question.** vLLM casts to bf16 regardless of
how the checkpoint is stored, and the cast is the same operation on the same
values, so re-evaluating an fp32 export reproduces the bf16 score exactly. fp32
storage matters for *downstream arithmetic and analysis*, not for eval fidelity.

What inference precision does change is the *effective* coefficient — the share
of ‖τ‖ that survives the cast, measured over the full model:

| nominal c | merge5 effective | MOPD effective |
|---|---|---|
| 0.4 | 0.2858 (71.5%) | 0.2417 (60.4%) |
| 0.6 | 0.4979 (83.0%) | 0.4347 (72.5%) |
| 0.8 | 0.7305 (91.3%) | 0.6457 (80.7%) |
| 1.0 | 0.9663 (96.6%) | 0.8718 (87.2%) |
| 1.2 | 1.1966 (99.7%) | 1.1075 (92.3%) |

Two consequences: the low end of the published coefficient sweep is handicapped
(c=0.4 loses 28.5% of its intended update), and a small task vector loses more
than a large one, so bf16 inference penalizes distillation more than merging.
`run_precision_control.sh` tests both.

Scaling τ up does recover fidelity, but far too slowly to be a fix —
cos(rounded, τ) goes 0.679 → 0.715 at α=1.15 and only reaches 0.909 at α=3,
where the normalized error is still 43.8%. Amplification is first-order,
fidelity recovery second-order; they cannot be separated. fp32 inference is the
clean control.

## Geometry (all endpoints fp32)

```
||theta_base|| = 1583.4033        (398 shared float tensors)

model      ||tau||     cos to MOPD    cos to merge5
MOPD        1.6356        1.0000         0.4580
merge5      2.3058        0.4580         1.0000
a1.15       1.8809        1.0000         0.4580
a1.30       2.1262        1.0000         0.4580
par         1.0560        1.0000         0.4580
par_nm      2.3058        1.0000         0.4580
perp        2.0497        0.0000         0.8890
perp_nm     2.3058        0.0000         0.8890
```

Every construction assertion holds exactly: ‖τ_a1.15‖ = 1.15 × 1.6356, the
parallel part is collinear with τ_MOPD, the perpendicular part is orthogonal to
it, and both norm-matched variants sit at ‖τ_merge‖.

Note cos(MOPD, merge5) = **0.4580** with both endpoints in fp32, against 0.3555
with only MOPD in fp32 and 0.3231 with both in bf16. Report the fp32 number; the
others understate the alignment. For scale, two random vectors in this dimension
have cos ≈ 1.6e-5, and the five experts are mutually orthogonal to within
[-0.0006, 0.0222] — so 0.458 is strong alignment, not near-orthogonality.

## Running

```bash
# H1 + H2 screen: 8 models, three columns, n=4, bf16 inference (the reported config)
bash research/experiments/20260920_mopd_extrapolation/run_h1h2_sweep.sh

# precision control: 3 models x {bfloat16, float32}, same columns
NGPUS_PER_NODE=16 bash research/experiments/20260920_mopd_extrapolation/run_precision_control.sh
```

Both need the code sandbox up (`bash setup_sandbox.sh`) or lcb_v5 is silently
halved, and both skip work that is already complete.

Columns are aime25 (most sensitive to sharpening — the merge loses 5.4 mean@16
points there between c=1.0 and c=1.2), ifbench (the only axis where merging
widened pass@16, +19.0), and lcb_v5 (largest code set, so least noisy).

## Decision rules, fixed in advance

- **H1**: α=1.15 must beat α=1.0 by more than +1.0 averaged over the three
  columns to justify a full n=16 run. Compare against the deployed bf16 MOPD;
  at bf16 inference α=1.15 delivers 1.20× the baseline's effective update.
- **H2**: if `proj_parallel_normmatched` stays within 2 points of merge5 on
  lcb_v5, the redundancy reading holds and earns a full n=16 run. If
  `proj_perp*` does better, H2 is falsified — record it either way.

## Files

| file | purpose |
|---|---|
| `run_h1h2_sweep.sh` | the 8-model screen |
| `run_precision_control.sh` | bf16 vs fp32 inference |
| `run_one_eval.sh` | single model, three columns, honours `ROLLOUT_DTYPE` |
| `../project_task_vector.py` | parallel / perpendicular decomposition |
| `../rescale_merged_update.py` | θ_base + α·(θ − θ_base) |
| `../../task_vector_geom.py` | norms and pairwise cosines — the gate |
