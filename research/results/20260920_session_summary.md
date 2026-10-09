# Session Findings — 2026-09-20

Analysis of the merge / distillation / mixed-RL comparison on Qwen3-4B-Instruct-2507.
Covers five threads: arithmetic errors in the reported table, what merging
actually buys (pass@16), why mixed-domain RL underperforms, the weight-space
geometry, and a numerical-precision problem that affects how coefficient sweeps
should be read.

Status: findings 1–5 are established. The extrapolation (H1) and projection (H2)
screens were still running when this was written — see [Open experiments](#open-experiments).

---

## 1. Six AVG8 values in the reported table are wrong

Recomputing the mean of the eight benchmark columns:

| row | reported | recomputed | delta |
|---|---:|---:|---:|
| expert_if | 54.75 | **59.14** | +4.39 |
| expert_agent | 56.82 | **58.85** | +2.03 |
| mcs3_c1.2 | 61.09 | 61.29 | +0.20 |
| merge5_c0.8 | 65.35 | 65.32 | −0.03 |
| **merge5_c1.0** | **67.51** | **66.00** | **−1.51** |
| merge5_c1.2 | 66.75 | 65.08 | −1.67 |

The other twelve rows check out. Two of these change conclusions:

- `expert_if` is not the worst expert. At 59.14 it is mid-pack: its bfcl collapse
  (−10.8) is more than offset by ifbench (+18.2) and ifeval (+5.9).
- `merge5_c1.0` is 66.00, not 67.51. That cuts its margin over `MOPD_merge1.0`
  from 2.06 to **0.55** — within noise on eight benchmarks, so the two should be
  described as comparable rather than merge being ahead.

Everything below uses recomputed values. **Check whichever script or spreadsheet
produced the published averages before reporting them.**

---

## 2. Merging sharpens the output distribution; it does not add capability

The `n16` evals record `best@16` (= pass@16) alongside `mean@16`, which separates
"solves more problems" from "solves the same problems more reliably".

| model | aime25 mean@16 | aime25 pass@16 | lcb_v5 mean@16 | lcb_v5 pass@16 | gpqa mean@16 | gpqa pass@16 |
|---|---:|---:|---:|---:|---:|---:|
| base | 47.08 | 71.46 | 58.78 | 79.28 | 56.69 | 83.89 |
| expert_math | 51.25 | **77.65** | 60.07 | 80.06 | 61.55 | 86.33 |
| expert_code | 51.46 | 72.69 | 63.73 | 82.58 | 59.00 | 85.37 |
| mcs3_c1.0 | **55.63** | 74.97 | **67.10** | 83.87 | **63.60** | 88.02 |
| MOPD_merge1.0 | 53.75 | 76.50 | **67.44** | 83.98 | **64.49** | 85.95 |

`mcs3_c1.0` has the best aime25 mean@16 in this group but a **lower** pass@16
than `expert_math` (74.97 vs 77.65). Aggregated over the family:

| family | mean@16 gain | pass@16 gain |
|---|---:|---:|
| mcs3, base → c=1.0 | +4.73 | +1.81 |
| merge5, c=0.4 → c=1.0 | +5.37 | +3.62 |

About **60% of the mean@16 gain is pure sharpening** — no expansion of the solved
set. And merge5's pass@16 gain is almost entirely one column: ifbench
+19.0 (46.91 → 65.87 across the sweep), while gpqa pass@16 actually *declines*
(87.29 → 86.16) and lcb_v5 gains only 3.1.

**Implication.** Instruction-following is the one axis where merging genuinely
widens the capability frontier. On reasoning axes the coefficient is a sharpness
knob over a fixed frontier. This is good news for distillation: sharpening is
exactly what a mode-seeking objective can reproduce, and `MOPD_merge1.0` already
exceeds its merge counterparts on lcb_v5 (67.44) and gpqa (64.49) mean@16.

### The c=1.2 regression is not capability damage

| | c=1.0 | c=1.2 | delta |
|---|---:|---:|---:|
| aime25 mean@16 | 57.08 | 51.67 | **−5.41** |
| aime25 pass@16 | 77.42 | 75.13 | −2.29 |
| gpqa mean@16 | 64.93 | 63.76 | −1.17 |
| gpqa pass@16 | 87.07 | 86.16 | −0.91 |

pass@16 barely moves while mean@16 drops 5.4 — the ability is intact, pass@1 is
just less reliable. c=1.2 **de-sharpens**; it does not destroy the math direction.
Meanwhile ifbench pass@16 is still climbing at c=1.2 (65.87, the sweep maximum).
So the trade at high c is "IF coverage up, reasoning output stability down",
which subsequent sharpening could in principle recover.

### An unreported checkpoint leads both metrics

`merge5_percoef_if1.2` (IF direction at 1.2, others at 1.0) tops the table on
**both** mean@16 (66.33) and pass@16 (80.67), and posts the highest gpqa pass@16
anywhere (89.36). It does what the two tables above predict: take the IF
coefficient gain without paying the reasoning de-sharpening. It was hand-tuned
and is not proposed as a result, but it corroborates the mechanism.

---

## 3. Mixed-domain RL: the IF domain never trains

`train_mix_8x8_20260919_162504.log`, per-domain validation (`mean@4`):

| step | aime25 | aime26 | gpqa | lcb_v5 | bfcl | **ifbench** | **ifeval** |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 43.33 | 48.33 | 56.69 | 58.52 | 63.00 | **31.08** | **84.20** |
| 100 | 50.00 | 53.33 | 60.48 | 61.62 | 69.50 | **31.67** | **84.20** |
| 225 | 51.67 | 50.83 | 61.11 | 61.75 | 70.37 | **33.67** | **85.07** |

Five of six domains improve normally (bfcl +7.4, gpqa +4.4). **ifbench moves 2.6
points in 225 steps and ifeval 0.9** — both still at base level, while a dedicated
`expert_if` run reaches ifbench 48.9 (+18).

This is not entropy collapse (`actor/entropy` holds 0.30–0.35 with no trend,
`grad_norm` stable) and not a compute shortfall. It is **gradient starvation from
degenerate GRPO groups.**

Per-domain group statistics from the rollout dumps (`research/group_stats.py`,
domain labelled from `gts.rm_type`), mix-RL steps 90–99:

| domain | groups | mean score | zero-variance % | mean group std | **share of gradient-carrying groups** |
|---|---:|---:|---:|---:|---:|
| Code | 327 | 0.353 | 59.0% | 0.105 | 25.6% |
| Science | 290 | 0.557 | 52.4% | 0.179 | 26.3% |
| Math | 283 | 0.593 | 38.2% | 0.241 | 33.4% |
| **IF** | 244 | 0.404 | **68.4%** | **0.117** | **14.7%** |

IF is 18.9% of the prompt mixture (Math 25.1 / Science 22.4 / Code 21.9 / IF 18.9
/ Agent 11.7, counted from the Mix parquet) but contributes only **14.7%** of
non-degenerate groups, and its groups have the lowest within-group spread — and
under GRPO the advantage scales with that spread, so its effective share of the
update is smaller still.

**The controlled comparison.** Training IF alone gives an *even higher*
degeneracy rate — 70.9% — yet it learns fine, because it receives 100% of the
gradient:

| run | IF reward trajectory |
|---|---|
| expert_if (IF only) | 0.378 → 0.459 → 0.498 → **0.559** |
| mix-RL (IF at 14.7%) | 0.365 → 0.370 → **0.404** |

So the mechanism is not "IF is a sparse-reward domain" in the abstract — the
same sparsity is trainable when the domain owns the batch.

**Correction — this is slowdown, not a threshold.** An earlier version of this
section claimed IF "falls below the threshold where it makes progress at all."
The slopes do not support that: IF reward rises at +0.00065/step in the mix
against +0.00226/step for expert_if, i.e. **0.29×** — while the effective
gradient share is 0.147, so the observed slowdown is ~1.95× *better* than a
linear-in-share prediction. The defensible claim is that **IF is diluted to
~0.29× learning rate, which inside a 250-step budget looks flat on benchmarks
while still moving in training reward.** Whether a genuine threshold exists is
open, and `../ideas/parallel_then_merge_vs_joint_rl.md` E2 is designed to answer it.

**Also unresolved: dilution vs interference.** mix-RL and expert_if use the same
`train_batch_size=128` / `rollout.n=16`, so IF gets ~1/5 the absolute effective
groups per step in the mix. "Too few groups per step" (fixable by a larger
batch) and "interference from sharing parameters" (not fixable that way) predict
the same observation. Separating them is the blocking experiment (E1).

**Data hygiene.** This run **diverged after step 300** — grad_norm 0.039 → 0.507,
response length to 15093 tokens (92% truncated), entropy to 0.009, and
aime25/aime26/gpqa/ifbench all to zero. `global_step_275` is the last healthy
checkpoint (aime25 46.67, ifbench 32.75). The IF finding is independent of the
divergence (visible from step 0), but any rerun needs gradient-norm clipping.

### Per-domain dynamics revise this into a two-factor account

Tracking group *composition* over training (`research/variance_trajectory.py`,
90-step windows, steps 1–270) shows IF is not the only sick domain, nor the
worst. Rates per step, against expert_if's dWrong/dt = −0.159:

| domain | prompt share | eff groups/step | dWrong/dt | vs expert | group std | verdict |
|---|---:|---:|---:|---:|---|---|
| Science | 19.5% | 15.8 | −0.0067 | 0.04× | 0.216 → **0.182** | healthy |
| Agent | 14.8% | — | — | — | 0.500 → **0.426** | healthy (score +0.280) |
| Math | 23.4% | 19.5 | +0.0078 | −0.05× | 0.243 → 0.233 | **saturated** |
| IF | 18.0% | 7.7 | −0.0267 | 0.17× | 0.120 → 0.124 | **starved** |
| **Code** | **24.2%** | 10.4 | **−0.0011** | **0.01×** | 0.085 → 0.083 | **frozen** |

Three corrections to the account above:

1. **Healthy learning is variance *decay*, not growth.** expert_if's group std
   falls 0.120 → 0.066 while zero-variance groups *rise* to 81%, because
   all-wrong groups convert to all-right (wrong 42.7% → 18.0%, right 25.3% →
   62.9%). The earlier "self-locking" narrative had the sign backwards.
2. **Code is worse than IF and is not starved.** Its all-wrong share is immobile
   (57.6% → 57.4%) despite the *largest* prompt share and more effective groups
   per step than IF. Gradient supply is adequate and it still does not learn:
   the task exceeds current capability. **More batch cannot fix this**, which
   refutes the single-factor "effective gradient share governs learning" claim.
3. **Two domains are healthy**, so joint RL is not broadly harmful — the failures
   are domain-specific. Agent, whose continuous reward yields 0.0% degenerate
   groups and std 2–4× the others, learns fastest, supporting within-group
   spread as the driver.

**So the claim becomes a 2×2**, with all informative cells populated by one run:

| | gradient adequate | gradient starved |
|---|---|---|
| task within capability | healthy (Science, Agent) | **starvation** (IF) — fixable by batch/mixture |
| task beyond capability | **difficulty freeze** (Code) — batch useless | both |

**Metric caveat:** `grp_std` is low both before a domain starts (IF, Code: high
all-wrong) and after it finishes (expert_if late: high all-right). Weighting
effective gradient share by mean std — as the analysis above does — is therefore
biased. Report **dWrong/dt** and **dRight/dt** instead.

**Fixes, now differentiated.** IF: balance by gradient-carrying group count
(~2.5× upsampling). Code: difficulty filtering or a curriculum — batching is
wasted on it, and at 24% of prompts it may be crowding out domains that could
learn. Extrapolation helps neither: scaling a task vector that never learned a
domain yields nothing.

Interim E1 evidence (`train_mix_8x8_20260921_164602`, bs=512, lr=2e-6, 78 steps):
IF effective groups rose to 30.6/step (**0.82×** expert parity) but recovery is
only **0.51×** by reward slope and **0.66×** by dWrong/dt — dilution is dominant
with roughly a third unexplained, and IF's zero-variance rate is unchanged at
~70%.

---

## 4. Weight-space geometry

All endpoints fp32 (see §5 for why that qualifier matters). 398 shared float
tensors, ‖θ_base‖ = 1583.4033.

| model | ‖τ‖ | ‖τ‖/‖θ_base‖ |
|---|---:|---:|
| expert_math | 0.9881 | 0.0624% |
| expert_code | 0.8501 | 0.0537% |
| expert_science | 1.0232 | 0.0646% |
| expert_if | **1.3220** | 0.0835% |
| expert_agent | 0.8694 | 0.0549% |
| merge5_c1.0 | 2.3058 | 0.1456% |
| MOPD_merge1.0 | 1.6356 | 0.1033% |

**The five experts are mutually orthogonal.** Pairwise cosines lie in
[−0.0006, 0.0222]; the largest is math–science, the rest are at the 1e-3 level.

**Task arithmetic is exactly a sum.** Predicting ‖Στᵢ‖ from the norms and the full
cosine matrix gives **2.3057** against a measured **2.3058**. (An earlier
statement of this check in-session was wrong on two counts — it omitted cross
terms and compared against a bf16 merge; corrected here.) Each expert's cosine to
the merge equals ‖τᵢ‖/‖τ_merge‖, exactly as an orthogonal sum requires — for
expert_if, the most orthogonal of the five, 0.5732 measured against 0.5734
predicted.

**Student–teacher alignment is strong, not near-orthogonal.**

| pair | cosine |
|---|---:|
| MOPD_merge1.0 ↔ merge5 (its own teacher) | **0.4580** |
| MOPD_vanilla ↔ merge5 (not its teacher) | 0.2299 (bf16 measurement) |
| two random vectors at this dimension | ~1.6e-5 |
| expert ↔ expert | ≤ 0.0222 |

Against a 1.6e-5 random baseline, 0.458 is alignment of a completely different
order than the genuinely-orthogonal expert pairs. The teacher is identifiable
from weights alone: the student trained on merge5 aligns to it substantially
better than the student that never saw it.

Interpretation: distillation constrains only the output distribution, and the
weight space has vast functionally-equivalent sets (permutation and scaling
symmetries), so a student descending from θ_base settles at whichever equivalent
point is nearest — not the teacher's. That 0.458 survives anyway says the
objective pins down weight geometry more than one would expect. And the student
reaches comparable quality with a **29% shorter** task vector (1.6356 vs 2.3058),
i.e. a shorter route to the same function.

---

## 5. Numerical precision: task vectors live below bfloat16 resolution

‖τ‖/‖θ_base‖ ≈ 0.1%, so a large share of τ's entries fall below one bfloat16
rounding unit. In a representative tensor (`layers.10.mlp.down_proj`), **70.7% of
τ's entries were exactly zero** in a bf16 export and 0% in fp32. Three distinct
defects followed from this; all were caught by a geometry check run *before*
spending GPU time, which is the main process lesson here.

**(a) The export pipeline discarded precision at the source.**
`verl/model_merger/fsdp_model_merger.py` cast fp32 FSDP master weights with
`.bfloat16()` while reassembling shards. Re-exporting in fp32:

| | ‖τ_MOPD‖ | cos to fp32 τ |
|---|---:|---:|
| bf16 export (what was evaluated) | 1.4259 | 0.6791 |
| fp32 export | 1.6356 | 1.0 |

The deployed MOPD checkpoint's task vector is only **0.68-aligned** with the
vector actually trained, retaining 87% of its norm.

**(b) `from_config` silently re-rounds.** `AutoModelForCausalLM.from_config`
honours `config.torch_dtype` — bfloat16 for the base model — so `load_state_dict`
rounded fp32 state into bf16 parameters and a later `.to(float32)` merely widened
already-rounded values. Result: a 15 GB "fp32" checkpoint holding bf16 data.
Detect with `(t.to(torch.bfloat16).to(t.dtype) == t).all()`. **This affects any
export path that builds an empty model from a base config and then loads
weights**, regardless of a trailing `.to(fp32)`.

**(c) Double rounding.** Computing θ_base + α·τ from an already-bf16 export and
storing bf16 collapses α = 1.15 to an effective 1.02.

### What is and is not affected

**Evaluation scores are unaffected.** vLLM casts to bf16 regardless of storage
(`rollout.dtype: bfloat16`), and "round at export" and "round at load" are the
same operation on the same values. Re-evaluating an fp32 re-export reproduces the
bf16 score exactly — so **no published number needs re-running on precision
grounds**. fp32 matters for downstream *arithmetic* and *analysis*, not fidelity
of a single eval.

**Coefficient sweeps are affected.** What inference sees is the share of ‖τ‖
surviving the cast, measured over the full model:

| nominal c | merge5 effective | surviving | MOPD effective | surviving |
|---|---:|---:|---:|---:|
| 0.4 | 0.2858 | **71.5%** | 0.2417 | **60.4%** |
| 0.6 | 0.4979 | 83.0% | 0.4347 | 72.5% |
| 0.8 | 0.7305 | 91.3% | 0.6457 | 80.7% |
| 1.0 | 0.9663 | 96.6% | 0.8718 | 87.2% |
| 1.2 | 1.1966 | 99.7% | 1.1075 | 92.3% |

Cross-check: 0.966 × 2.3058 = 2.228, matching the measured bf16 merge norm
2.2281; 0.872 × 1.6356 = 1.426 against 1.4259 measured.

This licensed two predictions, **both of which were then tested and refuted**:

1. ~~The published c-axis is non-linear, so the rising limb is steeper than the
   true relationship.~~
2. ~~bf16 inference penalizes distillation more than merging, biasing the
   66.00 vs 65.45 comparison.~~

### The precision control refutes both (measured, n=16)

Same fp32 weights, three diagnostic columns, changing only `ROLLOUT_DTYPE`:

| model | bf16 τ survival | bf16 | fp32 | fp32 − bf16 |
|---|---:|---:|---:|---:|
| merge5_c0.4 | **71.5%** | 49.16 | 48.31 | **−0.85** |
| merge5_c1.0 | 96.6% | 57.95 | 57.54 | −0.41 |
| MOPD_merge1.0 | 87.2% | 56.65 | 56.58 | −0.07 |

No model improves. `merge5_c0.4` is decisive: it loses 28.5% of ‖τ‖ to rounding
and gains **nothing** from fp32 inference (its one notable column, aime25 −3.75,
is 30 problems and within noise). MOPD's change (−0.07) is *smaller* than
merge5_c1.0's (−0.41), so there is no distillation-specific handicap either.

**Consequences, corrected:**

- **The reported c-sweep needs no correction.** `merge5_c0.4`'s 60.63 is the
  genuine performance of that coefficient; the curve shape is real.
- **The 66.00 vs 65.45 comparison is fair.** No re-running is required on
  precision grounds — every published score stands.
- **L2 norm is a poor proxy for the functional strength of a task vector.**
  Rounding truncates the smallest-magnitude coordinates; they carry 28.5% of the
  L2 norm while contributing almost nothing to the function. This is a stronger
  form of the DARE observation (drop most of the delta, keep the behaviour) —
  here no rescaling is needed at all. **Anything that normalizes by ‖τ‖ — the
  norm-matched coefficients in `../experiments/20260916_norm_matched_subsets`
  included — is therefore equalizing a quantity that is only loosely related to
  functional magnitude.** Worth stating as a caveat wherever norm matching is
  used as a control.
- The fp32 export fix remains worth keeping for **task-vector arithmetic and
  geometry** (§4's cosines shift materially: 0.679 → 1.0 against the true τ, and
  student–teacher 0.323 → 0.458), but it is not an urgent correctness bug for
  evaluation.

### Extrapolation does not repair this

Scaling τ up does push more entries above the rounding threshold, but far too
slowly to serve as a fix:

| α | ‖r‖/‖τ‖ | cos(r, τ) | ‖r/α − τ‖/‖τ‖ |
|---|---:|---:|---:|
| 1.0 | 0.872 | 0.679 | 0.759 |
| 1.15 | 1.048 | 0.715 | 0.726 |
| 1.3 | 1.226 | 0.746 | 0.695 |
| 2.0 | 2.034 | 0.842 | 0.568 |
| 3.0 | 3.131 | **0.909** | 0.438 |

At α = 1.15 fidelity improves by only +0.036 cosine while the update is amplified
1.20×. Reaching cos > 0.9 needs α ≈ 3, where normalized error is still 43.8%.
Amplification is first-order and fidelity recovery second-order — they cannot be
separated. **fp32 inference is the clean control**, and it is cheap.

---

## 6. Why merging works: orthogonality buys linear-regime headroom

A natural reading of §4 is "the updates are small, so the experts stay
orthogonal, so they add." The data splits that into one false half and one true
half.

**False: small displacement does not cause orthogonality.** If it did, the
largest task vector would be the least orthogonal. The opposite holds:

| expert | ‖τ‖ | max cosine to another expert |
|---|---:|---:|
| expert_if | **1.3220** (largest) | **0.0003** (most orthogonal) |
| expert_science | 1.0232 | 0.0222 |
| expert_math | 0.9881 | 0.0222 |
| expert_agent | 0.8694 | 0.0012 |
| expert_code | 0.8501 | 0.0088 |

The ordering tracks task semantics, not magnitude: the largest cosine is
math–science, then math–code, then code–science, with agent and IF far below. In
4e9 dimensions orthogonality is nearly free anyway (random cosine ≈ 1.6e-5), so
no smallness assumption is needed to obtain it.

**True: small displacement is what makes the sum additive.** Within a
small-enough neighbourhood the model responds approximately linearly, so
f(θ + τ₁ + τ₂) ≈ f(θ) + J·τ₁ + J·τ₂. The sweep locates that boundary:

| c | ‖τ_merge‖/‖θ_base‖ | AVG8 |
|---|---:|---:|
| 1.0 | **0.146%** | 66.00 (peak) |
| 1.2 | **0.175%** | 65.08 |

**How the halves connect.** Orthogonality matters not because it avoids
"conflict" but because it keeps the *summed* displacement inside that regime:

- five perfectly aligned vectors: ‖Στᵢ‖ = Σ‖τᵢ‖ = 5.0528 → **0.319%**
- actually orthogonal: ‖Στᵢ‖ = **2.3058** → **0.146%**

0.319% sits well past the observed 0.175% degradation point. Orthogonality
shrinks the total displacement 2.19× and thereby lands a five-expert sum exactly
inside the additive region.

**Direct evidence — `../experiments/20260916_norm_matched_subsets`.** That
experiment fixes total update norm to the five-expert λ=1.0 value and varies only
how many directions share it:

| variant | uniform coefficient | reasoning avg |
|---|---:|---:|
| five_all | 1.0 | **62.67** |
| mcs + agent | 1.2204 | 62.28 |
| mcs + if | 1.0800 | 61.71 |
| mcs | 1.3755 | 60.55 |

At equal displacement, more directions is monotonically better (5 > 4 > 3,
spanning 2.1 points); packing the same norm into three directions is worst. So
the operative variable is **how many independent directions fit inside the linear
budget**, not smallness per se.

**Counterexample against "smaller is better".** MOPD's ‖τ‖ is 29% below merge5's
(1.6356 vs 2.3058) at comparable quality (65.45 vs 66.00). Neither "small wins"
nor "large wins" survives.

### The projection experiment confirms it (measured, n=16, full suite)

Splitting τ_merge against τ_MOPD (c = 0.6449) and evaluating each part:

| model | ‖τ‖ | mean@16 | pass@16 |
|---|---:|---:|---:|
| **merge5_c1.0** (full τ_merge) | 2.3058 | **66.00** | **80.28** |
| `proj_parallel` (MOPD direction only) | 1.0560 | 63.68 | 78.56 |
| `proj_parallel_normmatched` (same direction, scaled to ‖τ_merge‖) | 2.3058 | 60.04 | 75.48 |
| `proj_perp_normmatched` (orthogonal remainder, scaled to ‖τ_merge‖) | 2.3058 | 60.23 | 76.91 |

**H2 is falsified — τ_merge is not mostly redundant.** `proj_parallel` reaching
96.5% of the merge's score on 46% of the norm looks superficially supportive, but
the norm-matched control settles it: scaling that same direction up to ‖τ_merge‖
makes it **worse**, 63.68 → 60.04. The MOPD direction is not a functional
subspace of τ_merge; it works only near its original scale. (This also explains
a runtime anomaly — `parallel_normmatched` generated pathologically long outputs
on long-CoT benchmarks, up to 2076 s/batch against ~200 s normally, with aime25
pass@16 falling to 63.61, below the base model's 71.46. Amplifying a single
direction damages convergence.)

**And the two halves are equally good, which is the real result.** At equal norm,
`perp_nm` 60.23 ≈ `parallel_nm` 60.04, and both sit ~6 points below the full
merge. There is no "essential direction plus redundant remainder": remove either
half, restore the norm, and you lose the same amount. The halves are
complementary by axis — `perp_nm` posts the table's best gpqa pass@16 (87.29,
above merge5's own 87.07) while `parallel_nm` leads on ifbench (52.71 / 61.26 vs
37.08 / 50.64) — consistent with distillation having absorbed some IF ability but
not enough to substitute for τ_if.

So performance comes from **many independent directions summing**, not from one
dominant direction. Together with the norm-matched subset result (5 > 4 > 3
directions at fixed norm), that is two independent lines of evidence.

**Suggested follow-up.** Measure the linearity directly: compare
f(θ + c·τ_math + c·τ_code) against the sum of the individual effects while
sweeping c. If the departure from additivity turns up near 0.15%, the linear
regime becomes a measurement rather than an inference.

---

## Code changes

| file | change |
|---|---|
| `verl/model_merger/fsdp_model_merger.py` | `.bfloat16()` → `.to(save_dtype)`; `_validate_state_dict` follows config (its hardcoded bf16 dtype assertion would otherwise fail) |
| `verl/model_merger/base_model_merger.py` | new `save_dtype` field (**default float32**) + `torch_save_dtype`; save path honours it; writes `config.torch_dtype` (previously `None` on merge outputs) |
| `verl/model_merger/__main__.py` (CLI) | new `--save_dtype {float32,bfloat16,float16}` |
| `examples/llm_fusion/merge/run_merge*.sh` (3 files) | `SAVE_DTYPE` default → `float32`; corrected the comment claiming bf16 is "lossless for vLLM eval" (true for evaluating that checkpoint, false once it feeds further arithmetic) |
| `examples/llm_fusion/merge/convert_experts_to_hf.py` | its 120-line `FP32FSDPModelMerger` existed only to work around the bug above; reduced to a thin wrapper now that upstream is fixed |
| `examples/llm_fusion/eval/eval.sh` | new `ROLLOUT_DTYPE` (default `bfloat16`, no behaviour change) for precision controls |
| `research/experiments/{rescale_merged_update,project_task_vector}.py` | pin `config.torch_dtype` before `from_config` (defect (b)) |

New tooling: `research/parse_train_log.py` (verl console log → metric series),
`research/group_stats.py` (rollout dumps → per-domain GRPO group statistics),
`research/task_vector_geom.py` (norms and pairwise cosines — the gate that caught
two modeling errors), `research/experiments/project_task_vector.py` (parallel /
perpendicular decomposition with norm matching).

Note `convert_experts_to_hf.py`'s docstring records that **the experts were always
exported in fp32** — every merge in the paper used fp32 inputs. Only the MOPD side,
which went through the unpatched `verl.model_merger`, was affected.

---

## Experiment status

| experiment | status | outcome |
|---|---|---|
| H2 — projection / redundancy | **done** (n=16, full suite) | **falsified**; see §6 |
| Precision control | **done** (n=16, 3 columns, bf16 vs fp32) | **both predictions refuted**; see §5 |
| H1 — extrapolation | **done** (n=16, full suite) | **falsified on mean@16**, but see below |

### H1 — extrapolation: falsified, with an unanticipated result

θ_base + α·τ_MOPD, rebuilt from fp32 endpoints, geometry verified exactly
(‖τ_1.15‖ = 1.8809 = 1.15 × 1.6356):

| model | ‖τ‖ | mean@16 | pass@16 |
|---|---:|---:|---:|
| MOPD α=1.0 | 1.6356 | **65.45** | 79.87 |
| MOPD α=1.15 | 1.8809 | 65.23 | **80.42** |
| MOPD α=1.30 | 2.1262 | 63.07 | 77.94 |

mean@16 declines monotonically, so extrapolation does not add capability — as
predicted. **But α=1.15 posts the second-highest pass@16 in the entire project
(80.42), above its own teacher merge5_c1.0 (80.28)**, gaining ifbench +3.22,
gpqa +1.70 and aime25 +1.00 in pass@16 while mean@16 barely moves (−0.22).

This is the *same* sharpness-versus-coverage trade-off that the merge
coefficient traces (§2): α buys coverage and pays in pass@1 reliability, and
α=1.30 overshoots into losing both. Two mechanically unrelated knobs — weight
merging coefficient and post-distillation extrapolation — produce the same curve
shape, which is considerably stronger evidence for the sharpening account than
either alone. Extrapolation is best described as **moving the operating point
along the trade-off, not increasing total capability.**

### Superseded: the abandoned n=4 screen

Kept only to document why the design changed. Three diagnostic columns, n=4, bf16:

| model | aime25 | ifbench | lcb_v5 | mean |
|---|---:|---:|---:|---:|
| merge5_c1.0 | 54.17 | 48.25 | 68.11 | 56.84 |
| MOPD_merge1.0 | 55.83 | 47.25 | 67.93 | **57.00** |

**These n=4 numbers are not usable for a decision, and the screen was abandoned
because of them.** On the *same three columns* at n=16 the ordering is the
opposite — merge5 58.34 vs MOPD 56.25, merge ahead by 2.10 — so n=4 reverses the
sign. Per column: ifbench (+2.02 → +1.00) and lcb_v5 (+0.94 → +0.18) keep their
sign and only shrink; aime25 swings from +3.33 to −1.66 and drags the mean with
it. aime25 has 30 problems, so one problem is 3.33 points and a 5-point swing is
1.5 problems — well inside sampling noise at 4 samples each.

**Everything was relaunched at n=16 over the full eight benchmarks**, using the
project's own `run_eval_experts.sh` — the script that produced every number in
the reported table. Two reasons the n=4 screen was the wrong call: it cannot
resolve gaps of this size, and it yields numbers that cannot be placed beside an
n=16 table, so the work would have to be redone anyway.

Because fp32 storage does not change eval scores (§5), three models already have
n=16 results and are not re-run: merge5_c0.4 (60.63), merge5_c1.0 (66.00),
MOPD_merge1.0 (65.45). That leaves nine jobs — six new H1/H2 models plus three
fp32-inference arms — at roughly 83 min each, split across three nodes.

**Precision control.** Only the fp32-inference arm needs running; it is compared
against the published bf16 numbers above. Results land in `eval_n16_float32/`
(`run_eval_experts.sh` now appends the dtype only when it is not bfloat16, so
existing paths are untouched). `merge5_c0.4` is the decisive one: a clear rise
under fp32 confirms the rising limb of the c-sweep is partly an artifact.

Decision rules were fixed before running; see
`research/experiments/20260920_mopd_extrapolation/README.md`.

---

## Recommendations

1. **Re-derive the AVG8 column** and audit whatever produced the published values.
   `merge5_c1.0` = 66.00 makes merge and MOPD_merge1.0 comparable, not ranked.
2. **Report pass@16 alongside mean@16.** It is the evidence that merging sharpens
   rather than extends, and it is already computed in every `n16` log.
3. **No correction to the c-sweep is needed**, and no score needs re-running for
   precision. The control settled this: bf16 rounding discards up to 28.5% of
   ‖τ‖ with no measurable effect on quality. The effective-coefficient table in
   `results.md` §2.1 is worth keeping as a *caveat about ‖τ‖ as a metric*, not as
   a correction to the scores.
4. **Stop treating ‖τ‖ as functional magnitude.** The rounding result shows the
   smallest-magnitude coordinates carry a large share of the L2 norm and almost
   none of the function. Any norm-matched control — including
   `../experiments/20260916_norm_matched_subsets` — equalizes a loosely-related
   quantity, and that should be stated where those controls are used.
5. **Frame merging as "many orthogonal directions summing inside a linear
   regime."** Three independent lines support it: more directions win at fixed
   norm (5 > 4 > 3), either half of a split τ_merge loses equally (§6), and
   orthogonality is what keeps the five-expert sum at 0.146% rather than 0.319%
   of ‖θ_base‖. Explicitly *not* supported: "small updates cause orthogonality,"
   and "merging works by avoiding conflict."
6. **Fix mixed-RL by rebalancing on gradient-carrying groups**, not prompt counts.
   The current run's IF domain is untrained, so it is not yet a fair baseline for
   "joint RL vs merging" — but it is a strong basis for a paper on *why* joint RL
   underperforms (threshold effect, not slowdown).
7. **Quote 0.4580** for student–teacher cosine (both endpoints fp32). Earlier
   values of 0.323 and 0.3555 understate it; the right reference scale is the
   1.6e-5 random baseline and the ≤0.0222 expert–expert pairs.
8. **Keep the geometry gate.** Asserting norms and cosines against closed-form
   predictions before evaluating caught two modeling defects that would otherwise
   have produced clean-looking but meaningless benchmark numbers.

## Predictions this session got wrong

Recorded because the pattern is informative: every wrong call over-weighted a
*measured intermediate quantity* (‖τ‖, cosine, rounding loss) as a proxy for
function, and each was caught only by running the control.

| prediction | outcome |
|---|---|
| bf16 rounding depresses low-c scores; c-sweep needs correcting | refuted (−0.85 at c=0.4) |
| bf16 penalizes MOPD more than merge, biasing the head-to-head | refuted (−0.07 vs −0.41) |
| τ_MOPD is a functional subspace of τ_merge, so the parallel part should retain most ability when norm-matched | refuted (60.04, worst in table) |
| merge5_c1.0's ‖τ‖ matched √Σ‖τᵢ‖² to four decimals | arithmetic error; correct check needs cross terms (2.3057 vs 2.3058) |
| the n=4 screen would be adequate to rank these models | refuted; it reversed the n=16 ordering |
| mix-RL's IF domain shows a *threshold* effect (no progress at all) | over-claimed; slopes show 0.29× slowdown, ~1.95× better than linear-in-share |
| IF stagnation is "self-locking": no progress → variance stays low → no gradient | sign backwards; healthy learning *decays* variance (expert_if 0.120 → 0.066 with zero-var rising to 81%) |
| effective gradient share governs which domains learn | insufficient; Code has more effective groups than IF and is completely frozen — difficulty is a second, independent factor |
| IF is the run's worst-off domain | Code is worse (dWrong/dt 0.01× vs IF's 0.17×) and holds the largest prompt share |
| extrapolating τ_MOPD would simply not help | half right: mean@16 falls as predicted, but α=1.15 gives the project's 2nd-best pass@16 (80.42 > teacher's 80.28) |
