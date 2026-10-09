# Why Parallel-then-Merge Beats Joint RL

Status: proposal. Parts 2–3 are measured and settled. Part 1 now rests on a
**two-factor diagnosis** supported by per-domain dynamics within a single run
(two failure modes, two healthy controls); its remaining gap is confirming that
the two modes are independently fixable, which is E7. Updated 2026-09-22.
Supersedes the framing in
[../results/20260920_session_summary.md](../results/20260920_session_summary.md),
whose numbers remain valid.

## Thesis

Multi-domain capability fusion is limited by two quantities that the existing
literature does not name:

1. **Joint RLVR fails per-domain, for two distinct and separately-fixable
   reasons.** Effective gradient share (not prompt share) governs one failure
   mode; task difficulty relative to current capability governs the other. A
   single five-domain run exhibits both, plus two healthy domains as controls.
2. **The linear-regime budget** of the base model — merging works because
   orthogonal task vectors sum to a displacement that stays inside the region
   where the model responds additively.

Parallel-then-merge wins because it is subject to neither: each expert owns its
full gradient at its own difficulty curriculum, and orthogonality keeps the sum
within budget.

---

## Part 1 — Why joint RL underperforms

### What is measured

`train_mix_8x8_20260919_162504` (Qwen3-4B, GRPO, 5 domains). Five of six
benchmark axes improve normally; IF does not.

| step | aime25 | gpqa | bfcl | **ifbench** | **ifeval** |
|---|---:|---:|---:|---:|---:|
| 0 | 43.33 | 56.69 | 63.00 | **31.08** | **84.20** |
| 275 | 46.67 | 60.10 | 71.13 | **32.75** | **85.58** |

Per-domain GRPO group statistics (rollout dumps, steps 90–99,
`research/group_stats.py`):

| domain | prompt share | zero-variance groups | mean group std | **effective gradient share** |
|---|---:|---:|---:|---:|
| Math | 25.1% | 38.2% | 0.241 | 33.4% |
| Science | 22.4% | 52.4% | 0.179 | 26.3% |
| Code | 21.9% | 59.0% | 0.105 | 25.6% |
| **IF** | **18.9%** | **68.4%** | **0.117** | **14.7%** |

Under GRPO a group with zero reward variance yields zero advantage, so nominal
mixture and effective update share decouple. IF is the sparsest domain and the
most diluted.

### Per-domain dynamics: two failure modes, two healthy controls

Static group counts hide what is actually happening. Tracking the *composition*
of each domain's groups over training (`research/variance_trajectory.py`, 90-step
windows, steps 1–270) separates three distinct behaviours. Rates are per step,
against the expert_if reference of dWrong/dt = −0.159, dRight/dt = +0.227:

| domain | prompt share | eff groups/step | dWrong/dt | vs expert | dRight/dt | vs expert | group std | verdict |
|---|---:|---:|---:|---:|---:|---:|---|---|
| **Science** | 19.5% | 15.8 | −0.0067 | 0.04× | **+0.0539** | **0.24×** | 0.216 → **0.182** | healthy |
| **Agent** | 14.8% | — | — | — | — | — | 0.500 → **0.426** | healthy (score +0.280) |
| **Math** | 23.4% | 19.5 | +0.0078 | −0.05× | +0.0039 | 0.02× | 0.243 → 0.233 | **saturated** |
| **IF** | 18.0% | 7.7 | −0.0267 | 0.17× | +0.0217 | 0.10× | 0.120 → 0.124 | **starved** |
| **Code** | **24.2%** | 10.4 | **−0.0011** | **0.01×** | **+0.0044** | 0.02× | 0.085 → 0.083 | **frozen** |

**Healthy learning looks like variance *decay*, not growth.** expert_if's group
std falls 0.120 → 0.087 → 0.066 while zero-variance groups *rise* 68% → 81%,
because all-wrong groups convert to all-right ones (wrong 42.7% → 18.0%, right
25.3% → 62.9%) — the domain is being solved. Science and Agent show the same
signature inside the mixed run, which is important: **joint RL is not broadly
harmful; two of five domains progress normally.**

**Code is the worse pathology, and it is not dilution.** Its all-wrong share is
immobile across 275 steps (57.6% → 57.4%, 0.01× the expert rate) and right%
never exceeds 3%, *despite* holding the largest prompt share (24.2%) and more
effective groups per step than IF (10.4 vs 7.7). **Gradient supply is adequate
and it still does not learn** — the task is simply beyond current capability, so
57% of prompts are permanently all-wrong. More batch cannot fix this.

**Math is saturated, not failing.** It starts at score 0.573 with only 16%
all-wrong, and moves ±0.01 — it is already near this data's ceiling.

**Agent explains the mechanism from the other side.** BFCL's reward is
continuous, so its group std is 0.43–0.50 (2–4× every other domain) and its
zero-variance rate is **0.0%** — every group carries gradient. It also learns
fastest (score 0.482 → 0.762). This is direct support for within-group spread,
not prompt count, driving learning rate.

### Revised claim: a 2×2 diagnosis

The single-factor story ("effective gradient share governs learning") is too
coarse — Code refutes it. The defensible claim is two-dimensional, and **one run
populates all the informative cells**:

| | gradient supply adequate | gradient supply starved |
|---|---|---|
| **task within capability** | healthy (Science, Agent) | **starvation** (IF) — fixable by batch/mixture |
| **task beyond capability** | **difficulty freeze** (Code) — batch is useless | both at once |

This is stronger than the original because the two failure modes predict
*different* remedies, which is testable (E3 vs E7). It also explains why E1's
batch increase only partially recovers IF: batching addresses one cell and leaves
Code — 24% of prompts and ~23% of effective gradient — untouched and largely
wasted.

### Metric caveat: group variance is two-sided

`grp_std` is low in two opposite situations — a domain that has not started
(IF/Code: high all-wrong) and one that has finished (expert_if late: high
all-right). Both show high zero-variance rates (65% vs 81%) with inverted
meaning. **Diagnosing progress requires the wrong%/right% decomposition, not
variance or group counts alone.** Consequently, weighting effective gradient
share by mean group std — as the earlier analysis did — is biased: IF's low
spread in the mixture means "not started," while in the expert run it means
"solved." We propose reporting **dWrong/dt** (curriculum unlock rate) and
**dRight/dt** (solve rate) as the primary diagnostics.

### The controlled comparison

Training IF alone has an **even higher** degeneracy rate, yet it learns:

| run | zero-var | gradient share | IF reward trajectory | slope |
|---|---:|---:|---|---:|
| expert_if (IF only) | **70.9%** | 100% | 0.378 → 0.459 → 0.498 → **0.559** | +0.00226/step |
| mix-RL | 68.4% | 14.7% | 0.365 → 0.370 → **0.404** | +0.00065/step |

This rules out "IF is intrinsically too hard" and points at dilution.

### Correction to an earlier claim

An earlier draft described this as a **threshold effect** (the domain stops
learning entirely below some share). **The slope data does not support that.**
mix-RL's IF reward rises at 0.29× the expert's rate, while its effective
gradient share is 0.147 — i.e. the observed slowdown is ~1.95× *better* than a
linear-in-gradient-share prediction. That is consistent with graceful linear
slowdown, not collapse.

The defensible claim is therefore weaker and more precise: **IF is diluted to
~0.29× learning rate, which within a 250-step budget renders it visually flat on
benchmarks while still moving in training reward.** Whether any genuine
threshold exists is an open question the dilution curve below is designed to
answer.

### The blocking experiment

**The current evidence cannot separate dilution from interference.** mix-RL and
expert_if use the *same* batch configuration (`train_batch_size=128`,
`rollout.n=16`), so IF receives roughly 1/5 the absolute number of effective
groups per step in the mix. Two hypotheses predict the same observation:

- **H-dilution** — IF simply gets too few effective groups per step. Scaling the
  batch restores it.
- **H-interference** — sharing parameters with four other domains degrades IF
  regardless of sample count. Scaling the batch does not help, since the
  proportion is unchanged.

A reviewer will ask "isn't this just undertraining?", and today there is no
answer. **This is the single highest-priority experiment.**

#### E1: batch-size scaling — running, interim result

`train_mix_8x8_20260921_164602`: bs=512, lr=2e-6, 78 steps so far, grad_norm
stable at 0.013–0.02 (healthy).

| run | IF eff groups/step | vs expert_if | IF reward slope | vs expert | dWrong/dt | vs expert |
|---|---:|---:|---:|---:|---:|---:|
| expert_if @bs128 | 37.2 | 1.00× | 0.00226 | 1.00× | −0.159 | 1.00× |
| mix @bs128 | 7.6 | 0.20× | 0.00065 | 0.29× | −0.027 | 0.17× |
| **mix @bs512** | **30.6** | **0.82×** | **0.00115** | **0.51×** | **−0.105** | **0.66×** |

Batching restored gradient supply to 0.82× parity, but recovery is partial:
0.51× by reward slope, 0.66× by curriculum unlock rate. **Pure dilution predicts
0.82×; observed is 62–80% of that.** So dilution is the dominant factor with a
residual gap of roughly a third. IF's zero-variance rate is unchanged at ~70%,
confirming that batching adds samples without shifting the difficulty
distribution.

Slopes are estimated over ~78 steps and are noisy; recheck at 150–200 steps.
Note also that bs512 raised lr to 2e-6, so batch and lr effects are currently
entangled — the B′ arm below separates them.

| arm | train_batch_size | lr | status |
|---|---|---|---|
| A (baseline) | 128 | 1e-6 | done, 0.29× / 0.17× |
| B | 512 | 2e-6 | **running**, 0.51× / 0.66× at step 78 |
| B′ | 512 | 1e-6 | not run — separates batch from lr |
| C | 512, 1/4 the steps | 2e-6 | not run — equal total samples |

**Decision rule, fixed in advance.** ≥0.8× supports pure dilution; ≤0.5×
indicates real interference. The interim 0.51–0.66× straddles this, so B′ and C
are needed before the claim is safe.

#### E2: dilution curve (functional form)

Hold IF's absolute prompt count fixed; vary the other domains so IF's share is
{100%, 50%, 25%, 19%, 10%}. Plot IF slope against effective gradient share.
A kink identifies a threshold; a straight line confirms graceful degradation.
This is what upgrades the finding from observation to mechanism, and it settles
the question raised by the correction above.

Rebalancing by effective group count and difficulty filtering — previously
listed here as separate E3/E4 — are now arms 2 and 1 of
[E7](#e7-difficulty-stratified-diagnosis-4x8-node-parallel), which runs them
together so their independence can be tested rather than assumed. If IF recovers
under rebalancing without harming other domains, the prescription is: **size the
batch and mixture by the sparsest domain's effective group count, not by prompt
count.**

### Why existing theory does not cover this

Multi-task gradient-conflict methods (PCGrad, CAGrad, and successors) assume
task gradients point in competing directions. Here the five task vectors are
mutually orthogonal to within cosine ≤ 0.0222 (§Part 2) — there is essentially
no conflict to resolve. The mechanism lives in the *reward* layer and is
specific to verifier-based RL, where group-relative advantages vanish on
all-correct and all-wrong groups.

### Data hygiene

The 8×8 run **diverged after step 300** and must not be used past that point:
grad_norm spiked 0.039 → 0.507, response length blew up to 15093 tokens (92%
truncated), entropy collapsed to 0.009, and aime25/aime26/gpqa/ifbench all went
to zero. `global_step_275` is the last healthy checkpoint (aime25 46.67, ifbench
32.75). The IF-stagnation finding is independent of this divergence — it is
visible from step 0 — but any rerun should add gradient-norm clipping, and the
paper's joint-RL baseline must be a healthy run.

---

## Part 2 — Why merging succeeds (settled)

All endpoints fp32; 398 shared tensors; ‖θ_base‖ = 1583.4033.

| model | ‖τ‖ | ‖τ‖/‖θ_base‖ |
|---|---:|---:|
| expert_math / code / science / if / agent | 0.9881 / 0.8501 / 1.0232 / **1.3220** / 0.8694 | 0.05–0.08% |
| merge5_c1.0 | 2.3058 | 0.146% |
| MOPD_merge1.0 | 1.6356 | 0.103% |

**The experts are mutually orthogonal** (pairwise cosine ∈ [−0.0006, 0.0222]),
and **task arithmetic is exactly a sum**: predicting ‖Στᵢ‖ from norms and the
full cosine matrix gives 2.3057 against 2.3058 measured.

**Orthogonality buys linear-regime headroom.** This is the causal chain, and it
is not "orthogonality avoids conflict":

- five aligned vectors would give ‖Στᵢ‖ = Σ‖τᵢ‖ = 5.0528 → **0.319%**
- actually orthogonal: **0.146%**
- measured degradation onset: ~**0.175%** (c=1.0 → 1.2, AVG8 66.00 → 65.08)

Orthogonality shrinks the sum 2.19× and lands it inside the additive region.

**Explicitly falsified: "small updates cause orthogonality."** The *largest*
task vector (expert_if, 1.3220) is the *most* orthogonal (max cosine 0.0003);
the largest cosine is math–science, two semantically adjacent mid-norm domains.
Orthogonality follows from task semantics and high dimension (random cosine
≈ 1.6e-5), not from displacement size.

**There is no dominant direction.** Splitting τ_merge against τ_MOPD
(c = 0.6449) and evaluating each part at n=16 over the full suite:

| model | ‖τ‖ | mean@16 | pass@16 |
|---|---:|---:|---:|
| merge5_c1.0 (full) | 2.3058 | **66.00** | 80.28 |
| proj_parallel | 1.0560 | 63.68 | 78.56 |
| proj_parallel_normmatched | 2.3058 | 60.04 | 75.48 |
| proj_perp | 2.0497 | 60.38 | 77.39 |
| proj_perp_normmatched | 2.3058 | 60.23 | 76.91 |

At equal norm the two halves are indistinguishable (60.04 vs 60.23) and both
lose ~6 points. Removing either half costs the same, so performance comes from
**many independent directions summing**. Scaling the parallel part *up* to
‖τ_merge‖ makes it worse (63.68 → 60.04) and damages convergence (pathologically
long generations; aime25 pass@16 63.61, below base's 71.46).

Independent confirmation: at fixed total norm, more directions is monotonically
better — 5 (62.67) > 4 (62.28 / 61.71) > 3 (60.55), from
`../experiments/20260916_norm_matched_subsets`.

---

## Part 3 — The gain is sharpening, not new capability (settled)

pass@16 separates "solves more problems" from "solves the same ones more
reliably":

| family | mean@16 gain | pass@16 gain |
|---|---:|---:|
| mcs3, base → c=1.0 | +4.73 | +1.81 |
| merge5, c=0.4 → c=1.0 | +5.37 | +3.62 |

~60% of the mean@16 gain is pure sharpening. merge5's pass@16 gain is almost
entirely one column (ifbench +19.0 across the sweep) while gpqa pass@16 actually
*declines* (87.29 → 86.16). Instruction-following is the only axis where merging
genuinely widens the frontier.

**Two independent mechanisms trace the same sharpness/coverage trade-off**,
which is what makes this convincing:

| knob | mean@16 | pass@16 |
|---|---|---|
| merge coefficient c: 1.0 → 1.2 | 66.00 → 65.08 | 80.28 → 80.11 |
| MOPD extrapolation α: 1.0 → 1.15 | 65.45 → 65.23 | 79.87 → **80.42** |
| MOPD extrapolation α: 1.0 → 1.30 | 65.45 → 63.07 | 79.87 → 77.94 |

α=1.15 buys coverage at the cost of sharpness — ifbench pass@16 +3.22, gpqa
+1.70, aime25 +1.00 — and posts **the second-highest pass@16 in the project
(80.42), above its own teacher (80.28)**. α=1.30 overshoots and loses both. So
extrapolation is not useless: it moves the operating point along the trade-off
rather than adding capability. Weight-space merging and post-distillation
extrapolation, two unrelated mechanisms, trace the same curve shape.

Corollary: sharpening is exactly what a mode-seeking distillation objective can
reproduce, which explains how MOPD_merge1.0 matches its teacher (65.45 vs 66.00)
and exceeds the three-expert merge on lcb_v5 (67.44 vs 67.10) and gpqa (64.49 vs
63.60).

---

## Methodological contribution: ‖τ‖ is a poor proxy for functional magnitude

bf16 inference discards every coordinate below one rounding unit — 28.5% of
‖τ‖ at c=0.4. A matched control (same fp32 weights, only `ROLLOUT_DTYPE`
changed, n=16) shows **no model improves under fp32**:

| model | ‖τ‖ surviving bf16 | fp32 − bf16 |
|---|---:|---:|
| merge5_c0.4 | 71.5% | **−0.85** |
| merge5_c1.0 | 96.6% | −0.41 |
| MOPD_merge1.0 | 87.2% | −0.07 |

Discarding 28.5% of the L2 norm costs nothing measurable. This is a stronger
form of the DARE observation — no rescaling is required.

**Consequence for this paper's own controls:** every norm-matched comparison —
including Part 2's `*_normmatched` variants and the subset experiment —
equalizes a quantity only loosely coupled to functional strength. The
conclusions there survive because the *contrasts* are large and consistent, but
the caveat must be stated. A better magnitude measure (Fisher-weighted norm, or
effective gradient norm along τ) is a natural follow-up.

This also explains why `proj_parallel` reaches 96.5% of the merge's score on 46%
of the norm.

---

## What to run next

| # | experiment | tests | status / cost |
|---|---|---|---|
| **E7** | **difficulty-stratified diagnosis (4 arms, parallel)** | **starvation vs difficulty freeze are separately fixable** | **next — 4×8 node** |
| E1 | batch-size scaling | dilution vs interference | B running; B′ and C outstanding |
| E2 | dilution curve, 5 mixture ratios | threshold vs linear slowdown | 5 short runs |
| E5 | healthy joint-RL baseline (resume 275 + grad clip) | usable baseline | 1 run |
| E6 | second scale, or a no-group-normalization variant | generality | 1–2 runs |
| E8 | pairwise domain ablation at fixed IF group count | residual interference, and whether it tracks τ orthogonality | 4 runs |

E1's partial recovery (0.51–0.66× against a 0.82× prediction) plus Code's freeze
make **E7 the highest-value next experiment**: it is the only design that tests
the 2×2 diagnosis directly, and it fits a 4-node machine exactly.

#### E7: difficulty-stratified diagnosis (4×8 node, parallel)

Fix everything except the intervention; measure **dWrong/dt** and **dRight/dt**
per domain. Reference rates: expert_if −0.159 (healthy), Science −0.007/+0.054
(healthy), Code −0.001 (frozen).

| arm | intervention | prediction if the 2×2 holds |
|---|---|---|
| 1 | Code restricted to prompts with pass rate ∈ (0,1) | Code unfreezes; IF unchanged |
| 2 | IF upsampled 2.5× (Code untouched) | IF recovers; Code still frozen |
| 3 | both interventions | both recover — the two fixes are independent |
| 4 | Code downsampled to 10% of its prompts | other domains improve on freed compute |

Arm 4 is the most interesting for practice: if removing a too-hard domain
*improves the others*, then in multi-domain RLVR an over-difficult domain does
not merely fail to learn — **it crowds out the domains that could**. That is a
directly actionable prescription, and it is not something the dilution story
alone predicts.

Arm 1 vs 2 is the crux: the 2×2 predicts each intervention fixes exactly one
domain. If arm 2 also unfreezes Code, difficulty and starvation are not
separable after all and the diagnosis collapses back to one factor.

E2 resolves the threshold-vs-slowdown question. E5 is required because the
current joint-RL run diverged. E8 was the previous top priority; it is now
superseded by E7, since the residual interference gap it targets is smaller than
the effect E7 measures.

Known weaknesses to address: single model scale (4B), single RL algorithm, and
public benchmarks used for both hypothesis selection and reporting — see the
development-data caveat in `../results/current_findings.md`.

## Provenance

Measurements: [../results/20260920_session_summary.md](../results/20260920_session_summary.md).
Tooling: `research/parse_train_log.py` (console log → metric series),
`research/group_stats.py` (per-domain GRPO group statistics),
`research/variance_trajectory.py` (per-domain group-composition dynamics),
`research/task_vector_geom.py` (norms and pairwise cosines),
`research/experiments/project_task_vector.py`.
Experiment README: `../experiments/20260920_mopd_extrapolation/README.md`.

**Domain labelling.** Only IF carries an explicit `rm_type` in the rollout
dumps; the rest are identified by ground-truth shape — Code by stdin/stdout test
vectors, Agent by an expected tool-call list, Science by a single answer letter,
Math by a bare expression. Agent must be checked before Math: a tool-call list
is also "not a letter" and an earlier version of this analysis misfiled all
Agent rollouts as Math, which inflated Math's apparent size and score. The
corrected labels reproduce the known mixture (Code 24.2 / Math 23.4 / Science
19.5 / IF 18.0 / Agent 14.8 against Math 25.1 / Science 22.4 / Code 21.9 / IF
18.9 / Agent 11.7 by prompt count).
