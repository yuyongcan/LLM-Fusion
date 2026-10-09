# E7 — Difficulty-Stratified Diagnosis of Joint Multi-Domain RL

Handoff for the agent operating the 4-node (4×8 GPU) machine. Written 2026-09-22.

You are running **four GRPO training arms in parallel, one per node.** Each arm
is a five-domain joint-RL run that differs from the others only in how the
training mixture is filtered or resampled. The goal is not a better model — it
is to determine *why* two of the five domains fail to learn, and whether the two
failures have independent causes.

Read [Background](#background) before touching anything: the whole experiment
turns on one specific metric, and the obvious metric is misleading.

---

## Background

A five-domain joint-RL run (`train_mix_8x8_20260919_162504`, Qwen3-4B, GRPO,
bs=128) was analysed per domain over steps 1–270. Rates are per step; the
reference is a single-domain IF expert run, which reached dWrong/dt = −0.159.

| domain | prompt share | eff. groups/step | dWrong/dt | vs expert | group std | verdict |
|---|---:|---:|---:|---:|---|---|
| Science | 19.5% | 15.8 | −0.0067 | 0.04× | 0.216 → **0.182** | healthy |
| Agent | 14.8% | — | — | — | 0.500 → **0.426** | healthy |
| Math | 23.4% | 19.5 | +0.0078 | −0.05× | 0.243 → 0.233 | **saturated** |
| IF | 18.0% | 7.7 | −0.0267 | 0.17× | 0.120 → 0.124 | **starved** |
| **Code** | **24.2%** | 10.4 | **−0.0011** | **0.01×** | 0.085 → 0.083 | **frozen** |

Two distinct pathologies, which we believe have different causes and different
fixes:

- **IF is starved.** It is the sparsest-reward domain: 68% of its GRPO groups
  have zero reward variance, so they yield zero advantage and no gradient. At
  18% of prompts it ends up with only ~15% of gradient-carrying groups. A
  separate run at bs=512 raised IF's effective groups to 0.82× expert parity and
  recovered learning to 0.51–0.66× — **partial**, consistent with dilution being
  the dominant cause.
- **Code is frozen, and starvation does not explain it.** Code holds the
  *largest* prompt share (24.2%) and more effective groups per step than IF
  (10.4 vs 7.7), yet its all-wrong group fraction is immobile across 275 steps
  (57.6% → 57.4%) and its all-right fraction never exceeds 3%. Gradient supply
  is adequate and it still does not learn. The hypothesis is that the task is
  simply beyond the model's current capability, so most prompts are permanently
  all-wrong. **If true, more batch or more data cannot fix Code.**

### The hypothesis you are testing

| | gradient supply adequate | gradient supply starved |
|---|---|---|
| **task within capability** | healthy (Science, Agent) | **starvation** (IF) — fix by rebalancing |
| **task beyond capability** | **difficulty freeze** (Code) — rebalancing is useless | both at once |

The prediction is that **each failure has its own fix and the fixes do not
substitute for one another.** Arm 1 should unfreeze Code and leave IF alone;
arm 2 should recover IF and leave Code frozen. If a single intervention fixes
both, the two-factor account collapses and we should know that.

### The metric trap — read this carefully

**Do not use mean group reward std as the health signal, and do not use
`zero_var_frac` on its own.** Both are two-sided:

- A domain that has *not started* has low spread and high zero-variance, because
  its groups are uniformly wrong.
- A domain that has *finished* also has low spread and high zero-variance,
  because its groups are uniformly right.

The single-domain IF expert demonstrates this: as it learned well, group std
*fell* 0.120 → 0.087 → 0.066 and zero-variance *rose* 68% → 81%, because
all-wrong groups were converting into all-right ones (wrong 42.7% → 18.0%, right
25.3% → 62.9%). An earlier version of this analysis read the falling variance as
deterioration; it was the opposite.

**Use the decomposition instead:**

- `all_wrong_frac` ↓ = the domain is unlocking problems it could not solve
  ("curriculum unlock rate", **dWrong/dt** — the primary signal)
- `all_right_frac` ↑ = the domain is consolidating solutions (**dRight/dt**)
- `effective_groups` = groups actually contributing gradient this step

These are now logged per domain by the trainer (added 2026-09-22, see
[Metrics](#metrics)), so you do not need to post-process rollout dumps.

---

## The four arms

All arms: Qwen3-4B-Instruct-2507, GRPO, five domains, **identical
hyperparameters**. The only difference is the training mixture. One node each,
run in parallel.

| arm | name | intervention | prediction if the 2×2 holds |
|---|---|---|---|
| 1 | `code_filtered` | keep only Code prompts with pass rate ∈ (0,1) | Code unfreezes; IF unchanged |
| 2 | `if_upsampled` | duplicate IF prompts 2.5×; Code untouched | IF recovers; Code still frozen |
| 3 | `both` | arms 1 and 2 together | both recover — fixes are independent |
| 4 | `code_downsampled` | keep 10% of Code prompts | **other domains improve** on freed compute |

Arm 4 is the one with a practical payoff: if dropping a too-hard domain makes
the others better, then in multi-domain RLVR an over-difficult domain does not
merely fail to learn, it **crowds out domains that could**. Plain dilution does
not predict that.

### Step 1 — measure Code's per-prompt pass rate

Arms 1 and 3 need to know which Code prompts are solvable at all. Do **not**
guess a difficulty proxy from prompt text. Measure it:

Use the existing rollout dumps from the completed baseline run at
`runs/llm_fusion/rollout_data/mix_rl_qwen3_4b_8x8/` (steps 1–382; **use only
steps ≤ 275**, see [Data hygiene](#data-hygiene)). Each line is one rollout with
`input`, `score`, `gts`. Group by `input`, compute the mean score per prompt, and
classify:

- `pass_rate == 0` → never solved; these are what arm 1 removes
- `0 < pass_rate < 1` → learnable signal; arm 1 keeps these
- `pass_rate == 1` → already solved

`research/group_stats.py` and `research/variance_trajectory.py` already contain
the domain-labelling logic — **import `domain()` from
`research/variance_trajectory.py` rather than rewriting it.** Labelling is
subtle: only IF carries an explicit `rm_type`; Code is identified by
stdin/stdout test vectors in `gts`, Agent by an expected tool-call list, Science
by a single answer letter, Math by a bare expression. **Agent must be checked
before Math** — a tool-call list is also "not a letter" and an earlier analysis
misfiled every Agent rollout as Math.

Sanity check your labelling against the known mixture before proceeding:
Math 25.1% / Science 22.4% / Code 21.9% / IF 18.9% / Agent 11.7% by prompt count.

Report how many Code prompts fall in each bucket. If the all-wrong bucket is
*not* a large majority (we expect ~57%), stop and say so — the premise is wrong.

### Step 2 — build the four mixtures

Source data: `data/LLM-Fusion/train/Mix/train-*.parquet` (16 shards, 87,699 rows).
Schema: `data_source`, `prompt`, `ability`, `reward_model`, `extra_info`,
`agent_name`. Domain is `data_source` (e.g. `LLM-Fusion/Code`) — for *building*
mixtures use this column directly; the `domain()` heuristic above is only needed
for rollout dumps, which lack it.

Write each arm's mixture to its own directory under
`data/LLM-Fusion/train/Mix_e7_<arm>/`. **Do not modify `Mix/`** — the baseline
and the running comparisons depend on it.

Constraints:

- Preserve every column, including `agent_name` (Agent rows carry
  `agent_name=tool_agent` and are routed to the multi-turn tool loop; losing it
  silently breaks Agent).
- Match Code prompts to their pass-rate bucket by the same key you grouped on in
  step 1. The dumps store a *rendered* prompt while the parquet stores structured
  `prompt`; verify your join actually matches before trusting it, and report the
  match rate. A low match rate means the join is wrong, not that the data is.
- Arm 2's upsampling is row duplication of `LLM-Fusion/IF` rows only.
- Keep shard sizes roughly comparable to the original (~5.5k rows/shard).

Record the resulting per-domain row counts for all four arms in your report.

### Step 3 — launch

Launcher: `examples/llm_fusion/train/train_mix_rl.sh`, driven by
`TRAIN_FILE`. Look at `run_train_multinode.sh` for how this repo wires up a
run; it handles the Ray/sandbox/env preflight.

Per-node settings (**one 8-GPU job per node — do not span nodes**):

```bash
export NNODES=1
export NGPUS_PER_NODE=8
export TRAIN_BATCH_SIZE=512          # matches the bs512 comparison run
export ACTOR_LR=2e-6                 # matches it too; see the caveat below
export TOTAL_TRAINING_STEPS=250
export SAVE_FREQ=50
export TEST_FREQ=25
export TRAIN_FILE="/taijifs_sz34/.../data/LLM-Fusion/train/Mix_e7_<arm>/train-*.parquet"
export EXPERIMENT_NAME=e7_<arm>_qwen3_4b
bash examples/llm_fusion/train/train_mix_rl.sh
```

Prerequisites per node: the code sandbox must be up on `localhost:8080`
(`bash setup_sandbox.sh`) or **Code rewards are silently near-zero — which would
fake the exact result we are testing for.** Verify with the curl probe in
`run_eval_experts.sh` before launching, and state in your report that you did.

**Add gradient-norm clipping.** The baseline run diverged at step 300 (see
[Data hygiene](#data-hygiene)). Check what the launcher already sets and make
sure a finite clip is active.

Two caveats to carry into the analysis:

1. bs=512 with lr=2e-6 changes *both* batch and learning rate relative to the
   bs=128/1e-6 baseline, so those effects are entangled. All four arms share the
   setting, so **cross-arm comparisons are clean**; comparisons against the
   bs=128 baseline are not.
2. 250 steps at bs=512 is ~4× the samples of 250 steps at bs=128. Compare arms to
   each other, and to the bs=512 run described below — not to bs=128 step counts.

### Step 4 — analyse

Primary signal, per arm, per domain:

```bash
python research/parse_train_log.py <log> \
    --grep 'grpo/group_by_domain/.*(all_wrong_frac|all_right_frac|effective_groups)' \
    --every 10
```

Fit **dWrong/dt** over a window well clear of the warm-up (e.g. steps 50–250)
and compare across arms. Reference points: expert_if −0.159 (healthy
single-domain), Science −0.007 (healthy in-mixture), Code −0.001 (frozen),
IF −0.027 (starved).

**Decision rules, fixed now — do not adjust after seeing results.**

| observation | conclusion |
|---|---|
| arm 1 lifts Code's dWrong/dt to ≥10× its baseline (−0.001 → ≤−0.01) and leaves IF within noise | difficulty freeze confirmed as a distinct mode |
| arm 2 lifts IF toward ≥0.8× expert and leaves Code frozen | starvation confirmed as a distinct mode |
| arm 3 ≈ arm 1's Code effect + arm 2's IF effect | the two fixes are independent — the 2×2 holds |
| arm 2 *also* unfreezes Code | **2×2 falsified**; one factor, report it plainly |
| arm 4 improves non-Code domains beyond arm-to-arm noise | over-difficult domains crowd out learnable ones |
| no arm moves anything | something is misconfigured — check the sandbox first |

Also report `val-core/<benchmark>/acc/mean@4` trends per arm, but treat training
dynamics as primary: 250 steps is unlikely to move benchmarks much, and the
mechanism lives in the group statistics.

---

## Metrics

Added to `verl/trainer/ppo/metric_utils.py` on 2026-09-22 (covered by tests in
`tests/trainer/ppo/test_metric_utils_on_cpu.py`). Per domain, per step:

| key under `grpo/group_by_domain/<domain>/` | meaning |
|---|---|
| `all_wrong_frac` | share of groups uniformly scoring 0 — **primary signal** |
| `all_right_frac` | share of groups uniformly scoring nonzero |
| `zero_var_frac` | degenerate groups (= the two above summed) |
| `reward_std_mean` | mean within-group reward std — advantage magnitude |
| `num_groups` | groups for this domain this step |
| `effective_groups` | non-degenerate groups — the gradient actually available |

`<domain>` is the lowercased tail of `data_source`: `math`, `science`, `code`,
`if`, `agent`. Note `reward_std_mean` is the **within-group** std averaged over
groups (this is the quantity GRPO's advantage scales with), not the std over all
samples in the batch.

With partial-credit verifiers `all_right_frac` means "uniformly nonzero", not
"perfect".

---

## Data hygiene

**The baseline run diverged after step 300.** grad_norm spiked 0.039 → 0.507,
response length blew up to 15,093 tokens (92% truncated), entropy collapsed to
0.009, and aime25/aime26/gpqa/ifbench all went to zero. **Use only steps ≤ 275**
from `mix_rl_qwen3_4b_8x8`; `global_step_275` is the last healthy checkpoint. The
IF and Code findings are unaffected — both are visible from step 0 — but do not
fit rates through the collapse.

**The bs=512 comparison run stopped at step 78** (`train_mix_8x8_20260921_164602`,
log idle since 11:41, no live process). Its numbers (IF recovery 0.51–0.66×) are
from 78 steps and are noisy. Do not treat them as settled; if you want a firmer
comparison, that run needs resuming — coordinate before doing so, it is not your
node.

---

## Deliverables

Write results to `research/experiments/20260922_difficulty_stratified/README.md`
following the format of `../20260920_mopd_extrapolation/README.md` — question,
method, decision rules stated up front, then results.

Include:

1. Code pass-rate distribution from step 1, and the parquet join match rate.
2. Per-domain row counts for all four mixtures.
3. dWrong/dt and dRight/dt per arm per domain, with the fitting window stated.
4. Which decision-rule row fired, including if it was a falsifying one.
5. Anything that contradicts the background above. **A clean negative is a
   result here** — the 2×2 is a hypothesis, not a conclusion to be confirmed.

Report what the runs actually did. If an arm crashed, was misconfigured, or you
skipped something, say so explicitly rather than reporting partial numbers as
complete.

## Context

Full research plan and the merge-side results:
`research/ideas/parallel_then_merge_vs_joint_rl.md`.
All measurements behind the background section:
`research/results/20260920_session_summary.md`.

Tools you should reuse rather than reimplement:

| tool | purpose |
|---|---|
| `research/parse_train_log.py` | verl console log → metric time series |
| `research/variance_trajectory.py` | per-domain group-composition dynamics from dumps; also the source of `domain()` |
| `research/group_stats.py` | per-domain GRPO group statistics for one window |
