# Method Development Plan

Updated: 2026-09-17. These are prospective criteria, not claims that existing
experiments were designed to meet them.

## Deliverable

Implement a better static merge of Math, Science, Code, IF, and Agent RLVR
experts on Qwen3-4B. Deliver a runnable coefficient/component selection
procedure, one exported checkpoint, a manifest sufficient to rebuild it, and
controlled evidence of improvement. No extra inference-time expert routing or
ensemble is needed. Qwen3-8B remains paused until its thinking-mode issue is
resolved.

Success is an empirical improvement, not the existence of a geometry
correlation, shared/private decomposition, or positive interaction estimate.
If scalar calibration works best, ship it and describe the contribution
honestly. A novel structured component must beat that calibrated scalar
control under the same selection budget.

## Prospective acceptance rules

Freeze these rules and dataset identities before evaluating a finalist:

| Dimension | Rule |
|---|---|
| Public comparability | Target >= +1.0 AVG8 point over the strongest eligible baseline; report every benchmark. This remains a development result. |
| Independent confirmation | Positive five-domain macro gain with a paired prompt-bootstrap 95% lower bound above zero on newly reserved prompts. |
| Domain retention | No domain point estimate below the reference by >1.0 pp; require simultaneous one-sided lower bounds >= -1.0 pp to claim retention. |
| Generation budget | Same decoding limits/settings; equal-domain mean generated tokens <=105% of reference, length-cap-hit rate <= reference +0.5 pp. Also disclose per-domain lengths and tool calls. |
| Repetition | At least two explicitly recorded decoding seeds with n>=16 for finalists, subject to a prospective prompt-count/power assessment. |
| Deployment | One static checkpoint; fp32 merge arithmetic, common bf16 export, measured update norm after export. |

For retention, one conservative option is a one-sided 99% bound for each of
five domains (Bonferroni family-wise 95%). A small pilot cannot establish this
automatically. If uncertainty is too large, report inconclusive and size the
confirmation sample before viewing its results. The +1.0 pp target is an
engineering milestone, not an estimate of the effect already achieved.

The reference is selected on calibration validation and frozen alongside the
candidate. Always report fixed TA lambda=1 and the paper's lambda=0.6 as
anchors. Eligible stronger references include calibration-selected uniform TA
and RAM+/ResMerge with calibrated global scale. For structured candidates,
include the best calibrated five-scalar method as an additional reference.
Match the number of candidate evaluations and selection prompts for
methods that tune parameters; report merge/calibration GPU-hours separately
from inference cost.

AVG8 gives equal weight to eight benchmarks; the five-domain macro gives equal
weight to Math, Science, Code, IF, and Agent, averaging a domain's benchmarks
first. The reasoning AVG5 screen excludes IF and Agent and cannot establish
multi-domain retention. Calibration rewards, especially fractional Code
rewards and Agent rewards near saturation, are not interchangeable with public
benchmark accuracies.

## Data roles

1. Existing public benchmarks: development diagnostics and paper comparison.
   They have already influenced lambda choice, module hypotheses, and the
   choice of factorial benchmarks. Fresh decoding seeds on them do not create
   an untouched test set.
2. Existing 128-row/domain calibration manifest: pilot data, possibly seen by
   experts during training. All rows have already been evaluated. Cross-fold
   analysis estimates pilot stability, not clean generalization.
3. New calibration validation prompts: reserve before fitting/selecting new
   candidates. Prefer independent prompts with executable/rule verifiers, or
   a verifiable pre-expert-training holdout. Check overlap by normalized
   content, not just source row IDs; store provenance and hashes.
4. New final confirmation prompts: disjoint from expert training, calibration,
   validation, and previous method selection. Freeze the method before using
   them. If these prompts are unavailable, label the result exploratory.

## Candidate family

Let `tau_i = theta_i - theta_0`. Begin with:

`theta(a) = theta_0 + sum_i a_i tau_i`, with `a_i` initially in `[0.8, 1.2]`.

TA is recovered exactly at `a_i=1` before the common export operation. The
verifier reward is nonlinear in `a` even though the parameter merge is linear.
A quadratic response surface can capture coefficient interactions without
inventing a pairwise parameter vector.

Only after this control works, consider

`theta(a, b) = theta_0 + sum_i a_i tau_i + sum_(i,g) b_(i,g) M_g tau_i`.

`M_g` is an explicit module mask; existing `merge_group_coefficients.py`
implements this by passing coefficient `a_i + b_(i,g)` for the selected group.
At `b=0` the scalar merge is recovered. Limit initial residuals to one or two
groups. Existing embedding specificity is a hypothesis source, not evidence
that increasing the embedding coefficient improves IF.

Optimize average verifier reward subject to retention and budget constraints,
with regularization toward the TA anchor. Use per-domain constraints to avoid
trading away a domain for a small mean gain. Start with main effects and
diagonal curvature; retain cross terms only when held-out candidate settings
improve predictive accuracy. Confirm any surrogate optimum with an actual
merge/evaluation; never report surrogate predictions as measured scores.

## Execution stages

| Stage | Concrete output | Advance condition |
|---|---|---|
| 0: reconcile evidence | Recomputed tables, uncertainty, source hashes, current findings | Completed in this revision; no improved method established. |
| 1: coefficient measurements | TA anchor plus ten one-expert +/-0.2 changes; four prespecified mixed settings for prediction checks | Per-domain verifier metrics, lengths, prompt IDs, seeds, and saved checkpoint norms are complete. |
| 2: scalar optimizer | Regularized local fit, constrained selector, actual evaluation of its selected checkpoint | Beats calibration-selected uniform TA on reserved validation within budget; otherwise retain TA and diagnose the failed constraint. |
| 3: optional structure | At most two module residuals using the existing merger | Beats the scalar control at equal calibration budget; verify IF/Agent directly. |
| 4: freeze and confirm | Locked formula/config/checkpoint plus independent confirmation | Meets the prospective acceptance rules; otherwise report a failed or inconclusive candidate. |

The Stage 1 candidate list is in
`../experiments/calibration/next_design.json`. It is prepared, not executed.
The three completed uniform settings lie on a single line in five-dimensional
coefficient space. They identify neither five expert effects nor pairwise
interactions. Eleven axial settings identify intercept, five main effects,
and five diagonal curvatures but saturate that fit; use regularization and the
four held-out settings to check predictions. They do not identify a full
21-parameter quadratic. A full quadratic requires a separately frozen,
full-rank design with more than 21 settings and out-of-design validation.

Implement the scalar optimizer only after these measurements exist; the
current repository has merge/evaluation primitives, not a validated optimizer.
Before GPU scheduling, verify current cluster processes and resources rather
than trusting historical PID records.

## Bounded mechanism work

Repeat functional geometry on disjoint prompt batches and compare raw weight
cosine, functional cosine, and simple update norms using held-out domains or
expert pairs. Proceed to functional subspaces only if this changes which
component to evaluate and survives stability checks. Current global functional
correlation superiority disappears in some leave-expert-out checks.

Do not resume the current Soft-Overlap rule without a concrete change and a
new falsifiable prediction. Do not infer a useful interaction vector from the
inconclusive IF x Agent factorial contrast. Do not require a novelty story
before implementing and measuring the simplest useful merge.
