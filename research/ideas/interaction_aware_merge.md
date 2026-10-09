# Verifier-Calibrated Merge Around Task Arithmetic

Updated: 2026-09-17. Active proposal; coefficient selection is not yet
implemented or validated. The filename is retained for existing references.

## Goal

Produce a static merged checkpoint with better measured multi-domain
performance than strong, comparably calibrated baselines. Follow the
[research plan](../notes/research_plan.md) for acceptance rules. Geometry and
interactions guide candidate selection only when useful; they are not required
mechanisms or success criteria.

## Evidence driving the design

- TA lambda=1 remains the strongest observed reasoning-screen point estimate.
  Norm-matched RAM+ and ResMerge reach 61.98 and 62.16 AVG5, versus TA's 64.50
  at n=4. This is insufficient to conclude they are universally worse.
- Soft-Overlap reaches 60.05 at roughly comparable update norm. Drop its
  current coordinate replacement rule from the primary implementation path.
- The n=16 IF x Agent interaction is +1.06 AVG3 with a confidence interval
  spanning zero. It does not justify a mandatory pairwise parameter component.
- Functional cosine correlates more strongly with transfer on all 20 directed
  pairs, but this ordering reverses when IF->Agent is removed. Treat it as an
  exploratory diagnostic, not a validated coefficient-selection signal.
- The pilot uniform sweep has domain tradeoffs and only varies one direction
  in five-dimensional coefficient space. Measure separate expert responses
  before fitting an adaptive merge.

See [current findings](../results/current_findings.md) for uncertainty and
reproducible source tables.

## First implementation

`theta(a) = theta_0 + sum_i a_i tau_i`.

Use fp32 arithmetic and a common bf16 export. Begin within a small trust region
`0.8 <= a_i <= 1.2`; `a_i=1` recovers TA. Use
`experiments/calibration/next_design.json` to measure one expert at a time,
followed by mixed settings reserved for prediction checks.

For each domain, fit verifier reward and generation cost as functions of
`z = a - 1`. Start with regularized main effects and diagonal quadratic terms.
Optimize mean reward under per-domain retention and length/cap-hit constraints.
Actually evaluate the selected coefficients on reserved validation prompts,
and fall back to the frozen reference when the correction fails.

The model's behavior is nonlinear even when checkpoint weights are an additive
merge. Cross terms in a response model can represent this behavior without
adding a constructed interaction vector to the checkpoint. A full quadratic
over five coefficients has 21 parameters including its intercept. Three
uniform settings cannot identify it; fitting cross terms requires a new,
overdetermined design and successful out-of-design prediction checks.

## Optional extension

After establishing the scalar control, permit one or two module corrections:

`theta(a,b) = theta_0 + sum_i a_i tau_i + sum_(i,g) b_(i,g) M_g tau_i`.

The existing module-coefficient merger provides these masks. Its `embedding`
group includes token embeddings and `lm_head` when present. A correction is
retained only if it beats scalar calibration at equal search cost. Evaluate
IF and Agent directly: a reasoning-only embedding screen cannot establish IF
retention. Activation subspaces and trajectory components remain alternative
future candidates, with their own implementation and validation costs.

## Method comparisons

Compare fixed TA, calibrated uniform TA, calibrated five-expert scalars, and
scaled RAM+/ResMerge under the same selection data and budget. For a structured
extension, report ablations removing its residual and its constraints. Report
post-export norm, every domain score, generated tokens, cap hits/finish
reasons, tool calls, and total calibration cost.

Improved engineering performance is sufficient for the first deliverable.
Any publication claim must separately establish what the procedure adds over
existing coefficient search, verifier-guided selection, and structured merging.
