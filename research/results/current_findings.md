# Current Findings and Method Decisions

Updated: 2026-09-17. The objective is an implemented, measurably better model
merge. No current candidate has demonstrated that outcome across all five
domains under independent confirmation.

## Evidence sources and scope

[Recomputed tables](20260917_reassessment/tables.md) and
[machine-readable results](20260917_reassessment/summary.json) come from
`experiments/reassess_existing_results.py`. The JSON records source hashes,
prompt counts, all domain/benchmark metrics, and paired contrasts. Reproduce:

```bash
python3 research/experiments/reassess_existing_results.py
```

Intervals use 20,000 prompt-cluster bootstrap replicates, NumPy RNG seed
20260917, pairing models by input and resampling separately within each
benchmark. Benchmarks/domains receive equal weight. Identical inputs stay in
the same cluster; duplicate source-row weights are retained. These exploratory
intervals condition on saved generations, do not correct for prior selection
or multiple comparisons, and are not independent-seed replication. Minor
differences from the older factorial intervals reflect the new RNG stream.

Existing public tests are development data because lambda and hypotheses were
chosen using them. Increasing n or changing the decoding seed does not make
their prompts untouched. Existing calibration rows were sampled after expert
training, so they may be in-training examples. The standard recorded decoding
uses temperature 0.6, top-p 0.95 and a 16,384-token response limit. Historical
launchers do not freeze every provenance field required by the new protocol.
The reported cap-hit rate uses response length, not an explicit finish reason.

## Baselines: direction, scale, and uncertainty

All numbers below are the five reasoning benchmarks at n=4. They exclude IF
and Agent and are not AVG8 or the calibration five-domain macro.

| Method | AVG5 | Delta vs TA (95% CI), pp | Mean tokens | Cap hits |
|---|---:|---:|---:|---:|
| TA lambda=1 | 64.50 | reference | 6,407 | 2.65% |
| RAM+ | 58.61 | -5.89 [-8.58, -3.31] | 4,738 | 2.29% |
| RAM+ norm matched | 61.98 | -2.52 [-4.71, -0.42] | 5,131 | 0.55% |
| ResMerge | 56.05 | -8.46 [-11.17, -5.88] | 4,828 | 4.61% |
| ResMerge norm matched | 62.16 | -2.34 [-4.77, +0.05] | 5,895 | 0.76% |
| Soft-Overlap norm matched | 60.05 | -4.46 [-7.08, -1.89] | 5,115 | 0.55% |
| IF embedding coefficient 1.2 | 63.09 | -1.42 [-4.08, +1.08] | 6,458 | 2.63% |

Norm matching recovers 3.37 points for RAM+ and 6.12 for ResMerge. Their original
deficits mix scale and direction effects. ResMerge's residual gap is not
resolved at this sample size, and both scaled baselines have shorter outputs
and fewer cap hits than TA. Keep them as competitors, including calibrated
global-scale controls and a comparison at similar generation cost.

The build script scales previously bf16-exported updates by 3.121 and 8.234.
Post-rescaling export quantization can change their norms again; a requested
norm match is not proof of an exact deployed match. Record both pre-export and
post-export norms in future comparisons. Historical natural deployed norms
were TA 2.228, RAM+ 0.714, ResMerge 0.271, and Soft-Overlap 2.163.

Another saved TA lambda=1 run scores 62.67, a -1.83 difference from the selected
64.50 run, with interval [-4.17, +0.38]. These runs are not a controlled seed
sweep. The difference nevertheless shows why each experiment needs a
contemporaneous anchor and why an n=4 ordering should not be treated as final.

Decision: retire this Soft-Overlap configuration; retain scaled RAM+/ResMerge;
prioritize calibrated additive merges with direct budget constraints.

## Factorial evidence is inconclusive

At n=16 on AIME25/AIME26/GPQA:

| Contrast | AVG3 delta, pp | 95% CI |
|---|---:|---:|
| IF x Agent interaction | +1.06 | [-1.26, +3.54] |
| All five minus M/S/C | +1.64 | [-0.62, +4.02] |
| IF added to M/S/C | -0.07 | [-2.03, +1.89] |
| Agent added to M/S/C | +0.64 | [-1.54, +2.76] |

The earlier description of All5-MCS as a robust gain overstated the evidence.
Both that gain and the interaction remain uncertain. Selection of these three
benchmarks based on the n=4 screen further limits a confirmation claim.

Decision: keep all experts in the starting merge, but do not make positive
IF x Agent synergy a premise of the method. A nonlinear reward response does
not require an explicit interaction vector in parameter space.

## Calibration exposes domain and cost tradeoffs

The saved pilot has 128 source rows/domain with two generations/row:

| Lambda | Math | Science | Code | IF | Agent | Domain macro | Tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.8 | 51.17 | 53.91 | 21.93 | 69.14 | 97.27 | 58.68 | 2,529 |
| 1.0 | 48.83 | 55.08 | 24.26 | 71.09 | 95.31 | 58.91 | 2,792 |
| 1.2 | 47.66 | 56.25 | 21.74 | 73.44 | 89.45 | 57.71 | 3,121 |

Relative to lambda=1, lambda=0.8 changes the mean by -0.23 pp with interval
[-1.97, +1.49], while reducing mean generated tokens by about 9.4%. Lambda=1.2
raises IF by 2.34 pp but lowers Agent by 5.86 pp, with about 11.8% more tokens.
These are effects of changing all five coefficients simultaneously, not
identified IF-expert or Agent-expert causal effects. Lambda=1 is a useful
anchor, not a confidently superior cost-adjusted calibration choice.

Agent has only 125 unique input strings: three inputs each occur twice among
the 128 source rows and have matching ground truths. The analysis clusters
them and retains source-row weighting so point scores match the recorded
evaluation. Future splits need content deduplication and provenance checks.

The uniform sweep's linear design rank is 2/6 and full quadratic rank is 3/21.
It cannot support a fit of separate expert effects or pairwise interactions.
The prepared next design measures ten axial changes plus an anchor, then four
mixed settings for prediction checks. It has not been run.

Decision: implement verifier-calibrated expert coefficients first; optimize
mean reward with per-domain and generation-budget constraints. Benchmark it
against calibrated uniform TA, including the cheaper lambda=0.8 candidate.

## Functional geometry has no stable superiority yet

The analysis uses base-model activations from eight prompts and 64 sampled
tokens per domain, covering 252 transformer linear matrices. It measures
local `Delta W x`, not end-to-end logit or verifier changes, and omits embedding
and normalization perturbations. The saved artifact lacks a prompt-offset
field; do not invent one from the current script default.

| Pair subset | Raw weight r | Functional r |
|---|---:|---:|
| All 20 directed pairs | 0.389 | 0.523 |
| Remove IF->Agent | 0.570 | 0.497 |
| Remove every IF pair | 0.586 | 0.367 |
| Remove every Agent pair | 0.606 | 0.555 |

The apparent advantage depends on which experts/pairs are included. Directed
pairs share experts and targets, so they are not 20 independent observations.
This is a sensitivity check, not proof that the outlying pair should be
discarded or that raw cosine is intrinsically better.

Decision: use geometry only to nominate small, testable module corrections.
Before making it part of coefficient selection, repeat on disjoint prompt
batches and evaluate predictions on held-out domains/pairs.

## IF module ablation remains incomplete

Only the embedding correction has saved evaluations here, on reasoning tasks.
Its -1.42 AVG5 change is uncertain, and there is no IF/Agent result establishing
whether it protects the capability it was designed to help. Attention and MLP
build definitions exist, but no corresponding saved evaluations were found.

Decision: evaluate all five domains before selecting a module correction.
Embedding specificity alone does not justify amplification. Keep this extension
behind the five-scalar control in the implementation order.

## Implementation priority

Follow [the research plan](../notes/research_plan.md): obtain defensible
validation/confirmation prompts, measure separate coefficient effects, fit and
validate a constrained scalar selector, then admit module residuals only if
they improve the scalar control. The current evidence supports this workflow;
it does not establish that its future checkpoint will beat TA.
