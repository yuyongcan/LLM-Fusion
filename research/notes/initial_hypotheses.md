# Initial Hypotheses

Status: historical starting hypotheses. The active goal is a better measured
merge, defined in [research_plan.md](research_plan.md); none of these proposed
mechanisms is a prerequisite. Current evidence rejects the first Soft-Overlap
configuration and leaves stable IF x Agent synergy unconfirmed.

## Observations to explain

1. The apparent IF degradation at the paper's `lambda=0.6` largely disappears at larger global coefficients.
2. Adding low-cosine IF and Agent experts improves the reasoning-domain merge relative to a Math/Science/Code-only merge.
3. Different benchmarks prefer different global coefficients, suggesting one scalar is too restrictive.

## Important caveats

- A single expert's eight-benchmark average is not comparable to the paper's per-domain-RL row, which selects the matching expert for each domain.
- Coefficients were selected on the reported test suite; generalization claims
  need independently reserved prompts. Repeated decoding seeds measure
  sampling stability but do not repair test-set selection bias.
- Larger coefficients increase response length. On the reproduced 4B sweep, reasoning-task mean length rises from about 4.3k tokens at `lambda=0.6` to 5.9k at `lambda=1.0`; LiveCodeBench truncation also rises.

## Candidate mechanism

Global cosine similarity can hide structured agreement: two task vectors may be nearly orthogonal overall while sharing useful directions in selected layers, modules, singular subspaces, or parameter coordinates. Decompose each task vector into:

`tau_i = shared_i + private_i`

where `shared_i` lies in a consensus subspace learned from all experts and `private_i` is its residual. Merge the consensus only once, then scale private components independently. Candidate coefficients should depend on local agreement, update magnitude, and a small calibration objective rather than a single global lambda.

## Falsifiable predictions

- Five-expert gains on reasoning tasks should concentrate in particular layers/modules rather than arise uniformly across parameters.
- Shared components should transfer across domains with lower interference than raw task-vector sums.
- IF-specific residual scaling should improve IFBench without requiring equal amplification of reasoning vectors.
- Any accuracy gain from larger coefficients should be evaluated jointly with response length and truncation rate.
