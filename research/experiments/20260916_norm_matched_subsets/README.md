# Norm-Matched Expert-Subset Controls

## Question

Do the low-global-cosine IF and Agent experts improve reasoning when added to the Math/Science/Code (MCS) merge, or is the apparent gain explained by a larger total update norm?

## Fixed reference

The target norm is the five-expert Task Arithmetic update at global `lambda=1.0`.
Coefficients are computed exactly from the global task-vector Gram matrix in
`research/results/geometry_4b/geometry.json`.

| Expert subset | Uniform coefficient | Purpose |
|---|---:|---|
| Math, Science, Code | 1.3755469137 | Same update norm as five-expert λ=1.0 |
| Math, Science, Code, IF | 1.0800244928 | Isolate Agent's contribution |
| Math, Science, Code, Agent | 1.2203741254 | Isolate IF's contribution |

These are causal controls, not proposed final methods. Screen with `n=4`; confirm any material effect with fresh `n=16+` samples.

