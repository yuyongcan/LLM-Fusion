# 4B Task-Vector Geometry

The statistics in this directory were computed directly from the fp32 base and five fp32 RLVR experts using `analyze_task_vector_geometry.py`.

## Initial findings

- Global task-vector norms are Math 0.988, Science 1.023, Code 0.850, IF 1.322, and Agent 0.869.
- Reasoning vectors are weakly but consistently aligned: Math–Science cosine 0.0222, Math–Code 0.0088, Science–Code 0.0023.
- Pairs involving IF or Agent have global cosine close to zero; IF–Agent is slightly negative (-0.00057).
- IF and Agent have less than 0.001% of their global energy projected into the span of Math/Science/Code. A global PCA/shared-vector decomposition is therefore unlikely to explain their apparent reasoning gains.
- The five-expert λ=1.0 update norm is 1.376 times the MCS λ=1.0 update norm. Expert-count comparisons require a norm-matched control.
- Local structure is stronger than global structure: some attention matrices reach pairwise cosine around 0.1, while one-dimensional normalization tensors show larger but low-energy positive and negative correlations.

These findings motivate matrix-level spectral or functional analysis and rule out treating one global cosine as a sufficient measure of transfer.

## Threshold-Free Overlap

`continuous_overlap.json`, `continuous_pair_summary.csv`, and
`continuous_tensor_hotspots.csv` separate magnitude overlap from signed
alignment. Across all expert pairs, absolute cosine is about `0.63` even when
ordinary cosine is nearly zero. Importantly, `2/pi = 0.6366` is also the
expected absolute cosine of independent zero-mean Gaussian coordinates, so
this value alone is **not** evidence of shared support. Positive and negative
products almost exactly cancel (`same-sign product fraction ≈ 0.50`).

The energy-weighted effective number of experts is `2.445`, compared with
about `2.561` in a matched five-vector Gaussian simulation; update dominance
is `0.581` versus a `0.555` null. Most dense transformer matrices are therefore
close to a random-independent coordinate baseline, with only a modest excess
of task concentration. Embeddings are the clear exception: their pairwise
absolute cosine and energy Jaccard are far below the null-like matrix values,
especially for IF and Agent. Static coordinate overlap is consequently too
weak a basis for declaring a shared vector; sharedness should be measured in a
module subspace or on the data-induced function response.
