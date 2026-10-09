# Verifier-Calibrated Shared/Private Merge (working proposal)

Status: historical proposal, superseded by
[verifier-calibrated merge around TA](interaction_aware_merge.md). Shared/private
decomposition is an optional future component, not the project's required
outcome. No validated shared/private merge is implemented here.

## Motivation

The reproduced 4B sweep shows that a single global coefficient under-scales some capabilities, while global parameter cosine does not explain all cross-domain gains. With five final task vectors, global cosine is 0.0222 for Math–Science, 0.0088 for Math–Code, and near zero for pairs involving IF or Agent. IF and Agent have essentially zero energy projection onto the span of the three reasoning vectors. Therefore, low global cosine cannot be interpreted as either irrelevance or independence in function space.

## Proposed representation

For each layer or matrix, decompose expert update `delta_i` into a stable component and a private residual. Candidate definitions will be compared rather than assumed:

1. Cross-expert spectral consensus: directions supported by multiple expert row/column subspaces.
2. Training-trajectory stability: directions whose sign/subspace persists across intermediate RL checkpoints.
3. Functional sensitivity: directions producing consistent logit or verifier changes on calibration prompts.

The merged update is

`delta_merge = alpha_shared * shared + sum_i alpha_i * private_i`.

Shared components are counted once; private components retain independent scales.

## Adaptive nonlinear calibration

Use small, fixed calibration subsets sampled from the original training corpora, never the final benchmarks. Evaluate a space-filling set of coefficient vectors and fit, for each domain, a regularized quadratic surrogate:

`reward_d(alpha) = b_d + g_d^T alpha + 0.5 * alpha^T H_d alpha`.

The off-diagonal terms explicitly model the observed possibility that near-orthogonal experts interact functionally. Select coefficients by maximizing normalized mean reward subject to a worst-domain retention constraint and a response-length penalty. This makes the method verifier-supervised but requires no gradient updates to model parameters.

## Required controls

- Uniform Task Arithmetic at matched global update norm.
- Fixed-coefficient leave-one-in/out merges.
- Task-wise scalar search without decomposition.
- RAM and ResMerge, if their released implementations support these checkpoints.
- Shared-only and private-only ablations.
- Equal inference budget and reporting of response length/truncation.

## Evidence threshold

A promising result must improve over uniform TA and task-wise scalar calibration
on newly reserved evaluation prompts, retain every domain within a predeclared
tolerance, and reproduce across independent decoding seeds. The existing public
benchmarks have already influenced selection and cannot serve as untouched
evaluation. Follow the current research plan for the prospective protocol.
