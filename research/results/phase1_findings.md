# Phase 1 Findings

Updated: 2026-09-17. This file preserves the Phase 1 experiment narrative.
See [current findings](current_findings.md) and
[recomputed evidence](20260917_reassessment/tables.md) for the latest synthesis
and uncertainty. All public-benchmark screens below are development evidence.

## Reproduction anchor

The reproduced 4B base, diagonal experts, and released five-expert TA (`lambda=0.6`) match the paper within roughly 0.2 AVG8 points. This makes the local setup suitable for controlled follow-up experiments.

## Geometry

- Global task-vector cosine is small even among reasoning experts: Math–Science `0.0222`, Math–Code `0.0088`, Science–Code `0.0023`.
- Pairs involving IF or Agent are nearly orthogonal globally. Their projection energy onto the M/S/C span is effectively zero.
- Local matrix structure is stronger than global geometry: selected attention tensors reach cosine around `0.1`, while different modules and layers contain both positive and negative alignment.
- Global parameter cosine only moderately correlates with directed cross-domain transfer (`r≈0.39`; layer 16 `r≈0.41`). It is informative but insufficient.

## Norm-matched expert-subset screening (`n=4`)

All subset updates were scaled to the same task-vector norm as five-expert TA at `lambda=1.0`.

| Model | Reasoning AVG5 | Mean tokens | Truncation |
|---|---:|---:|---:|
| M/S/C | 60.55 | 6,926 | 7.09% |
| M/S/C + IF | 61.71 | 5,916 | 1.98% |
| M/S/C + Agent | 62.28 | 7,308 | 5.97% |
| All five | 62.67 | 6,476 | 2.75% |

Relative to norm-matched M/S/C, adding IF gives `+1.16` points and adding Agent
gives `+1.73`; all five give `+2.12`. With only four samples per prompt, these
are screening signals rather than confirmation. The point estimates motivate
effects beyond total update norm, but do not exclude sampling variation. IF
also reduces response length and cap hits, which may itself help code accuracy.

## Support-overlap diagnosis

RAM-style hard support is unstable under the activation threshold. For IF, the fraction of update energy classified as unique changes from `1.0%` at `1e-6`, to `7.2%` at `1e-5`, to `41.8%` at `2e-5`. The corresponding global partition shifts from 69.2% of coordinates active in all experts to only 0.05%. Any hard-threshold shared/private claim must therefore include sensitivity analysis.

IF has unusually concentrated private support in embeddings: at threshold `1e-5`, about 57% of embedding-update energy is classified unique, versus roughly 4–9% globally. This motivates module-aware treatment of IF rather than a uniform expert scalar.

## Working interpretation

Low global cosine does not imply functional irrelevance. However, the evidence
does not establish a literal shared parameter vector between IF/Agent and
reasoning. Benefits may arise from input-conditioned nonlinear effects,
length/style control, or localized module interactions. The continuous
sharedness candidate was subsequently screened and did not improve TA; the
active plan now prioritizes verifier-calibrated coefficients.

## Fixed-coefficient factorial control (`lambda=1`, `n=4`)

| Model | Reasoning AVG5 | Mean tokens | Truncation |
|---|---:|---:|---:|
| M/S/C | 62.51 | 5,932 | 2.63% |
| M/S/C + IF | 61.91 | 5,777 | 1.48% |
| M/S/C + Agent | 62.43 | 6,517 | 3.41% |
| All five | 64.50 | 6,407 | 2.65% |

At the same coefficient, IF alone changes AVG5 by `-0.60` and Agent alone by `-0.08`, but adding both changes it by `+1.99`. The factorial interaction

`All5 - (MCS+IF) - (MCS+Agent) + MCS`

is `+2.67` AVG5 points, concentrated in AIME and GPQA; it is approximately zero
on the two code benchmarks. This suggested an interaction to test, without
establishing a higher-order mechanism. Because the run uses only four samples
per prompt, the rough AVG5 standard error is about `1.98` points; the larger
diagnostic below does not confirm a stable positive effect.

An additive parameter merge can have nonlinear, subset-dependent verifier
responses. Optimizing expert coefficients with a suitable response model can
account for these effects without constructing an interaction vector. The
method target remains improved measured performance, not mandatory interactions.

## Factorial confirmation (`lambda=1`, `n=16`)

The larger run weakens the exploratory interaction claim:

| Model | AIME25 | AIME26 | GPQA | AVG3 |
|---|---:|---:|---:|---:|
| M/S/C | 52.50 | 62.50 | 64.08 | 59.69 |
| M/S/C + IF | 54.79 | 59.79 | 64.30 | 59.63 |
| M/S/C + Agent | 53.54 | 63.12 | 64.33 | 60.33 |
| All five | 58.75 | 61.67 | 63.57 | 61.33 |

The factorial interaction is `+2.92`, `+1.25`, and `-0.98` points on the three
benchmarks, respectively, or `+1.06` AVG3. The original prompt bootstrap gave
`[-1.20, +3.55]`; the unified reanalysis gives `[-1.26, +3.54]`. All five exceeds
M/S/C by a point estimate of `+1.64` AVG3, but its interval `[-0.62, +4.02]`
also spans zero. Neither effect is confirmed. Explicit pairwise components
remain optional and require evidence on calibration validation.

## Continuous geometry and null-model check

Threshold-free coordinate statistics show pairwise absolute cosine near
`0.63`, despite signed cosine near zero. This initially resembles common
magnitude support, but an independent Gaussian null has expected absolute
cosine `2/pi = 0.6366` and nearly matching effective-expert statistics. Dense
attention and MLP task vectors are therefore close to random-independent at
the coordinate level, not demonstrably shared. Embeddings are substantially
more task-specific, especially for IF and Agent.

This weakens absolute overlap as the proposed method's main signal. The
subsequent `delta_W x` analysis is complete, but its global correlation advantage
does not survive several sensitivity checks. Any proposed decomposition still
needs direct verifier validation and should recover TA at its null correction.

## Strong static baselines (`n=4`, reasoning suite)

| Model | Reasoning AVG5 | Mean tokens | Truncation |
|---|---:|---:|---:|
| TA, `lambda=1` | 64.50 | 6,407 | 2.65% |
| RAM+ | 58.61 | 4,738 | 2.29% |
| RAM+, norm matched | 61.98 | 5,131 | 0.55% |
| ResMerge | 56.05 | 4,828 | 4.61% |
| ResMerge, norm matched | 62.16 | 5,895 | 0.76% |
| Soft-Overlap, norm matched | 60.05 | 5,115 | 0.55% |

The deployed update norms relative to the bf16 base are `2.228` (TA), `0.714`
(RAM+), `0.271` (ResMerge), and `2.163` (Soft-Overlap). Norm-matched controls
are now complete and recover much of RAM+/ResMerge's loss. Their remaining
gaps are `-2.52` (CI `[-4.71, -0.42]`) and `-2.34` (CI `[-4.77, +0.05]`).
Soft-Overlap trails TA by `4.46` points using unrounded scores. Reject that
tested configuration as the primary candidate; retain scaled baselines and
report their lower generation cost. This does not reject either method family.
