# Soft-Overlap Merge (SOM): First Candidate

Status: implemented and screened; rejected as the primary candidate in its
current form. The norm-matched checkpoint scores 60.05 reasoning AVG5 at n=4,
versus 64.50 for the selected TA lambda=1 run. This result applies to the tested
configuration, not every possible soft-overlap method. The design below is
retained as a historical record. See [current findings](../results/current_findings.md).

## Problem diagnosed locally

RAM classifies a coordinate as active with `abs(delta) > epsilon`, then treats coordinates active in one expert as private and those active in at least two experts as shared. On the local five fp32 RLVR experts, this partition is highly threshold-sensitive:

| Threshold | Coordinates active in exactly one expert | Active in all five | IF unique-energy fraction |
|---:|---:|---:|---:|
| `1e-6` | 3.9% | 69.2% | 1.0% |
| `5e-6` | 5.1% | 22.2% | 1.9% |
| `1e-5` | 13.8% | 4.1% | 7.2% |
| `2e-5` | 37.3% | 0.05% | 41.8% |

The definition of private knowledge therefore changes drastically under a modest absolute threshold change.

## Threshold-free formulation

For each parameter coordinate, let `a_i = |delta_i|`. Define soft participation

`p_i = a_i^q / (a_i^q + s^q)`,

where `s` is a coordinate-local robust scale such as the RMS or maximum magnitude across experts. This makes participation invariant to layer scale. The effective number of active experts is

`n_eff = (sum_i p_i)^2 / sum_i p_i^2`,

which ranges continuously from 1 (private) to N (shared).

Define a consensus update from sign-compatible weighted deltas, and a private update from the strongest expert contribution. A smooth gate `g(n_eff)` interpolates between them:

`delta_merge = g(n_eff) * consensus + (1 - g(n_eff)) * private`.

Coordinates with large sign conflict receive lower consensus confidence. A small global gain may be determined by norm matching rather than benchmark tuning.

## Intended distinctions (not an established novelty claim)

- Unlike RAM, no absolute active/inactive threshold or discrete overlap count is used.
- Unlike TIES, it does not discard all minority-sign updates after a global sign election.
- Unlike ResMerge/SVC, it operates on continuous coordinate participation rather than singular components.
- Unlike generic coefficient search, its parameterization is weight-derived and scale-invariant.

## Risks and falsification

The method may merely resemble a smooth magnitude-weighted average and lose strong private updates. It must beat RAM, ResMerge, and norm-matched Task Arithmetic under the same evaluation. Ablations should compare hard versus soft participation, no sign-confidence gate, and no norm matching.
