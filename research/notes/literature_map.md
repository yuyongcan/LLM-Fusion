# Literature Map: Adaptive and Structured Model Merging

Updated: 2026-09-16. Searches used the arXiv API and full arXiv HTML/source where available.

## Closest prior work

| Work | Core idea | Data needed | Relation to this project |
|---|---|---:|---|
| Task Arithmetic, arXiv:2212.04089 | Add scaled task vectors | No | Current baseline; one global scale |
| AdaMerging, arXiv:2310.02575 | Learn task-wise or layer-wise coefficients by entropy minimization | Unlabeled test inputs | Covers generic adaptive coefficients |
| TALL-masks / Consensus Merging, arXiv:2405.07813 | Identify task-relevant coordinates; retain parameters supported by multiple tasks | Validation data for mask threshold | Covers coordinate-level shared/private support |
| MetaGPT, arXiv:2406.11385 | Closed-form task coefficients derived under local linearity and orthogonality | No | Covers norm-based per-task coefficients |
| Adaptive Weight Disentanglement, arXiv:2411.18729 | Learn a redundant vector whose removal orthogonalizes task vectors | No task data | Covers common-vector subtraction |
| Isotropic Model Merging, arXiv:2502.04959 | Combine common and task-specific singular subspaces | No | Covers static spectral shared/private decomposition |
| Subspace Boosting, arXiv:2506.16506 | Restore task-specific spectral rank lost when common directions dominate | No | Covers rank-collapse correction |
| RAM, arXiv:2601.13572 | For RL experts, average shared active coordinates and preserve/rescale unique ones | No | Closest coordinate-space RL-specific baseline |
| SVC, arXiv:2602.05536 | Detect and correct shared spectral over-accumulation after merging | No | Closest projection-preservation/spectral-calibration baseline |
| ResMerge, arXiv:2606.02252 | Split RL task matrices into spectral head and residual; merge them with different rules | No | Closest RL-specific spectral baseline |
| SAR, arXiv:2607.03065 | Project RL updates into the base model's spectral basis | No | Relevant to reasoning-effective versus off-manifold updates |
| Expert Training Duration, arXiv:2607.11997 | Shows merge quality depends on expert checkpoint age | No additional training | Motivates trajectory analysis but does not decompose one trajectory into components |
| Sparsity Curse / SAR-Merging, arXiv:2606.18521 | Resolve overlapping RLVR updates with diagonal-Fisher sensitivity, then sparsify and rescale private regions | Small validation set | Covers RLVR-specific sensitivity-aware conflict arbitration |
| Essential Subspace Merging, arXiv:2606.19164 | Decompose each task matrix using PCA of output shifts `X Delta W^T`; orthogonalize retained components | Unlabeled proxy inputs | Directly covers activation-aware task-vector decomposition |
| When Do Task Vectors Interfere?, arXiv:2608.09490 | Measure input-conditioned non-additivity with two-dimensional coefficient surfaces and logit-space counterfactuals | Prompt sets | Closest analysis of functional pair interactions; diagnostic rather than a merge optimizer |

## Novelty boundary

This section concerns a potential publication contribution. The primary
engineering objective is to implement a better measured merge; it does not
require inventing a new decomposition. Literature conclusions below are the
existing reading record, not a newly verified exhaustive search.

The following ideas alone are not sufficiently novel: assigning one coefficient per expert, assigning one coefficient per layer, splitting coordinates into shared/unique support, subtracting a common redundant vector, or separating leading singular directions from residuals.

A more differentiated direction is **verifier-calibrated, robust merging for RLVR experts**:

1. Use weight/trajectory geometry to define a small number of interpretable components rather than thousands of free coefficients.
2. Estimate domain-specific response surfaces directly with the native RLVR
   verifiers on training-only calibration prompts.
3. Fit a nonlinear response surface over component coefficients and optimize a robust objective that rewards average gain, protects the weakest domain, and penalizes excessive generation length.
4. Evaluate once on newly reserved prompts after freezing the method. Existing
   public benchmarks already used for selection remain development data.

Activation-aware decomposition alone is no longer a viable novelty claim because
ESM already optimizes low-rank approximation of `Delta W x`. Likewise,
functional non-additivity itself is covered by arXiv:2608.09490. The remaining
gap is to use the defining property of RLVR—domain-specific executable or rule
verifiers—to choose a single static merge under explicit worst-domain and
generation-budget constraints. This differs from AdaMerging's entropy
objective, SAR's Fisher conflict arbitration, and generic black-box coefficient
search. Any interaction term must earn inclusion on calibration validation
rather than being assumed from parameter geometry. This possible gap is a
hypothesis for positioning, not an established novelty claim or a requirement
that the first useful implementation contain interactions.
