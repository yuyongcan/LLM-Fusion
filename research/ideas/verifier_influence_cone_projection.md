# Verifier Influence Cone Projection (VICP)

Status: active hypothesis; not yet validated. The scalar axial experiment is a
diagnostic/control, not the paper contribution.

## Claim to test

RLVR experts are not interchangeable task vectors. A signed perturbation of one
expert can improve one verifier while harming another, and the effect is only
partly visible in parameter geometry. We therefore estimate a **verifier
influence vector** for each merge component and retain only directions whose
uncertainty-adjusted influence lies in a robust multi-domain improvement cone.

For base parameters `theta_0`, component updates `u_j` and coefficients `a`,

`theta(a) = theta_0 + sum_j a_j u_j`.

At the TA anchor `a=1`, evaluate signed perturbations `a +/- h e_j` on the
same prompts. For domain `d`, the first-order influence is

`G[d,j] = (R_d(1+h e_j) - R_d(1-h e_j)) / (2h)`.

The measured object is a vector over native verifiers, not a weight cosine or
an unlabeled entropy proxy. Bootstrap intervals are computed on prompt clusters
so duplicate inputs cannot create fake precision. Generation length and cap-hit
rate are additional influence coordinates with a budget constraint.

## Robust cone projection

Let `L[d,j]` be a lower confidence bound for `G[d,j]`, and `q[j]` the signed
cost influence. We solve a small linear robust projection around the anchor:

`max_delta,t  t - lambda ||delta||_2^2`

subject to

`L[d,:] delta - lambda_cost q delta >= t` for every domain `d`,
`q delta <= budget`, `-rho <= delta_j <= rho`, and `sum(delta_j^2) <= rho^2`.

The checkpoint is exported once after this projection. The cone is not allowed
to trade an uncertain gain in one domain for a point estimate loss in another.
The selected direction must predict prespecified mixed settings that were not
used to estimate `G`; poor held-out prediction falsifies the mechanism.

## Required distinction from coefficient tuning

Five scalar coefficients are a control because they use the same additive
parameterization as Task Arithmetic. VICP earns a distinct contribution only if
all of the following hold:

1. influence vectors predict held-out mixed perturbations better than a scalar
   response model with no verifier structure;
2. the confidence-cone projection improves the weakest domain at matched budget;
3. the gain survives a frozen confirmation set and decoding repetition; and
4. removing signed verifier influence, uncertainty bounds, or the worst-domain
   constraint removes the gain.

If these tests fail, report the method as a constrained calibration baseline and
do not claim a new merge algorithm.

## Component progression

The first experiment uses five expert updates because it is cheap and tests
whether verifier influence is useful at all. Only after successful held-out
prediction may the basis expand to expert x module groups or a geometry-
nominated low-rank basis. Expanding the basis before that test would be a
high-dimensional coefficient search and would not support the proposed claim.
