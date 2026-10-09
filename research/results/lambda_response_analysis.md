# Uniform-Lambda Response Analysis

This is a diagnostic fit to the existing public-test sweep, not a source of
final hyperparameters. Five-expert Task Arithmetic was evaluated at
`lambda = {0.4, 0.6, 0.8, 1.0, 1.2}` with `n=16`.

For the completed training-pilot sweep and current method decisions, see
[current findings](current_findings.md). The public sweep below has already
influenced method selection and is development evidence.

Quadratic fits over the five points give approximate task optima:

| Benchmark | Fitted optimum | Best observed grid point |
|---|---:|---:|
| AIME 2025 | 0.805 | 1.0 |
| AIME 2026 | 0.863 | 0.8 |
| GPQA | 0.880 | 1.0 |
| LiveCodeBench v5 | 1.068 | 1.0 |
| LiveCodeBench v6 | 0.878 | 0.8 |
| IFEval | 0.979 | 1.0 |
| IFBench | at least 1.2 | 1.2 |
| BFCL v3 | 0.904 | 0.8/1.0 |

The exploratory evidence suggests a low-dimensional parameterization with a shared scale
near `0.9–1.0` and an IF-specific residual scale above it. It does not justify
independent layer-wise coefficients: those would greatly increase search
dimension without current evidence.

Response length rises monotonically in aggregate from 3,203 tokens at
`lambda=0.4` to 4,595 at `lambda=1.2`. LiveCodeBench v5 accuracy and length are
strongly correlated across this five-point sweep (`r=0.91`), while IFBench
accuracy increases as response length decreases (`r=-0.96`). Accuracy-only
selection can therefore confound capability with generation policy; the
calibration objective must include length and truncation diagnostics.

The training-pilot lambda=0.8/1.0 comparison has an uncertain mean reward gap
and roughly 9.4% fewer tokens at 0.8. Keep calibrated uniform TA as a
cost-aware baseline rather than assuming lambda=1 is optimal under every
objective. Neither uniform sweep identifies separate expert effects.
