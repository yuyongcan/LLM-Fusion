# IF × Agent Factorial Confirmation

This experiment repeats the exploratory four-cell comparison at `n=16` on
AIME 2025, AIME 2026, and GPQA, the benchmarks where the earlier `n=4`
interaction appeared.

All models use Task Arithmetic coefficient `1.0`:

- `mcs_lambda1`: Math + Science + Code.
- `mcsi_lambda1`: MCS + IF.
- `mcsa_lambda1`: MCS + Agent.
- `five_lambda1`: MCS + IF + Agent.

The interaction contrast is

`five - mcsi - mcsa + mcs`.

This is a higher-sample diagnostic on existing public prompts, not independent
confirmation. The three benchmarks were chosen after the n=4 screen. Do not
use their results to select final method coefficients.

The unified reanalysis gives interaction +1.06 pp with 95% CI [-1.26, +3.54].
All5-MCS is +1.64 pp with CI [-0.62, +4.02]; neither is a confirmed gain.
See [recomputed evidence](../../results/20260917_reassessment/tables.md).
