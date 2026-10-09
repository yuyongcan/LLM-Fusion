# IF Module-Group Ablation

All models start from five-expert Task Arithmetic with coefficient `1.0`.
Exactly one module group receives an additional `0.2 * tau_IF`:

- `if_embedding_1.2`
- `if_attention_1.2`
- `if_mlp_1.2`

Compare against the existing TA `lambda=1` model and the existing global
per-expert endpoint with IF coefficient `1.2`. The experiment tests whether the
IF gain can be localized to a task-specific module group rather than requiring
a global expert scalar.

## Recorded outcome

Only `if_embedding_1.2` has saved evaluation results here. Its reasoning AVG5
at n=4 is 63.09, versus TA's 64.50; the paired difference is -1.42 pp with
95% CI [-4.08, +1.08]. No IF or Agent evaluation is present, so this screen
does not answer whether the correction improves IF or preserves Agent. No
attention/MLP evaluation is recorded here.

Treat this as an incomplete exploratory ablation. The embedding group in the
merger includes `embed_tokens` and `lm_head` when present. Direct all-domain
evaluation and comparison to calibrated expert scalars are required before
selecting any module correction.
