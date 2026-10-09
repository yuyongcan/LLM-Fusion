# Functional Geometry Diagnostic

Completed pilot analysis of `Delta W x` on base-model activations. Recorded
settings: 8 prompts/domain, 64 tokens/domain, maximum prompt length 1024;
252 transformer linear modules. No prompt offset is recorded in the JSON.
The metric omits embedding and normalization perturbations and does not
propagate each perturbation to final logits or verifier rewards.

`functional_vs_transfer.csv` joins the target-domain functional cosine to
existing directed expert transfer on public evaluation data. Pearson r is
0.5231 versus 0.3887 for global weight cosine across 20 directed pairs. Removing
IF->Agent reverses that ordering (0.4973 versus 0.5701); removing all IF pairs
gives 0.3671 versus 0.5855. Pairs share experts and domains and are not
independent samples. The diagnostic does not establish predictive superiority
or a useful merge direction.

Reproduce sensitivity checks with
`python3 research/experiments/reassess_existing_results.py`; see
[current findings](../current_findings.md). Before selecting components from
this metric, repeat on disjoint prompt batches and check prediction on reserved
domains/pairs. More geometry work should be tied to a concrete method decision.
