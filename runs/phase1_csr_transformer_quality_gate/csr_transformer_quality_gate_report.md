# CSR Selector Model Evaluation

- status: `passed`
- schema_version: `phase1_csr_selector_model_eval_v1`
- evaluation_id: `csr_selector_model_quality_gate_v1`
- baseline_model_id: `candidate_prior_success_median_v1`
- challenger_model_id: `csr_masked_self_attention_ranker_v1`
- best_offline_model_id: `candidate_prior_success_median_v1`
- runtime_selected_model_id: ``
- runtime_selector_changed: `False`
- num_eval_predictions: `5`
- num_eval_oracle_requests: `4`
- min_required_eval_oracle_requests: `4`
- min_runtime_oracle_top1_accuracy: `0.5`
- min_runtime_profiled_success_rate: `0.8`
- challenger_beats_baseline: `False`
- challenger_runtime_eligible: `False`
- challenger_gate_failures: `['below_min_profiled_success_rate', 'below_baseline_oracle_top1', 'below_baseline_profiled_success_rate', 'non_success_eval_selections']`
- recommendation: `keep_runtime_artifact_backed_and_expand_benchmark_coverage`

| role | model | oracle_top1 | success_rate | non_success | runtime_eligible | failures |
|---|---|---:|---:|---:|---|---|
| baseline | candidate_prior_success_median_v1 | 0.6 | 0.8 | 1 | False | comparison_baseline_not_runtime_candidate |
| challenger | csr_masked_self_attention_ranker_v1 | 0.5 | 0.6 | 2 | False | below_min_profiled_success_rate, below_baseline_oracle_top1, below_baseline_profiled_success_rate, non_success_eval_selections |
