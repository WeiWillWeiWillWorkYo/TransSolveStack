# CSR Model Contract

- status: `passed`
- schema_version: `phase1_csr_model_contract_v1`
- prediction_source: `csr_learning_baseline_adapter_v1`
- model_required: `False`
- runtime_selector_changed: `False`
- requests: `12`
- targets: `12`
- predictions: `3`
- train_requests: `9`
- eval_requests: `3`
- request_candidates: `108`
- candidates_per_request: `9` to `9`
- target_label_class_counts: `{'not_applicable': 34, 'not_profiled': 44, 'screened_out': 15, 'success_non_oracle': 11, 'success_oracle': 4}`
- eval_oracle_top1_accuracy: `0.333333`
- eval_profiled_selection_rate: `0.666667`
- eval_mean_regret_ms: `2376.54`
- eval_max_regret_ms: `4753.09`
- validation_error_count: `0`

| matrix | selected | oracle | status | regret_ms | model_id |
|---|---|---|---|---:|---|
| suitesparse:FIDAP/ex5 | taichi_csr_pcg_jacobi_float64 | taichi_csr_pcg_jacobi_float64 | oracle_match | 0 | csr_learning_baseline_adapter_v1 |
| suitesparse:HB/curtis54 | taichi_csr_gmres_jacobi_restart16_float64 | taichi_csr_bicgstab_none_float64 | profiled_success_non_oracle | 4753.09 | csr_learning_baseline_adapter_v1 |
| suitesparse:Zitney/extr1b |  |  | no_profiled_success_candidate |  | csr_learning_baseline_adapter_v1 |
