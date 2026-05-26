# CSR Model Contract

- status: `passed`
- schema_version: `phase1_csr_model_contract_v1`
- prediction_source: `csr_transformer_ready_baseline_adapter_v1`
- model_required: `False`
- runtime_selector_changed: `False`
- requests: `20`
- targets: `20`
- predictions: `5`
- train_requests: `15`
- eval_requests: `5`
- request_candidates: `132`
- candidates_per_request: `3` to `9`
- target_label_class_counts: `{'not_applicable': 34, 'not_profiled': 44, 'screened_out': 31, 'success_non_oracle': 15, 'success_oracle': 8}`
- eval_oracle_top1_accuracy: `0.6`
- eval_profiled_selection_rate: `0.8`
- eval_mean_regret_ms: `27.4227`
- eval_max_regret_ms: `109.691`
- validation_error_count: `0`

| matrix | selected | oracle | status | regret_ms | model_id |
|---|---|---|---|---:|---|
| suitesparse:FIDAP/ex5 | taichi_csr_pcg_jacobi_float64 | taichi_csr_pcg_jacobi_float64 | oracle_match | 0 | csr_transformer_ready_baseline_adapter_v1 |
| suitesparse:HB/curtis54 | taichi_csr_bicgstab_none_float64 | taichi_csr_bicgstab_none_float64 | oracle_match | 0 | csr_transformer_ready_baseline_adapter_v1 |
| suitesparse:HB/jgl009 | taichi_csr_bicgstab_none_float64 | taichi_csr_gmres_jacobi_restart16_float64 | profiled_success_non_oracle | 109.691 | csr_transformer_ready_baseline_adapter_v1 |
| suitesparse:MathWorks/tomography | taichi_csr_gmres_jacobi_restart16_float64 | taichi_csr_gmres_jacobi_restart16_float64 | oracle_match | 0 | csr_transformer_ready_baseline_adapter_v1 |
| suitesparse:Zitney/extr1b |  |  | no_profiled_success_candidate |  | csr_transformer_ready_baseline_adapter_v1 |
