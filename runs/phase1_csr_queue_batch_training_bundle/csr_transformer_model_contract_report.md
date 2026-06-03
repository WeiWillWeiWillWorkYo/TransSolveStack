# CSR Model Contract

- status: `passed`
- schema_version: `phase1_csr_model_contract_v1`
- prediction_source: `csr_transformer_ready_baseline_adapter_v1`
- model_required: `False`
- runtime_selector_changed: `False`
- requests: `100`
- targets: `100`
- predictions: `25`
- train_requests: `75`
- eval_requests: `25`
- request_candidates: `372`
- candidates_per_request: `3` to `9`
- target_label_class_counts: `{'not_applicable': 34, 'not_profiled': 44, 'screened_out': 223, 'success_non_oracle': 39, 'success_oracle': 32}`
- eval_oracle_top1_accuracy: `0.08`
- eval_profiled_selection_rate: `0.16`
- eval_mean_regret_ms: `102.177`
- eval_max_regret_ms: `244.544`
- validation_error_count: `0`

| matrix | selected | oracle | status | regret_ms | model_id |
|---|---|---|---|---:|---|
| suitesparse:FIDAP/ex5 | taichi_csr_pcg_jacobi_float64 | taichi_csr_pcg_jacobi_float64 | oracle_match | 0 | csr_transformer_ready_baseline_adapter_v1 |
| suitesparse:HB/bcsstk10 | taichi_csr_pcg_jacobi_float64 | taichi_csr_pcg_jacobi_float64 | oracle_match | 0 | csr_transformer_ready_baseline_adapter_v1 |
| suitesparse:HB/bcsstm09 | taichi_csr_pcg_jacobi_float64 | taichi_csr_richardson_jacobi_float64 | profiled_success_non_oracle | 164.163 | csr_transformer_ready_baseline_adapter_v1 |
| suitesparse:HB/bcsstm13 |  |  | no_profiled_success_candidate |  | csr_transformer_ready_baseline_adapter_v1 |
| suitesparse:HB/bcsstm25 | taichi_csr_pcg_jacobi_float64 | taichi_csr_richardson_jacobi_float64 | profiled_success_non_oracle | 244.544 | csr_transformer_ready_baseline_adapter_v1 |
| suitesparse:HB/bcsstm27 |  |  | no_profiled_success_candidate |  | csr_transformer_ready_baseline_adapter_v1 |
| suitesparse:HB/blckhole |  |  | no_profiled_success_candidate |  | csr_transformer_ready_baseline_adapter_v1 |
| suitesparse:HB/bp_0 |  |  | no_profiled_success_candidate |  | csr_transformer_ready_baseline_adapter_v1 |
| suitesparse:HB/bp_1000 |  |  | no_profiled_success_candidate |  | csr_transformer_ready_baseline_adapter_v1 |
| suitesparse:HB/bp_1200 |  |  | no_profiled_success_candidate |  | csr_transformer_ready_baseline_adapter_v1 |
| suitesparse:HB/bp_1400 |  |  | no_profiled_success_candidate |  | csr_transformer_ready_baseline_adapter_v1 |
| suitesparse:HB/bp_1600 |  |  | no_profiled_success_candidate |  | csr_transformer_ready_baseline_adapter_v1 |
| suitesparse:HB/bp_200 |  |  | no_profiled_success_candidate |  | csr_transformer_ready_baseline_adapter_v1 |
| suitesparse:HB/bp_400 |  |  | no_profiled_success_candidate |  | csr_transformer_ready_baseline_adapter_v1 |
| suitesparse:HB/bp_600 |  |  | no_profiled_success_candidate |  | csr_transformer_ready_baseline_adapter_v1 |
| suitesparse:HB/bp_800 |  |  | no_profiled_success_candidate |  | csr_transformer_ready_baseline_adapter_v1 |
| suitesparse:HB/can_1054 |  |  | no_profiled_success_candidate |  | csr_transformer_ready_baseline_adapter_v1 |
| suitesparse:HB/fs_183_1 |  |  | no_profiled_success_candidate |  | csr_transformer_ready_baseline_adapter_v1 |
| suitesparse:HB/lshp1009 |  |  | no_profiled_success_candidate |  | csr_transformer_ready_baseline_adapter_v1 |
| suitesparse:HB/young3c |  |  | no_profiled_success_candidate |  | csr_transformer_ready_baseline_adapter_v1 |
| suitesparse:Hamrle/Hamrle1 |  |  | no_profiled_success_candidate |  | csr_transformer_ready_baseline_adapter_v1 |
| suitesparse:Negre/dendrimer |  |  | no_profiled_success_candidate |  | csr_transformer_ready_baseline_adapter_v1 |
| suitesparse:SNAP/email-Eu-core |  |  | no_profiled_success_candidate |  | csr_transformer_ready_baseline_adapter_v1 |
| suitesparse:Sandia/oscil_dcop_01 |  |  | no_profiled_success_candidate |  | csr_transformer_ready_baseline_adapter_v1 |
| suitesparse:Zitney/extr1b |  |  | no_profiled_success_candidate |  | csr_transformer_ready_baseline_adapter_v1 |
