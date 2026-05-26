# CSR Training Tensor Export

- status: `passed`
- schema_version: `phase1_csr_training_tensors_v1`
- source_contract_schema_version: `phase1_csr_model_contract_v1`
- storage_format: `json_numeric_arrays_v1`
- requests: `12`
- train_requests: `9`
- eval_requests: `3`
- global_candidates: `9`
- active_candidate_slots: `108`
- matrix_feature_dim: `21`
- candidate_feature_dim: `16`
- oracle_targets: `4`
- requests_without_oracle: `8`
- label_class_counts: `{'not_applicable': 34, 'not_profiled': 44, 'screened_out': 15, 'success_non_oracle': 11, 'success_oracle': 4}`
- target_status_counts: `{'not_applicable': 34, 'not_profiled': 44, 'screened_out': 15, 'success': 15}`
- has_missing_candidate_slots: `False`
- model_required: `False`
- runtime_selector_changed: `False`

| row | split | matrix | oracle_index | active_candidates |
|---:|---|---|---:|---:|
| 0 | eval | suitesparse:FIDAP/ex5 | 7 | 9 |
| 1 | eval | suitesparse:HB/curtis54 | 1 | 9 |
| 2 | eval | suitesparse:Zitney/extr1b | -1 | 9 |
| 3 | train | suitesparse:Bai/cdde1 | -1 | 9 |
| 4 | train | suitesparse:Goodwin/Goodwin_010 | -1 | 9 |
| 5 | train | suitesparse:Grund/b1_ss | 4 | 9 |
| 6 | train | suitesparse:HB/fs_183_1 | -1 | 9 |
| 7 | train | suitesparse:HB/young3c | -1 | 9 |
| 8 | train | suitesparse:Hamrle/Hamrle1 | -1 | 9 |
| 9 | train | suitesparse:JGD_Trefethen/Trefethen_20b | 3 | 9 |
| 10 | train | suitesparse:Negre/dendrimer | -1 | 9 |
| 11 | train | suitesparse:Sandia/oscil_dcop_01 | -1 | 9 |
