# CSR Training Tensor Export

- status: `passed`
- schema_version: `phase1_csr_training_tensors_v1`
- source_contract_schema_version: `phase1_csr_model_contract_v1`
- storage_format: `json_numeric_arrays_v1`
- requests: `20`
- train_requests: `15`
- eval_requests: `5`
- global_candidates: `9`
- active_candidate_slots: `132`
- matrix_feature_dim: `21`
- candidate_feature_dim: `16`
- oracle_targets: `8`
- requests_without_oracle: `12`
- label_class_counts: `{'not_applicable': 34, 'not_profiled': 44, 'screened_out': 31, 'success_non_oracle': 15, 'success_oracle': 8}`
- target_status_counts: `{'not_applicable': 34, 'not_profiled': 44, 'screened_out': 31, 'success': 23}`
- has_missing_candidate_slots: `True`
- model_required: `False`
- runtime_selector_changed: `False`

| row | split | matrix | oracle_index | active_candidates |
|---:|---|---|---:|---:|
| 0 | eval | suitesparse:FIDAP/ex5 | 7 | 9 |
| 1 | eval | suitesparse:HB/curtis54 | 1 | 9 |
| 2 | eval | suitesparse:HB/jgl009 | 4 | 3 |
| 3 | eval | suitesparse:MathWorks/tomography | 4 | 3 |
| 4 | eval | suitesparse:Zitney/extr1b | -1 | 9 |
| 5 | train | suitesparse:Bai/cdde1 | -1 | 9 |
| 6 | train | suitesparse:Goodwin/Goodwin_010 | -1 | 9 |
| 7 | train | suitesparse:Grund/b1_ss | 4 | 9 |
| 8 | train | suitesparse:Gset/G17 | -1 | 3 |
| 9 | train | suitesparse:HB/bcsstk07 | -1 | 3 |
| 10 | train | suitesparse:HB/fs_183_1 | -1 | 9 |
| 11 | train | suitesparse:HB/ibm32 | 1 | 3 |
| 12 | train | suitesparse:HB/lshp1009 | -1 | 3 |
| 13 | train | suitesparse:HB/young3c | -1 | 9 |
| 14 | train | suitesparse:Hamrle/Hamrle1 | -1 | 9 |
| 15 | train | suitesparse:JGD_Trefethen/Trefethen_20b | 3 | 9 |
| 16 | train | suitesparse:Negre/dendrimer | -1 | 9 |
| 17 | train | suitesparse:Oberwolfach/t2dal_e | 8 | 3 |
| 18 | train | suitesparse:SNAP/email-Eu-core | -1 | 3 |
| 19 | train | suitesparse:Sandia/oscil_dcop_01 | -1 | 9 |
