# CSR Learning Readiness

- status: `passed`
- schema_version: `phase1_csr_learning_features_v1`
- baseline_id: `candidate_prior_success_median_v1`
- rows: `108`
- train_rows: `81`
- eval_rows: `27`
- train_matrices: `9`
- eval_matrices: `3`
- label_class_counts: `{'not_applicable': 34, 'not_profiled': 44, 'screened_out': 15, 'success_non_oracle': 11, 'success_oracle': 4}`
- eval_oracle_top1_accuracy: `0.333333`
- eval_profiled_success_rate: `0.666667`
- eval_mean_regret_ms: `2376.54`
- eval_max_regret_ms: `4753.09`

| matrix | predicted | oracle | status | regret_ms | reason |
|---|---|---|---|---:|---|
| suitesparse:FIDAP/ex5 | taichi_csr_pcg_jacobi_float64 | taichi_csr_pcg_jacobi_float64 | oracle_match | 0 | first_training_candidate_prior_with_eval_profiled_success |
| suitesparse:HB/curtis54 | taichi_csr_gmres_jacobi_restart16_float64 | taichi_csr_bicgstab_none_float64 | profiled_success_non_oracle | 4753.09 | first_training_candidate_prior_with_eval_profiled_success |
| suitesparse:Zitney/extr1b |  |  | no_profiled_success_candidate |  | no_training_prior_candidate_available_in_eval_success_rows |
