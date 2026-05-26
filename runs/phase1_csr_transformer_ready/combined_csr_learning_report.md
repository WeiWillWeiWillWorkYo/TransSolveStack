# CSR Learning Readiness

- status: `passed`
- schema_version: `phase1_csr_learning_features_v1`
- baseline_id: `candidate_prior_success_median_v1`
- rows: `132`
- train_rows: `99`
- eval_rows: `33`
- train_matrices: `15`
- eval_matrices: `5`
- label_class_counts: `{'not_applicable': 34, 'not_profiled': 44, 'screened_out': 31, 'success_non_oracle': 15, 'success_oracle': 8}`
- eval_oracle_top1_accuracy: `0.6`
- eval_profiled_success_rate: `0.8`
- eval_mean_regret_ms: `27.4227`
- eval_max_regret_ms: `109.691`

| matrix | predicted | oracle | status | regret_ms | reason |
|---|---|---|---|---:|---|
| suitesparse:FIDAP/ex5 | taichi_csr_pcg_jacobi_float64 | taichi_csr_pcg_jacobi_float64 | oracle_match | 0 | first_training_candidate_prior_with_eval_profiled_success |
| suitesparse:HB/curtis54 | taichi_csr_bicgstab_none_float64 | taichi_csr_bicgstab_none_float64 | oracle_match | 0 | first_training_candidate_prior_with_eval_profiled_success |
| suitesparse:HB/jgl009 | taichi_csr_bicgstab_none_float64 | taichi_csr_gmres_jacobi_restart16_float64 | profiled_success_non_oracle | 109.691 | first_training_candidate_prior_with_eval_profiled_success |
| suitesparse:MathWorks/tomography | taichi_csr_gmres_jacobi_restart16_float64 | taichi_csr_gmres_jacobi_restart16_float64 | oracle_match | 0 | first_training_candidate_prior_with_eval_profiled_success |
| suitesparse:Zitney/extr1b |  |  | no_profiled_success_candidate |  | no_training_prior_candidate_available_in_eval_success_rows |
