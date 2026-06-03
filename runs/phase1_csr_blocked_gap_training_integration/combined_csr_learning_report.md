# CSR Learning Readiness

- status: `passed`
- schema_version: `phase1_csr_learning_features_v1`
- baseline_id: `candidate_prior_success_median_v1`
- rows: `380`
- train_rows: `261`
- eval_rows: `119`
- train_matrices: `74`
- eval_matrices: `25`
- label_class_counts: `{'not_applicable': 34, 'not_profiled': 44, 'screened_out': 223, 'success_non_oracle': 42, 'success_oracle': 37}`
- eval_oracle_top1_accuracy: `0`
- eval_profiled_success_rate: `0.16`
- eval_mean_regret_ms: `86.0838`
- eval_max_regret_ms: `260.885`

| matrix | predicted | oracle | status | regret_ms | reason |
|---|---|---|---|---:|---|
| suitesparse:FIDAP/ex5 | taichi_csr_pcg_jacobi_float64 | taichi_csr_bicgstab_ilu0_float64 | profiled_success_non_oracle | -239.456 | first_training_candidate_prior_with_eval_profiled_success |
| suitesparse:HB/bcsstk09 | taichi_csr_pcg_jacobi_float64 | taichi_csr_cg_none_float64 | profiled_success_non_oracle | 260.539 | first_training_candidate_prior_with_eval_profiled_success |
| suitesparse:HB/bcsstm08 | taichi_csr_pcg_jacobi_float64 | taichi_csr_richardson_jacobi_float64 | profiled_success_non_oracle | 260.885 | first_training_candidate_prior_with_eval_profiled_success |
| suitesparse:HB/bcsstm13 |  |  | no_profiled_success_candidate |  | no_training_prior_candidate_available_in_eval_success_rows |
| suitesparse:HB/bcsstm24 | taichi_csr_pcg_jacobi_float64 | taichi_csr_richardson_jacobi_float64 | profiled_success_non_oracle | 62.3663 | first_training_candidate_prior_with_eval_profiled_success |
| suitesparse:HB/bcsstm27 |  |  | no_profiled_success_candidate |  | no_training_prior_candidate_available_in_eval_success_rows |
| suitesparse:HB/blckhole |  |  | no_profiled_success_candidate |  | no_training_prior_candidate_available_in_eval_success_rows |
| suitesparse:HB/bp_0 |  |  | no_profiled_success_candidate |  | no_training_prior_candidate_available_in_eval_success_rows |
| suitesparse:HB/bp_1000 |  |  | no_profiled_success_candidate |  | no_training_prior_candidate_available_in_eval_success_rows |
| suitesparse:HB/bp_1200 |  |  | no_profiled_success_candidate |  | no_training_prior_candidate_available_in_eval_success_rows |
| suitesparse:HB/bp_1400 |  |  | no_profiled_success_candidate |  | no_training_prior_candidate_available_in_eval_success_rows |
| suitesparse:HB/bp_1600 |  |  | no_profiled_success_candidate |  | no_training_prior_candidate_available_in_eval_success_rows |
| suitesparse:HB/bp_200 |  |  | no_profiled_success_candidate |  | no_training_prior_candidate_available_in_eval_success_rows |
| suitesparse:HB/bp_400 |  |  | no_profiled_success_candidate |  | no_training_prior_candidate_available_in_eval_success_rows |
| suitesparse:HB/bp_600 |  |  | no_profiled_success_candidate |  | no_training_prior_candidate_available_in_eval_success_rows |
| suitesparse:HB/bp_800 |  |  | no_profiled_success_candidate |  | no_training_prior_candidate_available_in_eval_success_rows |
| suitesparse:HB/can_1054 |  |  | no_profiled_success_candidate |  | no_training_prior_candidate_available_in_eval_success_rows |
| suitesparse:HB/fs_183_1 |  |  | no_profiled_success_candidate |  | no_training_prior_candidate_available_in_eval_success_rows |
| suitesparse:HB/lshp1009 |  |  | no_profiled_success_candidate |  | no_training_prior_candidate_available_in_eval_success_rows |
| suitesparse:HB/young3c |  |  | no_profiled_success_candidate |  | no_training_prior_candidate_available_in_eval_success_rows |
| suitesparse:Hamrle/Hamrle1 |  |  | no_profiled_success_candidate |  | no_training_prior_candidate_available_in_eval_success_rows |
| suitesparse:Negre/dendrimer |  |  | no_profiled_success_candidate |  | no_training_prior_candidate_available_in_eval_success_rows |
| suitesparse:SNAP/email-Eu-core |  |  | no_profiled_success_candidate |  | no_training_prior_candidate_available_in_eval_success_rows |
| suitesparse:Sandia/oscil_dcop_01 |  |  | no_profiled_success_candidate |  | no_training_prior_candidate_available_in_eval_success_rows |
| suitesparse:Zitney/extr1b |  |  | no_profiled_success_candidate |  | no_training_prior_candidate_available_in_eval_success_rows |
