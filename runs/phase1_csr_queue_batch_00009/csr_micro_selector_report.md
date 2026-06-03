# CSR Selector Readiness

- status: `passed`
- schema_version: `phase1_csr_selector_features_v8`
- diagnostic_rows: `8`
- selector_rows: `24`
- success_rows: `8`
- failed_rows: `16`
- applicability_rows: `0`
- oracle_rows: `4`
- matrices_with_selector_rows: `8`
- max_final_relative_residual: `5.64503e-06`
- max_cpu_recomputed_relative_residual: `5.64503e-06`
- max_solution_relative_error: `5.64503e-06`
- min_success_rate: `0`
- max_solve_time_iqr_ms: `0`
- failure_reason_counts: `{'cpu_reference_bicgstab_screen_failed': 4, 'cpu_reference_cg_screen_failed': 6, 'cpu_reference_gmres_screen_failed': 2, 'cpu_reference_pcg_screen_failed': 2, 'cpu_reference_richardson_screen_failed': 2}`
- applicability_status_counts: `{}`
- applicability_reason_counts: `{}`

| matrix | candidate | oracle | status | failure | applicability | reason | success_rate | repeats | median_ms | regret_ms | rel_res | sol_err |
|---|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|
| suitesparse:HB/bcsstm23 | taichi_csr_cg_none_float64 | no | screened_out | cpu_reference_cg_screen_failed | applicable |  | 0 | 1 |  |  | 0.000125777 | 0.107985 |
| suitesparse:HB/bcsstm23 | taichi_csr_pcg_jacobi_float64 | no | success |  | applicable |  | 1 | 1 | 804.592 | 322.01 | 0 | 0 |
| suitesparse:HB/bcsstm23 | taichi_csr_richardson_jacobi_float64 | yes | success |  | applicable |  | 1 | 1 | 482.582 | 0 | 5.64503e-06 | 5.64503e-06 |
| suitesparse:HB/bcsstm24 | taichi_csr_cg_none_float64 | no | screened_out | cpu_reference_cg_screen_failed | applicable |  | 0 | 1 |  |  | 6.10311e-05 | 0.243523 |
| suitesparse:HB/bcsstm24 | taichi_csr_pcg_jacobi_float64 | no | success |  | applicable |  | 1 | 1 | 1065.13 | 62.3663 | 0 | 0 |
| suitesparse:HB/bcsstm24 | taichi_csr_richardson_jacobi_float64 | yes | success |  | applicable |  | 1 | 1 | 1002.76 | 0 | 5.64503e-06 | 5.64503e-06 |
| suitesparse:HB/bcsstm25 | taichi_csr_cg_none_float64 | no | screened_out | cpu_reference_cg_screen_failed | applicable |  | 0 | 1 |  |  | 1.71332e-05 | 0.626442 |
| suitesparse:HB/bcsstm25 | taichi_csr_pcg_jacobi_float64 | no | success |  | applicable |  | 1 | 1 | 842.865 | 244.544 | 0 | 0 |
| suitesparse:HB/bcsstm25 | taichi_csr_richardson_jacobi_float64 | yes | success |  | applicable |  | 1 | 1 | 598.321 | 0 | 5.64503e-06 | 5.64503e-06 |
| suitesparse:HB/bcsstm26 | taichi_csr_cg_none_float64 | no | screened_out | cpu_reference_cg_screen_failed | applicable |  | 0 | 1 |  |  | 6.4667e-05 | 0.0752745 |
| suitesparse:HB/bcsstm26 | taichi_csr_pcg_jacobi_float64 | no | success |  | applicable |  | 1 | 1 | 605.401 | 249.156 | 0 | 0 |
| suitesparse:HB/bcsstm26 | taichi_csr_richardson_jacobi_float64 | yes | success |  | applicable |  | 1 | 1 | 356.244 | 0 | 5.64503e-06 | 5.64503e-06 |
| suitesparse:HB/bcsstm27 | taichi_csr_cg_none_float64 | no | screened_out | cpu_reference_cg_screen_failed | applicable |  | 0 | 1 |  |  | 0.825259 | 5.21415 |
| suitesparse:HB/bcsstm27 | taichi_csr_pcg_jacobi_float64 | no | screened_out | cpu_reference_pcg_screen_failed | applicable |  | 0 | 1 |  |  | 0.242063 | 2.54063 |
| suitesparse:HB/bcsstm27 | taichi_csr_richardson_jacobi_float64 | no | screened_out | cpu_reference_richardson_screen_failed | applicable |  | 0 | 1 |  |  | 0.0456137 | 1.39914 |
| suitesparse:HB/blckhole | taichi_csr_cg_none_float64 | no | screened_out | cpu_reference_cg_screen_failed | applicable |  | 0 | 1 |  |  | 0.0505377 | 0.137607 |
| suitesparse:HB/blckhole | taichi_csr_pcg_jacobi_float64 | no | screened_out | cpu_reference_pcg_screen_failed | applicable |  | 0 | 1 |  |  | 0.0505377 | 0.137607 |
| suitesparse:HB/blckhole | taichi_csr_richardson_jacobi_float64 | no | screened_out | cpu_reference_richardson_screen_failed | applicable |  | 0 | 1 |  |  | 7.58857e+09 | 1.61397e+10 |
| suitesparse:HB/bp_0 | taichi_csr_bicgstab_none_float64 | no | screened_out | cpu_reference_bicgstab_screen_failed | applicable |  | 0 | 1 |  |  | 68.5917 | 704.897 |
| suitesparse:HB/bp_0 | taichi_csr_bicgstab_jacobi_float64 | no | screened_out | cpu_reference_bicgstab_screen_failed | applicable |  | 0 | 1 |  |  | 68.5917 | 704.897 |
| suitesparse:HB/bp_0 | taichi_csr_gmres_jacobi_restart16_float64 | no | screened_out | cpu_reference_gmres_screen_failed | applicable |  | 0 | 1 |  |  | 0.808301 | 3.0399 |
| suitesparse:HB/bp_1000 | taichi_csr_bicgstab_none_float64 | no | screened_out | cpu_reference_bicgstab_screen_failed | applicable |  | 0 | 1 |  |  | 34.381 | 257.442 |
| suitesparse:HB/bp_1000 | taichi_csr_bicgstab_jacobi_float64 | no | screened_out | cpu_reference_bicgstab_screen_failed | applicable |  | 0 | 1 |  |  | 38.2616 | 144.054 |
| suitesparse:HB/bp_1000 | taichi_csr_gmres_jacobi_restart16_float64 | no | screened_out | cpu_reference_gmres_screen_failed | applicable |  | 0 | 1 |  |  | 0.685843 | 4.05122 |
