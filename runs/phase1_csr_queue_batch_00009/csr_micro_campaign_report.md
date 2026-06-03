# CSR Micro-Campaign

- status: `passed`
- imported_matrices: `8`
- candidate_jobs: `24`
- gpu_success_rows: `8`
- cpu_screened_out_rows: `16`
- gpu_failed_rows: `0`
- selector_rows: `24`
- selector_oracle_rows: `4`
- max_final_relative_residual: `5.64503e-06`
- max_cpu_recomputed_relative_residual: `5.64503e-06`
- max_solution_relative_error: `5.64503e-06`

| matrix | candidate | status | solver | preconditioner | repeats | median_ms | iters | rel_res | sol_err |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| suitesparse:HB/bcsstm23 | taichi_csr_cg_none_float64 | screened_out | cg | none | 1 |  | 300 | 0.000125777 | 0.107985 |
| suitesparse:HB/bcsstm23 | taichi_csr_pcg_jacobi_float64 | success | pcg | jacobi | 1 | 804.592 | 1 | 0 | 0 |
| suitesparse:HB/bcsstm23 | taichi_csr_richardson_jacobi_float64 | success | richardson | jacobi | 1 | 482.582 | 11 | 5.64503e-06 | 5.64503e-06 |
| suitesparse:HB/bcsstm24 | taichi_csr_cg_none_float64 | screened_out | cg | none | 1 |  | 300 | 6.10311e-05 | 0.243523 |
| suitesparse:HB/bcsstm24 | taichi_csr_pcg_jacobi_float64 | success | pcg | jacobi | 1 | 1065.13 | 1 | 0 | 0 |
| suitesparse:HB/bcsstm24 | taichi_csr_richardson_jacobi_float64 | success | richardson | jacobi | 1 | 1002.76 | 11 | 5.64503e-06 | 5.64503e-06 |
| suitesparse:HB/bcsstm25 | taichi_csr_cg_none_float64 | screened_out | cg | none | 1 |  | 300 | 1.71332e-05 | 0.626442 |
| suitesparse:HB/bcsstm25 | taichi_csr_pcg_jacobi_float64 | success | pcg | jacobi | 1 | 842.865 | 1 | 0 | 0 |
| suitesparse:HB/bcsstm25 | taichi_csr_richardson_jacobi_float64 | success | richardson | jacobi | 1 | 598.321 | 11 | 5.64503e-06 | 5.64503e-06 |
| suitesparse:HB/bcsstm26 | taichi_csr_cg_none_float64 | screened_out | cg | none | 1 |  | 300 | 6.4667e-05 | 0.0752745 |
| suitesparse:HB/bcsstm26 | taichi_csr_pcg_jacobi_float64 | success | pcg | jacobi | 1 | 605.401 | 1 | 0 | 0 |
| suitesparse:HB/bcsstm26 | taichi_csr_richardson_jacobi_float64 | success | richardson | jacobi | 1 | 356.244 | 11 | 5.64503e-06 | 5.64503e-06 |
| suitesparse:HB/bcsstm27 | taichi_csr_cg_none_float64 | screened_out | cg | none | 1 |  | 62 | 0.825259 | 5.21415 |
| suitesparse:HB/bcsstm27 | taichi_csr_pcg_jacobi_float64 | screened_out | pcg | jacobi | 1 |  | 20 | 0.242063 | 2.54063 |
| suitesparse:HB/bcsstm27 | taichi_csr_richardson_jacobi_float64 | screened_out | richardson | jacobi | 1 |  | 300 | 0.0456137 | 1.39914 |
| suitesparse:HB/blckhole | taichi_csr_cg_none_float64 | screened_out | cg | none | 1 |  | 2 | 0.0505377 | 0.137607 |
| suitesparse:HB/blckhole | taichi_csr_pcg_jacobi_float64 | screened_out | pcg | jacobi | 1 |  | 2 | 0.0505377 | 0.137607 |
| suitesparse:HB/blckhole | taichi_csr_richardson_jacobi_float64 | screened_out | richardson | jacobi | 1 |  | 300 | 7.58857e+09 | 1.61397e+10 |
| suitesparse:HB/bp_0 | taichi_csr_bicgstab_none_float64 | screened_out | bicgstab | none | 1 |  | 300 | 68.5917 | 704.897 |
| suitesparse:HB/bp_0 | taichi_csr_bicgstab_jacobi_float64 | screened_out | bicgstab | jacobi | 1 |  | 300 | 68.5917 | 704.897 |
| suitesparse:HB/bp_0 | taichi_csr_gmres_jacobi_restart16_float64 | screened_out | gmres | jacobi | 1 |  | 300 | 0.808301 | 3.0399 |
| suitesparse:HB/bp_1000 | taichi_csr_bicgstab_none_float64 | screened_out | bicgstab | none | 1 |  | 300 | 34.381 | 257.442 |
| suitesparse:HB/bp_1000 | taichi_csr_bicgstab_jacobi_float64 | screened_out | bicgstab | jacobi | 1 |  | 300 | 38.2616 | 144.054 |
| suitesparse:HB/bp_1000 | taichi_csr_gmres_jacobi_restart16_float64 | screened_out | gmres | jacobi | 1 |  | 300 | 0.685843 | 4.05122 |
