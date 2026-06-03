# CSR Micro-Campaign

- status: `failed`
- imported_matrices: `8`
- candidate_jobs: `24`
- gpu_success_rows: `0`
- cpu_screened_out_rows: `24`
- gpu_failed_rows: `0`
- selector_rows: `24`
- selector_oracle_rows: `0`
- max_final_relative_residual: `0`
- max_cpu_recomputed_relative_residual: `0`
- max_solution_relative_error: `0`

| matrix | candidate | status | solver | preconditioner | repeats | median_ms | iters | rel_res | sol_err |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| suitesparse:HB/bp_1200 | taichi_csr_bicgstab_none_float64 | screened_out | bicgstab | none | 1 |  | 300 | 2255.21 | 21605.9 |
| suitesparse:HB/bp_1200 | taichi_csr_bicgstab_jacobi_float64 | screened_out | bicgstab | jacobi | 1 |  | 300 | 66.2673 | 404.836 |
| suitesparse:HB/bp_1200 | taichi_csr_gmres_jacobi_restart16_float64 | screened_out | gmres | jacobi | 1 |  | 300 | 0.849478 | 2.52897 |
| suitesparse:HB/bp_1400 | taichi_csr_bicgstab_none_float64 | screened_out | bicgstab | none | 1 |  | 300 | 508.467 | 1551.01 |
| suitesparse:HB/bp_1400 | taichi_csr_bicgstab_jacobi_float64 | screened_out | bicgstab | jacobi | 1 |  | 300 | 18.045 | 101.705 |
| suitesparse:HB/bp_1400 | taichi_csr_gmres_jacobi_restart16_float64 | screened_out | gmres | jacobi | 1 |  | 300 | 0.807191 | 3.1923 |
| suitesparse:HB/bp_1600 | taichi_csr_bicgstab_none_float64 | screened_out | bicgstab | none | 1 |  | 300 | 36.0645 | 165.728 |
| suitesparse:HB/bp_1600 | taichi_csr_bicgstab_jacobi_float64 | screened_out | bicgstab | jacobi | 1 |  | 300 | 11.7315 | 48.1003 |
| suitesparse:HB/bp_1600 | taichi_csr_gmres_jacobi_restart16_float64 | screened_out | gmres | jacobi | 1 |  | 300 | 0.81715 | 3.03109 |
| suitesparse:HB/bp_200 | taichi_csr_bicgstab_none_float64 | screened_out | bicgstab | none | 1 |  | 300 | 106.693 | 714.358 |
| suitesparse:HB/bp_200 | taichi_csr_bicgstab_jacobi_float64 | screened_out | bicgstab | jacobi | 1 |  | 300 | 134.205 | 491.131 |
| suitesparse:HB/bp_200 | taichi_csr_gmres_jacobi_restart16_float64 | screened_out | gmres | jacobi | 1 |  | 300 | 0.827915 | 4.51018 |
| suitesparse:HB/bp_400 | taichi_csr_bicgstab_none_float64 | screened_out | bicgstab | none | 1 |  | 300 | 284.689 | 2545.45 |
| suitesparse:HB/bp_400 | taichi_csr_bicgstab_jacobi_float64 | screened_out | bicgstab | jacobi | 1 |  | 300 | 708.369 | 4880.63 |
| suitesparse:HB/bp_400 | taichi_csr_gmres_jacobi_restart16_float64 | screened_out | gmres | jacobi | 1 |  | 300 | 0.739576 | 4.86005 |
| suitesparse:HB/bp_600 | taichi_csr_bicgstab_none_float64 | screened_out | bicgstab | none | 1 |  | 300 | 70.8259 | 265.062 |
| suitesparse:HB/bp_600 | taichi_csr_bicgstab_jacobi_float64 | screened_out | bicgstab | jacobi | 1 |  | 300 | 30.9947 | 177.055 |
| suitesparse:HB/bp_600 | taichi_csr_gmres_jacobi_restart16_float64 | screened_out | gmres | jacobi | 1 |  | 300 | 0.823465 | 4.26118 |
| suitesparse:HB/bp_800 | taichi_csr_bicgstab_none_float64 | screened_out | bicgstab | none | 1 |  | 300 | 921.696 | 4073.42 |
| suitesparse:HB/bp_800 | taichi_csr_bicgstab_jacobi_float64 | screened_out | bicgstab | jacobi | 1 |  | 300 | 101.811 | 698.342 |
| suitesparse:HB/bp_800 | taichi_csr_gmres_jacobi_restart16_float64 | screened_out | gmres | jacobi | 1 |  | 300 | 0.809933 | 2.11101 |
| suitesparse:HB/can_1054 | taichi_csr_cg_none_float64 | screened_out | cg | none | 1 |  | 2 | 0.0821189 | 0.498182 |
| suitesparse:HB/can_1054 | taichi_csr_pcg_jacobi_float64 | screened_out | pcg | jacobi | 1 |  | 2 | 0.0821189 | 0.498182 |
| suitesparse:HB/can_1054 | taichi_csr_richardson_jacobi_float64 | screened_out | richardson | jacobi | 1 |  | 300 | 8.23578e+14 | 2.34717e+15 |
