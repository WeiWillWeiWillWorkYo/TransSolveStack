# CSR Micro-Campaign

- status: `failed`
- imported_matrices: `7`
- candidate_jobs: `14`
- gpu_success_rows: `0`
- cpu_screened_out_rows: `14`
- gpu_failed_rows: `0`
- selector_rows: `14`
- selector_oracle_rows: `0`
- max_final_relative_residual: `0`
- max_cpu_recomputed_relative_residual: `0`
- max_solution_relative_error: `0`

| matrix | candidate | status | solver | preconditioner | repeats | median_ms | iters | rel_res | sol_err |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| suitesparse:HB/bp_1200 | taichi_csr_gmres_jacobi_restart32_float64 | screened_out | gmres | jacobi | 1 |  | 300 | 0.666409 | 6.42368 |
| suitesparse:HB/bp_1200 | taichi_csr_gmres_jacobi_restart64_float64 | screened_out | gmres | jacobi | 1 |  | 300 | 0.527699 | 5.80837 |
| suitesparse:HB/bp_1400 | taichi_csr_gmres_jacobi_restart32_float64 | screened_out | gmres | jacobi | 1 |  | 300 | 0.635971 | 4.01777 |
| suitesparse:HB/bp_1400 | taichi_csr_gmres_jacobi_restart64_float64 | screened_out | gmres | jacobi | 1 |  | 300 | 0.527041 | 4.72344 |
| suitesparse:HB/bp_1600 | taichi_csr_gmres_jacobi_restart32_float64 | screened_out | gmres | jacobi | 1 |  | 300 | 0.66306 | 3.06277 |
| suitesparse:HB/bp_1600 | taichi_csr_gmres_jacobi_restart64_float64 | screened_out | gmres | jacobi | 1 |  | 300 | 0.550223 | 9.11372 |
| suitesparse:HB/bp_200 | taichi_csr_gmres_jacobi_restart32_float64 | screened_out | gmres | jacobi | 1 |  | 300 | 0.703558 | 25.4326 |
| suitesparse:HB/bp_200 | taichi_csr_gmres_jacobi_restart64_float64 | screened_out | gmres | jacobi | 1 |  | 300 | 0.438621 | 23.4343 |
| suitesparse:HB/bp_400 | taichi_csr_gmres_jacobi_restart32_float64 | screened_out | gmres | jacobi | 1 |  | 300 | 0.601882 | 12.7312 |
| suitesparse:HB/bp_400 | taichi_csr_gmres_jacobi_restart64_float64 | screened_out | gmres | jacobi | 1 |  | 300 | 0.447162 | 6.93138 |
| suitesparse:HB/bp_600 | taichi_csr_gmres_jacobi_restart32_float64 | screened_out | gmres | jacobi | 1 |  | 300 | 0.656484 | 6.92126 |
| suitesparse:HB/bp_600 | taichi_csr_gmres_jacobi_restart64_float64 | screened_out | gmres | jacobi | 1 |  | 300 | 0.461241 | 7.89127 |
| suitesparse:HB/bp_800 | taichi_csr_gmres_jacobi_restart32_float64 | screened_out | gmres | jacobi | 1 |  | 300 | 0.650069 | 4.43349 |
| suitesparse:HB/bp_800 | taichi_csr_gmres_jacobi_restart64_float64 | screened_out | gmres | jacobi | 1 |  | 300 | 0.548828 | 4.0117 |
