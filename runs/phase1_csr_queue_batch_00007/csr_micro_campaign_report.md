# CSR Micro-Campaign

- status: `passed`
- imported_matrices: `8`
- candidate_jobs: `24`
- gpu_success_rows: `16`
- cpu_screened_out_rows: `8`
- gpu_failed_rows: `0`
- selector_rows: `24`
- selector_oracle_rows: `6`
- max_final_relative_residual: `9.97184e-06`
- max_cpu_recomputed_relative_residual: `9.97184e-06`
- max_solution_relative_error: `0.00071122`

| matrix | candidate | status | solver | preconditioner | repeats | median_ms | iters | rel_res | sol_err |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| suitesparse:HB/bcsstm02 | taichi_csr_cg_none_float64 | success | cg | none | 1 | 993.57 | 12 | 2.07415e-08 | 1.21438e-08 |
| suitesparse:HB/bcsstm02 | taichi_csr_pcg_jacobi_float64 | success | pcg | jacobi | 1 | 579.495 | 1 | 0 | 0 |
| suitesparse:HB/bcsstm02 | taichi_csr_richardson_jacobi_float64 | success | richardson | jacobi | 1 | 637.509 | 11 | 5.64503e-06 | 5.64503e-06 |
| suitesparse:HB/bcsstm03 | taichi_csr_cg_none_float64 | screened_out | cg | none | 1 |  | 27 | 8.53849e-06 | 0.598443 |
| suitesparse:HB/bcsstm03 | taichi_csr_pcg_jacobi_float64 | screened_out | pcg | jacobi | 1 |  | 1 | 0 | 0.597614 |
| suitesparse:HB/bcsstm03 | taichi_csr_richardson_jacobi_float64 | screened_out | richardson | jacobi | 1 |  |  | inf | inf |
| suitesparse:HB/bcsstm04 | taichi_csr_cg_none_float64 | screened_out | cg | none | 1 |  | 12 | 1.9084e-16 | 0.707107 |
| suitesparse:HB/bcsstm04 | taichi_csr_pcg_jacobi_float64 | screened_out | pcg | jacobi | 1 |  | 1 | 0 | 0.707107 |
| suitesparse:HB/bcsstm04 | taichi_csr_richardson_jacobi_float64 | screened_out | richardson | jacobi | 1 |  |  | inf | inf |
| suitesparse:HB/bcsstm05 | taichi_csr_cg_none_float64 | success | cg | none | 1 | 708.349 | 16 | 3.98604e-06 | 1.012e-05 |
| suitesparse:HB/bcsstm05 | taichi_csr_pcg_jacobi_float64 | success | pcg | jacobi | 1 | 813.824 | 1 | 0 | 0 |
| suitesparse:HB/bcsstm05 | taichi_csr_richardson_jacobi_float64 | success | richardson | jacobi | 1 | 367.957 | 11 | 5.64503e-06 | 5.64503e-06 |
| suitesparse:HB/bcsstm06 | taichi_csr_cg_none_float64 | screened_out | cg | none | 1 |  | 51 | 4.85334e-06 | 0.217263 |
| suitesparse:HB/bcsstm06 | taichi_csr_pcg_jacobi_float64 | success | pcg | jacobi | 1 | 855.372 | 1 | 0 | 0 |
| suitesparse:HB/bcsstm06 | taichi_csr_richardson_jacobi_float64 | success | richardson | jacobi | 1 | 361.296 | 11 | 5.64503e-06 | 5.64503e-06 |
| suitesparse:HB/bcsstm07 | taichi_csr_cg_none_float64 | success | cg | none | 1 | 1940.34 | 156 | 9.97184e-06 | 0.00071122 |
| suitesparse:HB/bcsstm07 | taichi_csr_pcg_jacobi_float64 | success | pcg | jacobi | 1 | 1353.44 | 28 | 8.60409e-06 | 9.4417e-05 |
| suitesparse:HB/bcsstm07 | taichi_csr_richardson_jacobi_float64 | success | richardson | jacobi | 1 | 1325.16 | 139 | 9.5507e-06 | 0.00023176 |
| suitesparse:HB/bcsstm08 | taichi_csr_cg_none_float64 | screened_out | cg | none | 1 |  | 29 | 3.32255e-06 | 0.203383 |
| suitesparse:HB/bcsstm08 | taichi_csr_pcg_jacobi_float64 | success | pcg | jacobi | 1 | 721.41 | 1 | 0 | 0 |
| suitesparse:HB/bcsstm08 | taichi_csr_richardson_jacobi_float64 | success | richardson | jacobi | 1 | 460.524 | 11 | 5.64503e-06 | 5.64503e-06 |
| suitesparse:HB/bcsstm09 | taichi_csr_cg_none_float64 | success | cg | none | 1 | 799.353 | 3 | 1.1626e-11 | 3.66166e-08 |
| suitesparse:HB/bcsstm09 | taichi_csr_pcg_jacobi_float64 | success | pcg | jacobi | 1 | 616.821 | 1 | 0 | 0 |
| suitesparse:HB/bcsstm09 | taichi_csr_richardson_jacobi_float64 | success | richardson | jacobi | 1 | 452.657 | 11 | 5.64503e-06 | 5.64503e-06 |
