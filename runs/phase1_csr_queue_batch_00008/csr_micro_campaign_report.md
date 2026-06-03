# CSR Micro-Campaign

- status: `passed`
- imported_matrices: `8`
- candidate_jobs: `24`
- gpu_success_rows: `12`
- cpu_screened_out_rows: `12`
- gpu_failed_rows: `0`
- selector_rows: `24`
- selector_oracle_rows: `5`
- max_final_relative_residual: `6.87101e-06`
- max_cpu_recomputed_relative_residual: `6.87101e-06`
- max_solution_relative_error: `2.87597e-05`

| matrix | candidate | status | solver | preconditioner | repeats | median_ms | iters | rel_res | sol_err |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| suitesparse:HB/bcsstm10 | taichi_csr_cg_none_float64 | screened_out | cg | none | 1 |  | 11 | 1.00749 | 1.15239 |
| suitesparse:HB/bcsstm10 | taichi_csr_pcg_jacobi_float64 | screened_out | pcg | jacobi | 1 |  | 4 | 0.457834 | 0.962214 |
| suitesparse:HB/bcsstm10 | taichi_csr_richardson_jacobi_float64 | screened_out | richardson | jacobi | 1 |  | 300 | 85310.8 | 79223.1 |
| suitesparse:HB/bcsstm11 | taichi_csr_cg_none_float64 | screened_out | cg | none | 1 |  | 11 | 9.5631e-06 | 0.183107 |
| suitesparse:HB/bcsstm11 | taichi_csr_pcg_jacobi_float64 | success | pcg | jacobi | 1 | 794.58 | 1 | 0 | 0 |
| suitesparse:HB/bcsstm11 | taichi_csr_richardson_jacobi_float64 | success | richardson | jacobi | 1 | 468.001 | 11 | 5.64503e-06 | 5.64503e-06 |
| suitesparse:HB/bcsstm12 | taichi_csr_cg_none_float64 | screened_out | cg | none | 1 |  | 300 | 0.000435499 | 0.114988 |
| suitesparse:HB/bcsstm12 | taichi_csr_pcg_jacobi_float64 | screened_out | pcg | jacobi | 1 |  | 88 | 9.75569e-06 | 0.0115877 |
| suitesparse:HB/bcsstm12 | taichi_csr_richardson_jacobi_float64 | screened_out | richardson | jacobi | 1 |  | 300 | 0.000163549 | 0.473675 |
| suitesparse:HB/bcsstm13 | taichi_csr_cg_none_float64 | screened_out | cg | none | 1 |  | 213 | 9.0685e-06 | 0.616791 |
| suitesparse:HB/bcsstm13 | taichi_csr_pcg_jacobi_float64 | screened_out | pcg | jacobi | 1 |  | 25 | 8.42238e-06 | 0.61679 |
| suitesparse:HB/bcsstm13 | taichi_csr_richardson_jacobi_float64 | screened_out | richardson | jacobi | 1 |  |  | inf | inf |
| suitesparse:HB/bcsstm19 | taichi_csr_cg_none_float64 | screened_out | cg | none | 1 |  | 164 | 7.30233e-06 | 0.0311531 |
| suitesparse:HB/bcsstm19 | taichi_csr_pcg_jacobi_float64 | success | pcg | jacobi | 1 | 895.083 | 1 | 0 | 0 |
| suitesparse:HB/bcsstm19 | taichi_csr_richardson_jacobi_float64 | success | richardson | jacobi | 1 | 1010.16 | 11 | 5.64503e-06 | 5.64503e-06 |
| suitesparse:HB/bcsstm20 | taichi_csr_cg_none_float64 | screened_out | cg | none | 1 |  | 126 | 7.0591e-06 | 0.0270976 |
| suitesparse:HB/bcsstm20 | taichi_csr_pcg_jacobi_float64 | success | pcg | jacobi | 1 | 600.919 | 1 | 0 | 0 |
| suitesparse:HB/bcsstm20 | taichi_csr_richardson_jacobi_float64 | success | richardson | jacobi | 1 | 562.688 | 11 | 5.64503e-06 | 5.64503e-06 |
| suitesparse:HB/bcsstm21 | taichi_csr_cg_none_float64 | success | cg | none | 1 | 1033.82 | 3 | 3.38303e-06 | 2.3964e-06 |
| suitesparse:HB/bcsstm21 | taichi_csr_pcg_jacobi_float64 | success | pcg | jacobi | 1 | 1173.92 | 1 | 0 | 0 |
| suitesparse:HB/bcsstm21 | taichi_csr_richardson_jacobi_float64 | success | richardson | jacobi | 1 | 1091.73 | 11 | 5.64503e-06 | 5.64503e-06 |
| suitesparse:HB/bcsstm22 | taichi_csr_cg_none_float64 | success | cg | none | 1 | 1108.25 | 50 | 6.87101e-06 | 2.87597e-05 |
| suitesparse:HB/bcsstm22 | taichi_csr_pcg_jacobi_float64 | success | pcg | jacobi | 1 | 632.086 | 1 | 0 | 0 |
| suitesparse:HB/bcsstm22 | taichi_csr_richardson_jacobi_float64 | success | richardson | jacobi | 1 | 363.388 | 11 | 5.64503e-06 | 5.64503e-06 |
