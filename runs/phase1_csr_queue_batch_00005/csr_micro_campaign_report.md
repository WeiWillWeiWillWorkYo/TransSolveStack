# CSR Micro-Campaign

- status: `passed`
- imported_matrices: `8`
- candidate_jobs: `24`
- gpu_success_rows: `1`
- cpu_screened_out_rows: `23`
- gpu_failed_rows: `0`
- selector_rows: `24`
- selector_oracle_rows: `1`
- max_final_relative_residual: `9.23618e-06`
- max_cpu_recomputed_relative_residual: `9.23618e-06`
- max_solution_relative_error: `6.52908e-05`

| matrix | candidate | status | solver | preconditioner | repeats | median_ms | iters | rel_res | sol_err |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| suitesparse:HB/bcsstk16 | taichi_csr_cg_none_float64 | screened_out | cg | none | 1 |  | 188 | 9.94167e-06 | 0.123091 |
| suitesparse:HB/bcsstk16 | taichi_csr_pcg_jacobi_float64 | success | pcg | jacobi | 1 | 1124.29 | 124 | 9.23618e-06 | 6.52908e-05 |
| suitesparse:HB/bcsstk16 | taichi_csr_richardson_jacobi_float64 | screened_out | richardson | jacobi | 1 |  | 300 | 0.00629634 | 0.286278 |
| suitesparse:HB/bcsstk17 | taichi_csr_cg_none_float64 | screened_out | cg | none | 1 |  | 300 | 0.00241781 | 0.660408 |
| suitesparse:HB/bcsstk17 | taichi_csr_pcg_jacobi_float64 | screened_out | pcg | jacobi | 1 |  | 300 | 0.000373221 | 0.39466 |
| suitesparse:HB/bcsstk17 | taichi_csr_richardson_jacobi_float64 | screened_out | richardson | jacobi | 1 |  | 300 | 0.00423632 | 0.651831 |
| suitesparse:HB/bcsstk18 | taichi_csr_cg_none_float64 | screened_out | cg | none | 1 |  | 300 | 0.000226271 | 0.729581 |
| suitesparse:HB/bcsstk18 | taichi_csr_pcg_jacobi_float64 | screened_out | pcg | jacobi | 1 |  | 258 | 9.40418e-06 | 0.159251 |
| suitesparse:HB/bcsstk18 | taichi_csr_richardson_jacobi_float64 | screened_out | richardson | jacobi | 1 |  | 300 | 0.000291833 | 1.15947 |
| suitesparse:HB/bcsstk19 | taichi_csr_cg_none_float64 | screened_out | cg | none | 1 |  | 96 | 9.68271e-06 | 0.815041 |
| suitesparse:HB/bcsstk19 | taichi_csr_pcg_jacobi_float64 | screened_out | pcg | jacobi | 1 |  | 78 | 9.81307e-06 | 1.45474 |
| suitesparse:HB/bcsstk19 | taichi_csr_richardson_jacobi_float64 | screened_out | richardson | jacobi | 1 |  | 300 | 5.43943e-05 | 2.69761 |
| suitesparse:HB/bcsstk20 | taichi_csr_cg_none_float64 | screened_out | cg | none | 1 |  | 35 | 8.60687e-06 | 0.817149 |
| suitesparse:HB/bcsstk20 | taichi_csr_pcg_jacobi_float64 | screened_out | pcg | jacobi | 1 |  | 140 | 9.83816e-06 | 14.8738 |
| suitesparse:HB/bcsstk20 | taichi_csr_richardson_jacobi_float64 | screened_out | richardson | jacobi | 1 |  | 300 | 0.00019983 | 10.782 |
| suitesparse:HB/bcsstk21 | taichi_csr_cg_none_float64 | screened_out | cg | none | 1 |  | 300 | 6.35618e-05 | 0.352362 |
| suitesparse:HB/bcsstk21 | taichi_csr_pcg_jacobi_float64 | screened_out | pcg | jacobi | 1 |  | 171 | 9.4143e-06 | 0.188692 |
| suitesparse:HB/bcsstk21 | taichi_csr_richardson_jacobi_float64 | screened_out | richardson | jacobi | 1 |  | 300 | 0.00225332 | 0.401614 |
| suitesparse:HB/bcsstk22 | taichi_csr_cg_none_float64 | screened_out | cg | none | 1 |  | 168 | 8.10679e-06 | 0.0418933 |
| suitesparse:HB/bcsstk22 | taichi_csr_pcg_jacobi_float64 | screened_out | pcg | jacobi | 1 |  | 74 | 7.27955e-06 | 0.0231212 |
| suitesparse:HB/bcsstk22 | taichi_csr_richardson_jacobi_float64 | screened_out | richardson | jacobi | 1 |  | 300 | 0.00013449 | 0.393149 |
| suitesparse:HB/bcsstk23 | taichi_csr_cg_none_float64 | screened_out | cg | none | 1 |  | 300 | 1.6162e-05 | 0.750538 |
| suitesparse:HB/bcsstk23 | taichi_csr_pcg_jacobi_float64 | screened_out | pcg | jacobi | 1 |  | 185 | 9.23225e-06 | 16.928 |
| suitesparse:HB/bcsstk23 | taichi_csr_richardson_jacobi_float64 | screened_out | richardson | jacobi | 1 |  | 300 | 0.000207002 | 24.7046 |
