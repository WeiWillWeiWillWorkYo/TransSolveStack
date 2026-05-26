# CSR Non-Success Fallback Probe

- status: `passed`
- target_matrices: `4`
- candidate_attempts: `15`
- gpu_success_rows: `4`
- cpu_screened_out_rows: `11`
- gpu_failed_rows: `0`
- resolved_matrices: `suitesparse:Bai/cdde1, suitesparse:FIDAP/ex5`
- unresolved_matrices: `suitesparse:Gset/G17, suitesparse:Zitney/extr1b`
- max_final_relative_residual: `9.95364e-06`
- max_cpu_recomputed_relative_residual: `9.95359e-06`
- max_solution_relative_error: `0.000736196`

| matrix | candidate | role | status | solver | preconditioner | gpu | rel_res | sol_err | screen_breakdown |
|---|---|---|---|---|---|---|---:|---:|---|
| suitesparse:FIDAP/ex5 | taichi_csr_bicgstab_none_float64 | learned_non_success_candidate | screened_out | bicgstab | none | False | 378895 | 218966 |  |
| suitesparse:FIDAP/ex5 | taichi_csr_bicgstab_jacobi_float64 | fallback_probe | screened_out | bicgstab | jacobi | False | 0.616818 | 0.947567 |  |
| suitesparse:FIDAP/ex5 | taichi_csr_gmres_jacobi_restart32_float64 | fallback_probe | success | gmres | jacobi | True | 9.95364e-06 | 3.31453e-05 |  |
| suitesparse:Bai/cdde1 | taichi_csr_gmres_jacobi_restart8_float64 | learned_non_success_candidate | screened_out | gmres | jacobi | False | 0.000411457 | 0.0410976 |  |
| suitesparse:Bai/cdde1 | taichi_csr_bicgstab_none_float64 | fallback_probe | success | bicgstab | none | True | 7.84296e-07 | 2.06263e-05 |  |
| suitesparse:Bai/cdde1 | taichi_csr_bicgstab_jacobi_float64 | fallback_probe | success | bicgstab | jacobi | True | 1.55458e-06 | 4.40981e-05 |  |
| suitesparse:Bai/cdde1 | taichi_csr_gmres_jacobi_restart32_float64 | fallback_probe | success | gmres | jacobi | True | 9.8388e-06 | 0.000736196 |  |
| suitesparse:Zitney/extr1b | taichi_csr_pcg_jacobi_float64 | learned_non_success_candidate | screened_out | pcg | jacobi | False | 1 | 1 | non_positive_curvature |
| suitesparse:Zitney/extr1b | taichi_csr_bicgstab_none_float64 | fallback_probe | screened_out | bicgstab | none | False | 2.69412e+09 | 4.48957e+12 |  |
| suitesparse:Zitney/extr1b | taichi_csr_bicgstab_jacobi_float64 | fallback_probe | screened_out | bicgstab | jacobi | False | 5.04585e+08 | 1.70035e+11 |  |
| suitesparse:Zitney/extr1b | taichi_csr_gmres_jacobi_restart32_float64 | fallback_probe | screened_out | gmres | jacobi | False | 0.934 | 3051.75 |  |
| suitesparse:Gset/G17 | taichi_csr_richardson_jacobi_float64 | learned_non_success_candidate | screened_out | richardson | jacobi | False | inf | inf | zero_diagonal |
| suitesparse:Gset/G17 | taichi_csr_bicgstab_none_float64 | fallback_probe | screened_out | bicgstab | none | False | 7.16595e-05 | 0.0240158 |  |
| suitesparse:Gset/G17 | taichi_csr_bicgstab_jacobi_float64 | fallback_probe | screened_out | bicgstab | jacobi | False | 7.16595e-05 | 0.0240158 |  |
| suitesparse:Gset/G17 | taichi_csr_gmres_jacobi_restart32_float64 | fallback_probe | screened_out | gmres | jacobi | False | 0.000298925 | 0.0630637 |  |
