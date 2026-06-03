# CSR Micro-Campaign

- status: `passed`
- imported_matrices: `8`
- candidate_jobs: `24`
- gpu_success_rows: `1`
- cpu_screened_out_rows: `23`
- gpu_failed_rows: `0`
- selector_rows: `24`
- selector_oracle_rows: `1`
- max_final_relative_residual: `9.58594e-06`
- max_cpu_recomputed_relative_residual: `9.58594e-06`
- max_solution_relative_error: `0.000154675`

| matrix | candidate | status | solver | preconditioner | repeats | median_ms | iters | rel_res | sol_err |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| suitesparse:HB/bcsstk24 | taichi_csr_cg_none_float64 | screened_out | cg | none | 1 |  | 300 | 3.18224e-05 | 0.740676 |
| suitesparse:HB/bcsstk24 | taichi_csr_pcg_jacobi_float64 | screened_out | pcg | jacobi | 1 |  | 250 | 9.21286e-06 | 2.5502 |
| suitesparse:HB/bcsstk24 | taichi_csr_richardson_jacobi_float64 | screened_out | richardson | jacobi | 1 |  | 300 | 0.000723424 | 11.3113 |
| suitesparse:HB/bcsstk25 | taichi_csr_cg_none_float64 | screened_out | cg | none | 1 |  | 300 | 7.43124e-05 | 0.728571 |
| suitesparse:HB/bcsstk25 | taichi_csr_pcg_jacobi_float64 | screened_out | pcg | jacobi | 1 |  | 131 | 8.72074e-06 | 0.646381 |
| suitesparse:HB/bcsstk25 | taichi_csr_richardson_jacobi_float64 | screened_out | richardson | jacobi | 1 |  | 300 | 0.00370092 | 3.08146 |
| suitesparse:HB/bcsstk26 | taichi_csr_cg_none_float64 | screened_out | cg | none | 1 |  | 300 | 0.00054983 | 0.650528 |
| suitesparse:HB/bcsstk26 | taichi_csr_pcg_jacobi_float64 | screened_out | pcg | jacobi | 1 |  | 300 | 3.22243e-05 | 0.157974 |
| suitesparse:HB/bcsstk26 | taichi_csr_richardson_jacobi_float64 | screened_out | richardson | jacobi | 1 |  | 300 | 0.00140829 | 1.45385 |
| suitesparse:HB/bcsstk27 | taichi_csr_cg_none_float64 | screened_out | cg | none | 1 |  | 300 | 0.00012016 | 0.00429484 |
| suitesparse:HB/bcsstk27 | taichi_csr_pcg_jacobi_float64 | success | pcg | jacobi | 1 | 1703.62 | 172 | 9.58594e-06 | 0.000154675 |
| suitesparse:HB/bcsstk27 | taichi_csr_richardson_jacobi_float64 | screened_out | richardson | jacobi | 1 |  | 300 | 0.0135761 | 0.420013 |
| suitesparse:HB/bcsstk28 | taichi_csr_cg_none_float64 | screened_out | cg | none | 1 |  | 300 | 0.00928338 | 0.874905 |
| suitesparse:HB/bcsstk28 | taichi_csr_pcg_jacobi_float64 | screened_out | pcg | jacobi | 1 |  | 300 | 0.00170142 | 0.791927 |
| suitesparse:HB/bcsstk28 | taichi_csr_richardson_jacobi_float64 | screened_out | richardson | jacobi | 1 |  | 300 | 0.0103604 | 0.9439 |
| suitesparse:HB/bcsstk29 | taichi_csr_cg_none_float64 | screened_out | cg | none | 1 |  | 3 | 0.34147 | 1.00046 |
| suitesparse:HB/bcsstk29 | taichi_csr_pcg_jacobi_float64 | screened_out | pcg | jacobi | 1 |  | 3 | 0.34147 | 1.00046 |
| suitesparse:HB/bcsstk29 | taichi_csr_richardson_jacobi_float64 | screened_out | richardson | jacobi | 1 |  | 300 | 2.55167e+56 | 6.57984e+56 |
| suitesparse:HB/bcsstk33 | taichi_csr_cg_none_float64 | screened_out | cg | none | 1 |  | 2 | 0.0515266 | 0.207475 |
| suitesparse:HB/bcsstk33 | taichi_csr_pcg_jacobi_float64 | screened_out | pcg | jacobi | 1 |  | 2 | 0.0515266 | 0.207475 |
| suitesparse:HB/bcsstk33 | taichi_csr_richardson_jacobi_float64 | screened_out | richardson | jacobi | 1 |  | 300 | 2.10361e+72 | 5.91464e+72 |
| suitesparse:HB/bcsstm01 | taichi_csr_cg_none_float64 | screened_out | cg | none | 1 |  | 2 | 1.58882e-17 | 0.707107 |
| suitesparse:HB/bcsstm01 | taichi_csr_pcg_jacobi_float64 | screened_out | pcg | jacobi | 1 |  | 1 | 0 | 0.707107 |
| suitesparse:HB/bcsstm01 | taichi_csr_richardson_jacobi_float64 | screened_out | richardson | jacobi | 1 |  |  | inf | inf |
