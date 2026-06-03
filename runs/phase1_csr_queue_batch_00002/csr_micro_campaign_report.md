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
| suitesparse:HB/bcspwr02 | taichi_csr_cg_none_float64 | screened_out | cg | none | 1 |  | 2 | 0.139728 | 0.388987 |
| suitesparse:HB/bcspwr02 | taichi_csr_pcg_jacobi_float64 | screened_out | pcg | jacobi | 1 |  | 2 | 0.139728 | 0.388987 |
| suitesparse:HB/bcspwr02 | taichi_csr_richardson_jacobi_float64 | screened_out | richardson | jacobi | 1 |  | 300 | 392221 | 795876 |
| suitesparse:HB/bcspwr03 | taichi_csr_cg_none_float64 | screened_out | cg | none | 1 |  | 2 | 0.221032 | 0.461983 |
| suitesparse:HB/bcspwr03 | taichi_csr_pcg_jacobi_float64 | screened_out | pcg | jacobi | 1 |  | 2 | 0.221032 | 0.461983 |
| suitesparse:HB/bcspwr03 | taichi_csr_richardson_jacobi_float64 | screened_out | richardson | jacobi | 1 |  | 300 | 3.60783e+07 | 6.8262e+07 |
| suitesparse:HB/bcspwr04 | taichi_csr_cg_none_float64 | screened_out | cg | none | 1 |  | 3 | 0.171643 | 0.49805 |
| suitesparse:HB/bcspwr04 | taichi_csr_pcg_jacobi_float64 | screened_out | pcg | jacobi | 1 |  | 3 | 0.171643 | 0.49805 |
| suitesparse:HB/bcspwr04 | taichi_csr_richardson_jacobi_float64 | screened_out | richardson | jacobi | 1 |  | 300 | 6.12735e+08 | 1.53404e+09 |
| suitesparse:HB/bcspwr05 | taichi_csr_cg_none_float64 | screened_out | cg | none | 1 |  | 2 | 0.17217 | 0.405119 |
| suitesparse:HB/bcspwr05 | taichi_csr_pcg_jacobi_float64 | screened_out | pcg | jacobi | 1 |  | 2 | 0.17217 | 0.405119 |
| suitesparse:HB/bcspwr05 | taichi_csr_richardson_jacobi_float64 | screened_out | richardson | jacobi | 1 |  | 300 | 1.00933e+08 | 1.63376e+08 |
| suitesparse:HB/bcspwr06 | taichi_csr_cg_none_float64 | screened_out | cg | none | 1 |  | 2 | 0.188134 | 0.410039 |
| suitesparse:HB/bcspwr06 | taichi_csr_pcg_jacobi_float64 | screened_out | pcg | jacobi | 1 |  | 2 | 0.188134 | 0.410039 |
| suitesparse:HB/bcspwr06 | taichi_csr_richardson_jacobi_float64 | screened_out | richardson | jacobi | 1 |  | 300 | 1.50591e+10 | 1.93472e+10 |
| suitesparse:HB/bcspwr07 | taichi_csr_cg_none_float64 | screened_out | cg | none | 1 |  | 2 | 0.218942 | 0.448802 |
| suitesparse:HB/bcspwr07 | taichi_csr_pcg_jacobi_float64 | screened_out | pcg | jacobi | 1 |  | 2 | 0.218942 | 0.448802 |
| suitesparse:HB/bcspwr07 | taichi_csr_richardson_jacobi_float64 | screened_out | richardson | jacobi | 1 |  | 300 | 1.27161e+10 | 1.63738e+10 |
| suitesparse:HB/bcspwr08 | taichi_csr_cg_none_float64 | screened_out | cg | none | 1 |  | 2 | 0.225546 | 0.463039 |
| suitesparse:HB/bcspwr08 | taichi_csr_pcg_jacobi_float64 | screened_out | pcg | jacobi | 1 |  | 2 | 0.225546 | 0.463039 |
| suitesparse:HB/bcspwr08 | taichi_csr_richardson_jacobi_float64 | screened_out | richardson | jacobi | 1 |  | 300 | 1.52222e+10 | 2.0216e+10 |
| suitesparse:HB/bcspwr09 | taichi_csr_cg_none_float64 | screened_out | cg | none | 1 |  | 2 | 0.216078 | 0.450993 |
| suitesparse:HB/bcspwr09 | taichi_csr_pcg_jacobi_float64 | screened_out | pcg | jacobi | 1 |  | 2 | 0.216078 | 0.450993 |
| suitesparse:HB/bcspwr09 | taichi_csr_richardson_jacobi_float64 | screened_out | richardson | jacobi | 1 |  | 300 | 1.91444e+10 | 2.57443e+10 |
