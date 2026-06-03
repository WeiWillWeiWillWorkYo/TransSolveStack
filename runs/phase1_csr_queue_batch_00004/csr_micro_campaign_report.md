# CSR Micro-Campaign

- status: `passed`
- imported_matrices: `8`
- candidate_jobs: `24`
- gpu_success_rows: `3`
- cpu_screened_out_rows: `21`
- gpu_failed_rows: `0`
- selector_rows: `24`
- selector_oracle_rows: `2`
- max_final_relative_residual: `9.83561e-06`
- max_cpu_recomputed_relative_residual: `9.83561e-06`
- max_solution_relative_error: `0.00434613`

| matrix | candidate | status | solver | preconditioner | repeats | median_ms | iters | rel_res | sol_err |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| suitesparse:HB/bcsstk08 | taichi_csr_cg_none_float64 | screened_out | cg | none | 1 |  | 300 | 4.24404e-05 | 0.398501 |
| suitesparse:HB/bcsstk08 | taichi_csr_pcg_jacobi_float64 | screened_out | pcg | jacobi | 1 |  | 78 | 8.80762e-06 | 0.0191595 |
| suitesparse:HB/bcsstk08 | taichi_csr_richardson_jacobi_float64 | screened_out | richardson | jacobi | 1 |  | 300 | 0.000211145 | 1.41184 |
| suitesparse:HB/bcsstk09 | taichi_csr_cg_none_float64 | success | cg | none | 1 | 1049.39 | 152 | 9.50179e-06 | 0.000186387 |
| suitesparse:HB/bcsstk09 | taichi_csr_pcg_jacobi_float64 | success | pcg | jacobi | 1 | 1309.93 | 98 | 8.60083e-06 | 0.000432482 |
| suitesparse:HB/bcsstk09 | taichi_csr_richardson_jacobi_float64 | screened_out | richardson | jacobi | 1 |  | 300 | 0.00177009 | 0.448329 |
| suitesparse:HB/bcsstk10 | taichi_csr_cg_none_float64 | screened_out | cg | none | 1 |  | 300 | 0.00020868 | 0.426307 |
| suitesparse:HB/bcsstk10 | taichi_csr_pcg_jacobi_float64 | success | pcg | jacobi | 1 | 1268.32 | 182 | 9.83561e-06 | 0.00434613 |
| suitesparse:HB/bcsstk10 | taichi_csr_richardson_jacobi_float64 | screened_out | richardson | jacobi | 1 |  | 300 | 9.78173e-05 | 0.421047 |
| suitesparse:HB/bcsstk11 | taichi_csr_cg_none_float64 | screened_out | cg | none | 1 |  | 300 | 0.00011027 | 0.240938 |
| suitesparse:HB/bcsstk11 | taichi_csr_pcg_jacobi_float64 | screened_out | pcg | jacobi | 1 |  | 197 | 9.77271e-06 | 0.0846525 |
| suitesparse:HB/bcsstk11 | taichi_csr_richardson_jacobi_float64 | screened_out | richardson | jacobi | 1 |  | 300 | 0.000156493 | 0.472661 |
| suitesparse:HB/bcsstk12 | taichi_csr_cg_none_float64 | screened_out | cg | none | 1 |  | 300 | 0.00011027 | 0.240938 |
| suitesparse:HB/bcsstk12 | taichi_csr_pcg_jacobi_float64 | screened_out | pcg | jacobi | 1 |  | 197 | 9.77271e-06 | 0.0846525 |
| suitesparse:HB/bcsstk12 | taichi_csr_richardson_jacobi_float64 | screened_out | richardson | jacobi | 1 |  | 300 | 0.000156493 | 0.472661 |
| suitesparse:HB/bcsstk13 | taichi_csr_cg_none_float64 | screened_out | cg | none | 1 |  | 300 | 0.000247178 | 0.804508 |
| suitesparse:HB/bcsstk13 | taichi_csr_pcg_jacobi_float64 | screened_out | pcg | jacobi | 1 |  | 300 | 2.60519e-05 | 0.484548 |
| suitesparse:HB/bcsstk13 | taichi_csr_richardson_jacobi_float64 | screened_out | richardson | jacobi | 1 |  | 300 | 0.00145464 | 0.918848 |
| suitesparse:HB/bcsstk14 | taichi_csr_cg_none_float64 | screened_out | cg | none | 1 |  | 300 | 7.00446e-05 | 0.623772 |
| suitesparse:HB/bcsstk14 | taichi_csr_pcg_jacobi_float64 | screened_out | pcg | jacobi | 1 |  | 103 | 9.78603e-06 | 0.0307678 |
| suitesparse:HB/bcsstk14 | taichi_csr_richardson_jacobi_float64 | screened_out | richardson | jacobi | 1 |  | 300 | 5.50747e-05 | 0.492926 |
| suitesparse:HB/bcsstk15 | taichi_csr_cg_none_float64 | screened_out | cg | none | 1 |  | 300 | 0.000152786 | 0.635238 |
| suitesparse:HB/bcsstk15 | taichi_csr_pcg_jacobi_float64 | screened_out | pcg | jacobi | 1 |  | 300 | 3.14182e-05 | 0.110869 |
| suitesparse:HB/bcsstk15 | taichi_csr_richardson_jacobi_float64 | screened_out | richardson | jacobi | 1 |  | 300 | 0.00133123 | 1.26746 |
