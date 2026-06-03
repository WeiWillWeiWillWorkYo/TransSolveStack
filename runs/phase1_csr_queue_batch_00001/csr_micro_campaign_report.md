# CSR Micro-Campaign

- status: `passed`
- imported_matrices: `8`
- candidate_jobs: `24`
- gpu_success_rows: `2`
- cpu_screened_out_rows: `22`
- gpu_failed_rows: `0`
- selector_rows: `24`
- selector_oracle_rows: `2`
- max_final_relative_residual: `9.80226e-06`
- max_cpu_recomputed_relative_residual: `9.80226e-06`
- max_solution_relative_error: `0.00255227`

| matrix | candidate | status | solver | preconditioner | repeats | median_ms | iters | rel_res | sol_err |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| suitesparse:HB/1138_bus | taichi_csr_cg_none_float64 | screened_out | cg | none | 1 |  | 300 | 0.00151996 | 0.949524 |
| suitesparse:HB/1138_bus | taichi_csr_pcg_jacobi_float64 | screened_out | pcg | jacobi | 1 |  | 300 | 0.00163456 | 0.118493 |
| suitesparse:HB/1138_bus | taichi_csr_richardson_jacobi_float64 | screened_out | richardson | jacobi | 1 |  | 300 | 0.000661028 | 0.995963 |
| suitesparse:HB/494_bus | taichi_csr_cg_none_float64 | screened_out | cg | none | 1 |  | 300 | 0.00341783 | 0.345386 |
| suitesparse:HB/494_bus | taichi_csr_pcg_jacobi_float64 | screened_out | pcg | jacobi | 1 |  | 300 | 2.31396e-05 | 0.00178397 |
| suitesparse:HB/494_bus | taichi_csr_richardson_jacobi_float64 | screened_out | richardson | jacobi | 1 |  | 300 | 0.000671536 | 0.966961 |
| suitesparse:HB/662_bus | taichi_csr_cg_none_float64 | screened_out | cg | none | 1 |  | 300 | 0.000129812 | 0.00310619 |
| suitesparse:HB/662_bus | taichi_csr_pcg_jacobi_float64 | success | pcg | jacobi | 1 | 662.573 | 148 | 9.45184e-06 | 0.000143747 |
| suitesparse:HB/662_bus | taichi_csr_richardson_jacobi_float64 | screened_out | richardson | jacobi | 1 |  | 300 | 0.000177846 | 0.979474 |
| suitesparse:HB/685_bus | taichi_csr_cg_none_float64 | screened_out | cg | none | 1 |  | 300 | 2.29664e-05 | 0.00549479 |
| suitesparse:HB/685_bus | taichi_csr_pcg_jacobi_float64 | success | pcg | jacobi | 1 | 650.156 | 150 | 9.80226e-06 | 0.00255227 |
| suitesparse:HB/685_bus | taichi_csr_richardson_jacobi_float64 | screened_out | richardson | jacobi | 1 |  | 300 | 0.000170132 | 0.937174 |
| suitesparse:HB/arc130 | taichi_csr_bicgstab_none_float64 | screened_out | bicgstab | none | 1 |  | 7 | 3.8921e-06 | 0.500077 |
| suitesparse:HB/arc130 | taichi_csr_bicgstab_jacobi_float64 | screened_out | bicgstab | jacobi | 1 |  | 3 | 2.11164e-06 | 97.4098 |
| suitesparse:HB/arc130 | taichi_csr_gmres_jacobi_restart16_float64 | screened_out | gmres | jacobi | 1 |  | 2 | 3.02451e-06 | 80810.8 |
| suitesparse:HB/ash292 | taichi_csr_cg_none_float64 | screened_out | cg | none | 1 |  | 3 | 0.0547811 | 0.2302 |
| suitesparse:HB/ash292 | taichi_csr_pcg_jacobi_float64 | screened_out | pcg | jacobi | 1 |  | 3 | 0.0547811 | 0.2302 |
| suitesparse:HB/ash292 | taichi_csr_richardson_jacobi_float64 | screened_out | richardson | jacobi | 1 |  | 300 | 4.78449e+06 | 1.56276e+07 |
| suitesparse:HB/ash85 | taichi_csr_cg_none_float64 | screened_out | cg | none | 1 |  | 3 | 0.172619 | 0.579531 |
| suitesparse:HB/ash85 | taichi_csr_pcg_jacobi_float64 | screened_out | pcg | jacobi | 1 |  | 3 | 0.172619 | 0.579531 |
| suitesparse:HB/ash85 | taichi_csr_richardson_jacobi_float64 | screened_out | richardson | jacobi | 1 |  | 300 | 216399 | 712193 |
| suitesparse:HB/bcspwr01 | taichi_csr_cg_none_float64 | screened_out | cg | none | 1 |  | 2 | 0.120381 | 0.309944 |
| suitesparse:HB/bcspwr01 | taichi_csr_pcg_jacobi_float64 | screened_out | pcg | jacobi | 1 |  | 2 | 0.120381 | 0.309944 |
| suitesparse:HB/bcspwr01 | taichi_csr_richardson_jacobi_float64 | screened_out | richardson | jacobi | 1 |  | 300 | 77388.7 | 165919 |
