# CSR Selector Readiness

- status: `passed`
- schema_version: `phase1_csr_selector_features_v8`
- diagnostic_rows: `8`
- selector_rows: `24`
- success_rows: `2`
- failed_rows: `22`
- applicability_rows: `0`
- oracle_rows: `2`
- matrices_with_selector_rows: `8`
- max_final_relative_residual: `9.80226e-06`
- max_cpu_recomputed_relative_residual: `9.80226e-06`
- max_solution_relative_error: `0.00255227`
- min_success_rate: `0`
- max_solve_time_iqr_ms: `0`
- failure_reason_counts: `{'cpu_reference_bicgstab_screen_failed': 2, 'cpu_reference_cg_screen_failed': 7, 'cpu_reference_gmres_screen_failed': 1, 'cpu_reference_pcg_screen_failed': 5, 'cpu_reference_richardson_screen_failed': 7}`
- applicability_status_counts: `{}`
- applicability_reason_counts: `{}`

| matrix | candidate | oracle | status | failure | applicability | reason | success_rate | repeats | median_ms | regret_ms | rel_res | sol_err |
|---|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|
| suitesparse:HB/1138_bus | taichi_csr_cg_none_float64 | no | screened_out | cpu_reference_cg_screen_failed | applicable |  | 0 | 1 |  |  | 0.00151996 | 0.949524 |
| suitesparse:HB/1138_bus | taichi_csr_pcg_jacobi_float64 | no | screened_out | cpu_reference_pcg_screen_failed | applicable |  | 0 | 1 |  |  | 0.00163456 | 0.118493 |
| suitesparse:HB/1138_bus | taichi_csr_richardson_jacobi_float64 | no | screened_out | cpu_reference_richardson_screen_failed | applicable |  | 0 | 1 |  |  | 0.000661028 | 0.995963 |
| suitesparse:HB/494_bus | taichi_csr_cg_none_float64 | no | screened_out | cpu_reference_cg_screen_failed | applicable |  | 0 | 1 |  |  | 0.00341783 | 0.345386 |
| suitesparse:HB/494_bus | taichi_csr_pcg_jacobi_float64 | no | screened_out | cpu_reference_pcg_screen_failed | applicable |  | 0 | 1 |  |  | 2.31396e-05 | 0.00178397 |
| suitesparse:HB/494_bus | taichi_csr_richardson_jacobi_float64 | no | screened_out | cpu_reference_richardson_screen_failed | applicable |  | 0 | 1 |  |  | 0.000671536 | 0.966961 |
| suitesparse:HB/662_bus | taichi_csr_cg_none_float64 | no | screened_out | cpu_reference_cg_screen_failed | applicable |  | 0 | 1 |  |  | 0.000129812 | 0.00310619 |
| suitesparse:HB/662_bus | taichi_csr_pcg_jacobi_float64 | yes | success |  | applicable |  | 1 | 1 | 662.573 | 0 | 9.45184e-06 | 0.000143747 |
| suitesparse:HB/662_bus | taichi_csr_richardson_jacobi_float64 | no | screened_out | cpu_reference_richardson_screen_failed | applicable |  | 0 | 1 |  |  | 0.000177846 | 0.979474 |
| suitesparse:HB/685_bus | taichi_csr_cg_none_float64 | no | screened_out | cpu_reference_cg_screen_failed | applicable |  | 0 | 1 |  |  | 2.29664e-05 | 0.00549479 |
| suitesparse:HB/685_bus | taichi_csr_pcg_jacobi_float64 | yes | success |  | applicable |  | 1 | 1 | 650.156 | 0 | 9.80226e-06 | 0.00255227 |
| suitesparse:HB/685_bus | taichi_csr_richardson_jacobi_float64 | no | screened_out | cpu_reference_richardson_screen_failed | applicable |  | 0 | 1 |  |  | 0.000170132 | 0.937174 |
| suitesparse:HB/arc130 | taichi_csr_bicgstab_none_float64 | no | screened_out | cpu_reference_bicgstab_screen_failed | applicable |  | 0 | 1 |  |  | 3.8921e-06 | 0.500077 |
| suitesparse:HB/arc130 | taichi_csr_bicgstab_jacobi_float64 | no | screened_out | cpu_reference_bicgstab_screen_failed | applicable |  | 0 | 1 |  |  | 2.11164e-06 | 97.4098 |
| suitesparse:HB/arc130 | taichi_csr_gmres_jacobi_restart16_float64 | no | screened_out | cpu_reference_gmres_screen_failed | applicable |  | 0 | 1 |  |  | 3.02451e-06 | 80810.8 |
| suitesparse:HB/ash292 | taichi_csr_cg_none_float64 | no | screened_out | cpu_reference_cg_screen_failed | applicable |  | 0 | 1 |  |  | 0.0547811 | 0.2302 |
| suitesparse:HB/ash292 | taichi_csr_pcg_jacobi_float64 | no | screened_out | cpu_reference_pcg_screen_failed | applicable |  | 0 | 1 |  |  | 0.0547811 | 0.2302 |
| suitesparse:HB/ash292 | taichi_csr_richardson_jacobi_float64 | no | screened_out | cpu_reference_richardson_screen_failed | applicable |  | 0 | 1 |  |  | 4.78449e+06 | 1.56276e+07 |
| suitesparse:HB/ash85 | taichi_csr_cg_none_float64 | no | screened_out | cpu_reference_cg_screen_failed | applicable |  | 0 | 1 |  |  | 0.172619 | 0.579531 |
| suitesparse:HB/ash85 | taichi_csr_pcg_jacobi_float64 | no | screened_out | cpu_reference_pcg_screen_failed | applicable |  | 0 | 1 |  |  | 0.172619 | 0.579531 |
| suitesparse:HB/ash85 | taichi_csr_richardson_jacobi_float64 | no | screened_out | cpu_reference_richardson_screen_failed | applicable |  | 0 | 1 |  |  | 216399 | 712193 |
| suitesparse:HB/bcspwr01 | taichi_csr_cg_none_float64 | no | screened_out | cpu_reference_cg_screen_failed | applicable |  | 0 | 1 |  |  | 0.120381 | 0.309944 |
| suitesparse:HB/bcspwr01 | taichi_csr_pcg_jacobi_float64 | no | screened_out | cpu_reference_pcg_screen_failed | applicable |  | 0 | 1 |  |  | 0.120381 | 0.309944 |
| suitesparse:HB/bcspwr01 | taichi_csr_richardson_jacobi_float64 | no | screened_out | cpu_reference_richardson_screen_failed | applicable |  | 0 | 1 |  |  | 77388.7 | 165919 |
