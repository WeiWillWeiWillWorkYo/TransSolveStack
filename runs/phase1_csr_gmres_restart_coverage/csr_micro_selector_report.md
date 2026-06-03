# CSR Selector Readiness

- status: `failed`
- schema_version: `phase1_csr_selector_features_v8`
- diagnostic_rows: `7`
- selector_rows: `14`
- success_rows: `0`
- failed_rows: `14`
- applicability_rows: `0`
- oracle_rows: `0`
- matrices_with_selector_rows: `7`
- max_final_relative_residual: `0`
- max_cpu_recomputed_relative_residual: `0`
- max_solution_relative_error: `0`
- min_success_rate: `0`
- max_solve_time_iqr_ms: `0`
- failure_reason_counts: `{'cpu_reference_gmres_screen_failed': 14}`
- applicability_status_counts: `{}`
- applicability_reason_counts: `{}`

| matrix | candidate | oracle | status | failure | applicability | reason | success_rate | repeats | median_ms | regret_ms | rel_res | sol_err |
|---|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|
| suitesparse:HB/bp_1200 | taichi_csr_gmres_jacobi_restart32_float64 | no | screened_out | cpu_reference_gmres_screen_failed | applicable |  | 0 | 1 |  |  | 0.666409 | 6.42368 |
| suitesparse:HB/bp_1200 | taichi_csr_gmres_jacobi_restart64_float64 | no | screened_out | cpu_reference_gmres_screen_failed | applicable |  | 0 | 1 |  |  | 0.527699 | 5.80837 |
| suitesparse:HB/bp_1400 | taichi_csr_gmres_jacobi_restart32_float64 | no | screened_out | cpu_reference_gmres_screen_failed | applicable |  | 0 | 1 |  |  | 0.635971 | 4.01777 |
| suitesparse:HB/bp_1400 | taichi_csr_gmres_jacobi_restart64_float64 | no | screened_out | cpu_reference_gmres_screen_failed | applicable |  | 0 | 1 |  |  | 0.527041 | 4.72344 |
| suitesparse:HB/bp_1600 | taichi_csr_gmres_jacobi_restart32_float64 | no | screened_out | cpu_reference_gmres_screen_failed | applicable |  | 0 | 1 |  |  | 0.66306 | 3.06277 |
| suitesparse:HB/bp_1600 | taichi_csr_gmres_jacobi_restart64_float64 | no | screened_out | cpu_reference_gmres_screen_failed | applicable |  | 0 | 1 |  |  | 0.550223 | 9.11372 |
| suitesparse:HB/bp_200 | taichi_csr_gmres_jacobi_restart32_float64 | no | screened_out | cpu_reference_gmres_screen_failed | applicable |  | 0 | 1 |  |  | 0.703558 | 25.4326 |
| suitesparse:HB/bp_200 | taichi_csr_gmres_jacobi_restart64_float64 | no | screened_out | cpu_reference_gmres_screen_failed | applicable |  | 0 | 1 |  |  | 0.438621 | 23.4343 |
| suitesparse:HB/bp_400 | taichi_csr_gmres_jacobi_restart32_float64 | no | screened_out | cpu_reference_gmres_screen_failed | applicable |  | 0 | 1 |  |  | 0.601882 | 12.7312 |
| suitesparse:HB/bp_400 | taichi_csr_gmres_jacobi_restart64_float64 | no | screened_out | cpu_reference_gmres_screen_failed | applicable |  | 0 | 1 |  |  | 0.447162 | 6.93138 |
| suitesparse:HB/bp_600 | taichi_csr_gmres_jacobi_restart32_float64 | no | screened_out | cpu_reference_gmres_screen_failed | applicable |  | 0 | 1 |  |  | 0.656484 | 6.92126 |
| suitesparse:HB/bp_600 | taichi_csr_gmres_jacobi_restart64_float64 | no | screened_out | cpu_reference_gmres_screen_failed | applicable |  | 0 | 1 |  |  | 0.461241 | 7.89127 |
| suitesparse:HB/bp_800 | taichi_csr_gmres_jacobi_restart32_float64 | no | screened_out | cpu_reference_gmres_screen_failed | applicable |  | 0 | 1 |  |  | 0.650069 | 4.43349 |
| suitesparse:HB/bp_800 | taichi_csr_gmres_jacobi_restart64_float64 | no | screened_out | cpu_reference_gmres_screen_failed | applicable |  | 0 | 1 |  |  | 0.548828 | 4.0117 |
