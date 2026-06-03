# CSR Selector Readiness

- status: `failed`
- schema_version: `phase1_csr_selector_features_v8`
- diagnostic_rows: `8`
- selector_rows: `24`
- success_rows: `0`
- failed_rows: `24`
- applicability_rows: `0`
- oracle_rows: `0`
- matrices_with_selector_rows: `8`
- max_final_relative_residual: `0`
- max_cpu_recomputed_relative_residual: `0`
- max_solution_relative_error: `0`
- min_success_rate: `0`
- max_solve_time_iqr_ms: `0`
- failure_reason_counts: `{'cpu_reference_bicgstab_screen_failed': 14, 'cpu_reference_cg_screen_failed': 1, 'cpu_reference_gmres_screen_failed': 7, 'cpu_reference_pcg_screen_failed': 1, 'cpu_reference_richardson_screen_failed': 1}`
- applicability_status_counts: `{}`
- applicability_reason_counts: `{}`

| matrix | candidate | oracle | status | failure | applicability | reason | success_rate | repeats | median_ms | regret_ms | rel_res | sol_err |
|---|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|
| suitesparse:HB/bp_1200 | taichi_csr_bicgstab_none_float64 | no | screened_out | cpu_reference_bicgstab_screen_failed | applicable |  | 0 | 1 |  |  | 2255.21 | 21605.9 |
| suitesparse:HB/bp_1200 | taichi_csr_bicgstab_jacobi_float64 | no | screened_out | cpu_reference_bicgstab_screen_failed | applicable |  | 0 | 1 |  |  | 66.2673 | 404.836 |
| suitesparse:HB/bp_1200 | taichi_csr_gmres_jacobi_restart16_float64 | no | screened_out | cpu_reference_gmres_screen_failed | applicable |  | 0 | 1 |  |  | 0.849478 | 2.52897 |
| suitesparse:HB/bp_1400 | taichi_csr_bicgstab_none_float64 | no | screened_out | cpu_reference_bicgstab_screen_failed | applicable |  | 0 | 1 |  |  | 508.467 | 1551.01 |
| suitesparse:HB/bp_1400 | taichi_csr_bicgstab_jacobi_float64 | no | screened_out | cpu_reference_bicgstab_screen_failed | applicable |  | 0 | 1 |  |  | 18.045 | 101.705 |
| suitesparse:HB/bp_1400 | taichi_csr_gmres_jacobi_restart16_float64 | no | screened_out | cpu_reference_gmres_screen_failed | applicable |  | 0 | 1 |  |  | 0.807191 | 3.1923 |
| suitesparse:HB/bp_1600 | taichi_csr_bicgstab_none_float64 | no | screened_out | cpu_reference_bicgstab_screen_failed | applicable |  | 0 | 1 |  |  | 36.0645 | 165.728 |
| suitesparse:HB/bp_1600 | taichi_csr_bicgstab_jacobi_float64 | no | screened_out | cpu_reference_bicgstab_screen_failed | applicable |  | 0 | 1 |  |  | 11.7315 | 48.1003 |
| suitesparse:HB/bp_1600 | taichi_csr_gmres_jacobi_restart16_float64 | no | screened_out | cpu_reference_gmres_screen_failed | applicable |  | 0 | 1 |  |  | 0.81715 | 3.03109 |
| suitesparse:HB/bp_200 | taichi_csr_bicgstab_none_float64 | no | screened_out | cpu_reference_bicgstab_screen_failed | applicable |  | 0 | 1 |  |  | 106.693 | 714.358 |
| suitesparse:HB/bp_200 | taichi_csr_bicgstab_jacobi_float64 | no | screened_out | cpu_reference_bicgstab_screen_failed | applicable |  | 0 | 1 |  |  | 134.205 | 491.131 |
| suitesparse:HB/bp_200 | taichi_csr_gmres_jacobi_restart16_float64 | no | screened_out | cpu_reference_gmres_screen_failed | applicable |  | 0 | 1 |  |  | 0.827915 | 4.51018 |
| suitesparse:HB/bp_400 | taichi_csr_bicgstab_none_float64 | no | screened_out | cpu_reference_bicgstab_screen_failed | applicable |  | 0 | 1 |  |  | 284.689 | 2545.45 |
| suitesparse:HB/bp_400 | taichi_csr_bicgstab_jacobi_float64 | no | screened_out | cpu_reference_bicgstab_screen_failed | applicable |  | 0 | 1 |  |  | 708.369 | 4880.63 |
| suitesparse:HB/bp_400 | taichi_csr_gmres_jacobi_restart16_float64 | no | screened_out | cpu_reference_gmres_screen_failed | applicable |  | 0 | 1 |  |  | 0.739576 | 4.86005 |
| suitesparse:HB/bp_600 | taichi_csr_bicgstab_none_float64 | no | screened_out | cpu_reference_bicgstab_screen_failed | applicable |  | 0 | 1 |  |  | 70.8259 | 265.062 |
| suitesparse:HB/bp_600 | taichi_csr_bicgstab_jacobi_float64 | no | screened_out | cpu_reference_bicgstab_screen_failed | applicable |  | 0 | 1 |  |  | 30.9947 | 177.055 |
| suitesparse:HB/bp_600 | taichi_csr_gmres_jacobi_restart16_float64 | no | screened_out | cpu_reference_gmres_screen_failed | applicable |  | 0 | 1 |  |  | 0.823465 | 4.26118 |
| suitesparse:HB/bp_800 | taichi_csr_bicgstab_none_float64 | no | screened_out | cpu_reference_bicgstab_screen_failed | applicable |  | 0 | 1 |  |  | 921.696 | 4073.42 |
| suitesparse:HB/bp_800 | taichi_csr_bicgstab_jacobi_float64 | no | screened_out | cpu_reference_bicgstab_screen_failed | applicable |  | 0 | 1 |  |  | 101.811 | 698.342 |
| suitesparse:HB/bp_800 | taichi_csr_gmres_jacobi_restart16_float64 | no | screened_out | cpu_reference_gmres_screen_failed | applicable |  | 0 | 1 |  |  | 0.809933 | 2.11101 |
| suitesparse:HB/can_1054 | taichi_csr_cg_none_float64 | no | screened_out | cpu_reference_cg_screen_failed | applicable |  | 0 | 1 |  |  | 0.0821189 | 0.498182 |
| suitesparse:HB/can_1054 | taichi_csr_pcg_jacobi_float64 | no | screened_out | cpu_reference_pcg_screen_failed | applicable |  | 0 | 1 |  |  | 0.0821189 | 0.498182 |
| suitesparse:HB/can_1054 | taichi_csr_richardson_jacobi_float64 | no | screened_out | cpu_reference_richardson_screen_failed | applicable |  | 0 | 1 |  |  | 8.23578e+14 | 2.34717e+15 |
