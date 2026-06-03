# CSR Selector Readiness

- status: `failed`
- schema_version: `phase1_csr_selector_features_v8`
- diagnostic_rows: `8`
- selector_rows: `16`
- success_rows: `0`
- failed_rows: `16`
- applicability_rows: `0`
- oracle_rows: `0`
- matrices_with_selector_rows: `8`
- max_final_relative_residual: `0`
- max_cpu_recomputed_relative_residual: `0`
- max_solution_relative_error: `0`
- min_success_rate: `0`
- max_solve_time_iqr_ms: `0`
- failure_reason_counts: `{'cpu_reference_bicgstab_screen_failed': 14, 'cpu_reference_chebyshev_screen_failed': 1, 'cpu_reference_pcg_screen_failed': 1}`
- applicability_status_counts: `{}`
- applicability_reason_counts: `{}`

| matrix | candidate | oracle | status | failure | applicability | reason | success_rate | repeats | median_ms | regret_ms | rel_res | sol_err |
|---|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|
| suitesparse:HB/bp_1200 | taichi_csr_bicgstab_ilu0_float64 | no | screened_out | cpu_reference_bicgstab_screen_failed | applicable |  | 0 | 1 |  |  | inf | inf |
| suitesparse:HB/bp_1200 | taichi_csr_bicgstab_row_column_equilibration_float64 | no | screened_out | cpu_reference_bicgstab_screen_failed | applicable |  | 0 | 1 |  |  | 11.1564 | 306.996 |
| suitesparse:HB/bp_1400 | taichi_csr_bicgstab_ilu0_float64 | no | screened_out | cpu_reference_bicgstab_screen_failed | applicable |  | 0 | 1 |  |  | inf | inf |
| suitesparse:HB/bp_1400 | taichi_csr_bicgstab_row_column_equilibration_float64 | no | screened_out | cpu_reference_bicgstab_screen_failed | applicable |  | 0 | 1 |  |  | 60.6812 | 1509.32 |
| suitesparse:HB/bp_1600 | taichi_csr_bicgstab_ilu0_float64 | no | screened_out | cpu_reference_bicgstab_screen_failed | applicable |  | 0 | 1 |  |  | inf | inf |
| suitesparse:HB/bp_1600 | taichi_csr_bicgstab_row_column_equilibration_float64 | no | screened_out | cpu_reference_bicgstab_screen_failed | applicable |  | 0 | 1 |  |  | 13.6299 | 376.834 |
| suitesparse:HB/bp_200 | taichi_csr_bicgstab_ilu0_float64 | no | screened_out | cpu_reference_bicgstab_screen_failed | applicable |  | 0 | 1 |  |  | inf | inf |
| suitesparse:HB/bp_200 | taichi_csr_bicgstab_row_column_equilibration_float64 | no | screened_out | cpu_reference_bicgstab_screen_failed | applicable |  | 0 | 1 |  |  | 32.4531 | 935.599 |
| suitesparse:HB/bp_400 | taichi_csr_bicgstab_ilu0_float64 | no | screened_out | cpu_reference_bicgstab_screen_failed | applicable |  | 0 | 1 |  |  | inf | inf |
| suitesparse:HB/bp_400 | taichi_csr_bicgstab_row_column_equilibration_float64 | no | screened_out | cpu_reference_bicgstab_screen_failed | applicable |  | 0 | 1 |  |  | 47.3449 | 1150.45 |
| suitesparse:HB/bp_600 | taichi_csr_bicgstab_ilu0_float64 | no | screened_out | cpu_reference_bicgstab_screen_failed | applicable |  | 0 | 1 |  |  | inf | inf |
| suitesparse:HB/bp_600 | taichi_csr_bicgstab_row_column_equilibration_float64 | no | screened_out | cpu_reference_bicgstab_screen_failed | applicable |  | 0 | 1 |  |  | 11.3422 | 297.373 |
| suitesparse:HB/bp_800 | taichi_csr_bicgstab_ilu0_float64 | no | screened_out | cpu_reference_bicgstab_screen_failed | applicable |  | 0 | 1 |  |  | inf | inf |
| suitesparse:HB/bp_800 | taichi_csr_bicgstab_row_column_equilibration_float64 | no | screened_out | cpu_reference_bicgstab_screen_failed | applicable |  | 0 | 1 |  |  | 9.77964 | 196.672 |
| suitesparse:HB/can_1054 | taichi_csr_chebyshev_jacobi_float64 | no | screened_out | cpu_reference_chebyshev_screen_failed | applicable |  | 0 | 1 |  |  | inf | inf |
| suitesparse:HB/can_1054 | taichi_csr_pcg_symmetric_equilibration_float64 | no | screened_out | cpu_reference_pcg_screen_failed | applicable |  | 0 | 1 |  |  | 0.0821189 | 0.498182 |
