# CSR Selector Readiness

- status: `passed`
- schema_version: `phase1_csr_selector_features_v8`
- diagnostic_rows: `8`
- selector_rows: `24`
- success_rows: `1`
- failed_rows: `23`
- applicability_rows: `0`
- oracle_rows: `1`
- matrices_with_selector_rows: `8`
- max_final_relative_residual: `9.23618e-06`
- max_cpu_recomputed_relative_residual: `9.23618e-06`
- max_solution_relative_error: `6.52908e-05`
- min_success_rate: `0`
- max_solve_time_iqr_ms: `0`
- failure_reason_counts: `{'cpu_reference_cg_screen_failed': 8, 'cpu_reference_pcg_screen_failed': 7, 'cpu_reference_richardson_screen_failed': 8}`
- applicability_status_counts: `{}`
- applicability_reason_counts: `{}`

| matrix | candidate | oracle | status | failure | applicability | reason | success_rate | repeats | median_ms | regret_ms | rel_res | sol_err |
|---|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|
| suitesparse:HB/bcsstk16 | taichi_csr_cg_none_float64 | no | screened_out | cpu_reference_cg_screen_failed | applicable |  | 0 | 1 |  |  | 9.94167e-06 | 0.123091 |
| suitesparse:HB/bcsstk16 | taichi_csr_pcg_jacobi_float64 | yes | success |  | applicable |  | 1 | 1 | 1124.29 | 0 | 9.23618e-06 | 6.52908e-05 |
| suitesparse:HB/bcsstk16 | taichi_csr_richardson_jacobi_float64 | no | screened_out | cpu_reference_richardson_screen_failed | applicable |  | 0 | 1 |  |  | 0.00629634 | 0.286278 |
| suitesparse:HB/bcsstk17 | taichi_csr_cg_none_float64 | no | screened_out | cpu_reference_cg_screen_failed | applicable |  | 0 | 1 |  |  | 0.00241781 | 0.660408 |
| suitesparse:HB/bcsstk17 | taichi_csr_pcg_jacobi_float64 | no | screened_out | cpu_reference_pcg_screen_failed | applicable |  | 0 | 1 |  |  | 0.000373221 | 0.39466 |
| suitesparse:HB/bcsstk17 | taichi_csr_richardson_jacobi_float64 | no | screened_out | cpu_reference_richardson_screen_failed | applicable |  | 0 | 1 |  |  | 0.00423632 | 0.651831 |
| suitesparse:HB/bcsstk18 | taichi_csr_cg_none_float64 | no | screened_out | cpu_reference_cg_screen_failed | applicable |  | 0 | 1 |  |  | 0.000226271 | 0.729581 |
| suitesparse:HB/bcsstk18 | taichi_csr_pcg_jacobi_float64 | no | screened_out | cpu_reference_pcg_screen_failed | applicable |  | 0 | 1 |  |  | 9.40418e-06 | 0.159251 |
| suitesparse:HB/bcsstk18 | taichi_csr_richardson_jacobi_float64 | no | screened_out | cpu_reference_richardson_screen_failed | applicable |  | 0 | 1 |  |  | 0.000291833 | 1.15947 |
| suitesparse:HB/bcsstk19 | taichi_csr_cg_none_float64 | no | screened_out | cpu_reference_cg_screen_failed | applicable |  | 0 | 1 |  |  | 9.68271e-06 | 0.815041 |
| suitesparse:HB/bcsstk19 | taichi_csr_pcg_jacobi_float64 | no | screened_out | cpu_reference_pcg_screen_failed | applicable |  | 0 | 1 |  |  | 9.81307e-06 | 1.45474 |
| suitesparse:HB/bcsstk19 | taichi_csr_richardson_jacobi_float64 | no | screened_out | cpu_reference_richardson_screen_failed | applicable |  | 0 | 1 |  |  | 5.43943e-05 | 2.69761 |
| suitesparse:HB/bcsstk20 | taichi_csr_cg_none_float64 | no | screened_out | cpu_reference_cg_screen_failed | applicable |  | 0 | 1 |  |  | 8.60687e-06 | 0.817149 |
| suitesparse:HB/bcsstk20 | taichi_csr_pcg_jacobi_float64 | no | screened_out | cpu_reference_pcg_screen_failed | applicable |  | 0 | 1 |  |  | 9.83816e-06 | 14.8738 |
| suitesparse:HB/bcsstk20 | taichi_csr_richardson_jacobi_float64 | no | screened_out | cpu_reference_richardson_screen_failed | applicable |  | 0 | 1 |  |  | 0.00019983 | 10.782 |
| suitesparse:HB/bcsstk21 | taichi_csr_cg_none_float64 | no | screened_out | cpu_reference_cg_screen_failed | applicable |  | 0 | 1 |  |  | 6.35618e-05 | 0.352362 |
| suitesparse:HB/bcsstk21 | taichi_csr_pcg_jacobi_float64 | no | screened_out | cpu_reference_pcg_screen_failed | applicable |  | 0 | 1 |  |  | 9.4143e-06 | 0.188692 |
| suitesparse:HB/bcsstk21 | taichi_csr_richardson_jacobi_float64 | no | screened_out | cpu_reference_richardson_screen_failed | applicable |  | 0 | 1 |  |  | 0.00225332 | 0.401614 |
| suitesparse:HB/bcsstk22 | taichi_csr_cg_none_float64 | no | screened_out | cpu_reference_cg_screen_failed | applicable |  | 0 | 1 |  |  | 8.10679e-06 | 0.0418933 |
| suitesparse:HB/bcsstk22 | taichi_csr_pcg_jacobi_float64 | no | screened_out | cpu_reference_pcg_screen_failed | applicable |  | 0 | 1 |  |  | 7.27955e-06 | 0.0231212 |
| suitesparse:HB/bcsstk22 | taichi_csr_richardson_jacobi_float64 | no | screened_out | cpu_reference_richardson_screen_failed | applicable |  | 0 | 1 |  |  | 0.00013449 | 0.393149 |
| suitesparse:HB/bcsstk23 | taichi_csr_cg_none_float64 | no | screened_out | cpu_reference_cg_screen_failed | applicable |  | 0 | 1 |  |  | 1.6162e-05 | 0.750538 |
| suitesparse:HB/bcsstk23 | taichi_csr_pcg_jacobi_float64 | no | screened_out | cpu_reference_pcg_screen_failed | applicable |  | 0 | 1 |  |  | 9.23225e-06 | 16.928 |
| suitesparse:HB/bcsstk23 | taichi_csr_richardson_jacobi_float64 | no | screened_out | cpu_reference_richardson_screen_failed | applicable |  | 0 | 1 |  |  | 0.000207002 | 24.7046 |
