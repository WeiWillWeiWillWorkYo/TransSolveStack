# CSR Selector Readiness

- status: `passed`
- schema_version: `phase1_csr_selector_features_v8`
- diagnostic_rows: `8`
- selector_rows: `24`
- success_rows: `16`
- failed_rows: `8`
- applicability_rows: `0`
- oracle_rows: `6`
- matrices_with_selector_rows: `8`
- max_final_relative_residual: `9.97184e-06`
- max_cpu_recomputed_relative_residual: `9.97184e-06`
- max_solution_relative_error: `0.00071122`
- min_success_rate: `0`
- max_solve_time_iqr_ms: `0`
- failure_reason_counts: `{'cpu_reference_cg_screen_failed': 4, 'cpu_reference_pcg_screen_failed': 2, 'cpu_reference_richardson_screen_failed': 2}`
- applicability_status_counts: `{}`
- applicability_reason_counts: `{}`

| matrix | candidate | oracle | status | failure | applicability | reason | success_rate | repeats | median_ms | regret_ms | rel_res | sol_err |
|---|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|
| suitesparse:HB/bcsstm02 | taichi_csr_cg_none_float64 | no | success |  | applicable |  | 1 | 1 | 993.57 | 414.075 | 2.07415e-08 | 1.21438e-08 |
| suitesparse:HB/bcsstm02 | taichi_csr_pcg_jacobi_float64 | yes | success |  | applicable |  | 1 | 1 | 579.495 | 0 | 0 | 0 |
| suitesparse:HB/bcsstm02 | taichi_csr_richardson_jacobi_float64 | no | success |  | applicable |  | 1 | 1 | 637.509 | 58.014 | 5.64503e-06 | 5.64503e-06 |
| suitesparse:HB/bcsstm03 | taichi_csr_cg_none_float64 | no | screened_out | cpu_reference_cg_screen_failed | applicable |  | 0 | 1 |  |  | 8.53849e-06 | 0.598443 |
| suitesparse:HB/bcsstm03 | taichi_csr_pcg_jacobi_float64 | no | screened_out | cpu_reference_pcg_screen_failed | applicable |  | 0 | 1 |  |  | 0 | 0.597614 |
| suitesparse:HB/bcsstm03 | taichi_csr_richardson_jacobi_float64 | no | screened_out | cpu_reference_richardson_screen_failed | applicable |  | 0 | 1 |  |  | inf | inf |
| suitesparse:HB/bcsstm04 | taichi_csr_cg_none_float64 | no | screened_out | cpu_reference_cg_screen_failed | applicable |  | 0 | 1 |  |  | 1.9084e-16 | 0.707107 |
| suitesparse:HB/bcsstm04 | taichi_csr_pcg_jacobi_float64 | no | screened_out | cpu_reference_pcg_screen_failed | applicable |  | 0 | 1 |  |  | 0 | 0.707107 |
| suitesparse:HB/bcsstm04 | taichi_csr_richardson_jacobi_float64 | no | screened_out | cpu_reference_richardson_screen_failed | applicable |  | 0 | 1 |  |  | inf | inf |
| suitesparse:HB/bcsstm05 | taichi_csr_cg_none_float64 | no | success |  | applicable |  | 1 | 1 | 708.349 | 340.392 | 3.98604e-06 | 1.012e-05 |
| suitesparse:HB/bcsstm05 | taichi_csr_pcg_jacobi_float64 | no | success |  | applicable |  | 1 | 1 | 813.824 | 445.868 | 0 | 0 |
| suitesparse:HB/bcsstm05 | taichi_csr_richardson_jacobi_float64 | yes | success |  | applicable |  | 1 | 1 | 367.957 | 0 | 5.64503e-06 | 5.64503e-06 |
| suitesparse:HB/bcsstm06 | taichi_csr_cg_none_float64 | no | screened_out | cpu_reference_cg_screen_failed | applicable |  | 0 | 1 |  |  | 4.85334e-06 | 0.217263 |
| suitesparse:HB/bcsstm06 | taichi_csr_pcg_jacobi_float64 | no | success |  | applicable |  | 1 | 1 | 855.372 | 494.077 | 0 | 0 |
| suitesparse:HB/bcsstm06 | taichi_csr_richardson_jacobi_float64 | yes | success |  | applicable |  | 1 | 1 | 361.296 | 0 | 5.64503e-06 | 5.64503e-06 |
| suitesparse:HB/bcsstm07 | taichi_csr_cg_none_float64 | no | success |  | applicable |  | 1 | 1 | 1940.34 | 615.183 | 9.97184e-06 | 0.00071122 |
| suitesparse:HB/bcsstm07 | taichi_csr_pcg_jacobi_float64 | no | success |  | applicable |  | 1 | 1 | 1353.44 | 28.2796 | 8.60409e-06 | 9.4417e-05 |
| suitesparse:HB/bcsstm07 | taichi_csr_richardson_jacobi_float64 | yes | success |  | applicable |  | 1 | 1 | 1325.16 | 0 | 9.5507e-06 | 0.00023176 |
| suitesparse:HB/bcsstm08 | taichi_csr_cg_none_float64 | no | screened_out | cpu_reference_cg_screen_failed | applicable |  | 0 | 1 |  |  | 3.32255e-06 | 0.203383 |
| suitesparse:HB/bcsstm08 | taichi_csr_pcg_jacobi_float64 | no | success |  | applicable |  | 1 | 1 | 721.41 | 260.885 | 0 | 0 |
| suitesparse:HB/bcsstm08 | taichi_csr_richardson_jacobi_float64 | yes | success |  | applicable |  | 1 | 1 | 460.524 | 0 | 5.64503e-06 | 5.64503e-06 |
| suitesparse:HB/bcsstm09 | taichi_csr_cg_none_float64 | no | success |  | applicable |  | 1 | 1 | 799.353 | 346.695 | 1.1626e-11 | 3.66166e-08 |
| suitesparse:HB/bcsstm09 | taichi_csr_pcg_jacobi_float64 | no | success |  | applicable |  | 1 | 1 | 616.821 | 164.163 | 0 | 0 |
| suitesparse:HB/bcsstm09 | taichi_csr_richardson_jacobi_float64 | yes | success |  | applicable |  | 1 | 1 | 452.657 | 0 | 5.64503e-06 | 5.64503e-06 |
