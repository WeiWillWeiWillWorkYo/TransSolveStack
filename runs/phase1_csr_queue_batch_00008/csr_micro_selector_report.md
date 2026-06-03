# CSR Selector Readiness

- status: `passed`
- schema_version: `phase1_csr_selector_features_v8`
- diagnostic_rows: `8`
- selector_rows: `24`
- success_rows: `12`
- failed_rows: `12`
- applicability_rows: `0`
- oracle_rows: `5`
- matrices_with_selector_rows: `8`
- max_final_relative_residual: `6.87101e-06`
- max_cpu_recomputed_relative_residual: `6.87101e-06`
- max_solution_relative_error: `2.87597e-05`
- min_success_rate: `0`
- max_solve_time_iqr_ms: `0`
- failure_reason_counts: `{'cpu_reference_cg_screen_failed': 6, 'cpu_reference_pcg_screen_failed': 3, 'cpu_reference_richardson_screen_failed': 3}`
- applicability_status_counts: `{}`
- applicability_reason_counts: `{}`

| matrix | candidate | oracle | status | failure | applicability | reason | success_rate | repeats | median_ms | regret_ms | rel_res | sol_err |
|---|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|
| suitesparse:HB/bcsstm10 | taichi_csr_cg_none_float64 | no | screened_out | cpu_reference_cg_screen_failed | applicable |  | 0 | 1 |  |  | 1.00749 | 1.15239 |
| suitesparse:HB/bcsstm10 | taichi_csr_pcg_jacobi_float64 | no | screened_out | cpu_reference_pcg_screen_failed | applicable |  | 0 | 1 |  |  | 0.457834 | 0.962214 |
| suitesparse:HB/bcsstm10 | taichi_csr_richardson_jacobi_float64 | no | screened_out | cpu_reference_richardson_screen_failed | applicable |  | 0 | 1 |  |  | 85310.8 | 79223.1 |
| suitesparse:HB/bcsstm11 | taichi_csr_cg_none_float64 | no | screened_out | cpu_reference_cg_screen_failed | applicable |  | 0 | 1 |  |  | 9.5631e-06 | 0.183107 |
| suitesparse:HB/bcsstm11 | taichi_csr_pcg_jacobi_float64 | no | success |  | applicable |  | 1 | 1 | 794.58 | 326.579 | 0 | 0 |
| suitesparse:HB/bcsstm11 | taichi_csr_richardson_jacobi_float64 | yes | success |  | applicable |  | 1 | 1 | 468.001 | 0 | 5.64503e-06 | 5.64503e-06 |
| suitesparse:HB/bcsstm12 | taichi_csr_cg_none_float64 | no | screened_out | cpu_reference_cg_screen_failed | applicable |  | 0 | 1 |  |  | 0.000435499 | 0.114988 |
| suitesparse:HB/bcsstm12 | taichi_csr_pcg_jacobi_float64 | no | screened_out | cpu_reference_pcg_screen_failed | applicable |  | 0 | 1 |  |  | 9.75569e-06 | 0.0115877 |
| suitesparse:HB/bcsstm12 | taichi_csr_richardson_jacobi_float64 | no | screened_out | cpu_reference_richardson_screen_failed | applicable |  | 0 | 1 |  |  | 0.000163549 | 0.473675 |
| suitesparse:HB/bcsstm13 | taichi_csr_cg_none_float64 | no | screened_out | cpu_reference_cg_screen_failed | applicable |  | 0 | 1 |  |  | 9.0685e-06 | 0.616791 |
| suitesparse:HB/bcsstm13 | taichi_csr_pcg_jacobi_float64 | no | screened_out | cpu_reference_pcg_screen_failed | applicable |  | 0 | 1 |  |  | 8.42238e-06 | 0.61679 |
| suitesparse:HB/bcsstm13 | taichi_csr_richardson_jacobi_float64 | no | screened_out | cpu_reference_richardson_screen_failed | applicable |  | 0 | 1 |  |  | inf | inf |
| suitesparse:HB/bcsstm19 | taichi_csr_cg_none_float64 | no | screened_out | cpu_reference_cg_screen_failed | applicable |  | 0 | 1 |  |  | 7.30233e-06 | 0.0311531 |
| suitesparse:HB/bcsstm19 | taichi_csr_pcg_jacobi_float64 | yes | success |  | applicable |  | 1 | 1 | 895.083 | 0 | 0 | 0 |
| suitesparse:HB/bcsstm19 | taichi_csr_richardson_jacobi_float64 | no | success |  | applicable |  | 1 | 1 | 1010.16 | 115.075 | 5.64503e-06 | 5.64503e-06 |
| suitesparse:HB/bcsstm20 | taichi_csr_cg_none_float64 | no | screened_out | cpu_reference_cg_screen_failed | applicable |  | 0 | 1 |  |  | 7.0591e-06 | 0.0270976 |
| suitesparse:HB/bcsstm20 | taichi_csr_pcg_jacobi_float64 | no | success |  | applicable |  | 1 | 1 | 600.919 | 38.2307 | 0 | 0 |
| suitesparse:HB/bcsstm20 | taichi_csr_richardson_jacobi_float64 | yes | success |  | applicable |  | 1 | 1 | 562.688 | 0 | 5.64503e-06 | 5.64503e-06 |
| suitesparse:HB/bcsstm21 | taichi_csr_cg_none_float64 | yes | success |  | applicable |  | 1 | 1 | 1033.82 | 0 | 3.38303e-06 | 2.3964e-06 |
| suitesparse:HB/bcsstm21 | taichi_csr_pcg_jacobi_float64 | no | success |  | applicable |  | 1 | 1 | 1173.92 | 140.102 | 0 | 0 |
| suitesparse:HB/bcsstm21 | taichi_csr_richardson_jacobi_float64 | no | success |  | applicable |  | 1 | 1 | 1091.73 | 57.9079 | 5.64503e-06 | 5.64503e-06 |
| suitesparse:HB/bcsstm22 | taichi_csr_cg_none_float64 | no | success |  | applicable |  | 1 | 1 | 1108.25 | 744.864 | 6.87101e-06 | 2.87597e-05 |
| suitesparse:HB/bcsstm22 | taichi_csr_pcg_jacobi_float64 | no | success |  | applicable |  | 1 | 1 | 632.086 | 268.698 | 0 | 0 |
| suitesparse:HB/bcsstm22 | taichi_csr_richardson_jacobi_float64 | yes | success |  | applicable |  | 1 | 1 | 363.388 | 0 | 5.64503e-06 | 5.64503e-06 |
