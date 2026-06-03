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
- max_final_relative_residual: `9.58594e-06`
- max_cpu_recomputed_relative_residual: `9.58594e-06`
- max_solution_relative_error: `0.000154675`
- min_success_rate: `0`
- max_solve_time_iqr_ms: `0`
- failure_reason_counts: `{'cpu_reference_cg_screen_failed': 8, 'cpu_reference_pcg_screen_failed': 7, 'cpu_reference_richardson_screen_failed': 8}`
- applicability_status_counts: `{}`
- applicability_reason_counts: `{}`

| matrix | candidate | oracle | status | failure | applicability | reason | success_rate | repeats | median_ms | regret_ms | rel_res | sol_err |
|---|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|
| suitesparse:HB/bcsstk24 | taichi_csr_cg_none_float64 | no | screened_out | cpu_reference_cg_screen_failed | applicable |  | 0 | 1 |  |  | 3.18224e-05 | 0.740676 |
| suitesparse:HB/bcsstk24 | taichi_csr_pcg_jacobi_float64 | no | screened_out | cpu_reference_pcg_screen_failed | applicable |  | 0 | 1 |  |  | 9.21286e-06 | 2.5502 |
| suitesparse:HB/bcsstk24 | taichi_csr_richardson_jacobi_float64 | no | screened_out | cpu_reference_richardson_screen_failed | applicable |  | 0 | 1 |  |  | 0.000723424 | 11.3113 |
| suitesparse:HB/bcsstk25 | taichi_csr_cg_none_float64 | no | screened_out | cpu_reference_cg_screen_failed | applicable |  | 0 | 1 |  |  | 7.43124e-05 | 0.728571 |
| suitesparse:HB/bcsstk25 | taichi_csr_pcg_jacobi_float64 | no | screened_out | cpu_reference_pcg_screen_failed | applicable |  | 0 | 1 |  |  | 8.72074e-06 | 0.646381 |
| suitesparse:HB/bcsstk25 | taichi_csr_richardson_jacobi_float64 | no | screened_out | cpu_reference_richardson_screen_failed | applicable |  | 0 | 1 |  |  | 0.00370092 | 3.08146 |
| suitesparse:HB/bcsstk26 | taichi_csr_cg_none_float64 | no | screened_out | cpu_reference_cg_screen_failed | applicable |  | 0 | 1 |  |  | 0.00054983 | 0.650528 |
| suitesparse:HB/bcsstk26 | taichi_csr_pcg_jacobi_float64 | no | screened_out | cpu_reference_pcg_screen_failed | applicable |  | 0 | 1 |  |  | 3.22243e-05 | 0.157974 |
| suitesparse:HB/bcsstk26 | taichi_csr_richardson_jacobi_float64 | no | screened_out | cpu_reference_richardson_screen_failed | applicable |  | 0 | 1 |  |  | 0.00140829 | 1.45385 |
| suitesparse:HB/bcsstk27 | taichi_csr_cg_none_float64 | no | screened_out | cpu_reference_cg_screen_failed | applicable |  | 0 | 1 |  |  | 0.00012016 | 0.00429484 |
| suitesparse:HB/bcsstk27 | taichi_csr_pcg_jacobi_float64 | yes | success |  | applicable |  | 1 | 1 | 1703.62 | 0 | 9.58594e-06 | 0.000154675 |
| suitesparse:HB/bcsstk27 | taichi_csr_richardson_jacobi_float64 | no | screened_out | cpu_reference_richardson_screen_failed | applicable |  | 0 | 1 |  |  | 0.0135761 | 0.420013 |
| suitesparse:HB/bcsstk28 | taichi_csr_cg_none_float64 | no | screened_out | cpu_reference_cg_screen_failed | applicable |  | 0 | 1 |  |  | 0.00928338 | 0.874905 |
| suitesparse:HB/bcsstk28 | taichi_csr_pcg_jacobi_float64 | no | screened_out | cpu_reference_pcg_screen_failed | applicable |  | 0 | 1 |  |  | 0.00170142 | 0.791927 |
| suitesparse:HB/bcsstk28 | taichi_csr_richardson_jacobi_float64 | no | screened_out | cpu_reference_richardson_screen_failed | applicable |  | 0 | 1 |  |  | 0.0103604 | 0.9439 |
| suitesparse:HB/bcsstk29 | taichi_csr_cg_none_float64 | no | screened_out | cpu_reference_cg_screen_failed | applicable |  | 0 | 1 |  |  | 0.34147 | 1.00046 |
| suitesparse:HB/bcsstk29 | taichi_csr_pcg_jacobi_float64 | no | screened_out | cpu_reference_pcg_screen_failed | applicable |  | 0 | 1 |  |  | 0.34147 | 1.00046 |
| suitesparse:HB/bcsstk29 | taichi_csr_richardson_jacobi_float64 | no | screened_out | cpu_reference_richardson_screen_failed | applicable |  | 0 | 1 |  |  | 2.55167e+56 | 6.57984e+56 |
| suitesparse:HB/bcsstk33 | taichi_csr_cg_none_float64 | no | screened_out | cpu_reference_cg_screen_failed | applicable |  | 0 | 1 |  |  | 0.0515266 | 0.207475 |
| suitesparse:HB/bcsstk33 | taichi_csr_pcg_jacobi_float64 | no | screened_out | cpu_reference_pcg_screen_failed | applicable |  | 0 | 1 |  |  | 0.0515266 | 0.207475 |
| suitesparse:HB/bcsstk33 | taichi_csr_richardson_jacobi_float64 | no | screened_out | cpu_reference_richardson_screen_failed | applicable |  | 0 | 1 |  |  | 2.10361e+72 | 5.91464e+72 |
| suitesparse:HB/bcsstm01 | taichi_csr_cg_none_float64 | no | screened_out | cpu_reference_cg_screen_failed | applicable |  | 0 | 1 |  |  | 1.58882e-17 | 0.707107 |
| suitesparse:HB/bcsstm01 | taichi_csr_pcg_jacobi_float64 | no | screened_out | cpu_reference_pcg_screen_failed | applicable |  | 0 | 1 |  |  | 0 | 0.707107 |
| suitesparse:HB/bcsstm01 | taichi_csr_richardson_jacobi_float64 | no | screened_out | cpu_reference_richardson_screen_failed | applicable |  | 0 | 1 |  |  | inf | inf |
