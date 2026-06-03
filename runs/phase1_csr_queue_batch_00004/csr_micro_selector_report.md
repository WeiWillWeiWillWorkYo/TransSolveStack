# CSR Selector Readiness

- status: `passed`
- schema_version: `phase1_csr_selector_features_v8`
- diagnostic_rows: `8`
- selector_rows: `24`
- success_rows: `3`
- failed_rows: `21`
- applicability_rows: `0`
- oracle_rows: `2`
- matrices_with_selector_rows: `8`
- max_final_relative_residual: `9.83561e-06`
- max_cpu_recomputed_relative_residual: `9.83561e-06`
- max_solution_relative_error: `0.00434613`
- min_success_rate: `0`
- max_solve_time_iqr_ms: `0`
- failure_reason_counts: `{'cpu_reference_cg_screen_failed': 7, 'cpu_reference_pcg_screen_failed': 6, 'cpu_reference_richardson_screen_failed': 8}`
- applicability_status_counts: `{}`
- applicability_reason_counts: `{}`

| matrix | candidate | oracle | status | failure | applicability | reason | success_rate | repeats | median_ms | regret_ms | rel_res | sol_err |
|---|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|
| suitesparse:HB/bcsstk08 | taichi_csr_cg_none_float64 | no | screened_out | cpu_reference_cg_screen_failed | applicable |  | 0 | 1 |  |  | 4.24404e-05 | 0.398501 |
| suitesparse:HB/bcsstk08 | taichi_csr_pcg_jacobi_float64 | no | screened_out | cpu_reference_pcg_screen_failed | applicable |  | 0 | 1 |  |  | 8.80762e-06 | 0.0191595 |
| suitesparse:HB/bcsstk08 | taichi_csr_richardson_jacobi_float64 | no | screened_out | cpu_reference_richardson_screen_failed | applicable |  | 0 | 1 |  |  | 0.000211145 | 1.41184 |
| suitesparse:HB/bcsstk09 | taichi_csr_cg_none_float64 | yes | success |  | applicable |  | 1 | 1 | 1049.39 | 0 | 9.50179e-06 | 0.000186387 |
| suitesparse:HB/bcsstk09 | taichi_csr_pcg_jacobi_float64 | no | success |  | applicable |  | 1 | 1 | 1309.93 | 260.539 | 8.60083e-06 | 0.000432482 |
| suitesparse:HB/bcsstk09 | taichi_csr_richardson_jacobi_float64 | no | screened_out | cpu_reference_richardson_screen_failed | applicable |  | 0 | 1 |  |  | 0.00177009 | 0.448329 |
| suitesparse:HB/bcsstk10 | taichi_csr_cg_none_float64 | no | screened_out | cpu_reference_cg_screen_failed | applicable |  | 0 | 1 |  |  | 0.00020868 | 0.426307 |
| suitesparse:HB/bcsstk10 | taichi_csr_pcg_jacobi_float64 | yes | success |  | applicable |  | 1 | 1 | 1268.32 | 0 | 9.83561e-06 | 0.00434613 |
| suitesparse:HB/bcsstk10 | taichi_csr_richardson_jacobi_float64 | no | screened_out | cpu_reference_richardson_screen_failed | applicable |  | 0 | 1 |  |  | 9.78173e-05 | 0.421047 |
| suitesparse:HB/bcsstk11 | taichi_csr_cg_none_float64 | no | screened_out | cpu_reference_cg_screen_failed | applicable |  | 0 | 1 |  |  | 0.00011027 | 0.240938 |
| suitesparse:HB/bcsstk11 | taichi_csr_pcg_jacobi_float64 | no | screened_out | cpu_reference_pcg_screen_failed | applicable |  | 0 | 1 |  |  | 9.77271e-06 | 0.0846525 |
| suitesparse:HB/bcsstk11 | taichi_csr_richardson_jacobi_float64 | no | screened_out | cpu_reference_richardson_screen_failed | applicable |  | 0 | 1 |  |  | 0.000156493 | 0.472661 |
| suitesparse:HB/bcsstk12 | taichi_csr_cg_none_float64 | no | screened_out | cpu_reference_cg_screen_failed | applicable |  | 0 | 1 |  |  | 0.00011027 | 0.240938 |
| suitesparse:HB/bcsstk12 | taichi_csr_pcg_jacobi_float64 | no | screened_out | cpu_reference_pcg_screen_failed | applicable |  | 0 | 1 |  |  | 9.77271e-06 | 0.0846525 |
| suitesparse:HB/bcsstk12 | taichi_csr_richardson_jacobi_float64 | no | screened_out | cpu_reference_richardson_screen_failed | applicable |  | 0 | 1 |  |  | 0.000156493 | 0.472661 |
| suitesparse:HB/bcsstk13 | taichi_csr_cg_none_float64 | no | screened_out | cpu_reference_cg_screen_failed | applicable |  | 0 | 1 |  |  | 0.000247178 | 0.804508 |
| suitesparse:HB/bcsstk13 | taichi_csr_pcg_jacobi_float64 | no | screened_out | cpu_reference_pcg_screen_failed | applicable |  | 0 | 1 |  |  | 2.60519e-05 | 0.484548 |
| suitesparse:HB/bcsstk13 | taichi_csr_richardson_jacobi_float64 | no | screened_out | cpu_reference_richardson_screen_failed | applicable |  | 0 | 1 |  |  | 0.00145464 | 0.918848 |
| suitesparse:HB/bcsstk14 | taichi_csr_cg_none_float64 | no | screened_out | cpu_reference_cg_screen_failed | applicable |  | 0 | 1 |  |  | 7.00446e-05 | 0.623772 |
| suitesparse:HB/bcsstk14 | taichi_csr_pcg_jacobi_float64 | no | screened_out | cpu_reference_pcg_screen_failed | applicable |  | 0 | 1 |  |  | 9.78603e-06 | 0.0307678 |
| suitesparse:HB/bcsstk14 | taichi_csr_richardson_jacobi_float64 | no | screened_out | cpu_reference_richardson_screen_failed | applicable |  | 0 | 1 |  |  | 5.50747e-05 | 0.492926 |
| suitesparse:HB/bcsstk15 | taichi_csr_cg_none_float64 | no | screened_out | cpu_reference_cg_screen_failed | applicable |  | 0 | 1 |  |  | 0.000152786 | 0.635238 |
| suitesparse:HB/bcsstk15 | taichi_csr_pcg_jacobi_float64 | no | screened_out | cpu_reference_pcg_screen_failed | applicable |  | 0 | 1 |  |  | 3.14182e-05 | 0.110869 |
| suitesparse:HB/bcsstk15 | taichi_csr_richardson_jacobi_float64 | no | screened_out | cpu_reference_richardson_screen_failed | applicable |  | 0 | 1 |  |  | 0.00133123 | 1.26746 |
