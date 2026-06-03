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
- failure_reason_counts: `{'cpu_reference_cg_screen_failed': 8, 'cpu_reference_pcg_screen_failed': 8, 'cpu_reference_richardson_screen_failed': 8}`
- applicability_status_counts: `{}`
- applicability_reason_counts: `{}`

| matrix | candidate | oracle | status | failure | applicability | reason | success_rate | repeats | median_ms | regret_ms | rel_res | sol_err |
|---|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|
| suitesparse:HB/bcspwr02 | taichi_csr_cg_none_float64 | no | screened_out | cpu_reference_cg_screen_failed | applicable |  | 0 | 1 |  |  | 0.139728 | 0.388987 |
| suitesparse:HB/bcspwr02 | taichi_csr_pcg_jacobi_float64 | no | screened_out | cpu_reference_pcg_screen_failed | applicable |  | 0 | 1 |  |  | 0.139728 | 0.388987 |
| suitesparse:HB/bcspwr02 | taichi_csr_richardson_jacobi_float64 | no | screened_out | cpu_reference_richardson_screen_failed | applicable |  | 0 | 1 |  |  | 392221 | 795876 |
| suitesparse:HB/bcspwr03 | taichi_csr_cg_none_float64 | no | screened_out | cpu_reference_cg_screen_failed | applicable |  | 0 | 1 |  |  | 0.221032 | 0.461983 |
| suitesparse:HB/bcspwr03 | taichi_csr_pcg_jacobi_float64 | no | screened_out | cpu_reference_pcg_screen_failed | applicable |  | 0 | 1 |  |  | 0.221032 | 0.461983 |
| suitesparse:HB/bcspwr03 | taichi_csr_richardson_jacobi_float64 | no | screened_out | cpu_reference_richardson_screen_failed | applicable |  | 0 | 1 |  |  | 3.60783e+07 | 6.8262e+07 |
| suitesparse:HB/bcspwr04 | taichi_csr_cg_none_float64 | no | screened_out | cpu_reference_cg_screen_failed | applicable |  | 0 | 1 |  |  | 0.171643 | 0.49805 |
| suitesparse:HB/bcspwr04 | taichi_csr_pcg_jacobi_float64 | no | screened_out | cpu_reference_pcg_screen_failed | applicable |  | 0 | 1 |  |  | 0.171643 | 0.49805 |
| suitesparse:HB/bcspwr04 | taichi_csr_richardson_jacobi_float64 | no | screened_out | cpu_reference_richardson_screen_failed | applicable |  | 0 | 1 |  |  | 6.12735e+08 | 1.53404e+09 |
| suitesparse:HB/bcspwr05 | taichi_csr_cg_none_float64 | no | screened_out | cpu_reference_cg_screen_failed | applicable |  | 0 | 1 |  |  | 0.17217 | 0.405119 |
| suitesparse:HB/bcspwr05 | taichi_csr_pcg_jacobi_float64 | no | screened_out | cpu_reference_pcg_screen_failed | applicable |  | 0 | 1 |  |  | 0.17217 | 0.405119 |
| suitesparse:HB/bcspwr05 | taichi_csr_richardson_jacobi_float64 | no | screened_out | cpu_reference_richardson_screen_failed | applicable |  | 0 | 1 |  |  | 1.00933e+08 | 1.63376e+08 |
| suitesparse:HB/bcspwr06 | taichi_csr_cg_none_float64 | no | screened_out | cpu_reference_cg_screen_failed | applicable |  | 0 | 1 |  |  | 0.188134 | 0.410039 |
| suitesparse:HB/bcspwr06 | taichi_csr_pcg_jacobi_float64 | no | screened_out | cpu_reference_pcg_screen_failed | applicable |  | 0 | 1 |  |  | 0.188134 | 0.410039 |
| suitesparse:HB/bcspwr06 | taichi_csr_richardson_jacobi_float64 | no | screened_out | cpu_reference_richardson_screen_failed | applicable |  | 0 | 1 |  |  | 1.50591e+10 | 1.93472e+10 |
| suitesparse:HB/bcspwr07 | taichi_csr_cg_none_float64 | no | screened_out | cpu_reference_cg_screen_failed | applicable |  | 0 | 1 |  |  | 0.218942 | 0.448802 |
| suitesparse:HB/bcspwr07 | taichi_csr_pcg_jacobi_float64 | no | screened_out | cpu_reference_pcg_screen_failed | applicable |  | 0 | 1 |  |  | 0.218942 | 0.448802 |
| suitesparse:HB/bcspwr07 | taichi_csr_richardson_jacobi_float64 | no | screened_out | cpu_reference_richardson_screen_failed | applicable |  | 0 | 1 |  |  | 1.27161e+10 | 1.63738e+10 |
| suitesparse:HB/bcspwr08 | taichi_csr_cg_none_float64 | no | screened_out | cpu_reference_cg_screen_failed | applicable |  | 0 | 1 |  |  | 0.225546 | 0.463039 |
| suitesparse:HB/bcspwr08 | taichi_csr_pcg_jacobi_float64 | no | screened_out | cpu_reference_pcg_screen_failed | applicable |  | 0 | 1 |  |  | 0.225546 | 0.463039 |
| suitesparse:HB/bcspwr08 | taichi_csr_richardson_jacobi_float64 | no | screened_out | cpu_reference_richardson_screen_failed | applicable |  | 0 | 1 |  |  | 1.52222e+10 | 2.0216e+10 |
| suitesparse:HB/bcspwr09 | taichi_csr_cg_none_float64 | no | screened_out | cpu_reference_cg_screen_failed | applicable |  | 0 | 1 |  |  | 0.216078 | 0.450993 |
| suitesparse:HB/bcspwr09 | taichi_csr_pcg_jacobi_float64 | no | screened_out | cpu_reference_pcg_screen_failed | applicable |  | 0 | 1 |  |  | 0.216078 | 0.450993 |
| suitesparse:HB/bcspwr09 | taichi_csr_richardson_jacobi_float64 | no | screened_out | cpu_reference_richardson_screen_failed | applicable |  | 0 | 1 |  |  | 1.91444e+10 | 2.57443e+10 |
