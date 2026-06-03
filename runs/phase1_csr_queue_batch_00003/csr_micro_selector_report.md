# CSR Selector Readiness

- status: `passed`
- schema_version: `phase1_csr_selector_features_v8`
- diagnostic_rows: `8`
- selector_rows: `24`
- success_rows: `5`
- failed_rows: `19`
- applicability_rows: `0`
- oracle_rows: `3`
- matrices_with_selector_rows: `8`
- max_final_relative_residual: `8.41872e-06`
- max_cpu_recomputed_relative_residual: `8.41872e-06`
- max_solution_relative_error: `0.00436904`
- min_success_rate: `0`
- max_solve_time_iqr_ms: `0`
- failure_reason_counts: `{'cpu_reference_cg_screen_failed': 6, 'cpu_reference_pcg_screen_failed': 5, 'cpu_reference_richardson_screen_failed': 8}`
- applicability_status_counts: `{}`
- applicability_reason_counts: `{}`

| matrix | candidate | oracle | status | failure | applicability | reason | success_rate | repeats | median_ms | regret_ms | rel_res | sol_err |
|---|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|
| suitesparse:HB/bcspwr10 | taichi_csr_cg_none_float64 | no | screened_out | cpu_reference_cg_screen_failed | applicable |  | 0 | 1 |  |  | 0.163044 | 0.345204 |
| suitesparse:HB/bcspwr10 | taichi_csr_pcg_jacobi_float64 | no | screened_out | cpu_reference_pcg_screen_failed | applicable |  | 0 | 1 |  |  | 0.163044 | 0.345204 |
| suitesparse:HB/bcspwr10 | taichi_csr_richardson_jacobi_float64 | no | screened_out | cpu_reference_richardson_screen_failed | applicable |  | 0 | 1 |  |  | 8.35394e+09 | 1.20478e+10 |
| suitesparse:HB/bcsstk01 | taichi_csr_cg_none_float64 | no | screened_out | cpu_reference_cg_screen_failed | applicable |  | 0 | 1 |  |  | 7.17601e-06 | 0.557175 |
| suitesparse:HB/bcsstk01 | taichi_csr_pcg_jacobi_float64 | no | screened_out | cpu_reference_pcg_screen_failed | applicable |  | 0 | 1 |  |  | 6.07376e-06 | 0.113325 |
| suitesparse:HB/bcsstk01 | taichi_csr_richardson_jacobi_float64 | no | screened_out | cpu_reference_richardson_screen_failed | applicable |  | 0 | 1 |  |  | 9.97229e-06 | 0.755891 |
| suitesparse:HB/bcsstk02 | taichi_csr_cg_none_float64 | yes | success |  | applicable |  | 1 | 1 | 655.122 | 0 | 5.73295e-06 | 1.62737e-06 |
| suitesparse:HB/bcsstk02 | taichi_csr_pcg_jacobi_float64 | no | success |  | applicable |  | 1 | 1 | 805.373 | 150.251 | 1.36088e-06 | 3.7795e-07 |
| suitesparse:HB/bcsstk02 | taichi_csr_richardson_jacobi_float64 | no | screened_out | cpu_reference_richardson_screen_failed | applicable |  | 0 | 1 |  |  | 0.00286737 | 0.617056 |
| suitesparse:HB/bcsstk03 | taichi_csr_cg_none_float64 | no | screened_out | cpu_reference_cg_screen_failed | applicable |  | 0 | 1 |  |  | 9.30613e-06 | 0.595813 |
| suitesparse:HB/bcsstk03 | taichi_csr_pcg_jacobi_float64 | yes | success |  | applicable |  | 1 | 1 | 947.337 | 0 | 7.40291e-06 | 0.00436904 |
| suitesparse:HB/bcsstk03 | taichi_csr_richardson_jacobi_float64 | no | screened_out | cpu_reference_richardson_screen_failed | applicable |  | 0 | 1 |  |  | 0.000565104 | 2.01236 |
| suitesparse:HB/bcsstk04 | taichi_csr_cg_none_float64 | no | screened_out | cpu_reference_cg_screen_failed | applicable |  | 0 | 1 |  |  | 7.28504e-06 | 0.578403 |
| suitesparse:HB/bcsstk04 | taichi_csr_pcg_jacobi_float64 | no | screened_out | cpu_reference_pcg_screen_failed | applicable |  | 0 | 1 |  |  | 3.48464e-06 | 0.0750214 |
| suitesparse:HB/bcsstk04 | taichi_csr_richardson_jacobi_float64 | no | screened_out | cpu_reference_richardson_screen_failed | applicable |  | 0 | 1 |  |  | 9.9413e-06 | 0.949327 |
| suitesparse:HB/bcsstk05 | taichi_csr_cg_none_float64 | no | success |  | applicable |  | 1 | 1 | 1974.22 | 1056.74 | 8.41872e-06 | 3.44912e-05 |
| suitesparse:HB/bcsstk05 | taichi_csr_pcg_jacobi_float64 | yes | success |  | applicable |  | 1 | 1 | 917.48 | 0 | 6.39843e-06 | 5.25257e-06 |
| suitesparse:HB/bcsstk05 | taichi_csr_richardson_jacobi_float64 | no | screened_out | cpu_reference_richardson_screen_failed | applicable |  | 0 | 1 |  |  | 0.0291065 | 0.618368 |
| suitesparse:HB/bcsstk06 | taichi_csr_cg_none_float64 | no | screened_out | cpu_reference_cg_screen_failed | applicable |  | 0 | 1 |  |  | 8.94851e-06 | 0.526635 |
| suitesparse:HB/bcsstk06 | taichi_csr_pcg_jacobi_float64 | no | screened_out | cpu_reference_pcg_screen_failed | applicable |  | 0 | 1 |  |  | 7.19328e-06 | 0.0808606 |
| suitesparse:HB/bcsstk06 | taichi_csr_richardson_jacobi_float64 | no | screened_out | cpu_reference_richardson_screen_failed | applicable |  | 0 | 1 |  |  | 1.06253e-05 | 0.839226 |
| suitesparse:HB/bcsstk07 | taichi_csr_cg_none_float64 | no | screened_out | cpu_reference_cg_screen_failed | applicable |  | 0 | 1 |  |  | 8.94851e-06 | 0.526635 |
| suitesparse:HB/bcsstk07 | taichi_csr_pcg_jacobi_float64 | no | screened_out | cpu_reference_pcg_screen_failed | applicable |  | 0 | 1 |  |  | 7.19328e-06 | 0.0808606 |
| suitesparse:HB/bcsstk07 | taichi_csr_richardson_jacobi_float64 | no | screened_out | cpu_reference_richardson_screen_failed | applicable |  | 0 | 1 |  |  | 1.06253e-05 | 0.839226 |
