# CSR Selector Readiness

- status: `passed`
- schema_version: `phase1_csr_selector_features_v8`
- diagnostic_rows: `8`
- selector_rows: `8`
- success_rows: `8`
- failed_rows: `0`
- applicability_rows: `0`
- oracle_rows: `5`
- matrices_with_selector_rows: `5`
- max_final_relative_residual: `8.11032e-06`
- max_cpu_recomputed_relative_residual: `8.11032e-06`
- max_solution_relative_error: `0.000472771`
- min_success_rate: `1`
- max_solve_time_iqr_ms: `0`
- failure_reason_counts: `{}`
- applicability_status_counts: `{}`
- applicability_reason_counts: `{}`

| matrix | candidate | oracle | status | failure | applicability | reason | success_rate | repeats | median_ms | regret_ms | rel_res | sol_err |
|---|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|
| suitesparse:HB/curtis54 | taichi_csr_bicgstab_row_column_equilibration_float64 | yes | success |  | applicable |  | 1 | 1 | 2713.66 | 0 | 5.90786e-06 | 0.000472771 |
| suitesparse:Grund/b1_ss | taichi_csr_bicgstab_row_column_equilibration_float64 | yes | success |  | applicable |  | 1 | 1 | 1933.83 | 0 | 1.52432e-06 | 1.54376e-05 |
| suitesparse:JGD_Trefethen/Trefethen_20b | taichi_csr_bicgstab_ilu0_float64 | no | success |  | applicable |  | 1 | 1 | 1331.34 | 428.076 | 1.08595e-06 | 4.93118e-06 |
| suitesparse:JGD_Trefethen/Trefethen_20b | taichi_csr_chebyshev_jacobi_float64 | no | success |  | applicable |  | 1 | 1 | 1369.56 | 466.294 | 4.7438e-06 | 5.96855e-06 |
| suitesparse:JGD_Trefethen/Trefethen_20b | taichi_csr_pcg_symmetric_equilibration_float64 | yes | success |  | applicable |  | 1 | 1 | 903.265 | 0 | 2.37775e-06 | 7.72059e-06 |
| suitesparse:FIDAP/ex5 | taichi_csr_bicgstab_ilu0_float64 | yes | success |  | applicable |  | 1 | 1 | 1359.48 | 0 | 4.48152e-06 | 2.30582e-06 |
| suitesparse:FIDAP/ex5 | taichi_csr_pcg_symmetric_equilibration_float64 | no | success |  | applicable |  | 1 | 1 | 2457.52 | 1098.04 | 2.30831e-06 | 1.24131e-07 |
| suitesparse:HB/bcsstk01 | taichi_csr_chebyshev_jacobi_float64 | yes | success |  | applicable |  | 1 | 1 | 1643.1 | 0 | 8.11032e-06 | 0.000218803 |
