# CSR Selector Policy Smoke

- status: `passed`
- selected_plans: `4`
- oracle_selected_plans: `4`
- solve_checks: `2`
- successful_solve_checks: `2`
- max_final_relative_residual: `8.72966e-06`
- max_cpu_recomputed_relative_residual: `8.72966e-06`
- max_solution_relative_error: `0.000422806`

| matrix | candidate | oracle | reason | fallbacks |
|---|---|---:|---|---:|
| suitesparse:FIDAP/ex5 | taichi_csr_pcg_jacobi_float64 | yes | profiled_success | 1 |
| suitesparse:Grund/b1_ss | taichi_csr_gmres_jacobi_restart16_float64 | yes | profiled_success | 4 |
| suitesparse:HB/curtis54 | taichi_csr_bicgstab_none_float64 | yes | profiled_success | 3 |
| suitesparse:JGD_Trefethen/Trefethen_20b | taichi_csr_chebyshev_jacobi_float64 | yes | profiled_success | 3 |

## Solve Checks

| matrix | candidate | status | rel_res | cpu_rel_res | sol_rel_err |
|---|---|---|---:|---:|---:|
| suitesparse:HB/curtis54 | taichi_csr_bicgstab_none_float64 | success | 8.72966e-06 | 8.72966e-06 | 0.000422806 |
| suitesparse:JGD_Trefethen/Trefethen_20b | taichi_csr_chebyshev_jacobi_float64 | success | 4.7438e-06 | 4.7438e-06 | 5.96855e-06 |
