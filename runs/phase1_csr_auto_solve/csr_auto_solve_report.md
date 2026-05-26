# CSR Auto Solve Smoke

- status: `passed`
- solves: `2`
- successes: `2`
- selected_solver_set: `bicgstab, chebyshev`
- oracle_selected: `2`
- max_final_relative_residual: `8.72966e-06`
- max_cpu_recomputed_relative_residual: `8.72966e-06`
- max_solution_relative_error: `0.000422806`

| matrix | candidate | solver | preconditioner | status | rel_res | sol_err |
|---|---|---|---|---|---:|---:|
| suitesparse:JGD_Trefethen/Trefethen_20b | taichi_csr_chebyshev_jacobi_float64 | chebyshev | jacobi | success | 4.7438e-06 | 5.96855e-06 |
| suitesparse:HB/curtis54 | taichi_csr_bicgstab_none_float64 | bicgstab | none | success | 8.72966e-06 | 0.000422806 |
