# Taichi CSR Solve Smoke

- status: `passed`
- selected_matrices: `2`
- solves: `4`
- success: `4`
- failed: `0`
- measurement_repeats: `3`
- total_csr_nnz: `852`
- max_final_relative_residual: `6.69498e-06`
- max_cpu_recomputed_relative_residual: `6.69528e-06`
- max_solution_relative_error: `7.72059e-06`

| matrix | solver | preconditioner | status | repeats | median_solve_ms | iters | rel_res | cpu_rel_res | sol_rel_err |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| suitesparse:JGD_Trefethen/Trefethen_20b | cg | none | success | 3 | 946.159 | 19 | 2.95876e-06 | 2.95876e-06 | 1.90716e-06 |
| suitesparse:JGD_Trefethen/Trefethen_20b | pcg | jacobi | success | 3 | 610.911 | 6 | 2.37775e-06 | 2.37775e-06 | 7.72059e-06 |
| suitesparse:FIDAP/ex5 | cg | none | success | 3 | 1505.02 | 89 | 3.33419e-06 | 3.33427e-06 | 3.21417e-07 |
| suitesparse:FIDAP/ex5 | pcg | jacobi | success | 3 | 1120.02 | 83 | 6.69498e-06 | 6.69528e-06 | 5.48859e-07 |

## Skipped Candidates

- `suitesparse:HB/curtis54`: `not_symmetric`
- `suitesparse:HB/fs_183_1`: `not_symmetric`
- `suitesparse:HB/young3c`: `not_symmetric`
- `suitesparse:Grund/b1_ss`: `not_symmetric`
- `suitesparse:Zitney/extr1b`: `not_symmetric`
- `suitesparse:Hamrle/Hamrle1`: `not_symmetric`
- `suitesparse:Sandia/oscil_dcop_01`: `not_symmetric`
- `suitesparse:Negre/dendrimer`: `cpu_reference_cg_screen_failed`
- `suitesparse:Goodwin/Goodwin_010`: `not_symmetric`
- `suitesparse:Bai/cdde1`: `after_selection_limit`
