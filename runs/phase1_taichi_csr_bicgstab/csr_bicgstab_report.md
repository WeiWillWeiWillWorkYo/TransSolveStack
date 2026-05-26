# Taichi CSR BiCGSTAB Smoke

- status: `passed`
- selected_matrices: `2`
- solves: `4`
- success: `4`
- failed: `0`
- measurement_repeats: `3`
- total_csr_nnz: `612`
- max_final_relative_residual: `8.72966e-06`
- max_cpu_recomputed_relative_residual: `8.72966e-06`
- max_solution_relative_error: `0.000422806`

| matrix | preconditioner | status | repeats | median_solve_ms | iters | rel_res | cpu_rel_res | sol_rel_err |
|---|---|---|---:|---:|---:|---:|---:|---:|
| suitesparse:HB/curtis54 | none | success | 3 | 1584.74 | 88 | 8.72966e-06 | 8.72966e-06 | 0.000422806 |
| suitesparse:HB/curtis54 | jacobi | success | 3 | 2182.62 | 88 | 8.72966e-06 | 8.72966e-06 | 0.000422806 |
| suitesparse:Grund/b1_ss | none | success | 3 | 1415.07 | 7 | 3.49738e-10 | 3.49737e-10 | 4.46119e-09 |
| suitesparse:Grund/b1_ss | jacobi | success | 3 | 1643.45 | 5 | 8.97836e-08 | 8.97836e-08 | 6.74364e-06 |

## Skipped Candidates

- `suitesparse:HB/fs_183_1`: `cpu_reference_bicgstab_screen_failed`
- `suitesparse:HB/young3c`: `cpu_reference_bicgstab_screen_failed`
- `suitesparse:Zitney/extr1b`: `after_selection_limit`
- `suitesparse:Hamrle/Hamrle1`: `after_selection_limit`
- `suitesparse:Sandia/oscil_dcop_01`: `after_selection_limit`
- `suitesparse:JGD_Trefethen/Trefethen_20b`: `after_selection_limit`
- `suitesparse:Negre/dendrimer`: `after_selection_limit`
- `suitesparse:Goodwin/Goodwin_010`: `after_selection_limit`
- `suitesparse:FIDAP/ex5`: `after_selection_limit`
- `suitesparse:Bai/cdde1`: `after_selection_limit`
