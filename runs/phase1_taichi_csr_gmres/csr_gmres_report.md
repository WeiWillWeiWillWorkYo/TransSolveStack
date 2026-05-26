# Taichi CSR GMRES Smoke

- status: `passed`
- selected_matrices: `2`
- solves: `5`
- restarts: `8, 16, 32`
- success: `5`
- failed: `0`
- measurement_repeats: `3`
- total_csr_nnz: `627`
- max_final_relative_residual: `9.74836e-06`
- max_cpu_recomputed_relative_residual: `9.74836e-06`
- max_solution_relative_error: `0.000817409`

| matrix | preconditioner | restart | status | repeats | median_solve_ms | iters | rel_res | cpu_rel_res | sol_rel_err |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| suitesparse:HB/curtis54 | jacobi | 16 | success | 3 | 6337.83 | 466 | 9.74836e-06 | 9.74836e-06 | 0.000817409 |
| suitesparse:HB/curtis54 | jacobi | 32 | success | 3 | 4062.14 | 128 | 9.3701e-06 | 9.3701e-06 | 0.000271967 |
| suitesparse:Grund/b1_ss | jacobi | 8 | success | 3 | 812.013 | 5 | 1.84509e-08 | 1.84509e-08 | 3.05751e-07 |
| suitesparse:Grund/b1_ss | jacobi | 16 | success | 3 | 799.065 | 5 | 1.84509e-08 | 1.84509e-08 | 3.05751e-07 |
| suitesparse:Grund/b1_ss | jacobi | 32 | success | 3 | 946.215 | 5 | 1.84509e-08 | 1.84509e-08 | 3.05751e-07 |

## Skipped Candidates

- `suitesparse:HB/fs_183_1`: `cpu_reference_gmres_screen_failed`
- `suitesparse:HB/young3c`: `cpu_reference_gmres_screen_failed`
- `suitesparse:Zitney/extr1b`: `after_selection_limit`
- `suitesparse:Hamrle/Hamrle1`: `after_selection_limit`
- `suitesparse:Sandia/oscil_dcop_01`: `after_selection_limit`
- `suitesparse:JGD_Trefethen/Trefethen_20b`: `after_selection_limit`
- `suitesparse:Negre/dendrimer`: `after_selection_limit`
- `suitesparse:Goodwin/Goodwin_010`: `after_selection_limit`
- `suitesparse:FIDAP/ex5`: `after_selection_limit`
- `suitesparse:Bai/cdde1`: `after_selection_limit`

## Screened-Out Restart Configs

- `suitesparse:HB/curtis54` `jacobi` `restart=8`
