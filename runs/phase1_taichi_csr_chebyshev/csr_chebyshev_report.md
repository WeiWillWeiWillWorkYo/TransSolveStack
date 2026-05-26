# Taichi CSR Chebyshev/Jacobi Smoke

- status: `passed`
- selected_matrices: `1`
- solves: `1`
- success: `1`
- failed: `0`
- measurement_repeats: `3`
- total_csr_nnz: `147`
- max_final_relative_residual: `4.7438e-06`
- max_cpu_recomputed_relative_residual: `4.7438e-06`
- max_solution_relative_error: `5.96855e-06`

| matrix | status | repeats | median_solve_ms | iters | lambda_min | lambda_max | rel_res | sol_rel_err |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| suitesparse:JGD_Trefethen/Trefethen_20b | success | 3 | 204.16 | 9 | 0.583885 | 1.6661 | 4.7438e-06 | 5.96855e-06 |

## Skipped Candidates

- `suitesparse:HB/curtis54`: `not_symmetric`
- `suitesparse:HB/fs_183_1`: `not_symmetric`
- `suitesparse:HB/young3c`: `not_symmetric`
- `suitesparse:Grund/b1_ss`: `not_symmetric`
- `suitesparse:Zitney/extr1b`: `not_symmetric`
- `suitesparse:Hamrle/Hamrle1`: `not_symmetric`
- `suitesparse:Sandia/oscil_dcop_01`: `not_symmetric`
- `suitesparse:Negre/dendrimer`: `above_smoke_size_limit`
- `suitesparse:Goodwin/Goodwin_010`: `not_symmetric`
- `suitesparse:FIDAP/ex5`: `cpu_reference_chebyshev_screen_failed`
- `suitesparse:Bai/cdde1`: `not_symmetric`
