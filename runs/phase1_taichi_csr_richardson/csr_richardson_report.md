# Taichi CSR Richardson/Jacobi Smoke

- status: `passed`
- selected_matrices: `1`
- solves: `1`
- success: `1`
- failed: `0`
- omega: `0.6666666666666666`
- measurement_repeats: `3`
- total_csr_nnz: `147`
- max_final_relative_residual: `8.13933e-06`
- max_cpu_recomputed_relative_residual: `8.13933e-06`
- max_solution_relative_error: `9.74851e-05`

| matrix | status | repeats | median_solve_ms | iters | rel_res | cpu_rel_res | sol_rel_err |
|---|---|---:|---:|---:|---:|---:|---:|
| suitesparse:JGD_Trefethen/Trefethen_20b | success | 3 | 259.811 | 11 | 8.13933e-06 | 8.13933e-06 | 9.74851e-05 |

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
- `suitesparse:FIDAP/ex5`: `cpu_reference_richardson_screen_failed`
- `suitesparse:Bai/cdde1`: `not_symmetric`
