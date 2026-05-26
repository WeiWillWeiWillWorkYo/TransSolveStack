# CSR Selector Readiness

- status: `passed`
- schema_version: `phase1_csr_selector_features_v8`
- diagnostic_rows: `12`
- selector_rows: `108`
- success_rows: `15`
- failed_rows: `15`
- applicability_rows: `78`
- oracle_rows: `4`
- matrices_with_selector_rows: `12`
- max_final_relative_residual: `9.74836e-06`
- max_cpu_recomputed_relative_residual: `9.74836e-06`
- max_solution_relative_error: `0.000817409`
- min_success_rate: `0`
- max_solve_time_iqr_ms: `1136.8`
- failure_reason_counts: `{'cpu_reference_bicgstab_screen_failed': 4, 'cpu_reference_cg_screen_failed': 2, 'cpu_reference_chebyshev_screen_failed': 1, 'cpu_reference_gmres_screen_failed': 7, 'cpu_reference_richardson_screen_failed': 1}`
- applicability_status_counts: `{'not_applicable': 34, 'not_profiled': 44}`
- applicability_reason_counts: `{'above_smoke_size_limit': 2, 'after_selection_limit': 42, 'not_symmetric': 34}`

| matrix | candidate | oracle | status | failure | applicability | reason | success_rate | repeats | median_ms | regret_ms | rel_res | sol_err |
|---|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|
| suitesparse:JGD_Trefethen/Trefethen_20b | taichi_csr_cg_none_float64 | no | success |  | applicable |  | 1 | 3 | 946.159 | 741.999 | 2.95876e-06 | 1.90716e-06 |
| suitesparse:JGD_Trefethen/Trefethen_20b | taichi_csr_pcg_jacobi_float64 | no | success |  | applicable |  | 1 | 3 | 610.911 | 406.751 | 2.37775e-06 | 7.72059e-06 |
| suitesparse:FIDAP/ex5 | taichi_csr_cg_none_float64 | no | success |  | applicable |  | 1 | 3 | 1505.02 | 384.996 | 3.33419e-06 | 3.21417e-07 |
| suitesparse:FIDAP/ex5 | taichi_csr_pcg_jacobi_float64 | yes | success |  | applicable |  | 1 | 3 | 1120.02 | 0 | 6.69498e-06 | 5.48859e-07 |
| suitesparse:HB/curtis54 | taichi_csr_bicgstab_none_float64 | yes | success |  | applicable |  | 1 | 3 | 1584.74 | 0 | 8.72966e-06 | 0.000422806 |
| suitesparse:HB/curtis54 | taichi_csr_bicgstab_jacobi_float64 | no | success |  | applicable |  | 1 | 3 | 2182.62 | 597.878 | 8.72966e-06 | 0.000422806 |
| suitesparse:Grund/b1_ss | taichi_csr_bicgstab_none_float64 | no | success |  | applicable |  | 1 | 3 | 1415.07 | 616 | 3.49738e-10 | 4.46119e-09 |
| suitesparse:Grund/b1_ss | taichi_csr_bicgstab_jacobi_float64 | no | success |  | applicable |  | 1 | 3 | 1643.45 | 844.389 | 8.97836e-08 | 6.74364e-06 |
| suitesparse:HB/curtis54 | taichi_csr_gmres_jacobi_restart16_float64 | no | success |  | applicable |  | 1 | 3 | 6337.83 | 4753.09 | 9.74836e-06 | 0.000817409 |
| suitesparse:HB/curtis54 | taichi_csr_gmres_jacobi_restart32_float64 | no | success |  | applicable |  | 1 | 3 | 4062.14 | 2477.4 | 9.3701e-06 | 0.000271967 |
| suitesparse:Grund/b1_ss | taichi_csr_gmres_jacobi_restart8_float64 | no | success |  | applicable |  | 1 | 3 | 812.013 | 12.9477 | 1.84509e-08 | 3.05751e-07 |
| suitesparse:Grund/b1_ss | taichi_csr_gmres_jacobi_restart16_float64 | yes | success |  | applicable |  | 1 | 3 | 799.065 | 0 | 1.84509e-08 | 3.05751e-07 |
| suitesparse:Grund/b1_ss | taichi_csr_gmres_jacobi_restart32_float64 | no | success |  | applicable |  | 1 | 3 | 946.215 | 147.15 | 1.84509e-08 | 3.05751e-07 |
| suitesparse:JGD_Trefethen/Trefethen_20b | taichi_csr_richardson_jacobi_float64 | no | success |  | applicable |  | 1 | 3 | 259.811 | 55.651 | 8.13933e-06 | 9.74851e-05 |
| suitesparse:JGD_Trefethen/Trefethen_20b | taichi_csr_chebyshev_jacobi_float64 | yes | success |  | applicable |  | 1 | 3 | 204.16 | 0 | 4.7438e-06 | 5.96855e-06 |
| suitesparse:HB/curtis54 | taichi_csr_cg_none_float64 | no | not_applicable |  | not_applicable | not_symmetric | 0 | 0 |  |  |  |  |
| suitesparse:HB/curtis54 | taichi_csr_pcg_jacobi_float64 | no | not_applicable |  | not_applicable | not_symmetric | 0 | 0 |  |  |  |  |
| suitesparse:HB/fs_183_1 | taichi_csr_cg_none_float64 | no | not_applicable |  | not_applicable | not_symmetric | 0 | 0 |  |  |  |  |
| suitesparse:HB/fs_183_1 | taichi_csr_pcg_jacobi_float64 | no | not_applicable |  | not_applicable | not_symmetric | 0 | 0 |  |  |  |  |
| suitesparse:HB/young3c | taichi_csr_cg_none_float64 | no | not_applicable |  | not_applicable | not_symmetric | 0 | 0 |  |  |  |  |
| suitesparse:HB/young3c | taichi_csr_pcg_jacobi_float64 | no | not_applicable |  | not_applicable | not_symmetric | 0 | 0 |  |  |  |  |
| suitesparse:Grund/b1_ss | taichi_csr_cg_none_float64 | no | not_applicable |  | not_applicable | not_symmetric | 0 | 0 |  |  |  |  |
| suitesparse:Grund/b1_ss | taichi_csr_pcg_jacobi_float64 | no | not_applicable |  | not_applicable | not_symmetric | 0 | 0 |  |  |  |  |
| suitesparse:Zitney/extr1b | taichi_csr_cg_none_float64 | no | not_applicable |  | not_applicable | not_symmetric | 0 | 0 |  |  |  |  |
| suitesparse:Zitney/extr1b | taichi_csr_pcg_jacobi_float64 | no | not_applicable |  | not_applicable | not_symmetric | 0 | 0 |  |  |  |  |
| suitesparse:Hamrle/Hamrle1 | taichi_csr_cg_none_float64 | no | not_applicable |  | not_applicable | not_symmetric | 0 | 0 |  |  |  |  |
| suitesparse:Hamrle/Hamrle1 | taichi_csr_pcg_jacobi_float64 | no | not_applicable |  | not_applicable | not_symmetric | 0 | 0 |  |  |  |  |
| suitesparse:Sandia/oscil_dcop_01 | taichi_csr_cg_none_float64 | no | not_applicable |  | not_applicable | not_symmetric | 0 | 0 |  |  |  |  |
| suitesparse:Sandia/oscil_dcop_01 | taichi_csr_pcg_jacobi_float64 | no | not_applicable |  | not_applicable | not_symmetric | 0 | 0 |  |  |  |  |
| suitesparse:Negre/dendrimer | taichi_csr_cg_none_float64 | no | screened_out | cpu_reference_cg_screen_failed | applicable |  | 0 | 1 |  |  | 0.00885149 | 1.10908 |
| suitesparse:Negre/dendrimer | taichi_csr_pcg_jacobi_float64 | no | screened_out | cpu_reference_cg_screen_failed | applicable |  | 0 | 1 |  |  | 0.0212097 | 1.31178 |
| suitesparse:Goodwin/Goodwin_010 | taichi_csr_cg_none_float64 | no | not_applicable |  | not_applicable | not_symmetric | 0 | 0 |  |  |  |  |
| suitesparse:Goodwin/Goodwin_010 | taichi_csr_pcg_jacobi_float64 | no | not_applicable |  | not_applicable | not_symmetric | 0 | 0 |  |  |  |  |
| suitesparse:Bai/cdde1 | taichi_csr_cg_none_float64 | no | not_profiled |  | not_profiled | after_selection_limit | 0 | 0 |  |  |  |  |
| suitesparse:Bai/cdde1 | taichi_csr_pcg_jacobi_float64 | no | not_profiled |  | not_profiled | after_selection_limit | 0 | 0 |  |  |  |  |
| suitesparse:HB/fs_183_1 | taichi_csr_bicgstab_jacobi_float64 | no | screened_out | cpu_reference_bicgstab_screen_failed | applicable |  | 0 | 1 |  |  | 1.30042e-06 | 32420.5 |
| suitesparse:HB/fs_183_1 | taichi_csr_bicgstab_none_float64 | no | screened_out | cpu_reference_bicgstab_screen_failed | applicable |  | 0 | 1 |  |  | 9.85755e-06 | 1.25567 |
| suitesparse:HB/young3c | taichi_csr_bicgstab_jacobi_float64 | no | screened_out | cpu_reference_bicgstab_screen_failed | applicable |  | 0 | 1 |  |  | 0.000811846 | 0.00961562 |
| suitesparse:HB/young3c | taichi_csr_bicgstab_none_float64 | no | screened_out | cpu_reference_bicgstab_screen_failed | applicable |  | 0 | 1 |  |  | 0.000139375 | 0.00336018 |
| suitesparse:Zitney/extr1b | taichi_csr_bicgstab_none_float64 | no | not_profiled |  | not_profiled | after_selection_limit | 0 | 0 |  |  |  |  |
| suitesparse:Zitney/extr1b | taichi_csr_bicgstab_jacobi_float64 | no | not_profiled |  | not_profiled | after_selection_limit | 0 | 0 |  |  |  |  |
| suitesparse:Hamrle/Hamrle1 | taichi_csr_bicgstab_none_float64 | no | not_profiled |  | not_profiled | after_selection_limit | 0 | 0 |  |  |  |  |
| suitesparse:Hamrle/Hamrle1 | taichi_csr_bicgstab_jacobi_float64 | no | not_profiled |  | not_profiled | after_selection_limit | 0 | 0 |  |  |  |  |
| suitesparse:Sandia/oscil_dcop_01 | taichi_csr_bicgstab_none_float64 | no | not_profiled |  | not_profiled | after_selection_limit | 0 | 0 |  |  |  |  |
| suitesparse:Sandia/oscil_dcop_01 | taichi_csr_bicgstab_jacobi_float64 | no | not_profiled |  | not_profiled | after_selection_limit | 0 | 0 |  |  |  |  |
| suitesparse:JGD_Trefethen/Trefethen_20b | taichi_csr_bicgstab_none_float64 | no | not_profiled |  | not_profiled | after_selection_limit | 0 | 0 |  |  |  |  |
| suitesparse:JGD_Trefethen/Trefethen_20b | taichi_csr_bicgstab_jacobi_float64 | no | not_profiled |  | not_profiled | after_selection_limit | 0 | 0 |  |  |  |  |
| suitesparse:Negre/dendrimer | taichi_csr_bicgstab_none_float64 | no | not_profiled |  | not_profiled | after_selection_limit | 0 | 0 |  |  |  |  |
| suitesparse:Negre/dendrimer | taichi_csr_bicgstab_jacobi_float64 | no | not_profiled |  | not_profiled | after_selection_limit | 0 | 0 |  |  |  |  |
| suitesparse:Goodwin/Goodwin_010 | taichi_csr_bicgstab_none_float64 | no | not_profiled |  | not_profiled | after_selection_limit | 0 | 0 |  |  |  |  |
| suitesparse:Goodwin/Goodwin_010 | taichi_csr_bicgstab_jacobi_float64 | no | not_profiled |  | not_profiled | after_selection_limit | 0 | 0 |  |  |  |  |
| suitesparse:FIDAP/ex5 | taichi_csr_bicgstab_none_float64 | no | not_profiled |  | not_profiled | after_selection_limit | 0 | 0 |  |  |  |  |
| suitesparse:FIDAP/ex5 | taichi_csr_bicgstab_jacobi_float64 | no | not_profiled |  | not_profiled | after_selection_limit | 0 | 0 |  |  |  |  |
| suitesparse:Bai/cdde1 | taichi_csr_bicgstab_none_float64 | no | not_profiled |  | not_profiled | after_selection_limit | 0 | 0 |  |  |  |  |
| suitesparse:Bai/cdde1 | taichi_csr_bicgstab_jacobi_float64 | no | not_profiled |  | not_profiled | after_selection_limit | 0 | 0 |  |  |  |  |
| suitesparse:HB/curtis54 | taichi_csr_gmres_jacobi_restart8_float64 | no | screened_out | cpu_reference_gmres_screen_failed | applicable |  | 0 | 1 |  |  | 5.85252e-05 | 0.00493156 |
| suitesparse:HB/fs_183_1 | taichi_csr_gmres_jacobi_restart16_float64 | no | screened_out | cpu_reference_gmres_screen_failed | applicable |  | 0 | 1 |  |  | 0.0339108 | 1.85334e+09 |
| suitesparse:HB/fs_183_1 | taichi_csr_gmres_jacobi_restart32_float64 | no | screened_out | cpu_reference_gmres_screen_failed | applicable |  | 0 | 1 |  |  | 4.17081e-08 | 454.149 |
| suitesparse:HB/fs_183_1 | taichi_csr_gmres_jacobi_restart8_float64 | no | screened_out | cpu_reference_gmres_screen_failed | applicable |  | 0 | 1 |  |  | 0.0361631 | 2.51875e+09 |
| suitesparse:HB/young3c | taichi_csr_gmres_jacobi_restart16_float64 | no | screened_out | cpu_reference_gmres_screen_failed | applicable |  | 0 | 1 |  |  | 0.00064705 | 0.0108773 |
| suitesparse:HB/young3c | taichi_csr_gmres_jacobi_restart32_float64 | no | screened_out | cpu_reference_gmres_screen_failed | applicable |  | 0 | 1 |  |  | 0.000150186 | 0.0027822 |
| suitesparse:HB/young3c | taichi_csr_gmres_jacobi_restart8_float64 | no | screened_out | cpu_reference_gmres_screen_failed | applicable |  | 0 | 1 |  |  | 0.00183221 | 0.023837 |
| suitesparse:Zitney/extr1b | taichi_csr_gmres_jacobi_restart8_float64 | no | not_profiled |  | not_profiled | after_selection_limit | 0 | 0 |  |  |  |  |
| suitesparse:Zitney/extr1b | taichi_csr_gmres_jacobi_restart16_float64 | no | not_profiled |  | not_profiled | after_selection_limit | 0 | 0 |  |  |  |  |
| suitesparse:Zitney/extr1b | taichi_csr_gmres_jacobi_restart32_float64 | no | not_profiled |  | not_profiled | after_selection_limit | 0 | 0 |  |  |  |  |
| suitesparse:Hamrle/Hamrle1 | taichi_csr_gmres_jacobi_restart8_float64 | no | not_profiled |  | not_profiled | after_selection_limit | 0 | 0 |  |  |  |  |
| suitesparse:Hamrle/Hamrle1 | taichi_csr_gmres_jacobi_restart16_float64 | no | not_profiled |  | not_profiled | after_selection_limit | 0 | 0 |  |  |  |  |
| suitesparse:Hamrle/Hamrle1 | taichi_csr_gmres_jacobi_restart32_float64 | no | not_profiled |  | not_profiled | after_selection_limit | 0 | 0 |  |  |  |  |
| suitesparse:Sandia/oscil_dcop_01 | taichi_csr_gmres_jacobi_restart8_float64 | no | not_profiled |  | not_profiled | after_selection_limit | 0 | 0 |  |  |  |  |
| suitesparse:Sandia/oscil_dcop_01 | taichi_csr_gmres_jacobi_restart16_float64 | no | not_profiled |  | not_profiled | after_selection_limit | 0 | 0 |  |  |  |  |
| suitesparse:Sandia/oscil_dcop_01 | taichi_csr_gmres_jacobi_restart32_float64 | no | not_profiled |  | not_profiled | after_selection_limit | 0 | 0 |  |  |  |  |
| suitesparse:JGD_Trefethen/Trefethen_20b | taichi_csr_gmres_jacobi_restart8_float64 | no | not_profiled |  | not_profiled | after_selection_limit | 0 | 0 |  |  |  |  |
| suitesparse:JGD_Trefethen/Trefethen_20b | taichi_csr_gmres_jacobi_restart16_float64 | no | not_profiled |  | not_profiled | after_selection_limit | 0 | 0 |  |  |  |  |
| suitesparse:JGD_Trefethen/Trefethen_20b | taichi_csr_gmres_jacobi_restart32_float64 | no | not_profiled |  | not_profiled | after_selection_limit | 0 | 0 |  |  |  |  |
| suitesparse:Negre/dendrimer | taichi_csr_gmres_jacobi_restart8_float64 | no | not_profiled |  | not_profiled | after_selection_limit | 0 | 0 |  |  |  |  |
| suitesparse:Negre/dendrimer | taichi_csr_gmres_jacobi_restart16_float64 | no | not_profiled |  | not_profiled | after_selection_limit | 0 | 0 |  |  |  |  |
| suitesparse:Negre/dendrimer | taichi_csr_gmres_jacobi_restart32_float64 | no | not_profiled |  | not_profiled | after_selection_limit | 0 | 0 |  |  |  |  |
| suitesparse:Goodwin/Goodwin_010 | taichi_csr_gmres_jacobi_restart8_float64 | no | not_profiled |  | not_profiled | after_selection_limit | 0 | 0 |  |  |  |  |
| suitesparse:Goodwin/Goodwin_010 | taichi_csr_gmres_jacobi_restart16_float64 | no | not_profiled |  | not_profiled | after_selection_limit | 0 | 0 |  |  |  |  |
| suitesparse:Goodwin/Goodwin_010 | taichi_csr_gmres_jacobi_restart32_float64 | no | not_profiled |  | not_profiled | after_selection_limit | 0 | 0 |  |  |  |  |
| suitesparse:FIDAP/ex5 | taichi_csr_gmres_jacobi_restart8_float64 | no | not_profiled |  | not_profiled | after_selection_limit | 0 | 0 |  |  |  |  |
| suitesparse:FIDAP/ex5 | taichi_csr_gmres_jacobi_restart16_float64 | no | not_profiled |  | not_profiled | after_selection_limit | 0 | 0 |  |  |  |  |
| suitesparse:FIDAP/ex5 | taichi_csr_gmres_jacobi_restart32_float64 | no | not_profiled |  | not_profiled | after_selection_limit | 0 | 0 |  |  |  |  |
| suitesparse:Bai/cdde1 | taichi_csr_gmres_jacobi_restart8_float64 | no | not_profiled |  | not_profiled | after_selection_limit | 0 | 0 |  |  |  |  |
| suitesparse:Bai/cdde1 | taichi_csr_gmres_jacobi_restart16_float64 | no | not_profiled |  | not_profiled | after_selection_limit | 0 | 0 |  |  |  |  |
| suitesparse:Bai/cdde1 | taichi_csr_gmres_jacobi_restart32_float64 | no | not_profiled |  | not_profiled | after_selection_limit | 0 | 0 |  |  |  |  |
| suitesparse:HB/curtis54 | taichi_csr_richardson_jacobi_float64 | no | not_applicable |  | not_applicable | not_symmetric | 0 | 0 |  |  |  |  |
| suitesparse:HB/fs_183_1 | taichi_csr_richardson_jacobi_float64 | no | not_applicable |  | not_applicable | not_symmetric | 0 | 0 |  |  |  |  |
| suitesparse:HB/young3c | taichi_csr_richardson_jacobi_float64 | no | not_applicable |  | not_applicable | not_symmetric | 0 | 0 |  |  |  |  |
| suitesparse:Grund/b1_ss | taichi_csr_richardson_jacobi_float64 | no | not_applicable |  | not_applicable | not_symmetric | 0 | 0 |  |  |  |  |
| suitesparse:Zitney/extr1b | taichi_csr_richardson_jacobi_float64 | no | not_applicable |  | not_applicable | not_symmetric | 0 | 0 |  |  |  |  |
| suitesparse:Hamrle/Hamrle1 | taichi_csr_richardson_jacobi_float64 | no | not_applicable |  | not_applicable | not_symmetric | 0 | 0 |  |  |  |  |
| suitesparse:Sandia/oscil_dcop_01 | taichi_csr_richardson_jacobi_float64 | no | not_applicable |  | not_applicable | not_symmetric | 0 | 0 |  |  |  |  |
| suitesparse:Negre/dendrimer | taichi_csr_richardson_jacobi_float64 | no | not_profiled |  | not_profiled | above_smoke_size_limit | 0 | 0 |  |  |  |  |
| suitesparse:Goodwin/Goodwin_010 | taichi_csr_richardson_jacobi_float64 | no | not_applicable |  | not_applicable | not_symmetric | 0 | 0 |  |  |  |  |
| suitesparse:FIDAP/ex5 | taichi_csr_richardson_jacobi_float64 | no | screened_out | cpu_reference_richardson_screen_failed | applicable |  | 0 | 1 |  |  | inf | 1.34555e+147 |
| suitesparse:Bai/cdde1 | taichi_csr_richardson_jacobi_float64 | no | not_applicable |  | not_applicable | not_symmetric | 0 | 0 |  |  |  |  |
| suitesparse:HB/curtis54 | taichi_csr_chebyshev_jacobi_float64 | no | not_applicable |  | not_applicable | not_symmetric | 0 | 0 |  |  |  |  |
| suitesparse:HB/fs_183_1 | taichi_csr_chebyshev_jacobi_float64 | no | not_applicable |  | not_applicable | not_symmetric | 0 | 0 |  |  |  |  |
| suitesparse:HB/young3c | taichi_csr_chebyshev_jacobi_float64 | no | not_applicable |  | not_applicable | not_symmetric | 0 | 0 |  |  |  |  |
| suitesparse:Grund/b1_ss | taichi_csr_chebyshev_jacobi_float64 | no | not_applicable |  | not_applicable | not_symmetric | 0 | 0 |  |  |  |  |
| suitesparse:Zitney/extr1b | taichi_csr_chebyshev_jacobi_float64 | no | not_applicable |  | not_applicable | not_symmetric | 0 | 0 |  |  |  |  |
| suitesparse:Hamrle/Hamrle1 | taichi_csr_chebyshev_jacobi_float64 | no | not_applicable |  | not_applicable | not_symmetric | 0 | 0 |  |  |  |  |
| suitesparse:Sandia/oscil_dcop_01 | taichi_csr_chebyshev_jacobi_float64 | no | not_applicable |  | not_applicable | not_symmetric | 0 | 0 |  |  |  |  |
| suitesparse:Negre/dendrimer | taichi_csr_chebyshev_jacobi_float64 | no | not_profiled |  | not_profiled | above_smoke_size_limit | 0 | 0 |  |  |  |  |
| suitesparse:Goodwin/Goodwin_010 | taichi_csr_chebyshev_jacobi_float64 | no | not_applicable |  | not_applicable | not_symmetric | 0 | 0 |  |  |  |  |
| suitesparse:FIDAP/ex5 | taichi_csr_chebyshev_jacobi_float64 | no | screened_out | cpu_reference_chebyshev_screen_failed | applicable |  | 0 | 1 |  |  | 0.765041 | 0.995003 |
| suitesparse:Bai/cdde1 | taichi_csr_chebyshev_jacobi_float64 | no | not_applicable |  | not_applicable | not_symmetric | 0 | 0 |  |  |  |  |
