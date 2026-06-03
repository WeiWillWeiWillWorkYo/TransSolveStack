# CSR Blocked Gap Positive Search

- status: `passed`
- positive_evidence_found: `True`
- selected_candidates: `8`
- gpu_success_rows: `8`
- gpu_failed_rows: `0`
- positive_gpu_gap_ids: `['general_bicgstab_ilu0', 'general_bicgstab_row_column_equilibration', 'symmetric_chebyshev_jacobi', 'symmetric_pcg_symmetric_equilibration']`
- queue_merge_ready: `False`
- next_step: `integrate positive blocked-gap rows into guarded selector evidence without generic queue merge`

| gap | matrix | candidate | status | iters | rel_res | sol_err |
|---|---|---|---|---:|---:|---:|
| general_bicgstab_row_column_equilibration | suitesparse:HB/curtis54 | taichi_csr_bicgstab_row_column_equilibration_float64 | success | 81 | 5.90786e-06 | 0.000472771 |
| general_bicgstab_row_column_equilibration | suitesparse:Grund/b1_ss | taichi_csr_bicgstab_row_column_equilibration_float64 | success | 9 | 1.52432e-06 | 1.54376e-05 |
| general_bicgstab_ilu0 | suitesparse:JGD_Trefethen/Trefethen_20b | taichi_csr_bicgstab_ilu0_float64 | success | 2 | 1.08595e-06 | 4.93118e-06 |
| symmetric_chebyshev_jacobi | suitesparse:JGD_Trefethen/Trefethen_20b | taichi_csr_chebyshev_jacobi_float64 | success | 9 | 4.7438e-06 | 5.96855e-06 |
| symmetric_pcg_symmetric_equilibration | suitesparse:JGD_Trefethen/Trefethen_20b | taichi_csr_pcg_symmetric_equilibration_float64 | success | 6 | 2.37775e-06 | 7.72059e-06 |
| general_bicgstab_ilu0 | suitesparse:FIDAP/ex5 | taichi_csr_bicgstab_ilu0_float64 | success | 15 | 4.48152e-06 | 2.30582e-06 |
| symmetric_pcg_symmetric_equilibration | suitesparse:FIDAP/ex5 | taichi_csr_pcg_symmetric_equilibration_float64 | success | 92 | 2.30831e-06 | 1.24131e-07 |
| symmetric_chebyshev_jacobi | suitesparse:HB/bcsstk01 | taichi_csr_chebyshev_jacobi_float64 | success | 222 | 8.11032e-06 | 0.000218803 |
