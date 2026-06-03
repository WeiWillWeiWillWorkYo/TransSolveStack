# CSR Blocked Gap Internal Execution

- status: `passed`
- imported_matrices: `8`
- candidate_jobs: `16`
- gpu_success_rows: `0`
- cpu_screened_out_rows: `16`
- gpu_failed_rows: `0`

| matrix | candidate | status | backend | failure | rel_res | sol_err |
|---|---|---|---|---|---:|---:|
| suitesparse:HB/bp_1200 | taichi_csr_bicgstab_ilu0_float64 | screened_out | cpu_reference_screen | cpu_reference_bicgstab_screen_failed | inf | inf |
| suitesparse:HB/bp_1200 | taichi_csr_bicgstab_row_column_equilibration_float64 | screened_out | cpu_reference_screen | cpu_reference_bicgstab_screen_failed | 11.1564 | 306.996 |
| suitesparse:HB/bp_1400 | taichi_csr_bicgstab_ilu0_float64 | screened_out | cpu_reference_screen | cpu_reference_bicgstab_screen_failed | inf | inf |
| suitesparse:HB/bp_1400 | taichi_csr_bicgstab_row_column_equilibration_float64 | screened_out | cpu_reference_screen | cpu_reference_bicgstab_screen_failed | 60.6812 | 1509.32 |
| suitesparse:HB/bp_1600 | taichi_csr_bicgstab_ilu0_float64 | screened_out | cpu_reference_screen | cpu_reference_bicgstab_screen_failed | inf | inf |
| suitesparse:HB/bp_1600 | taichi_csr_bicgstab_row_column_equilibration_float64 | screened_out | cpu_reference_screen | cpu_reference_bicgstab_screen_failed | 13.6299 | 376.834 |
| suitesparse:HB/bp_200 | taichi_csr_bicgstab_ilu0_float64 | screened_out | cpu_reference_screen | cpu_reference_bicgstab_screen_failed | inf | inf |
| suitesparse:HB/bp_200 | taichi_csr_bicgstab_row_column_equilibration_float64 | screened_out | cpu_reference_screen | cpu_reference_bicgstab_screen_failed | 32.4531 | 935.599 |
| suitesparse:HB/bp_400 | taichi_csr_bicgstab_ilu0_float64 | screened_out | cpu_reference_screen | cpu_reference_bicgstab_screen_failed | inf | inf |
| suitesparse:HB/bp_400 | taichi_csr_bicgstab_row_column_equilibration_float64 | screened_out | cpu_reference_screen | cpu_reference_bicgstab_screen_failed | 47.3449 | 1150.45 |
| suitesparse:HB/bp_600 | taichi_csr_bicgstab_ilu0_float64 | screened_out | cpu_reference_screen | cpu_reference_bicgstab_screen_failed | inf | inf |
| suitesparse:HB/bp_600 | taichi_csr_bicgstab_row_column_equilibration_float64 | screened_out | cpu_reference_screen | cpu_reference_bicgstab_screen_failed | 11.3422 | 297.373 |
| suitesparse:HB/bp_800 | taichi_csr_bicgstab_ilu0_float64 | screened_out | cpu_reference_screen | cpu_reference_bicgstab_screen_failed | inf | inf |
| suitesparse:HB/bp_800 | taichi_csr_bicgstab_row_column_equilibration_float64 | screened_out | cpu_reference_screen | cpu_reference_bicgstab_screen_failed | 9.77964 | 196.672 |
| suitesparse:HB/can_1054 | taichi_csr_chebyshev_jacobi_float64 | screened_out | cpu_reference_screen | cpu_reference_chebyshev_screen_failed | inf | inf |
| suitesparse:HB/can_1054 | taichi_csr_pcg_symmetric_equilibration_float64 | screened_out | cpu_reference_screen | cpu_reference_pcg_screen_failed | 0.0821189 | 0.498182 |
