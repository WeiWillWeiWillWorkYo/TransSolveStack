# CSR Selector Readiness

- status: `passed`
- schema_version: `phase1_csr_selector_features_v8`
- diagnostic_rows: `8`
- selector_rows: `24`
- success_rows: `8`
- failed_rows: `16`
- applicability_rows: `0`
- oracle_rows: `4`
- matrices_with_selector_rows: `8`
- max_final_relative_residual: `5.73677e-06`
- max_cpu_recomputed_relative_residual: `5.73677e-06`
- max_solution_relative_error: `0.00075577`
- min_success_rate: `0`
- max_solve_time_iqr_ms: `0`
- failure_reason_counts: `{'cpu_reference_bicgstab_screen_failed': 4, 'cpu_reference_cg_screen_failed': 4, 'cpu_reference_gmres_screen_failed': 2, 'cpu_reference_pcg_screen_failed': 3, 'cpu_reference_richardson_screen_failed': 3}`
- applicability_status_counts: `{}`
- applicability_reason_counts: `{}`

| matrix | candidate | oracle | status | failure | applicability | reason | success_rate | repeats | median_ms | regret_ms | rel_res | sol_err |
|---|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|
| suitesparse:MathWorks/tomography | taichi_csr_bicgstab_none_float64 | no | screened_out | cpu_reference_bicgstab_screen_failed | applicable |  | 0 | 1 |  |  | 2.82738e-06 | 0.827096 |
| suitesparse:MathWorks/tomography | taichi_csr_bicgstab_jacobi_float64 | no | screened_out | cpu_reference_bicgstab_screen_failed | applicable |  | 0 | 1 |  |  | 8.13054e-06 | 2.05748 |
| suitesparse:MathWorks/tomography | taichi_csr_gmres_jacobi_restart16_float64 | yes | success |  | applicable |  | 1 | 1 | 1021.73 | 0 | 1.01192e-06 | 0.00075577 |
| suitesparse:HB/jgl009 | taichi_csr_bicgstab_none_float64 | no | success |  | applicable |  | 1 | 1 | 1020.43 | 109.691 | 1.61094e-08 | 3.55943e-07 |
| suitesparse:HB/jgl009 | taichi_csr_bicgstab_jacobi_float64 | no | success |  | applicable |  | 1 | 1 | 1076.28 | 165.538 | 1.61094e-08 | 3.55943e-07 |
| suitesparse:HB/jgl009 | taichi_csr_gmres_jacobi_restart16_float64 | yes | success |  | applicable |  | 1 | 1 | 910.743 | 0 | 7.04831e-09 | 1.29953e-08 |
| suitesparse:HB/ibm32 | taichi_csr_bicgstab_none_float64 | yes | success |  | applicable |  | 1 | 1 | 1642.12 | 0 | 5.73677e-06 | 0.000322589 |
| suitesparse:HB/ibm32 | taichi_csr_bicgstab_jacobi_float64 | no | success |  | applicable |  | 1 | 1 | 1698.7 | 56.5853 | 5.73677e-06 | 0.000322589 |
| suitesparse:HB/ibm32 | taichi_csr_gmres_jacobi_restart16_float64 | no | screened_out | cpu_reference_gmres_screen_failed | applicable |  | 0 | 1 |  |  | 0.00293604 | 0.134923 |
| suitesparse:SNAP/email-Eu-core | taichi_csr_bicgstab_none_float64 | no | screened_out | cpu_reference_bicgstab_screen_failed | applicable |  | 0 | 1 |  |  | 0.321837 | 8.23796 |
| suitesparse:SNAP/email-Eu-core | taichi_csr_bicgstab_jacobi_float64 | no | screened_out | cpu_reference_bicgstab_screen_failed | applicable |  | 0 | 1 |  |  | 0.321837 | 8.23796 |
| suitesparse:SNAP/email-Eu-core | taichi_csr_gmres_jacobi_restart16_float64 | no | screened_out | cpu_reference_gmres_screen_failed | applicable |  | 0 | 1 |  |  | 0.0396667 | 0.833799 |
| suitesparse:Oberwolfach/t2dal_e | taichi_csr_cg_none_float64 | no | screened_out | cpu_reference_cg_screen_failed | applicable |  | 0 | 1 |  |  | 5.76569e-05 | 0.221708 |
| suitesparse:Oberwolfach/t2dal_e | taichi_csr_pcg_jacobi_float64 | no | success |  | applicable |  | 1 | 1 | 516.89 | 154.067 | 0 | 0 |
| suitesparse:Oberwolfach/t2dal_e | taichi_csr_richardson_jacobi_float64 | yes | success |  | applicable |  | 1 | 1 | 362.823 | 0 | 5.64503e-06 | 5.64503e-06 |
| suitesparse:HB/bcsstk07 | taichi_csr_cg_none_float64 | no | screened_out | cpu_reference_cg_screen_failed | applicable |  | 0 | 1 |  |  | 8.94851e-06 | 0.526635 |
| suitesparse:HB/bcsstk07 | taichi_csr_pcg_jacobi_float64 | no | screened_out | cpu_reference_pcg_screen_failed | applicable |  | 0 | 1 |  |  | 7.19328e-06 | 0.0808606 |
| suitesparse:HB/bcsstk07 | taichi_csr_richardson_jacobi_float64 | no | screened_out | cpu_reference_richardson_screen_failed | applicable |  | 0 | 1 |  |  | 1.37593e-05 | 0.882617 |
| suitesparse:HB/lshp1009 | taichi_csr_cg_none_float64 | no | screened_out | cpu_reference_cg_screen_failed | applicable |  | 0 | 1 |  |  | 0.00778825 | 0.0491823 |
| suitesparse:HB/lshp1009 | taichi_csr_pcg_jacobi_float64 | no | screened_out | cpu_reference_pcg_screen_failed | applicable |  | 0 | 1 |  |  | 0.00778825 | 0.0491823 |
| suitesparse:HB/lshp1009 | taichi_csr_richardson_jacobi_float64 | no | screened_out | cpu_reference_richardson_screen_failed | applicable |  | 0 | 1 |  |  | 846.813 | 3024.03 |
| suitesparse:Gset/G17 | taichi_csr_cg_none_float64 | no | screened_out | cpu_reference_cg_screen_failed | applicable |  | 0 | 1 |  |  | 0.224341 | 0.713351 |
| suitesparse:Gset/G17 | taichi_csr_pcg_jacobi_float64 | no | screened_out | cpu_reference_pcg_screen_failed | applicable |  | 0 | 1 |  |  | 0.224341 | 0.713351 |
| suitesparse:Gset/G17 | taichi_csr_richardson_jacobi_float64 | no | screened_out | cpu_reference_richardson_screen_failed | applicable |  | 0 | 1 |  |  | inf | inf |
