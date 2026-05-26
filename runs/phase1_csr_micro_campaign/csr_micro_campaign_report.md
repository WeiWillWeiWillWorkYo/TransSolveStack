# CSR Micro-Campaign

- status: `passed`
- imported_matrices: `8`
- candidate_jobs: `24`
- gpu_success_rows: `8`
- cpu_screened_out_rows: `16`
- gpu_failed_rows: `0`
- selector_rows: `24`
- selector_oracle_rows: `4`
- max_final_relative_residual: `5.73677e-06`
- max_cpu_recomputed_relative_residual: `5.73677e-06`
- max_solution_relative_error: `0.00075577`

| matrix | candidate | status | solver | preconditioner | repeats | median_ms | iters | rel_res | sol_err |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| suitesparse:MathWorks/tomography | taichi_csr_bicgstab_none_float64 | screened_out | bicgstab | none | 1 |  | 4 | 2.82738e-06 | 0.827096 |
| suitesparse:MathWorks/tomography | taichi_csr_bicgstab_jacobi_float64 | screened_out | bicgstab | jacobi | 1 |  | 5 | 8.13054e-06 | 2.05748 |
| suitesparse:MathWorks/tomography | taichi_csr_gmres_jacobi_restart16_float64 | success | gmres | jacobi | 1 | 1021.73 | 11 | 1.01192e-06 | 0.00075577 |
| suitesparse:HB/jgl009 | taichi_csr_bicgstab_none_float64 | success | bicgstab | none | 1 | 1020.43 | 6 | 1.61094e-08 | 3.55943e-07 |
| suitesparse:HB/jgl009 | taichi_csr_bicgstab_jacobi_float64 | success | bicgstab | jacobi | 1 | 1076.28 | 6 | 1.61094e-08 | 3.55943e-07 |
| suitesparse:HB/jgl009 | taichi_csr_gmres_jacobi_restart16_float64 | success | gmres | jacobi | 1 | 910.743 | 5 | 7.04831e-09 | 1.29953e-08 |
| suitesparse:HB/ibm32 | taichi_csr_bicgstab_none_float64 | success | bicgstab | none | 1 | 1642.12 | 88 | 5.73677e-06 | 0.000322589 |
| suitesparse:HB/ibm32 | taichi_csr_bicgstab_jacobi_float64 | success | bicgstab | jacobi | 1 | 1698.7 | 88 | 5.73677e-06 | 0.000322589 |
| suitesparse:HB/ibm32 | taichi_csr_gmres_jacobi_restart16_float64 | screened_out | gmres | jacobi | 1 |  | 256 | 0.00293604 | 0.134923 |
| suitesparse:SNAP/email-Eu-core | taichi_csr_bicgstab_none_float64 | screened_out | bicgstab | none | 1 |  | 256 | 0.321837 | 8.23796 |
| suitesparse:SNAP/email-Eu-core | taichi_csr_bicgstab_jacobi_float64 | screened_out | bicgstab | jacobi | 1 |  | 256 | 0.321837 | 8.23796 |
| suitesparse:SNAP/email-Eu-core | taichi_csr_gmres_jacobi_restart16_float64 | screened_out | gmres | jacobi | 1 |  | 256 | 0.0396667 | 0.833799 |
| suitesparse:Oberwolfach/t2dal_e | taichi_csr_cg_none_float64 | screened_out | cg | none | 1 |  | 256 | 5.76569e-05 | 0.221708 |
| suitesparse:Oberwolfach/t2dal_e | taichi_csr_pcg_jacobi_float64 | success | pcg | jacobi | 1 | 516.89 | 1 | 0 | 0 |
| suitesparse:Oberwolfach/t2dal_e | taichi_csr_richardson_jacobi_float64 | success | richardson | jacobi | 1 | 362.823 | 11 | 5.64503e-06 | 5.64503e-06 |
| suitesparse:HB/bcsstk07 | taichi_csr_cg_none_float64 | screened_out | cg | none | 1 |  | 248 | 8.94851e-06 | 0.526635 |
| suitesparse:HB/bcsstk07 | taichi_csr_pcg_jacobi_float64 | screened_out | pcg | jacobi | 1 |  | 103 | 7.19328e-06 | 0.0808606 |
| suitesparse:HB/bcsstk07 | taichi_csr_richardson_jacobi_float64 | screened_out | richardson | jacobi | 1 |  | 256 | 1.37593e-05 | 0.882617 |
| suitesparse:HB/lshp1009 | taichi_csr_cg_none_float64 | screened_out | cg | none | 1 |  | 4 | 0.00778825 | 0.0491823 |
| suitesparse:HB/lshp1009 | taichi_csr_pcg_jacobi_float64 | screened_out | pcg | jacobi | 1 |  | 4 | 0.00778825 | 0.0491823 |
| suitesparse:HB/lshp1009 | taichi_csr_richardson_jacobi_float64 | screened_out | richardson | jacobi | 1 |  | 256 | 846.813 | 3024.03 |
| suitesparse:Gset/G17 | taichi_csr_cg_none_float64 | screened_out | cg | none | 1 |  | 1 | 0.224341 | 0.713351 |
| suitesparse:Gset/G17 | taichi_csr_pcg_jacobi_float64 | screened_out | pcg | jacobi | 1 |  | 1 | 0.224341 | 0.713351 |
| suitesparse:Gset/G17 | taichi_csr_richardson_jacobi_float64 | screened_out | richardson | jacobi | 1 |  |  | inf | inf |
