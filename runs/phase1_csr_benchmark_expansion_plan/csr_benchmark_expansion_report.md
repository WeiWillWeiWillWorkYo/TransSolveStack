# CSR Benchmark Expansion Plan

- status: `passed`
- schema_version: `phase1_csr_benchmark_expansion_plan_v1`
- plan_id: `phase1_csr_benchmark_expansion_m49`
- runtime_selector_changed: `False`
- executes_gpu: `False`
- selected_source_matrices: `64`
- already_profiled_matrices: `12`
- eligible_unprofiled_matrices: `36`
- planned_matrices: `8`
- planned_candidate_jobs: `24`
- planned_gpu_solve_attempts: `24`
- total_planned_nnz: `82789`
- estimated_total_nnz_visits: `92343696`
- by_candidate_profile: `{'general': 4, 'symmetric': 4}`
- by_solver: `{'bicgstab': 8, 'gmres': 4, 'cg': 4, 'pcg': 4, 'richardson': 4}`
- skipped_existing_matrices: `12`
- skipped_resource_limit_matrices: `16`
- skipped_budget_matrices: `28`
- next_step: `import_planned_matrices_then_run_cpu_screened_taichi_csr_micro_campaign`

## Matrix Queue

| rank | matrix | shape | nnz | profile | symmetry | archive_mb |
|---:|---|---:|---:|---|---|---:|
| 13 | suitesparse:MathWorks/tomography | 500x500 | 28726 | general | partially_symmetric | 0.277 |
| 14 | suitesparse:HB/jgl009 | 9x9 | 50 | general | partially_symmetric | 0.001 |
| 15 | suitesparse:HB/ibm32 | 32x32 | 126 | general | partially_symmetric | 0.001 |
| 16 | suitesparse:SNAP/email-Eu-core | 1005x1005 | 25571 | general | partially_symmetric | 0.068 |
| 28 | suitesparse:Oberwolfach/t2dal_e | 4257x4257 | 4257 | symmetric | spd | 0.040 |
| 30 | suitesparse:HB/bcsstk07 | 420x420 | 7860 | symmetric | spd | 0.027 |
| 31 | suitesparse:HB/lshp1009 | 1009x1009 | 6865 | symmetric | symmetric | 0.009 |
| 32 | suitesparse:Gset/G17 | 800x800 | 9334 | symmetric | symmetric | 0.013 |

## Candidate Queue

| matrix | candidate | solver | preconditioner | repeats | max_iter | estimated_nnz_visits |
|---|---|---|---|---:|---:|---:|
| suitesparse:MathWorks/tomography | taichi_csr_bicgstab_none_float64 | bicgstab | none | 1 | 256 | 14707712 |
| suitesparse:MathWorks/tomography | taichi_csr_bicgstab_jacobi_float64 | bicgstab | jacobi | 1 | 256 | 14707712 |
| suitesparse:MathWorks/tomography | taichi_csr_gmres_jacobi_restart16_float64 | gmres | jacobi | 1 | 256 | 7813472 |
| suitesparse:HB/jgl009 | taichi_csr_bicgstab_none_float64 | bicgstab | none | 1 | 256 | 25600 |
| suitesparse:HB/jgl009 | taichi_csr_bicgstab_jacobi_float64 | bicgstab | jacobi | 1 | 256 | 25600 |
| suitesparse:HB/jgl009 | taichi_csr_gmres_jacobi_restart16_float64 | gmres | jacobi | 1 | 256 | 13600 |
| suitesparse:HB/ibm32 | taichi_csr_bicgstab_none_float64 | bicgstab | none | 1 | 256 | 64512 |
| suitesparse:HB/ibm32 | taichi_csr_bicgstab_jacobi_float64 | bicgstab | jacobi | 1 | 256 | 64512 |
| suitesparse:HB/ibm32 | taichi_csr_gmres_jacobi_restart16_float64 | gmres | jacobi | 1 | 256 | 34272 |
| suitesparse:SNAP/email-Eu-core | taichi_csr_bicgstab_none_float64 | bicgstab | none | 1 | 256 | 13092352 |
| suitesparse:SNAP/email-Eu-core | taichi_csr_bicgstab_jacobi_float64 | bicgstab | jacobi | 1 | 256 | 13092352 |
| suitesparse:SNAP/email-Eu-core | taichi_csr_gmres_jacobi_restart16_float64 | gmres | jacobi | 1 | 256 | 6955312 |
| suitesparse:Oberwolfach/t2dal_e | taichi_csr_cg_none_float64 | cg | none | 1 | 256 | 1089792 |
| suitesparse:Oberwolfach/t2dal_e | taichi_csr_pcg_jacobi_float64 | pcg | jacobi | 1 | 256 | 1089792 |
| suitesparse:Oberwolfach/t2dal_e | taichi_csr_richardson_jacobi_float64 | richardson | jacobi | 1 | 256 | 1089792 |
| suitesparse:HB/bcsstk07 | taichi_csr_cg_none_float64 | cg | none | 1 | 256 | 2012160 |
| suitesparse:HB/bcsstk07 | taichi_csr_pcg_jacobi_float64 | pcg | jacobi | 1 | 256 | 2012160 |
| suitesparse:HB/bcsstk07 | taichi_csr_richardson_jacobi_float64 | richardson | jacobi | 1 | 256 | 2012160 |
| suitesparse:HB/lshp1009 | taichi_csr_cg_none_float64 | cg | none | 1 | 256 | 1757440 |
| suitesparse:HB/lshp1009 | taichi_csr_pcg_jacobi_float64 | pcg | jacobi | 1 | 256 | 1757440 |
| suitesparse:HB/lshp1009 | taichi_csr_richardson_jacobi_float64 | richardson | jacobi | 1 | 256 | 1757440 |
| suitesparse:Gset/G17 | taichi_csr_cg_none_float64 | cg | none | 1 | 256 | 2389504 |
| suitesparse:Gset/G17 | taichi_csr_pcg_jacobi_float64 | pcg | jacobi | 1 | 256 | 2389504 |
| suitesparse:Gset/G17 | taichi_csr_richardson_jacobi_float64 | richardson | jacobi | 1 | 256 | 2389504 |
