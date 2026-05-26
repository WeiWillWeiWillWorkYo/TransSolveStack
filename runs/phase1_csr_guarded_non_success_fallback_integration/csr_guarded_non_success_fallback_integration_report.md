# CSR Guarded Non-Success Fallback Integration

- status: `passed`
- target_non_success_scenarios: `4`
- resolved_exact_fallback_scenarios: `2`
- unresolved_exact_fallback_scenarios: `2`
- executed_gpu_scenarios: `2`
- guard_only_scenarios: `2`
- learned_non_success_blocks: `4`
- artifact_fallback_solves: `2`
- selected_solver_set: `bicgstab, pcg`
- max_final_relative_residual: `6.69498e-06`
- max_cpu_recomputed_relative_residual: `6.69528e-06`
- max_solution_relative_error: `4.40981e-05`

| scenario | matrix | selected | exact fallback | executed_gpu | runtime candidate | solver | status | rel_res |
|---|---|---|---:|---:|---|---|---|---:|
| m57_non_success_block_01 | suitesparse:FIDAP/ex5 | taichi_csr_chebyshev_jacobi_float64 | 3 | True | taichi_csr_pcg_jacobi_float64 | pcg | success | 6.69498e-06 |
| m57_non_success_block_02 | suitesparse:Bai/cdde1 | taichi_csr_gmres_jacobi_restart8_float64 | 3 | True | taichi_csr_bicgstab_jacobi_float64 | bicgstab | success | 1.55458e-06 |
| m57_non_success_block_03 | suitesparse:Zitney/extr1b | taichi_csr_pcg_jacobi_float64 | 0 | False | taichi_csr_chebyshev_jacobi_float64 |  | success |  |
| m57_non_success_block_04 | suitesparse:Gset/G17 | taichi_csr_richardson_jacobi_float64 | 0 | False | taichi_csr_chebyshev_jacobi_float64 |  | success |  |
