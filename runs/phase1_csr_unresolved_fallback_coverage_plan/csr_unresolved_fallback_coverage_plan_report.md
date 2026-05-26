# CSR Unresolved Fallback Coverage Plan

- status: `passed`
- unresolved_matrices: `suitesparse:Gset/G17, suitesparse:Zitney/extr1b`
- candidate_plan_rows: `6`
- cpu_screen_ready_candidates: `4`
- future_dependency_candidates: `2`
- executes_gpu: `False`
- runtime_selector_changed: `False`

| row | matrix | kind | candidate | stage | blocker |
|---|---|---|---|---|---|
| m61_gset_g17_01 | suitesparse:Gset/G17 | existing_solver_parameter_sweep | taichi_csr_gmres_none_restart64_float64 | cpu_screen_ready | zero_diagonal_and_solution_error_gate |
| m61_gset_g17_02 | suitesparse:Gset/G17 | existing_solver_parameter_sweep | taichi_csr_bicgstab_none_maxiter2048_float64 | cpu_screen_ready | zero_diagonal_and_solution_error_gate |
| m61_gset_g17_03 | suitesparse:Gset/G17 | matrix_formulation_diagnostic | graph_laplacian_or_shifted_diagonal_formulation_check | diagnostic_only_no_gpu | zero_diagonal_and_solution_error_gate |
| m61_zitney_extr1b_01 | suitesparse:Zitney/extr1b | existing_solver_parameter_sweep | taichi_csr_gmres_none_restart64_float64 | cpu_screen_ready | solution_error_above_gate_despite_small_residual |
| m61_zitney_extr1b_02 | suitesparse:Zitney/extr1b | existing_solver_parameter_sweep | taichi_csr_gmres_jacobi_restart64_float64 | cpu_screen_ready | solution_error_above_gate_despite_small_residual |
| m61_zitney_extr1b_03 | suitesparse:Zitney/extr1b | future_preconditioner_required | taichi_csr_bicgstab_ilu0_or_scaling_float64 | blocked_until_preconditioner_exists | solution_error_above_gate_despite_small_residual |
