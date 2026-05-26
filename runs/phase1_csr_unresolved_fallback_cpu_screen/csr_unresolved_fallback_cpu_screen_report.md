# CSR Unresolved Fallback CPU Screen

- status: `passed`
- attempted_cpu_screens: `4`
- screen_success_rows: `0`
- cpu_screened_out_rows: `4`
- gpu_probe_ready_candidates: `0`
- executes_gpu: `False`
- runtime_selector_changed: `False`
- next_step: `defer_unresolved_to_future_preconditioners_or_formulation_diagnostics`

| matrix | candidate | status | solver | preconditioner | iters | rel_res | sol_err | reason |
|---|---|---|---|---|---:|---:|---:|---|
| suitesparse:Gset/G17 | taichi_csr_gmres_none_restart64_float64 | screened_out | gmres | none | 2048 | 3.12987e-05 | 0.0179331 | cpu_screen_residual_above_tolerance |
| suitesparse:Gset/G17 | taichi_csr_bicgstab_none_maxiter2048_float64 | screened_out | bicgstab | none | 1305 | 9.96165e-06 | 0.0082299 | cpu_screen_solution_error_above_tolerance |
| suitesparse:Zitney/extr1b | taichi_csr_gmres_none_restart64_float64 | screened_out | gmres | none | 2048 | 0.864132 | 4198.65 | cpu_screen_residual_above_tolerance |
| suitesparse:Zitney/extr1b | taichi_csr_gmres_jacobi_restart64_float64 | screened_out | gmres | jacobi | 2048 | 0.864132 | 4198.65 | cpu_screen_residual_above_tolerance |
