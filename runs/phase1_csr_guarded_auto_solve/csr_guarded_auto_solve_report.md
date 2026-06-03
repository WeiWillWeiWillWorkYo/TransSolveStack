# CSR Guarded Auto Solve Smoke

- status: `passed`
- solves: `2`
- successes: `2`
- quality_gate_blocks: `2`
- learned_prediction_source: `model_artifact`
- saved_model_loaded_count: `2`
- learned_runtime_promotions: `0`
- runtime_selector_changed: `False`
- selected_solver_set: `bicgstab, pcg`
- max_final_relative_residual: `8.72966e-06`
- max_cpu_recomputed_relative_residual: `8.72966e-06`
- max_solution_relative_error: `0.000422806`

| matrix | runtime candidate | learned candidate | guard | solver | status | rel_res |
|---|---|---|---|---|---|---:|
| suitesparse:FIDAP/ex5 | taichi_csr_pcg_jacobi_float64 | taichi_csr_chebyshev_jacobi_float64 | blocked_quality_gate | pcg | success | 6.69498e-06 |
| suitesparse:HB/curtis54 | taichi_csr_bicgstab_none_float64 | taichi_csr_bicgstab_none_float64 | blocked_quality_gate | bicgstab | success | 8.72966e-06 |
