# CSR Guarded Promotion Readiness

- status: `passed`
- scenarios: `3`
- successes: `3`
- current_quality_gate_blocks: `1`
- fixture_learned_promotions: `1`
- runtime_exception_fallbacks: `1`
- runtime_fallback_used_count: `1`
- current_runtime_selector_changed: `False`
- selected_solver_set: `bicgstab, pcg`
- max_final_relative_residual: `8.72966e-06`
- max_cpu_recomputed_relative_residual: `8.72966e-06`
- max_solution_relative_error: `0.000422806`

| scenario | matrix | source | guard | solver | fallback | status | rel_res |
|---|---|---|---|---|---|---|---:|
| current_ranker_blocked_by_quality_gate | suitesparse:FIDAP/ex5 | artifact | blocked_quality_gate | pcg | False | success | 6.69498e-06 |
| fixture_gate_promotes_profiled_success_candidate | suitesparse:FIDAP/ex5 | learned | promoted | pcg | False | success | 6.69498e-06 |
| runtime_exception_retries_profiled_gpu_fallback | suitesparse:HB/curtis54 | runtime_guard_fallback_fixture |  | bicgstab | True | success | 8.72966e-06 |
