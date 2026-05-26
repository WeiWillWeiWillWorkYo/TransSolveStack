# Taichi CSR Row/Column Equilibration

- status: `passed`
- solver: `bicgstab`
- gpu_executed_rows: `2`
- numeric_success_rows: `1`
- failed_numeric_gate_rows: `1`
- candidate_promoted_rows: `1`
- executes_gpu: `True`
- runtime_selector_changed: `False`
- next_step: `merge_successful_row_column_scaled_candidates_into_guarded_fallback only after exact-matrix selector rows and fallback-chain gates are updated`

| matrix | candidate | solver_status | numeric_status | iters | rel_res | cpu_rel_res | sol_err | failures |
|---|---|---|---|---:|---:|---:|---:|---|
| suitesparse:HB/curtis54 | taichi_csr_bicgstab_row_column_equilibration_float64 | success | success | 81 | 5.90786e-06 | 5.90786e-06 | 0.000472771 | none |
| suitesparse:Zitney/extr1b | taichi_csr_bicgstab_row_column_equilibration_float64 | failed | failed_numeric_gate | 512 | 12.0498 | 12.0498 | 130703 | not_converged,trace_residual_above_tolerance,cpu_recomputed_residual_above_tolerance,solution_error_above_tolerance,scaled_residual_did_not_drop |
