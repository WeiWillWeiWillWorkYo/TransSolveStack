# Taichi CSR Symmetric Equilibration

- status: `passed`
- gpu_executed_rows: `1`
- candidate_promoted: `False`
- failed_numeric_gate_rows: `1`
- executes_gpu: `True`
- runtime_selector_changed: `False`
- next_step: `do_not_promote_symmetric_equilibration_for_bcsstk07_without_ic0_or_better_gate`

| matrix | candidate | solver_status | numeric_status | iters | rel_res | cpu_rel_res | sol_err | failures |
|---|---|---|---|---:|---:|---:|---:|---|
| suitesparse:HB/bcsstk07 | taichi_csr_pcg_symmetric_equilibration_float64 | success | failed_numeric_gate | 104 | 6.16727e-06 | 6.16727e-06 | 0.0810366 | solution_error_above_tolerance |
