# Taichi CSR ILU0

- status: `passed`
- solver: `bicgstab`
- preconditioner: `ilu0`
- gpu_executed_rows: `2`
- numeric_success_rows: `1`
- failed_numeric_gate_rows: `1`
- candidate_promoted_rows: `1`
- executes_gpu: `True`
- runtime_selector_changed: `False`
- next_step: `use ILU0 success rows as benchmark evidence only; integrate into guarded fallback after exact selector rows and fallback-chain gates`

| matrix | solver_status | numeric_status | iters | rel_res | cpu_rel_res | sol_err | min_pivot | failures |
|---|---|---|---:|---:|---:|---:|---:|---|
| suitesparse:Bai/cdde1 | success | success | 22 | 2.87583e-06 | 2.87583e-06 | 0.000109312 | 3.38052 | none |
| suitesparse:MathWorks/tomography | success | failed_numeric_gate | 3 | 5.11355e-06 | 5.11355e-06 | 0.0258158 | 0.381492 | solution_error_above_tolerance |
