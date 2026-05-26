# CSR ILU0 Guarded Integration

- status: `passed`
- ilu0_selector_rows: `2`
- ilu0_success_rows_imported: `1`
- ilu0_failed_numeric_gate_rows_imported: `1`
- promoted_success_rows: `1`
- failed_numeric_gate_rows_blocked: `1`
- executed_gpu_scenarios: `1`
- production_runtime_selector_changed: `False`

| matrix | target_status | guard | gpu | solver | preconditioner | rel_res | sol_err | failures |
|---|---|---|---:|---|---|---:|---:|---|
| suitesparse:Bai/cdde1 | success | promoted | True | bicgstab | ilu0 | 2.87583e-06 | 0.000109312 | none |
| suitesparse:MathWorks/tomography | failed_numeric_gate | blocked_non_success_candidate | False | None | None | na | na | none |
