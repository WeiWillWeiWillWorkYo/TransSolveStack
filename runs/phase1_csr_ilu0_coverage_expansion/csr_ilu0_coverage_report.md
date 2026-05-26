# CSR ILU0 Coverage Expansion

- status: `passed`
- candidate_rows: `8`
- numeric_success_rows: `3`
- failed_numeric_gate_rows: `1`
- setup_failed_rows: `4`
- merge_ready_success_rows: `3`
- gpu_solve_rows: `4`
- runtime_selector_changed: `False`
- merged_into_main_transformer_ready: `False`
- next_step: `merge ILU0 success rows into the main Transformer-ready bundle only after adding enough exact context successes and a dedicated quality gate`

| matrix | status | solver_status | gpu_solve | iters | rel_res | cpu_rel_res | sol_err | failure |
|---|---|---|---:|---:|---:|---:|---:|---|
| suitesparse:Grund/b1_ss | setup_failed | setup_failed | False | None | nan | nan | nan | ilu0_setup_failed |
| suitesparse:JGD_Trefethen/Trefethen_20b | success | success | True | 2 | 1.08595e-06 | 1.08595e-06 | 4.93118e-06 | none |
| suitesparse:FIDAP/ex5 | success | success | True | 15 | 4.48152e-06 | 4.4814e-06 | 2.30582e-06 | none |
| suitesparse:Hamrle/Hamrle1 | setup_failed | setup_failed | False | None | nan | nan | nan | ilu0_setup_failed |
| suitesparse:HB/curtis54 | setup_failed | setup_failed | False | None | nan | nan | nan | ilu0_setup_failed |
| suitesparse:Sandia/oscil_dcop_01 | setup_failed | setup_failed | False | None | nan | nan | nan | ilu0_setup_failed |
| suitesparse:HB/young3c | failed_numeric_gate | failed | True | 384 | 7.13347 | 7.13347 | 19.6672 | not_converged,trace_residual_above_tolerance,cpu_recomputed_residual_above_tolerance,solution_error_above_tolerance,residual_did_not_drop |
| suitesparse:Bai/cdde1 | success | success | True | 22 | 2.87583e-06 | 2.87583e-06 | 0.000109312 | none |
