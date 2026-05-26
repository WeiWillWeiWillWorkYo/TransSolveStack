# CSR Guarded Promotion Coverage Exec

- status: `passed`
- scenarios: `12`
- executed_gpu_scenarios: `8`
- guard_only_scenarios: `4`
- current_quality_gate_blocks: `4`
- fixture_learned_promotions: `2`
- non_success_blocks: `4`
- runtime_exception_fallbacks: `2`
- runtime_fallback_used_count: `2`
- current_runtime_selector_changed: `False`
- selected_solver_set: `bicgstab, chebyshev, gmres, pcg`
- max_final_relative_residual: `8.72966e-06`
- max_cpu_recomputed_relative_residual: `8.72966e-06`
- max_solution_relative_error: `0.00075577`

| scenario | kind | matrix | source | solver | gpu | fallback | status | rel_res |
|---|---|---|---|---|---|---|---|---:|
| m57_current_blocked_01 | current_ranker_quality_gate_block | suitesparse:JGD_Trefethen/Trefethen_20b | artifact | chebyshev | True | False | success | 4.7438e-06 |
| m57_current_blocked_02 | current_ranker_quality_gate_block | suitesparse:MathWorks/tomography | artifact | gmres | True | False | success | 1.01192e-06 |
| m57_current_blocked_03 | current_ranker_quality_gate_block | suitesparse:HB/ibm32 | artifact | bicgstab | True | False | success | 5.73677e-06 |
| m57_current_blocked_04 | current_ranker_quality_gate_block | suitesparse:HB/jgl009 | artifact | gmres | True | False | success | 7.04831e-09 |
| m57_fixture_promotion_01 | fixture_profiled_success_promotion | suitesparse:HB/jgl009 | learned | bicgstab | True | False | success | 1.61094e-08 |
| m57_fixture_promotion_02 | fixture_profiled_success_promotion | suitesparse:Oberwolfach/t2dal_e | learned | pcg | True | False | success | 0 |
| m57_non_success_block_01 | fixture_non_success_candidate_block | suitesparse:FIDAP/ex5 | artifact |  | False | False | success |  |
| m57_non_success_block_02 | fixture_non_success_candidate_block | suitesparse:Bai/cdde1 | artifact |  | False | False | success |  |
| m57_non_success_block_03 | fixture_non_success_candidate_block | suitesparse:Zitney/extr1b | artifact |  | False | False | success |  |
| m57_non_success_block_04 | fixture_non_success_candidate_block | suitesparse:Gset/G17 | artifact |  | False | False | success |  |
| m57_runtime_fallback_01 | runtime_exception_fallback | suitesparse:Grund/b1_ss | runtime_guard_fallback | gmres | True | True | success | 1.84509e-08 |
| m57_runtime_fallback_02 | runtime_exception_fallback | suitesparse:HB/curtis54 | runtime_guard_fallback | bicgstab | True | True | success | 8.72966e-06 |
