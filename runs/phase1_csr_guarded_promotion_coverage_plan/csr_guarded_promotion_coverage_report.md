# CSR Guarded Promotion Coverage Plan

- status: `passed`
- schema_version: `phase1_csr_guarded_promotion_coverage_plan_v1`
- scenarios: `15`
- planned_gpu_final_solves: `11`
- planned_guard_only_scenarios: `4`
- current_blocked_scenarios: `4`
- fixture_promotion_scenarios: `4`
- non_success_block_scenarios: `4`
- runtime_fallback_scenarios: `3`
- quality_gate_runtime_eligible: `False`
- runtime_selector_changed: `False`
- executes_gpu: `False`

| scenario | kind | matrix | selected | runtime | gpu | status |
|---|---|---|---|---|---|---|
| m57_current_blocked_01 | current_ranker_quality_gate_block | suitesparse:JGD_Trefethen/Trefethen_20b | taichi_csr_chebyshev_jacobi_float64 | taichi_csr_chebyshev_jacobi_float64 | True | queued_current_gate_block_gpu_smoke |
| m57_current_blocked_02 | current_ranker_quality_gate_block | suitesparse:MathWorks/tomography | taichi_csr_gmres_jacobi_restart16_float64 | taichi_csr_gmres_jacobi_restart16_float64 | True | queued_current_gate_block_gpu_smoke |
| m57_current_blocked_03 | current_ranker_quality_gate_block | suitesparse:HB/ibm32 | taichi_csr_bicgstab_none_float64 | taichi_csr_bicgstab_none_float64 | True | queued_current_gate_block_gpu_smoke |
| m57_current_blocked_04 | current_ranker_quality_gate_block | suitesparse:HB/jgl009 | taichi_csr_bicgstab_none_float64 | taichi_csr_gmres_jacobi_restart16_float64 | True | queued_current_gate_block_gpu_smoke |
| m57_fixture_promotion_01 | fixture_profiled_success_promotion | suitesparse:HB/jgl009 | taichi_csr_bicgstab_none_float64 | taichi_csr_bicgstab_none_float64 | True | queued_fixture_promotion_gpu_smoke |
| m57_fixture_promotion_02 | fixture_profiled_success_promotion | suitesparse:Oberwolfach/t2dal_e | taichi_csr_pcg_jacobi_float64 | taichi_csr_pcg_jacobi_float64 | True | queued_fixture_promotion_gpu_smoke |
| m57_fixture_promotion_03 | fixture_profiled_success_promotion | suitesparse:Grund/b1_ss | taichi_csr_gmres_jacobi_restart8_float64 | taichi_csr_gmres_jacobi_restart8_float64 | True | queued_fixture_promotion_gpu_smoke |
| m57_fixture_promotion_04 | fixture_profiled_success_promotion | suitesparse:JGD_Trefethen/Trefethen_20b | taichi_csr_chebyshev_jacobi_float64 | taichi_csr_chebyshev_jacobi_float64 | True | queued_fixture_promotion_gpu_smoke |
| m57_non_success_block_01 | fixture_non_success_candidate_block | suitesparse:FIDAP/ex5 | taichi_csr_chebyshev_jacobi_float64 | taichi_csr_pcg_jacobi_float64 | False | queued_guard_only_no_exact_profiled_fallback |
| m57_non_success_block_02 | fixture_non_success_candidate_block | suitesparse:Bai/cdde1 | taichi_csr_gmres_jacobi_restart8_float64 |  | False | queued_guard_only_no_exact_profiled_fallback |
| m57_non_success_block_03 | fixture_non_success_candidate_block | suitesparse:Zitney/extr1b | taichi_csr_pcg_jacobi_float64 |  | False | queued_guard_only_no_exact_profiled_fallback |
| m57_non_success_block_04 | fixture_non_success_candidate_block | suitesparse:Gset/G17 | taichi_csr_richardson_jacobi_float64 |  | False | queued_guard_only_no_exact_profiled_fallback |
| m57_runtime_fallback_01 | runtime_exception_fallback | suitesparse:Grund/b1_ss | injected_invalid_primary | taichi_csr_gmres_jacobi_restart16_float64 | True | queued_runtime_exception_fallback_gpu_smoke |
| m57_runtime_fallback_02 | runtime_exception_fallback | suitesparse:HB/curtis54 | injected_invalid_primary | taichi_csr_bicgstab_none_float64 | True | queued_runtime_exception_fallback_gpu_smoke |
| m57_runtime_fallback_03 | runtime_exception_fallback | suitesparse:JGD_Trefethen/Trefethen_20b | injected_invalid_primary | taichi_csr_chebyshev_jacobi_float64 | True | queued_runtime_exception_fallback_gpu_smoke |
