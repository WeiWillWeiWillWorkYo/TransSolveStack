# Phase 1 Campaign Report

- campaign: `phase1_readiness`
- status: `passed`
- stages: `52`

| stage | status | artifact_kind | rows | artifact_dir |
|---|---|---|---:|---|
| benchmark | passed | taichi_smoke_benchmark | 21 | runs/phase1_smoke_taichi |
| expanded_solver_benchmark | passed | taichi_smoke_benchmark | 35 | runs/phase1_smoke_expanded_solvers |
| regression | passed | taichi_numerical_regression | 21 | runs/phase1_numerical_regression |
| sequence | passed | taichi_sequence_benchmark | 8 | runs/phase1_sequence_taichi |
| policy_selection | passed | policy_selection | 7 | runs/phase1_policy_selection |
| policy_solve | passed | policy_driven_taichi_solve | 7 | runs/phase1_policy_solve |
| dataset_plan | passed | dataset_download_plan | 3 | runs/phase1_dataset_plan |
| matrix_market_probe | passed | matrix_market_metadata_probe | 1 | runs/phase1_matrix_market_probe |
| suitesparse_selection | passed | suitesparse_matrix_subset_selection | 64 | runs/phase1_suitesparse_selection |
| suitesparse_header_probe | passed | suitesparse_archive_header_probe | 64 | runs/phase1_suitesparse_header_probe |
| suitesparse_csr_import | passed | suitesparse_csr_import_boundary | 12 | runs/phase1_suitesparse_csr_import |
| taichi_csr_matvec | passed | taichi_csr_matvec_smoke | 4 | runs/phase1_taichi_csr_matvec |
| taichi_csr_primitives | passed | taichi_csr_primitives_smoke | 4 | runs/phase1_taichi_csr_primitives |
| taichi_csr_solve | passed | taichi_csr_solve_smoke | 4 | runs/phase1_taichi_csr_solve |
| taichi_csr_bicgstab | passed | taichi_csr_bicgstab_smoke | 4 | runs/phase1_taichi_csr_bicgstab |
| taichi_csr_gmres | passed | taichi_csr_gmres_smoke | 5 | runs/phase1_taichi_csr_gmres |
| taichi_csr_richardson | passed | taichi_csr_richardson_smoke | 1 | runs/phase1_taichi_csr_richardson |
| taichi_csr_chebyshev | passed | taichi_csr_chebyshev_smoke | 1 | runs/phase1_taichi_csr_chebyshev |
| taichi_csr_symmetric_equilibration | passed | taichi_csr_symmetric_equilibration_smoke | 1 | runs/phase1_taichi_csr_symmetric_equilibration |
| taichi_csr_row_column_equilibration | passed | taichi_csr_row_column_equilibration_smoke | 2 | runs/phase1_taichi_csr_row_column_equilibration |
| taichi_csr_ilu0 | passed | taichi_csr_ilu0_smoke | 2 | runs/phase1_taichi_csr_ilu0 |
| csr_selector_readiness | passed | csr_selector_readiness_export | 108 | runs/phase1_csr_selector_readiness |
| csr_selector_policy | passed | csr_selector_policy_smoke | 4 | runs/phase1_csr_selector_policy |
| csr_auto_solve | passed | csr_auto_solve_smoke | 2 | runs/phase1_csr_auto_solve |
| csr_learning_readiness | passed | csr_learning_readiness_export | 108 | runs/phase1_csr_learning_readiness |
| csr_model_contract | passed | csr_model_contract_export | 12 | runs/phase1_csr_model_contract |
| csr_training_tensors | passed | csr_training_tensor_export | 12 | runs/phase1_csr_training_tensors |
| csr_linear_ranker | passed | csr_linear_ranker_baseline | 12 | runs/phase1_csr_linear_ranker |
| csr_selector_model_eval | passed | csr_selector_model_quality_gate | 2 | runs/phase1_csr_selector_model_eval |
| csr_benchmark_expansion_plan | passed | csr_benchmark_expansion_plan | 24 | runs/phase1_csr_benchmark_expansion_plan |
| csr_micro_campaign | passed | csr_micro_campaign | 24 | runs/phase1_csr_micro_campaign |
| csr_transformer_ready | passed | csr_transformer_ready_bundle | 20 | runs/phase1_csr_transformer_ready |
| csr_transformer_ranker | passed | csr_transformer_ranker | 20 | runs/phase1_csr_transformer_ranker |
| csr_transformer_quality_gate | passed | csr_transformer_quality_gate | 2 | runs/phase1_csr_transformer_quality_gate |
| csr_transformer_training_entrypoint | passed | csr_transformer_training_entrypoint | 6 | runs/phase1_csr_transformer_training_entrypoint |
| csr_learned_runtime_guard | passed | csr_learned_runtime_guard | 4 | runs/phase1_csr_learned_guard |
| csr_guarded_auto_solve | passed | csr_guarded_auto_solve_smoke | 2 | runs/phase1_csr_guarded_auto_solve |
| csr_guarded_promotion_readiness | passed | csr_guarded_promotion_readiness | 3 | runs/phase1_csr_guarded_promotion_readiness |
| csr_guarded_promotion_coverage_plan | passed | csr_guarded_promotion_coverage_plan | 15 | runs/phase1_csr_guarded_promotion_coverage_plan |
| csr_guarded_promotion_coverage_exec | passed | csr_guarded_promotion_coverage_exec | 12 | runs/phase1_csr_guarded_promotion_coverage_exec |
| csr_non_success_fallback_probe | passed | csr_non_success_fallback_probe | 15 | runs/phase1_csr_non_success_fallback_probe |
| csr_guarded_non_success_fallback_integration | passed | csr_guarded_non_success_fallback_integration | 4 | runs/phase1_csr_guarded_non_success_fallback_integration |
| csr_ilu0_guarded_integration | passed | csr_ilu0_guarded_integration | 2 | runs/phase1_csr_ilu0_guarded_integration |
| csr_ilu0_coverage_expansion | passed | csr_ilu0_coverage_expansion | 8 | runs/phase1_csr_ilu0_coverage_expansion |
| csr_unresolved_fallback_coverage_plan | passed | csr_unresolved_fallback_coverage_plan | 6 | runs/phase1_csr_unresolved_fallback_coverage_plan |
| csr_unresolved_fallback_cpu_screen | passed | csr_unresolved_fallback_cpu_screen | 4 | runs/phase1_csr_unresolved_fallback_cpu_screen |
| csr_unresolved_matrix_diagnostics | passed | csr_unresolved_matrix_diagnostics | 2 | runs/phase1_csr_unresolved_matrix_diagnostics |
| public_release_hygiene | passed | public_release_hygiene | 79 | runs/phase1_public_release_hygiene |
| transformer_readiness | passed | transformer_readiness_export | 21 | runs/phase1_transformer_readiness |
| public_api_smoke | passed | public_api_smoke | 1 | runs/phase1_public_api_smoke |
| csr_public_api_smoke | passed | csr_public_api_smoke | 1 | runs/phase1_csr_public_api |
| solver_functional | passed | solver_functional_smoke | 2 | runs/phase1_solver_functional |
