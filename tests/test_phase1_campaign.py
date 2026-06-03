from transsolvestack.profiling.campaign import build_phase1_campaign_report


def test_phase1_campaign_report_passes_current_artifacts():
    report = build_phase1_campaign_report("runs")
    assert report.campaign_id == "phase1_readiness"
    assert report.status == "passed"
    stage_ids = {stage.stage_id for stage in report.stages}
    queue_stage_ids = {
        stage_id
        for stage_id in stage_ids
        if stage_id.startswith("csr_queue_batch_")
        and stage_id.removeprefix("csr_queue_batch_").isdigit()
    }
    assert report.num_stages == 74 + len(queue_stage_ids)
    required_stage_ids = {
        "benchmark",
        "expanded_solver_benchmark",
        "regression",
        "sequence",
        "policy_selection",
        "policy_solve",
        "dataset_plan",
        "matrix_market_probe",
        "suitesparse_selection",
        "suitesparse_header_probe",
        "suitesparse_csr_import",
        "taichi_csr_matvec",
        "taichi_csr_primitives",
        "taichi_csr_solve",
        "taichi_csr_bicgstab",
        "taichi_csr_gmres",
        "taichi_csr_richardson",
        "taichi_csr_chebyshev",
        "taichi_csr_symmetric_equilibration",
        "taichi_csr_row_column_equilibration",
        "taichi_csr_ilu0",
        "csr_selector_readiness",
        "csr_selector_policy",
        "csr_auto_solve",
        "csr_learning_readiness",
        "csr_model_contract",
        "csr_training_tensors",
        "csr_linear_ranker",
        "csr_selector_model_eval",
        "csr_benchmark_expansion_plan",
        "csr_full_dataset_queue",
        "csr_queue_batch_00001",
        "csr_queue_batch_00002",
        "csr_queue_batch_00003",
        "csr_queue_training_pool",
        "csr_queue_batch_training_bundle",
        "csr_queue_batch_reference_ranker",
        "csr_queue_batch_model_replay",
        "csr_queue_candidate_coverage",
        "csr_gmres_restart_coverage",
        "csr_blocked_gap_probe",
        "csr_blocked_gap_positive_search",
        "csr_blocked_gap_training_integration",
        "csr_blocked_gap_augmented_ranker",
        "csr_blocked_gap_guarded_replay",
        "csr_transformer_handoff_bundle",
        "csr_transformer_training_package",
        "csr_transformer_package_consumer_dry_run",
        "csr_micro_campaign",
        "csr_transformer_ready",
        "csr_transformer_ranker",
        "csr_external_model_adapter",
        "csr_transformer_model_replay",
        "csr_transformer_quality_gate",
        "csr_policy_model_artifact",
        "csr_policy_model_acceptance",
        "csr_policy_model_submission",
        "csr_external_model_intake",
        "csr_transformer_training_entrypoint",
        "csr_transformer_reference_training_export",
        "csr_learned_runtime_guard",
        "csr_guarded_auto_solve",
        "csr_guarded_promotion_readiness",
        "csr_guarded_promotion_coverage_plan",
        "csr_guarded_promotion_coverage_exec",
        "csr_non_success_fallback_probe",
        "csr_guarded_non_success_fallback_integration",
        "csr_ilu0_guarded_integration",
        "csr_ilu0_coverage_expansion",
        "csr_unresolved_fallback_coverage_plan",
        "csr_unresolved_fallback_cpu_screen",
        "csr_unresolved_matrix_diagnostics",
        "public_release_hygiene",
        "transformer_readiness",
        "public_api_smoke",
        "csr_public_api_smoke",
        "solver_functional",
    }
    assert required_stage_ids <= stage_ids
    assert all(stage.row_count > 0 for stage in report.stages)
