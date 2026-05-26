from transsolvestack.policies.csr_transformer_training_entrypoint import (
    build_csr_transformer_training_entrypoint_from_files,
)


def test_csr_transformer_training_entrypoint_formalizes_training_boundary(tmp_path):
    export = build_csr_transformer_training_entrypoint_from_files(output_dir=tmp_path)
    summary = export["summary"]

    assert summary["status"] == "passed"
    assert summary["schema_version"] == "phase1_csr_transformer_training_entrypoint_v1"
    assert summary["training_entrypoint_ready"] is True
    assert summary["model_training_required"] is True
    assert summary["model_trained"] is False
    assert summary["runtime_selector_changed"] is False
    assert summary["transformer_connectable"] is True
    assert summary["quality_gate_enforced"] is True
    assert summary["current_quality_gate_runtime_eligible"] is False
    assert set(summary["current_quality_gate_failures"]) == {
        "below_min_profiled_success_rate",
        "below_baseline_oracle_top1",
        "below_baseline_profiled_success_rate",
        "non_success_eval_selections",
    }
    assert summary["num_selector_rows"] == 132
    assert summary["num_matrices"] == 20
    assert summary["num_model_requests"] == 20
    assert summary["num_tensor_requests"] == 20
    assert summary["num_global_candidates"] == 9
    assert summary["matrix_feature_dim"] == 21
    assert summary["candidate_feature_dim"] == 16
    assert summary["validation_error_count"] == 0

    assert {row["row_kind"] for row in export["rows"]} == {
        "input_tensor",
        "input_request_index",
        "input_model_contract",
        "offline_targets",
        "quality_gate",
        "runtime_guard",
    }
    assert export["schema"]["runtime_integration"]["selector_before_training"] == (
        "artifact_backed"
    )
    assert export["job_spec"]["runtime_selector_changed"] is False
    assert "do_not_train_inside_this_entrypoint_artifact" in export["job_spec"]["non_goals"]
    assert (
        export["quality_contract"]["current_gate_status"]["challenger_runtime_eligible"]
        is False
    )
