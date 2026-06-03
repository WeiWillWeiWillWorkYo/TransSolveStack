import transsolvestack as tss


def test_csr_external_model_intake_runs_full_shadow_pipeline(tmp_path):
    export = tss.run_csr_external_model_intake(output_dir=tmp_path)
    summary = export["summary"]

    assert summary["status"] == "passed"
    assert summary["schema_version"] == "phase1_csr_external_model_intake_v1"
    assert summary["intake_ready"] is True
    assert summary["shadow_submission_ready"] is True
    assert summary["runtime_promotion_ready"] is False
    assert summary["default_runtime_mode"] == "shadow"
    assert summary["adapter_ready"] is True
    assert summary["replay_exact"] is True
    assert summary["quality_gate_runtime_eligible"] is False
    assert summary["policy_model_artifact_ready"] is True
    assert summary["accepted_for_shadow"] is True
    assert summary["accepted_for_runtime_promotion"] is False
    assert summary["guarded_gpu_shadow_smoke_checked"] is True
    assert summary["guarded_gpu_smoke_successes"] == 2
    assert summary["num_predictions"] == 20
    assert summary["runtime_selector_changed"] is False
    assert set(summary["stage_statuses"]) == {
        "adapter",
        "replay",
        "quality_gate",
        "policy_model_artifact",
        "learned_guard",
        "acceptance",
        "submission",
    }
    assert all(status == "passed" for status in summary["stage_statuses"].values())
    assert "Promotion Blockers" in export["report"]
