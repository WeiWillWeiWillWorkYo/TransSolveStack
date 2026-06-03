import transsolvestack as tss


def test_csr_policy_model_acceptance_accepts_shadow_and_blocks_promotion():
    export = tss.accept_csr_policy_model_artifact()
    summary = export["summary"]

    assert summary["status"] == "passed"
    assert summary["schema_version"] == "phase1_csr_policy_model_acceptance_v1"
    assert summary["model_artifact_ready"] is True
    assert summary["model_loaded"] is True
    assert summary["prediction_contract_checked"] is True
    assert summary["num_predictions"] == 20
    assert summary["replay_exact"] is True
    assert summary["quality_gate_checked"] is True
    assert summary["quality_gate_runtime_eligible"] is False
    assert summary["guard_plan_shadow_checked"] is True
    assert summary["guarded_gpu_shadow_smoke_checked"] is True
    assert summary["accepted_for_shadow"] is True
    assert summary["accepted_for_runtime_promotion"] is False
    assert "quality_gate_not_runtime_eligible" in summary["promotion_blockers"]
    assert summary["runtime_selector_changed"] is False
    assert summary["validation_error_count"] == 0

    rows_by_kind = {}
    for row in export["rows"]:
        rows_by_kind.setdefault(row["row_kind"], []).append(row)
    assert len(rows_by_kind["guard_plan_shadow"]) == 2
    assert len(rows_by_kind["guarded_gpu_shadow_result"]) == 2
    assert all(
        row["learned_policy_source"]["source_kind"] == "model_artifact"
        for row in rows_by_kind["guard_plan_shadow"]
    )
    assert all(row["backend"] == "taichi_gpu" for row in rows_by_kind["guarded_gpu_shadow_result"])
