import transsolvestack as tss


def test_csr_transformer_reference_training_export_builds_intake_checkpoint(tmp_path):
    export = tss.build_csr_transformer_reference_training_export(output_dir=tmp_path)
    summary = export["summary"]

    assert summary["status"] == "passed"
    assert (
        summary["schema_version"]
        == "phase1_csr_transformer_reference_training_export_v1"
    )
    assert summary["reference_training_export_ready"] is True
    assert summary["model_training_executed"] is True
    assert summary["model_trained"] is True
    assert summary["training_entrypoint_ready"] is True
    assert summary["external_checkpoint_ready"] is True
    assert summary["adapter_ready"] is True
    assert summary["adapter_roundtrip_exact"] is True
    assert summary["intake_input_ready"] is True
    assert summary["quality_gate_required"] is True
    assert summary["current_quality_gate_runtime_eligible"] is False
    assert summary["default_runtime_mode"] == "shadow"
    assert summary["runtime_selector_changed"] is False
    assert summary["num_predictions"] == 20
    assert summary["num_eval_predictions"] == 5
    assert summary["num_eval_oracle_requests"] == 4
    assert summary["eval_oracle_top1_accuracy"] == 0.5
    assert summary["eval_profiled_success_selection_rate"] == 0.6
    assert summary["eval_non_success_selection_count"] == 2
    assert summary["validation_error_count"] == 0
    assert {row["row_kind"] for row in export["rows"]} == {
        "training_result",
        "external_checkpoint",
        "adapter_roundtrip",
        "intake_boundary",
        "runtime_boundary",
        "export_decision",
    }
    assert export["checkpoint"]["runtime_contract"]["default_mode"] == "shadow"
    assert export["adapter"]["summary"]["adapter_ready"] is True
    assert "Intake Input" in export["report"]
