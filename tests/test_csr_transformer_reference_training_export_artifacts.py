import json
from pathlib import Path

from transsolvestack.profiling.artifacts import read_jsonl
from transsolvestack.profiling.provenance import read_manifest, verify_manifest_hashes


def test_csr_transformer_reference_training_export_artifacts_are_valid():
    root = Path("runs/phase1_csr_transformer_reference_training_export")
    rows = read_jsonl(root / "csr_transformer_reference_training_export_rows.jsonl")
    summary = json.loads(
        (root / "csr_transformer_reference_training_export_summary.json").read_text(
            encoding="utf-8"
        )
    )
    schema = json.loads(
        (root / "csr_transformer_reference_training_export_schema.json").read_text(
            encoding="utf-8"
        )
    )
    checkpoint = json.loads(
        (root / "reference_csr_external_ranker_checkpoint.json").read_text(
            encoding="utf-8"
        )
    )
    adapter_summary = json.loads(
        (root / "reference_csr_external_adapter_summary.json").read_text(
            encoding="utf-8"
        )
    )
    report = (root / "csr_transformer_reference_training_export_report.md").read_text(
        encoding="utf-8"
    )
    manifest = read_manifest(root / "artifact_manifest.json")

    assert manifest.artifact_kind == "csr_transformer_reference_training_export"
    assert verify_manifest_hashes(manifest) == ()
    assert summary["status"] == "passed"
    assert (
        summary["schema_version"]
        == "phase1_csr_transformer_reference_training_export_v1"
    )
    assert schema["schema_version"] == summary["schema_version"]
    assert summary["reference_training_export_ready"] is True
    assert summary["model_training_executed"] is True
    assert summary["model_trained"] is True
    assert summary["external_checkpoint_ready"] is True
    assert summary["adapter_ready"] is True
    assert summary["adapter_roundtrip_exact"] is True
    assert summary["intake_input_ready"] is True
    assert summary["default_runtime_mode"] == "shadow"
    assert summary["quality_gate_required"] is True
    assert summary["runtime_selector_changed"] is False
    assert summary["num_predictions"] == 20
    assert summary["validation_error_count"] == 0
    assert {row["row_kind"] for row in rows} == {
        "training_result",
        "external_checkpoint",
        "adapter_roundtrip",
        "intake_boundary",
        "runtime_boundary",
        "export_decision",
    }
    assert checkpoint["schema_version"] == "phase1_csr_external_ranker_checkpoint_v1"
    assert checkpoint["runtime_contract"]["guard_required"] is True
    assert checkpoint["runtime_contract"]["runtime_selector_changed"] is False
    assert adapter_summary["adapter_ready"] is True
    assert "Intake Input" in report
    assert manifest.metadata["reference_training_export_ready"] is True
    assert manifest.metadata["intake_input_ready"] is True
    assert manifest.metadata["runtime_selector_changed"] is False
