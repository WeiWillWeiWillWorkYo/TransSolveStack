import json
from pathlib import Path

from transsolvestack.profiling.artifacts import read_jsonl
from transsolvestack.profiling.provenance import read_manifest, verify_manifest_hashes


def test_csr_transformer_training_entrypoint_artifacts_are_valid():
    root = Path("runs/phase1_csr_transformer_training_entrypoint")
    rows = read_jsonl(root / "csr_transformer_training_entrypoint_rows.jsonl")
    summary = json.loads(
        (root / "csr_transformer_training_entrypoint_summary.json").read_text(
            encoding="utf-8"
        )
    )
    schema = json.loads(
        (root / "csr_transformer_training_entrypoint_schema.json").read_text(
            encoding="utf-8"
        )
    )
    job_spec = json.loads(
        (root / "csr_transformer_training_job_spec.json").read_text(encoding="utf-8")
    )
    quality_contract = json.loads(
        (root / "csr_transformer_training_quality_contract.json").read_text(
            encoding="utf-8"
        )
    )
    manifest = read_manifest(root / "artifact_manifest.json")

    assert manifest.artifact_kind == "csr_transformer_training_entrypoint"
    assert verify_manifest_hashes(manifest) == ()
    assert summary["status"] == "passed"
    assert summary["schema_version"] == "phase1_csr_transformer_training_entrypoint_v1"
    assert schema["schema_version"] == summary["schema_version"]
    assert job_spec["schema_version"] == summary["schema_version"]
    assert quality_contract["schema_version"] == summary["schema_version"]
    assert len(rows) == 6
    assert summary["training_entrypoint_ready"] is True
    assert summary["model_training_required"] is True
    assert summary["model_trained"] is False
    assert summary["runtime_selector_changed"] is False
    assert summary["transformer_connectable"] is True
    assert summary["current_quality_gate_runtime_eligible"] is False
    assert summary["num_model_requests"] == 20
    assert summary["num_tensor_requests"] == 20
    assert summary["num_global_candidates"] == 9
    assert summary["validation_error_count"] == 0
    assert schema["runtime_integration"]["selector_before_training"] == "artifact_backed"
    assert job_spec["runtime_selector_changed"] is False
    assert (
        quality_contract["current_gate_status"]["challenger_runtime_eligible"]
        is False
    )
    assert manifest.metadata["training_entrypoint_ready"] is True
    assert manifest.metadata["runtime_selector_changed"] is False
