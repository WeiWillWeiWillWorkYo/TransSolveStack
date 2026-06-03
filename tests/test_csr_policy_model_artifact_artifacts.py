import json
from pathlib import Path

from transsolvestack.profiling.artifacts import read_jsonl
from transsolvestack.profiling.provenance import read_manifest, verify_manifest_hashes


def test_csr_policy_model_artifact_artifacts_are_valid():
    root = Path("runs/phase1_csr_policy_model_artifact")
    artifact = json.loads((root / "csr_policy_model_artifact.json").read_text(encoding="utf-8"))
    rows = read_jsonl(root / "csr_policy_model_artifact_rows.jsonl")
    summary = json.loads(
        (root / "csr_policy_model_artifact_summary.json").read_text(encoding="utf-8")
    )
    schema = json.loads(
        (root / "csr_policy_model_artifact_schema.json").read_text(encoding="utf-8")
    )
    manifest = read_manifest(root / "artifact_manifest.json")

    assert manifest.artifact_kind == "csr_policy_model_artifact"
    assert verify_manifest_hashes(manifest) == ()
    assert summary["status"] == "passed"
    assert summary["schema_version"] == "phase1_csr_policy_model_artifact_v1"
    assert schema["schema_version"] == summary["schema_version"]
    assert artifact["schema_version"] == summary["schema_version"]
    assert summary["artifact_ready"] is True
    assert summary["adapter"] == "csr_transformer_ranker_saved_model_v1"
    assert summary["model_loaded"] is True
    assert summary["prediction_contract_checked"] is True
    assert summary["num_predictions"] == 20
    assert summary["num_request_index_rows"] == 20
    assert summary["replay_exact"] is True
    assert summary["guard_required"] is True
    assert summary["runtime_selector_changed"] is False
    assert artifact["runtime_contract"]["guard_required"] is True
    assert artifact["runtime_contract"]["runtime_selector_changed"] is False
    assert {row["row_kind"] for row in rows} == {
        "model",
        "tensor_contract",
        "request_index",
        "quality_gate",
        "runtime_contract",
    }
    assert manifest.metadata["artifact_ready"] is True
    assert manifest.metadata["model_loaded"] is True
    assert manifest.metadata["replay_exact"] is True
    assert manifest.metadata["guard_required"] is True
    assert manifest.metadata["runtime_selector_changed"] is False
