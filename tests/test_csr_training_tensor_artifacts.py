import json
from pathlib import Path

from transsolvestack.profiling.artifacts import read_jsonl
from transsolvestack.profiling.provenance import read_manifest, verify_manifest_hashes


def test_csr_training_tensor_artifacts_are_valid():
    root = Path("runs/phase1_csr_training_tensors")
    arrays = json.loads((root / "csr_training_tensors.json").read_text(encoding="utf-8"))
    index_rows = read_jsonl(root / "csr_training_request_index.jsonl")
    summary = json.loads((root / "csr_training_tensor_summary.json").read_text(encoding="utf-8"))
    schema = json.loads((root / "csr_training_tensor_schema.json").read_text(encoding="utf-8"))
    manifest = read_manifest(root / "artifact_manifest.json")

    assert manifest.artifact_kind == "csr_training_tensor_export"
    assert verify_manifest_hashes(manifest) == ()
    assert summary["status"] == "passed"
    assert summary["schema_version"] == "phase1_csr_training_tensors_v1"
    assert schema["schema_version"] == summary["schema_version"]
    assert schema["model_required"] is False
    assert schema["runtime_selector_changed"] is False
    assert summary["model_required"] is False
    assert summary["runtime_selector_changed"] is False
    assert summary["num_requests"] == 12
    assert len(index_rows) == 12
    assert summary["num_global_candidates"] == 9
    assert summary["num_active_candidate_slots"] == 108
    assert summary["num_oracle_targets"] == 4
    assert summary["num_requests_without_oracle"] == 8
    assert summary["matrix_feature_dim"] == 21
    assert summary["candidate_feature_dim"] == 16
    assert summary["has_missing_candidate_slots"] is False
    assert len(arrays["matrix_feature_names"]) == summary["matrix_feature_dim"]
    assert len(arrays["candidate_feature_names"]) == summary["candidate_feature_dim"]
    assert len(arrays["matrix_features"]) == 12
    assert len(arrays["candidate_features"]) == 12
    assert len(arrays["candidate_features"][0]) == 9
    assert sum(sum(row) for row in arrays["candidate_mask"]) == 108
    assert sum(1 for value in arrays["oracle_index"] if value >= 0) == 4
    assert summary["label_class_counts"] == {
        "not_applicable": 34,
        "not_profiled": 44,
        "screened_out": 15,
        "success_non_oracle": 11,
        "success_oracle": 4,
    }
    assert summary["target_status_counts"] == {
        "not_applicable": 34,
        "not_profiled": 44,
        "screened_out": 15,
        "success": 15,
    }
