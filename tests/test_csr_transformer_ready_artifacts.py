import json
from pathlib import Path

from transsolvestack.profiling.artifacts import read_jsonl
from transsolvestack.profiling.provenance import read_manifest, verify_manifest_hashes


def test_csr_transformer_ready_artifacts_are_valid():
    root = Path("runs/phase1_csr_transformer_ready")
    selector_rows = read_jsonl(root / "combined_csr_selector_rows.jsonl")
    requests = read_jsonl(root / "csr_transformer_model_requests.jsonl")
    request_index = read_jsonl(root / "csr_transformer_request_index.jsonl")
    arrays = json.loads((root / "csr_transformer_training_tensors.json").read_text(encoding="utf-8"))
    summary = json.loads((root / "csr_transformer_ready_summary.json").read_text(encoding="utf-8"))
    manifest = read_manifest(root / "artifact_manifest.json")

    assert manifest.artifact_kind == "csr_transformer_ready_bundle"
    assert verify_manifest_hashes(manifest) == ()
    assert summary["status"] == "passed"
    assert summary["transformer_connectable"] is True
    assert summary["runtime_selector_changed"] is False
    assert summary["schema_version"] == "phase1_csr_transformer_ready_v1"
    assert summary["num_selector_rows"] == 132
    assert len(selector_rows) == 132
    assert summary["num_matrices"] == 20
    assert summary["num_success_rows"] == 23
    assert summary["num_screened_out_rows"] == 31
    assert summary["num_oracle_rows"] == 8
    assert summary["num_model_requests"] == 20
    assert len(requests) == 20
    assert summary["num_tensor_requests"] == 20
    assert len(request_index) == 20
    assert summary["num_global_candidates"] == 9
    assert summary["num_active_candidate_slots"] == 132
    assert summary["matrix_feature_dim"] == 21
    assert summary["candidate_feature_dim"] == 16
    assert summary["validation_error_count"] == 0
    assert summary["label_class_counts"] == {
        "not_applicable": 34,
        "not_profiled": 44,
        "screened_out": 31,
        "success_non_oracle": 15,
        "success_oracle": 8,
    }
    assert summary["target_status_counts"] == {
        "not_applicable": 34,
        "not_profiled": 44,
        "screened_out": 31,
        "success": 23,
    }
    assert arrays["storage_format"] == "json_numeric_arrays_v1"
    assert len(arrays["request_ids"]) == 20
    assert len(arrays["global_candidate_ids"]) == 9
    assert len(arrays["matrix_features"]) == 20
    assert all(len(row) == 21 for row in arrays["matrix_features"])
    assert sum(sum(row) for row in arrays["candidate_mask"]) == 132

