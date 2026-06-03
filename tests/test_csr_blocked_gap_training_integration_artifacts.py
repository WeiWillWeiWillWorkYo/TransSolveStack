import json
from pathlib import Path

from transsolvestack.profiling.artifacts import read_jsonl
from transsolvestack.profiling.provenance import read_manifest, verify_manifest_hashes


REQUIRED_GAPS = {
    "general_bicgstab_ilu0",
    "general_bicgstab_row_column_equilibration",
    "symmetric_chebyshev_jacobi",
    "symmetric_pcg_symmetric_equilibration",
}


def test_csr_blocked_gap_training_integration_artifacts_are_valid():
    root = Path("runs/phase1_csr_blocked_gap_training_integration")
    selector_rows = read_jsonl(root / "combined_csr_selector_rows.jsonl")
    requests = read_jsonl(root / "csr_blocked_gap_training_model_requests.jsonl")
    targets = read_jsonl(root / "csr_blocked_gap_training_model_targets.jsonl")
    request_index = read_jsonl(root / "csr_blocked_gap_training_request_index.jsonl")
    sources = read_jsonl(root / "csr_blocked_gap_training_sources.jsonl")
    positive_membership = read_jsonl(
        root / "csr_blocked_gap_training_positive_membership.jsonl"
    )
    arrays = json.loads(
        (root / "csr_blocked_gap_training_tensors.json").read_text(
            encoding="utf-8"
        )
    )
    tensor_summary = json.loads(
        (root / "csr_blocked_gap_training_tensor_summary.json").read_text(
            encoding="utf-8"
        )
    )
    summary = json.loads(
        (root / "csr_blocked_gap_training_integration_summary.json").read_text(
            encoding="utf-8"
        )
    )
    schema = json.loads(
        (root / "csr_blocked_gap_training_integration_schema.json").read_text(
            encoding="utf-8"
        )
    )
    manifest = read_manifest(root / "artifact_manifest.json")

    assert manifest.artifact_kind == "csr_blocked_gap_training_integration"
    assert verify_manifest_hashes(manifest) == ()
    assert summary["status"] == "passed"
    assert summary["schema_version"] == "phase1_csr_blocked_gap_training_integration_v1"
    assert schema["schema_version"] == summary["schema_version"]
    assert schema["runtime_boundary"]["runtime_selector_changed"] is False
    assert schema["runtime_boundary"]["generic_queue_merge"] is False
    assert summary["runtime_selector_changed"] is False
    assert summary["generic_queue_merge"] is False
    assert summary["executes_gpu"] is False
    assert summary["imports_matrices"] is False
    assert summary["training_only_integration"] is True

    assert len(sources) == 2
    assert summary["base_selector_rows"] == 372
    assert summary["positive_selector_rows"] == 8
    assert summary["integrated_selector_rows"] == len(selector_rows) == 380
    assert summary["base_matrix_contexts"] == 100
    assert summary["positive_matrix_contexts"] == 5
    assert summary["integrated_model_requests"] == len(requests) == 105
    assert summary["integrated_tensor_requests"] == len(request_index) == 105
    assert len(targets) == len(requests)
    assert summary["model_request_delta"] == 5
    assert summary["duplicate_key_count"] == 0
    assert summary["validation_error_count"] == 0

    assert summary["base_success_rows"] == 71
    assert summary["positive_success_rows"] == 8
    assert summary["integrated_success_rows"] == 79
    assert summary["base_oracle_rows"] == 32
    assert summary["positive_oracle_rows"] == 5
    assert summary["integrated_oracle_rows"] == 37
    assert summary["positive_target_status_counts"] == {"success": 8}
    assert set(summary["positive_gap_ids"]) == REQUIRED_GAPS
    assert summary["positive_gap_row_counts"] == {
        "general_bicgstab_ilu0": 2,
        "general_bicgstab_row_column_equilibration": 2,
        "symmetric_chebyshev_jacobi": 2,
        "symmetric_pcg_symmetric_equilibration": 2,
    }
    assert summary["positive_source_queue_merge_ready"] is False
    assert summary["positive_source_runtime_selector_changed"] is False

    assert summary["integrated_global_candidates"] == 12
    assert set(summary["new_global_candidate_ids"]) == {
        "taichi_csr_bicgstab_ilu0_float64",
        "taichi_csr_bicgstab_row_column_equilibration_float64",
        "taichi_csr_pcg_symmetric_equilibration_float64",
    }
    assert summary["existing_global_candidate_ids"] == [
        "taichi_csr_chebyshev_jacobi_float64"
    ]

    positive_rows = [
        row
        for row in selector_rows
        if row["context_id"] == "phase1_csr_blocked_gap_positive_search"
    ]
    assert len(positive_rows) == len(positive_membership) == 8
    assert {row["target_status"] for row in positive_rows} == {"success"}
    assert sum(1 for row in positive_rows if row["label_is_oracle"]) == 5
    assert {row["candidate_id"] for row in positive_rows} == set(
        summary["positive_candidate_ids"]
    )
    assert {row["target_status"] for row in positive_membership} == {"success"}

    assert tensor_summary["status"] == "passed"
    assert tensor_summary["runtime_selector_changed"] is False
    assert tensor_summary["num_requests"] == summary["integrated_tensor_requests"]
    assert tensor_summary["num_global_candidates"] == summary[
        "integrated_global_candidates"
    ]
    assert tensor_summary["num_active_candidate_slots"] == summary[
        "integrated_active_candidate_slots"
    ]
    assert tensor_summary["target_status_counts"]["success"] == summary[
        "integrated_success_rows"
    ]
    assert arrays["storage_format"] == "json_numeric_arrays_v1"
    assert len(arrays["request_ids"]) == len(request_index)
    assert len(arrays["global_candidate_ids"]) == 12
    assert set(summary["new_global_candidate_ids"]) <= set(
        arrays["global_candidate_ids"]
    )
    assert manifest.metadata["integrated_selector_rows"] == summary[
        "integrated_selector_rows"
    ]
    assert manifest.metadata["positive_success_rows"] == 8
    assert manifest.metadata["runtime_selector_changed"] is False
    assert manifest.metadata["generic_queue_merge"] is False
