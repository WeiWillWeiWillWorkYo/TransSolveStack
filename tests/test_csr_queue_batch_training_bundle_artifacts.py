import json
from pathlib import Path

from transsolvestack.profiling.artifacts import read_jsonl
from transsolvestack.profiling.provenance import read_manifest, verify_manifest_hashes


def test_csr_queue_batch_training_bundle_artifacts_are_valid():
    root = Path("runs/phase1_csr_queue_batch_training_bundle")
    selector_rows = read_jsonl(root / "combined_csr_selector_rows.jsonl")
    requests = read_jsonl(root / "csr_transformer_model_requests.jsonl")
    targets = read_jsonl(root / "csr_transformer_model_targets.jsonl")
    request_index = read_jsonl(root / "csr_transformer_request_index.jsonl")
    baseline_predictions = read_jsonl(root / "csr_transformer_baseline_predictions.jsonl")
    summary = json.loads(
        (root / "csr_transformer_ready_summary.json").read_text(encoding="utf-8")
    )
    tensor_summary = json.loads(
        (root / "csr_transformer_tensor_summary.json").read_text(encoding="utf-8")
    )
    queue_pool_summary = json.loads(
        Path("runs/phase1_csr_queue_training_pool/csr_queue_training_pool_summary.json")
        .read_text(encoding="utf-8")
    )
    manifest = read_manifest(root / "artifact_manifest.json")

    assert manifest.artifact_kind == "csr_queue_batch_training_bundle"
    assert verify_manifest_hashes(manifest) == ()
    assert summary["status"] == "passed"
    assert summary["transformer_connectable"] is True
    assert summary["runtime_selector_changed"] is False
    assert summary["model_required"] is False
    assert len(summary["source_selector_paths"]) == 1
    assert (
        "runs/phase1_csr_queue_training_pool/csr_queue_training_pool_selector_rows.jsonl"
        in summary["source_selector_paths"]
    )
    assert summary["num_selector_rows"] == len(selector_rows)
    assert summary["num_model_requests"] == len(requests)
    assert summary["num_model_targets"] == len(targets)
    assert summary["num_tensor_requests"] == len(request_index)
    assert summary["num_selector_rows"] == queue_pool_summary["selector_rows"]
    assert summary["num_matrices"] == queue_pool_summary["matrices"]
    assert summary["num_success_rows"] == queue_pool_summary["success_rows"]
    assert summary["num_screened_out_rows"] == queue_pool_summary["screened_out_rows"]
    assert summary["num_oracle_rows"] == queue_pool_summary["oracle_rows"]
    assert summary["num_learning_rows"] == len(selector_rows)
    assert summary["num_learning_train_rows"] + summary["num_learning_eval_rows"] == len(
        selector_rows
    )
    assert summary["num_model_predictions"] == len(baseline_predictions)
    assert summary["num_global_candidates"] == 9
    assert summary["num_active_candidate_slots"] == len(selector_rows)
    assert summary["matrix_feature_dim"] == 21
    assert summary["candidate_feature_dim"] == 16
    assert summary["target_status_counts"] == {
        status: sum(1 for row in selector_rows if row["target_status"] == status)
        for status in ("not_applicable", "not_profiled", "screened_out", "success")
    }
    assert summary["label_class_counts"]["success_oracle"] == queue_pool_summary[
        "oracle_rows"
    ]
    assert summary["label_class_counts"]["screened_out"] == queue_pool_summary[
        "screened_out_rows"
    ]
    assert 0.0 <= summary["eval_oracle_top1_accuracy"] <= 1.0
    assert 0.0 <= summary["eval_profiled_selection_rate"] <= 1.0
    assert summary["validation_error_count"] == 0
    assert tensor_summary["status"] == "passed"
    assert tensor_summary["runtime_selector_changed"] is False
    assert tensor_summary["num_requests"] == summary["num_tensor_requests"]
    assert manifest.metadata["queue_batch_selector_rows"] == queue_pool_summary[
        "queue_batch_selector_rows"
    ]
    assert manifest.metadata["queue_pool_ready"] is True
    assert (
        manifest.metadata["queue_pool_summary"]
        == "runs/phase1_csr_queue_training_pool/csr_queue_training_pool_summary.json"
    )
    assert manifest.metadata["num_selector_rows"] == summary["num_selector_rows"]
    assert manifest.metadata["transformer_connectable"] is True
    assert manifest.metadata["runtime_selector_changed"] is False
