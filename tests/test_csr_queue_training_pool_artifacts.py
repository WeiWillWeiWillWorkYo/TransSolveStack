import json
from pathlib import Path

from transsolvestack.profiling.artifacts import read_jsonl
from transsolvestack.profiling.provenance import read_manifest, verify_manifest_hashes


def _count_status(rows, status):
    return sum(1 for row in rows if row["target_status"] == status)


def test_csr_queue_training_pool_artifacts_are_valid():
    root = Path("runs/phase1_csr_queue_training_pool")
    selector_rows = read_jsonl(root / "csr_queue_training_pool_selector_rows.jsonl")
    sources = read_jsonl(root / "csr_queue_training_pool_sources.jsonl")
    membership = read_jsonl(root / "csr_queue_training_pool_membership.jsonl")
    full_queue_batches = read_jsonl(
        "runs/phase1_csr_full_dataset_queue/csr_full_dataset_queue_batches.jsonl"
    )
    summary = json.loads(
        (root / "csr_queue_training_pool_summary.json").read_text(encoding="utf-8")
    )
    state = json.loads(
        (root / "csr_queue_training_pool_state.json").read_text(encoding="utf-8")
    )
    manifest = read_manifest(root / "artifact_manifest.json")

    assert manifest.artifact_kind == "csr_queue_training_pool"
    assert verify_manifest_hashes(manifest) == ()
    assert summary["status"] == "passed"
    assert summary["schema_version"] == "phase1_csr_queue_training_pool_v1"
    assert summary["runtime_selector_changed"] is False
    assert summary["executes_gpu"] is False
    assert summary["imports_matrices"] is False
    assert summary["append_only"] is True
    assert summary["training_pool_ready"] is True
    queue_sources = [
        row for row in sources if row["source_kind"] == "queue_batch_execution"
    ]
    base_sources = [row for row in sources if row["source_kind"] == "base_selector"]
    completed_batch_ids = [row["batch_id"] for row in queue_sources]
    pending_batch_ids = [
        row["batch_id"]
        for row in full_queue_batches
        if row["batch_id"] not in set(completed_batch_ids)
    ]
    assert summary["source_count"] == len(sources)
    assert summary["base_source_count"] == len(base_sources) == 2
    assert summary["queue_batch_source_count"] == len(queue_sources)
    assert summary["completed_queue_batches"] == len(queue_sources)
    assert summary["completed_batch_ids"] == completed_batch_ids
    assert summary["queue_batch_oracle_batches"] == sum(
        1 for row in queue_sources if row["batch_outcome"] == "profiled_with_gpu_success"
    )
    assert summary["queue_batch_screen_only_batches"] == sum(
        1 for row in queue_sources if row["batch_outcome"] == "screen_only_no_oracle"
    )
    assert summary["full_queue_batches"] == len(full_queue_batches) == 194
    assert summary["remaining_queue_batches"] == len(pending_batch_ids)
    assert summary["next_pending_batch_id"] == pending_batch_ids[0]
    assert summary["selector_rows"] == len(selector_rows)
    assert summary["membership_rows"] == len(membership) == len(selector_rows)
    assert summary["matrices"] == len({row["matrix_id"] for row in selector_rows})
    assert summary["success_rows"] == _count_status(selector_rows, "success")
    assert summary["screened_out_rows"] == _count_status(selector_rows, "screened_out")
    assert summary["not_profiled_rows"] == _count_status(selector_rows, "not_profiled")
    assert summary["not_applicable_rows"] == _count_status(
        selector_rows, "not_applicable"
    )
    assert summary["oracle_rows"] == sum(
        1 for row in selector_rows if row["label_is_oracle"] is True
    )
    assert summary["queue_batch_selector_rows"] == sum(
        row["selector_rows"] for row in queue_sources
    )
    assert summary["non_queue_selector_rows"] == sum(
        row["selector_rows"] for row in base_sources
    )
    assert state["training_pool_ready"] is True
    assert state["queue_batch_screen_only_batches"] == summary[
        "queue_batch_screen_only_batches"
    ]
    assert state["next_pending_batch_id"] == summary["next_pending_batch_id"]
    sources_by_id = {row["source_id"]: row for row in sources}
    assert {"base_01", "base_02", "queue_batch_00001", "queue_batch_00002", "queue_batch_00003"} <= set(sources_by_id)
    assert sources_by_id["base_01"]["selector_rows"] == 108
    assert sources_by_id["base_02"]["selector_rows"] == 24
    assert sources_by_id["queue_batch_00001"]["selector_rows"] == 24
    assert sources_by_id["queue_batch_00001"]["executes_gpu"] is True
    assert sources_by_id["queue_batch_00001"]["batch_outcome"] == (
        "profiled_with_gpu_success"
    )
    assert sources_by_id["queue_batch_00001"]["runtime_selector_changed"] is False
    assert sources_by_id["queue_batch_00002"]["selector_rows"] == 24
    assert sources_by_id["queue_batch_00002"]["executes_gpu"] is True
    assert sources_by_id["queue_batch_00002"]["batch_outcome"] == (
        "screen_only_no_oracle"
    )
    assert sources_by_id["queue_batch_00002"]["completed_without_oracle"] is True
    assert sources_by_id["queue_batch_00002"]["oracle_rows"] == 0
    assert sources_by_id["queue_batch_00003"]["selector_rows"] == 24
    assert sources_by_id["queue_batch_00003"]["executes_gpu"] is True
    assert sources_by_id["queue_batch_00003"]["batch_outcome"] == (
        "profiled_with_gpu_success"
    )
    assert sources_by_id["queue_batch_00003"]["oracle_rows"] == 3
    assert all(row["selector_rows"] > 0 for row in queue_sources)
    assert all(row["executes_gpu"] is True for row in queue_sources)
    assert all(row["runtime_selector_changed"] is False for row in queue_sources)
    assert manifest.metadata["completed_batch_ids"] == summary["completed_batch_ids"]
    assert manifest.metadata["queue_batch_screen_only_batches"] == summary[
        "queue_batch_screen_only_batches"
    ]
    assert manifest.metadata["next_pending_batch_id"] == summary["next_pending_batch_id"]
    assert manifest.metadata["selector_rows"] == summary["selector_rows"]
