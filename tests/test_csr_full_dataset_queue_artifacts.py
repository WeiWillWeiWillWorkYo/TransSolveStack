import json
from pathlib import Path

from transsolvestack.profiling.artifacts import read_jsonl
from transsolvestack.profiling.provenance import read_manifest, verify_manifest_hashes


def test_csr_full_dataset_queue_artifacts_are_valid():
    root = Path("runs/phase1_csr_full_dataset_queue")
    matrices = read_jsonl(root / "csr_full_dataset_queue_matrices.jsonl")
    jobs = read_jsonl(root / "csr_full_dataset_queue_jobs.jsonl")
    batches = read_jsonl(root / "csr_full_dataset_queue_batches.jsonl")
    state = json.loads(
        (root / "csr_full_dataset_queue_state.json").read_text(encoding="utf-8")
    )
    summary = json.loads(
        (root / "csr_full_dataset_queue_summary.json").read_text(encoding="utf-8")
    )
    schema = json.loads(
        (root / "csr_full_dataset_queue_schema.json").read_text(encoding="utf-8")
    )
    report = (root / "csr_full_dataset_queue_report.md").read_text(encoding="utf-8")
    manifest = read_manifest(root / "artifact_manifest.json")

    assert manifest.artifact_kind == "csr_full_dataset_queue"
    assert verify_manifest_hashes(manifest) == ()
    assert summary["status"] == "passed"
    assert summary["schema_version"] == "phase1_csr_full_dataset_queue_v1"
    assert schema["schema_version"] == summary["schema_version"]
    assert summary["runtime_selector_changed"] is False
    assert summary["executes_gpu"] is False
    assert summary["imports_matrices"] is False
    assert summary["resumable"] is True
    assert summary["external_drive_required"] is True
    assert summary["index_matrices"] == 2904
    assert summary["index_present_archives"] == 2904
    assert summary["queued_matrices"] > 0
    assert summary["queued_jobs"] == len(jobs)
    assert summary["queued_batches"] == len(batches)
    assert len([row for row in matrices if row["queue_status"] == "queued"]) == summary[
        "queued_matrices"
    ]
    assert sum(row["job_count"] for row in batches) == summary["queued_jobs"]
    assert sum(row["matrix_count"] for row in batches) == summary["queued_matrices"]
    assert state["resume_policy"]["first_pending_batch_id"] == summary[
        "first_pending_batch_id"
    ]
    assert schema["execution_boundary"]["requires_explicit_gpu_benchmark_start"] is True
    assert all(row["requires_import"] is True for row in jobs)
    assert all(row["requires_cpu_screen"] is True for row in jobs)
    assert "Execution Boundary" in report
    assert manifest.metadata["queued_jobs"] == summary["queued_jobs"]
    assert manifest.metadata["runtime_selector_changed"] is False
