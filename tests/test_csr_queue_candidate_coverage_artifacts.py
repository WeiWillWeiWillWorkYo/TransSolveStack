import json
from pathlib import Path

from transsolvestack.profiling.artifacts import read_jsonl
from transsolvestack.profiling.provenance import read_manifest, verify_manifest_hashes


def test_csr_queue_candidate_coverage_artifacts_are_valid():
    root = Path("runs/phase1_csr_queue_candidate_coverage")
    batch_rows = read_jsonl(root / "csr_queue_candidate_coverage_batch_rows.jsonl")
    candidate_rows = read_jsonl(
        root / "csr_queue_candidate_coverage_candidate_rows.jsonl"
    )
    gap_rows = read_jsonl(root / "csr_queue_candidate_coverage_gap_rows.jsonl")
    summary = json.loads(
        (root / "csr_queue_candidate_coverage_summary.json").read_text(
            encoding="utf-8"
        )
    )
    schema = json.loads(
        (root / "csr_queue_candidate_coverage_schema.json").read_text(
            encoding="utf-8"
        )
    )
    pool_summary = json.loads(
        Path(
            "runs/phase1_csr_queue_training_pool/csr_queue_training_pool_summary.json"
        ).read_text(encoding="utf-8")
    )
    manifest = read_manifest(root / "artifact_manifest.json")

    assert manifest.artifact_kind == "csr_queue_candidate_coverage"
    assert verify_manifest_hashes(manifest) == ()
    assert summary["status"] == "passed"
    assert summary["schema_version"] == "phase1_csr_queue_candidate_coverage_v1"
    assert schema["schema_version"] == summary["schema_version"]
    assert schema["integration_boundary"]["does_not_modify_full_dataset_queue"] is True
    assert summary["runtime_selector_changed"] is False
    assert summary["executes_gpu"] is False
    assert summary["imports_matrices"] is False

    assert summary["completed_queue_batches"] == pool_summary["completed_queue_batches"]
    assert summary["next_pending_batch_id"] == pool_summary["next_pending_batch_id"]
    assert len(batch_rows) == summary["completed_queue_batches"]
    assert len(candidate_rows) == summary["candidate_coverage_rows"]
    assert len(gap_rows) == summary["planned_gap_count"]
    assert batch_rows[-1]["batch_id"] == summary["latest_batch_id"]

    assert "chebyshev" in set(summary["queue_missing_supported_solvers"])
    missing_preconditioners = set(summary["queue_missing_supported_preconditioners"])
    assert {"ilu0", "row_column_equilibration", "symmetric_equilibration"} <= (
        missing_preconditioners
    )

    gap_ids = {row["gap_id"] for row in gap_rows}
    assert {
        "general_gmres_jacobi_restart32",
        "general_gmres_jacobi_restart64",
        "general_bicgstab_ilu0",
        "symmetric_chebyshev_jacobi",
        "symmetric_pcg_symmetric_equilibration",
    } <= gap_ids

    queue_ready = {row["gap_id"] for row in gap_rows if row["queue_ready"]}
    assert {
        "general_gmres_jacobi_restart32",
        "general_gmres_jacobi_restart64",
    } <= queue_ready
    assert all(row["blocker"] is None for row in gap_rows if row["queue_ready"])
    assert all(row["blocker"] for row in gap_rows if not row["queue_ready"])
    assert manifest.metadata["planned_gap_count"] == summary["planned_gap_count"]
    assert manifest.metadata["runtime_selector_changed"] is False
