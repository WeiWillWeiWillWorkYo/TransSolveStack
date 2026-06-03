import json
from pathlib import Path

from transsolvestack.profiling.artifacts import read_jsonl
from transsolvestack.profiling.provenance import read_manifest, verify_manifest_hashes


def test_csr_blocked_gap_probe_artifacts_are_valid():
    root = Path("runs/phase1_csr_blocked_gap_probe")
    matrix_queue = read_jsonl(root / "csr_blocked_gap_matrix_queue.jsonl")
    candidate_queue = read_jsonl(root / "csr_blocked_gap_candidate_queue.jsonl")
    results = read_jsonl(root / "csr_micro_campaign_results.jsonl")
    selector_rows = read_jsonl(root / "csr_micro_selector_rows.jsonl")
    summary = json.loads(
        (root / "csr_blocked_gap_probe_summary.json").read_text(encoding="utf-8")
    )
    schema = json.loads(
        (root / "csr_blocked_gap_probe_schema.json").read_text(encoding="utf-8")
    )
    manifest = read_manifest(root / "artifact_manifest.json")

    assert manifest.artifact_kind == "csr_blocked_gap_probe"
    assert verify_manifest_hashes(manifest) == ()
    assert summary["status"] == "passed"
    assert summary["schema_version"] == "phase1_csr_blocked_gap_probe_v1"
    assert schema["schema_version"] == summary["schema_version"]
    assert schema["coverage_boundary"]["does_not_promote_runtime_selector"] is True
    assert schema["coverage_boundary"]["requires_cpu_screen_before_gpu_solve"] is True
    assert schema["coverage_boundary"]["gpu_solve_only_after_screen_success"] is True
    assert summary["runtime_selector_changed"] is False
    assert summary["executes_gpu"] is True
    assert summary["imports_matrices"] is True
    assert summary["cpu_screen_required"] is True

    assert summary["coverage_source_latest_batch_id"] == "batch_00010"
    assert summary["coverage_trigger_candidate_coverage"] is True
    assert summary["selected_general_matrices"] == 7
    assert summary["selected_symmetric_matrices"] == 1
    assert summary["selected_matrices"] == len(matrix_queue) == 8
    assert summary["candidate_jobs"] == len(candidate_queue) == len(results) == 16
    assert summary["selector_rows"] == len(selector_rows) == summary["candidate_jobs"]
    assert summary["newly_cpu_screen_integrated_gap_count"] == 4
    assert set(summary["blocked_gap_ids"]) == {
        "general_bicgstab_ilu0",
        "general_bicgstab_row_column_equilibration",
        "symmetric_chebyshev_jacobi",
        "symmetric_pcg_symmetric_equilibration",
    }

    accounted = (
        summary["gpu_success_rows"]
        + summary["cpu_screened_out_rows"]
        + summary["gpu_failed_rows"]
    )
    assert accounted == summary["candidate_jobs"]
    assert summary["gpu_failed_rows"] == 0
    assert summary["queue_merge_ready"] == (
        len(summary["queue_merge_ready_gap_ids"]) > 0
    )
    assert summary["coverage_outcome"] == "screen_only_no_oracle"
    assert summary["gpu_success_rows"] == 0
    assert summary["cpu_screened_out_rows"] == 16
    assert summary["queue_merge_ready_gap_ids"] == []

    by_gap = summary["by_gap_status"]
    assert by_gap["general_bicgstab_ilu0"] == {"screened_out": 7}
    assert by_gap["general_bicgstab_row_column_equilibration"] == {
        "screened_out": 7
    }
    assert by_gap["symmetric_chebyshev_jacobi"] == {"screened_out": 1}
    assert by_gap["symmetric_pcg_symmetric_equilibration"] == {"screened_out": 1}
    assert {row["backend"] for row in results} == {"cpu_reference_screen"}
    assert {row["status"] for row in results} == {"screened_out"}
    assert manifest.metadata["candidate_jobs"] == summary["candidate_jobs"]
    assert manifest.metadata["queue_merge_ready"] is False
