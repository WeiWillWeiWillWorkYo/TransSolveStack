import json
from pathlib import Path

from transsolvestack.profiling.artifacts import read_jsonl
from transsolvestack.profiling.provenance import read_manifest, verify_manifest_hashes


def test_csr_gmres_restart_coverage_artifacts_are_valid():
    root = Path("runs/phase1_csr_gmres_restart_coverage")
    matrix_queue = read_jsonl(root / "csr_gmres_restart_matrix_queue.jsonl")
    candidate_queue = read_jsonl(root / "csr_gmres_restart_candidate_queue.jsonl")
    results = read_jsonl(root / "csr_micro_campaign_results.jsonl")
    selector_rows = read_jsonl(root / "csr_micro_selector_rows.jsonl")
    summary = json.loads(
        (root / "csr_gmres_restart_coverage_summary.json").read_text(
            encoding="utf-8"
        )
    )
    schema = json.loads(
        (root / "csr_gmres_restart_coverage_schema.json").read_text(
            encoding="utf-8"
        )
    )
    manifest = read_manifest(root / "artifact_manifest.json")

    assert manifest.artifact_kind == "csr_gmres_restart_coverage"
    assert verify_manifest_hashes(manifest) == ()
    assert summary["status"] == "passed"
    assert summary["schema_version"] == "phase1_csr_gmres_restart_coverage_v1"
    assert schema["schema_version"] == summary["schema_version"]
    assert schema["coverage_boundary"]["does_not_promote_runtime_selector"] is True
    assert schema["coverage_boundary"]["requires_cpu_screen_before_gpu_solve"] is True
    assert summary["runtime_selector_changed"] is False
    assert summary["executes_gpu"] is True
    assert summary["imports_matrices"] is True
    assert summary["cpu_screen_required"] is True

    assert summary["coverage_source_latest_batch_id"] == "batch_00010"
    assert summary["coverage_trigger_candidate_coverage"] is True
    assert summary["selected_matrices"] == len(matrix_queue)
    assert summary["selected_matrices"] == 7
    assert summary["candidate_jobs"] == len(candidate_queue) == len(results) == 14
    assert summary["selector_rows"] == len(selector_rows) == summary["candidate_jobs"]
    assert summary["restarts"] == [32, 64]
    assert {row["solver"] for row in candidate_queue} == {"gmres"}
    assert {row["preconditioner"] for row in candidate_queue} == {"jacobi"}
    assert {int(row["solver_parameters"]["restart"]) for row in candidate_queue} == {
        32,
        64,
    }

    accounted = (
        summary["gpu_success_rows"]
        + summary["cpu_screened_out_rows"]
        + summary["gpu_failed_rows"]
    )
    assert accounted == summary["candidate_jobs"]
    assert summary["gpu_failed_rows"] == 0
    assert summary["queue_merge_ready"] == (
        summary["gpu_success_rows"] > 0 and summary["selector_oracle_rows"] > 0
    )
    assert set(summary["by_restart"]) == {"32", "64"}
    for restart in ("32", "64"):
        assert sum(summary["by_restart"][restart].values()) == summary["selected_matrices"]

    if summary["gpu_success_rows"] > 0:
        assert summary["coverage_outcome"] == "profiled_with_gpu_success"
        assert summary["max_final_relative_residual"] <= 1.0e-5
        assert summary["max_cpu_recomputed_relative_residual"] <= 1.0e-4
        assert summary["max_solution_relative_error"] <= 5.0e-3
    else:
        assert summary["coverage_outcome"] == "screen_only_no_oracle"
