import json
from pathlib import Path

from transsolvestack.profiling.artifacts import read_jsonl
from transsolvestack.profiling.provenance import read_manifest, verify_manifest_hashes


def _assert_queue_batch_execution_artifacts(
    root: Path,
    *,
    expected_batch_id: str,
    expected_outcome: str,
    expected_planned_total_nnz: int,
    expected_planned_max_matrix_nnz: int,
    expected_gpu_success_rows: int,
    expected_cpu_screened_out_rows: int,
    expected_selector_oracle_rows: int,
    expected_resume_status: str,
) -> None:
    matrices = read_jsonl(root / "csr_queue_batch_matrix_queue.jsonl")
    jobs = read_jsonl(root / "csr_queue_batch_candidate_queue.jsonl")
    csr_rows = read_jsonl(root / "csr_matrices.jsonl")
    results = read_jsonl(root / "csr_micro_campaign_results.jsonl")
    selector_rows = read_jsonl(root / "csr_micro_selector_rows.jsonl")
    summary = json.loads(
        (root / "csr_queue_batch_execution_summary.json").read_text(encoding="utf-8")
    )
    schema = json.loads(
        (root / "csr_queue_batch_execution_schema.json").read_text(encoding="utf-8")
    )
    state = json.loads(
        (root / "csr_queue_batch_execution_state.json").read_text(encoding="utf-8")
    )
    manifest = read_manifest(root / "artifact_manifest.json")

    assert manifest.artifact_kind == "csr_queue_batch_execution"
    assert verify_manifest_hashes(manifest) == ()
    assert summary["status"] == "passed"
    assert summary["schema_version"] == "phase1_csr_queue_batch_execution_v1"
    assert schema["schema_version"] == summary["schema_version"]
    assert summary["batch_id"] == expected_batch_id
    assert summary["batch_outcome"] == expected_outcome
    assert summary["runtime_selector_changed"] is False
    assert summary["executes_gpu"] is True
    assert summary["imports_matrices"] is True
    assert summary["cpu_screen_required"] is True
    assert summary["planned_matrices"] == 8
    assert summary["planned_jobs"] == 24
    assert summary["executed_matrices"] == len(matrices) == len(csr_rows) == 8
    assert summary["executed_jobs"] == len(jobs) == len(results) == 24
    assert summary["imported_matrices"] == 8
    assert summary["candidate_jobs"] == 24
    assert summary["gpu_success_rows"] == expected_gpu_success_rows
    assert summary["cpu_screened_out_rows"] == expected_cpu_screened_out_rows
    assert summary["gpu_success_rows"] + summary["cpu_screened_out_rows"] == 24
    assert summary["gpu_failed_rows"] == 0
    assert summary["selector_rows"] == len(selector_rows) == 24
    assert summary["selector_oracle_rows"] == expected_selector_oracle_rows
    assert summary["completed_without_oracle"] is (
        expected_outcome == "screen_only_no_oracle"
    )
    assert summary["has_oracle_rows"] is (expected_selector_oracle_rows > 0)
    assert summary["matrices_with_selector_rows"] == 8
    assert summary["planned_total_nnz"] == expected_planned_total_nnz
    assert summary["planned_max_matrix_nnz"] == expected_planned_max_matrix_nnz
    if expected_gpu_success_rows:
        assert summary["max_final_relative_residual"] <= 1.0e-5
        assert summary["max_cpu_recomputed_relative_residual"] <= 1.0e-4
        assert summary["max_solution_relative_error"] <= 5.0e-3
    else:
        assert summary["max_final_relative_residual"] == 0.0
        assert summary["max_cpu_recomputed_relative_residual"] == 0.0
        assert summary["max_solution_relative_error"] == 0.0
    assert state["resume_status"] == expected_resume_status
    assert state["batch_outcome"] == expected_outcome
    assert state["runtime_selector_changed"] is False
    assert sum(1 for row in results if row["backend"] == "taichi_gpu") == summary[
        "gpu_success_rows"
    ]
    assert sum(1 for row in results if row["backend"] == "cpu_reference_screen") == summary[
        "cpu_screened_out_rows"
    ]
    assert all(row["status"] != "failed" for row in results)
    assert all(row["success_rate"] == 1.0 for row in results if row["status"] == "success")
    assert all(
        row["success_rate"] == 0.0 for row in results if row["status"] == "screened_out"
    )
    assert manifest.metadata["batch_id"] == summary["batch_id"]
    assert manifest.metadata["batch_outcome"] == expected_outcome
    assert manifest.metadata["runtime_selector_changed"] is False


def test_csr_queue_batch_00001_execution_artifacts_are_valid():
    _assert_queue_batch_execution_artifacts(
        Path("runs/phase1_csr_queue_batch_00001"),
        expected_batch_id="batch_00001",
        expected_outcome="profiled_with_gpu_success",
        expected_planned_total_nnz=15342,
        expected_planned_max_matrix_nnz=4054,
        expected_gpu_success_rows=2,
        expected_cpu_screened_out_rows=22,
        expected_selector_oracle_rows=2,
        expected_resume_status="completed",
    )


def test_csr_queue_batch_00002_screen_only_artifacts_are_valid():
    _assert_queue_batch_execution_artifacts(
        Path("runs/phase1_csr_queue_batch_00002"),
        expected_batch_id="batch_00002",
        expected_outcome="screen_only_no_oracle",
        expected_planned_total_nnz=27563,
        expected_planned_max_matrix_nnz=6511,
        expected_gpu_success_rows=0,
        expected_cpu_screened_out_rows=24,
        expected_selector_oracle_rows=0,
        expected_resume_status="completed_no_oracle",
    )


def test_csr_queue_batch_00003_execution_artifacts_are_valid():
    _assert_queue_batch_execution_artifacts(
        Path("runs/phase1_csr_queue_batch_00003"),
        expected_batch_id="batch_00003",
        expected_outcome="profiled_with_gpu_success",
        expected_planned_total_nnz=49029,
        expected_planned_max_matrix_nnz=21842,
        expected_gpu_success_rows=5,
        expected_cpu_screened_out_rows=19,
        expected_selector_oracle_rows=3,
        expected_resume_status="completed",
    )
