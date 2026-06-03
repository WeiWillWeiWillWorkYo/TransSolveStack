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


def test_csr_blocked_gap_positive_search_artifacts_are_valid():
    root = Path("runs/phase1_csr_blocked_gap_positive_search")
    candidates = read_jsonl(root / "csr_blocked_gap_positive_candidates.jsonl")
    search_rows = read_jsonl(root / "csr_blocked_gap_positive_search_rows.jsonl")
    results = read_jsonl(root / "csr_blocked_gap_positive_results.jsonl")
    diagnostics = read_jsonl(root / "csr_blocked_gap_positive_diagnostic_rows.jsonl")
    selector_rows = read_jsonl(root / "csr_blocked_gap_positive_selector_rows.jsonl")
    summary = json.loads(
        (root / "csr_blocked_gap_positive_search_summary.json").read_text(
            encoding="utf-8"
        )
    )
    schema = json.loads(
        (root / "csr_blocked_gap_positive_search_schema.json").read_text(
            encoding="utf-8"
        )
    )
    selector_summary = json.loads(
        (root / "csr_blocked_gap_positive_selector_summary.json").read_text(
            encoding="utf-8"
        )
    )
    manifest = read_manifest(root / "artifact_manifest.json")

    assert manifest.artifact_kind == "csr_blocked_gap_positive_search"
    assert verify_manifest_hashes(manifest) == ()
    assert summary["status"] == "passed"
    assert summary["schema_version"] == "phase1_csr_blocked_gap_positive_search_v1"
    assert schema["schema_version"] == summary["schema_version"]
    assert schema["execution_boundary"]["executes_gpu"] is True
    assert schema["execution_boundary"]["cpu_screen_before_gpu"] is True
    assert schema["execution_boundary"]["runtime_selector_changed"] is False
    assert schema["execution_boundary"]["generic_queue_merge"] is False
    assert summary["runtime_selector_changed"] is False
    assert summary["executes_gpu"] is True
    assert summary["queue_merge_ready"] is False
    assert summary["positive_evidence_found"] is True

    assert set(summary["required_gap_ids"]) == REQUIRED_GAPS
    assert set(summary["positive_gpu_gap_ids"]) == REQUIRED_GAPS
    assert summary["source_records"] == 91
    assert summary["search_rows"] == len(search_rows) == 50
    assert summary["selected_candidates"] == len(candidates) == 8
    assert summary["candidate_jobs"] == len(results) == 8
    assert summary["selector_rows"] == len(selector_rows) == 8
    assert len(diagnostics) == 8
    assert summary["selector_oracle_rows"] == 5
    assert selector_summary["status"] == "passed"
    assert selector_summary["num_selector_rows"] == len(selector_rows)
    assert selector_summary["num_oracle_rows"] == summary["selector_oracle_rows"]
    assert selector_summary["num_success_rows"] == summary["gpu_success_rows"]

    assert summary["gpu_success_rows"] == 8
    assert summary["gpu_failed_rows"] == 0
    assert summary["by_status"] == {"success": 8}
    assert summary["by_search_status"] == {
        "screened_out": 19,
        "selected_positive": 8,
        "skipped": 23,
    }
    assert summary["by_gap_status"] == {
        "general_bicgstab_ilu0": {"success": 2},
        "general_bicgstab_row_column_equilibration": {"success": 2},
        "symmetric_chebyshev_jacobi": {"success": 2},
        "symmetric_pcg_symmetric_equilibration": {"success": 2},
    }

    assert {row["backend"] for row in results} == {"taichi_gpu"}
    assert {row["status"] for row in results} == {"success"}
    assert {row["coverage_gap_id"] for row in results} == REQUIRED_GAPS
    assert {row["coverage_gap_id"] for row in candidates} == REQUIRED_GAPS
    assert all(row["success_rate"] == 1.0 for row in results)
    assert all(float(row["final_relative_residual"]) <= 1.0e-5 for row in results)
    assert all(
        float(row["cpu_recomputed_relative_residual"]) <= 1.0e-5
        for row in results
    )
    assert all(float(row["solution_relative_error"]) <= 5.0e-3 for row in results)
    assert {(row["solver"], row["preconditioner"]) for row in results} == {
        ("bicgstab", "ilu0"),
        ("bicgstab", "row_column_equilibration"),
        ("chebyshev", "jacobi"),
        ("pcg", "symmetric_equilibration"),
    }
    assert set(summary["selected_matrix_ids"]) == {
        "suitesparse:FIDAP/ex5",
        "suitesparse:Grund/b1_ss",
        "suitesparse:HB/bcsstk01",
        "suitesparse:HB/curtis54",
        "suitesparse:JGD_Trefethen/Trefethen_20b",
    }
    assert manifest.metadata["candidate_jobs"] == summary["candidate_jobs"]
    assert manifest.metadata["positive_evidence_found"] is True
    assert manifest.metadata["queue_merge_ready"] is False
