import json
from pathlib import Path

from transsolvestack.profiling.artifacts import read_jsonl
from transsolvestack.profiling.provenance import read_manifest, verify_manifest_hashes


def test_csr_benchmark_expansion_artifacts_are_valid():
    root = Path("runs/phase1_csr_benchmark_expansion_plan")
    matrix_rows = read_jsonl(root / "csr_benchmark_matrix_queue.jsonl")
    candidate_rows = read_jsonl(root / "csr_benchmark_candidate_queue.jsonl")
    summary = json.loads((root / "csr_benchmark_expansion_summary.json").read_text(encoding="utf-8"))
    schema = json.loads((root / "csr_benchmark_expansion_schema.json").read_text(encoding="utf-8"))
    manifest = read_manifest(root / "artifact_manifest.json")

    assert manifest.artifact_kind == "csr_benchmark_expansion_plan"
    assert verify_manifest_hashes(manifest) == ()
    assert summary["status"] == "passed"
    assert summary["schema_version"] == "phase1_csr_benchmark_expansion_plan_v1"
    assert schema["schema_version"] == summary["schema_version"]
    assert schema["runtime_selector_changed"] is False
    assert schema["executes_gpu"] is False
    assert summary["runtime_selector_changed"] is False
    assert summary["executes_gpu"] is False
    assert summary["selected_source_matrices"] == 64
    assert summary["already_profiled_matrices"] == 12
    assert summary["eligible_unprofiled_matrices"] == 36
    assert summary["planned_matrices"] == 8
    assert len(matrix_rows) == 8
    assert summary["planned_candidate_jobs"] == 24
    assert len(candidate_rows) == 24
    assert summary["planned_gpu_solve_attempts"] == 24
    assert summary["measurement_repeats"] == 1
    assert summary["total_planned_nnz"] == 82789
    assert summary["estimated_total_nnz_visits"] == 92343696
    assert summary["by_candidate_profile"] == {"general": 4, "symmetric": 4}
    assert summary["by_solver"] == {
        "bicgstab": 8,
        "cg": 4,
        "gmres": 4,
        "pcg": 4,
        "richardson": 4,
    }

    expected_matrices = {
        "suitesparse:MathWorks/tomography",
        "suitesparse:HB/jgl009",
        "suitesparse:HB/ibm32",
        "suitesparse:SNAP/email-Eu-core",
        "suitesparse:Oberwolfach/t2dal_e",
        "suitesparse:HB/bcsstk07",
        "suitesparse:HB/lshp1009",
        "suitesparse:Gset/G17",
    }
    assert {row["matrix_id"] for row in matrix_rows} == expected_matrices
    assert {row["matrix_id"] for row in candidate_rows} == expected_matrices
    assert all(row["planned_import"] for row in matrix_rows)
    assert all(row["planned_gpu_benchmark"] for row in matrix_rows)
    assert all(row["skipped_reason"] is None for row in matrix_rows)
    assert all(row["requires_cpu_screen"] for row in candidate_rows)
    assert all(row["planned_status"] == "queued_cpu_screen_required" for row in candidate_rows)
    assert all(row["measurement_repeats"] == 1 for row in candidate_rows)
    assert all(row["max_iter"] == 256 for row in candidate_rows)
    assert all(row["precision"] == "float64" for row in candidate_rows)
    assert sum(row["estimated_nnz_visits"] for row in candidate_rows) == (
        summary["estimated_total_nnz_visits"]
    )
