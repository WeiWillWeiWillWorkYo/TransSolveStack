import json
from pathlib import Path

from transsolvestack.profiling.artifacts import read_jsonl
from transsolvestack.profiling.provenance import read_manifest, verify_manifest_hashes


def test_taichi_csr_chebyshev_artifacts_are_valid():
    root = Path("runs/phase1_taichi_csr_chebyshev")
    rows = read_jsonl(root / "csr_chebyshev_results.jsonl")
    summary = json.loads((root / "csr_chebyshev_summary.json").read_text(encoding="utf-8"))
    manifest = read_manifest(root / "artifact_manifest.json")

    assert manifest.artifact_kind == "taichi_csr_chebyshev_smoke"
    assert verify_manifest_hashes(manifest) == ()
    assert summary["status"] == "passed"
    assert summary["num_selected_matrices"] == 1
    assert summary["num_solves"] == 1
    assert summary["num_success"] == 1
    assert summary["num_failed"] == 0
    assert summary["precision"] == "float64"
    assert summary["measurement_repeats"] == 3
    assert float(summary["max_final_relative_residual"]) <= float(
        summary["tolerance_rel"]
    )
    assert float(summary["max_cpu_recomputed_relative_residual"]) <= 1.0e-4
    assert float(summary["max_solution_relative_error"]) <= 5.0e-3
    assert len(rows) == 1
    assert {row["solver"] for row in rows} == {"chebyshev"}
    assert {row["preconditioner"] for row in rows} == {"jacobi"}
    assert all(row["status"] == "success" for row in rows)
    assert all(row["measurement_repeats"] == 3 for row in rows)
    assert all(row["success_rate"] == 1.0 for row in rows)
    assert all(len(row["repeat_records"]) == 3 for row in rows)
    assert all(row["solve_time_ms"] == row["median_solve_time_ms"] for row in rows)
    assert all(row["solve_time_iqr_ms"] >= 0.0 for row in rows)
    assert all(row["backend"] == "taichi_gpu" for row in rows)
    assert all(row["cpu_screen"]["success"] is True for row in rows)
    assert rows[0]["lambda_min"] > 0.0
    assert rows[0]["lambda_min"] < rows[0]["lambda_max"]
    assert any(
        row["reason"] == "cpu_reference_chebyshev_screen_failed"
        for row in summary["skipped_candidates"]
    )
