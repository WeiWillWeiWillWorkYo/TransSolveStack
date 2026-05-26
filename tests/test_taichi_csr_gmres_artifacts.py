from pathlib import Path
import json

from transsolvestack.profiling.artifacts import read_jsonl
from transsolvestack.profiling.provenance import read_manifest, verify_manifest_hashes


def test_taichi_csr_gmres_artifacts_are_valid():
    root = Path("runs/phase1_taichi_csr_gmres")
    rows = read_jsonl(root / "csr_gmres_results.jsonl")
    summary = json.loads((root / "csr_gmres_summary.json").read_text(encoding="utf-8"))
    manifest = read_manifest(root / "artifact_manifest.json")

    assert manifest.artifact_kind == "taichi_csr_gmres_smoke"
    assert verify_manifest_hashes(manifest) == ()
    assert summary["status"] == "passed"
    assert summary["num_selected_matrices"] == 2
    assert summary["num_solves"] == 5
    assert summary["num_success"] == 5
    assert summary["num_failed"] == 0
    assert summary["precision"] == "float64"
    assert summary["restarts"] == [8, 16, 32]
    assert summary["num_restarts"] == 3
    assert summary["measurement_repeats"] == 3
    assert len(rows) == 5
    assert {row["solver"] for row in rows} == {"gmres"}
    assert {row["preconditioner"] for row in rows} == {"jacobi"}
    assert {row["restart"] for row in rows} == {8, 16, 32}
    assert any(
        row["matrix_id"] == "suitesparse:HB/curtis54" and row["restart"] == 16
        for row in rows
    )
    assert any(
        row["matrix_id"] == "suitesparse:HB/curtis54" and row["restart"] == 32
        for row in rows
    )
    assert any(
        row["matrix_id"] == "suitesparse:HB/curtis54" and row["restart"] == 8
        for row in summary["screened_out_restart_configs"]
    )
    assert all(row["status"] == "success" for row in rows)
    assert all(row["measurement_repeats"] == 3 for row in rows)
    assert all(row["success_rate"] == 1.0 for row in rows)
    assert all(len(row["repeat_records"]) == 3 for row in rows)
    assert all(row["solve_time_ms"] == row["median_solve_time_ms"] for row in rows)
    assert all(row["solve_time_iqr_ms"] >= 0.0 for row in rows)
    assert all(row["backend"] == "taichi_gpu" for row in rows)
    assert all(row["precision"] == "float64" for row in rows)
    assert all(row["symmetry"] != "symmetric" for row in rows)
    assert all(row["final_relative_residual"] <= summary["tolerance_rel"] for row in rows)
    assert all(row["cpu_recomputed_relative_residual"] <= 1.0e-4 for row in rows)
    assert all(row["solution_relative_error"] <= 5.0e-3 for row in rows)
    assert all(row["cpu_screen"]["success"] is True for row in rows)
    assert all(
        row["orthogonalization_backend"] == "taichi_gpu_modified_gram_schmidt"
        for row in rows
    )
    assert all(row["least_squares_backend"] == "numpy_lstsq_dense_hessenberg" for row in rows)
