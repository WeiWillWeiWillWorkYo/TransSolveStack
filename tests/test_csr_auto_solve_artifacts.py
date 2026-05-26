import json
from pathlib import Path

from transsolvestack.profiling.artifacts import read_jsonl
from transsolvestack.profiling.provenance import read_manifest, verify_manifest_hashes


def test_csr_auto_solve_artifacts_are_valid():
    root = Path("runs/phase1_csr_auto_solve")
    rows = read_jsonl(root / "csr_auto_solve_results.jsonl")
    summary = json.loads((root / "csr_auto_solve_summary.json").read_text(encoding="utf-8"))
    manifest = read_manifest(root / "artifact_manifest.json")

    assert manifest.artifact_kind == "csr_auto_solve_smoke"
    assert verify_manifest_hashes(manifest) == ()
    assert summary["status"] == "passed"
    assert summary["num_solves"] == 2
    assert summary["num_success"] == 2
    assert summary["num_failed"] == 0
    assert set(summary["selected_solver_set"]).issubset(
        {"cg", "pcg", "bicgstab", "gmres", "richardson", "chebyshev"}
    )
    assert summary["num_oracle_selected"] == 2
    assert float(summary["max_final_relative_residual"]) <= 1.0e-5
    assert float(summary["max_cpu_recomputed_relative_residual"]) <= 1.0e-4
    assert float(summary["max_solution_relative_error"]) <= 5.0e-3
    assert len(rows) == 2
    assert all(row["selection_reason"] == "profiled_success" for row in rows)
    assert all(row["selected_is_oracle"] is True for row in rows)
    assert all(row["trace"]["backend"] == "taichi_gpu" for row in rows)
    assert all("auto_solve_csr" in row["trace"]["metadata"] for row in rows)
    trefethen = next(
        row
        for row in rows
        if row["matrix_id"] == "suitesparse:JGD_Trefethen/Trefethen_20b"
    )
    auto_meta = trefethen["trace"]["metadata"]["auto_solve_csr"]
    assert trefethen["candidate_id"] == "taichi_csr_chebyshev_jacobi_float64"
    assert set(auto_meta["fallback_candidate_ids"]) == {
        "taichi_csr_richardson_jacobi_float64",
        "taichi_csr_cg_none_float64",
        "taichi_csr_pcg_jacobi_float64",
    }
