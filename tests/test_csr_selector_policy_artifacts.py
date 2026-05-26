import json
from pathlib import Path

from transsolvestack.profiling.artifacts import read_jsonl
from transsolvestack.profiling.provenance import read_manifest, verify_manifest_hashes


def test_csr_selector_policy_artifacts_are_valid():
    root = Path("runs/phase1_csr_selector_policy")
    plans = read_jsonl(root / "csr_selected_policy_plans.jsonl")
    solve_checks = read_jsonl(root / "csr_selector_policy_solve_checks.jsonl")
    summary = json.loads((root / "csr_selector_policy_summary.json").read_text(encoding="utf-8"))
    manifest = read_manifest(root / "artifact_manifest.json")

    assert manifest.artifact_kind == "csr_selector_policy_smoke"
    assert verify_manifest_hashes(manifest) == ()
    assert summary["status"] == "passed"
    assert summary["num_selected_plans"] == 4
    assert summary["num_oracle_selected_plans"] == 4
    assert summary["num_solve_checks"] == 2
    assert summary["num_successful_solve_checks"] == 2
    assert set(summary["selected_solver_set"]).issubset(
        {"cg", "pcg", "bicgstab", "gmres", "richardson", "chebyshev"}
    )
    assert float(summary["max_final_relative_residual"]) <= 1.0e-5
    assert float(summary["max_cpu_recomputed_relative_residual"]) <= 1.0e-4
    assert float(summary["max_solution_relative_error"]) <= 5.0e-3
    assert len(plans) == 4
    assert all(row["audit"]["selector"] == "csr_artifact" for row in plans)
    assert all(row["audit"]["is_oracle"] is True for row in plans)
    assert all(row["status"] == "success" for row in solve_checks)
    trefethen = next(
        row
        for row in plans
        if row["matrix_id"] == "suitesparse:JGD_Trefethen/Trefethen_20b"
    )
    assert trefethen["candidate_id"] == "taichi_csr_chebyshev_jacobi_float64"
    assert trefethen["plan"]["solver"]["lambda_min"] > 0.0
    assert (
        trefethen["plan"]["solver"]["lambda_min"]
        < trefethen["plan"]["solver"]["lambda_max"]
    )
