from pathlib import Path

from transsolvestack.profiling.artifacts import read_jsonl
from transsolvestack.profiling.provenance import read_manifest, verify_manifest_hashes


def test_solver_functional_artifacts_are_valid():
    root = Path("runs/phase1_solver_functional")
    rows = read_jsonl(root / "solver_functional_results.jsonl")
    manifest = read_manifest(root / "artifact_manifest.json")
    assert len(rows) == 2
    assert manifest.artifact_kind == "solver_functional_smoke"
    assert manifest.metadata["num_records"] == 2
    assert manifest.metadata["num_success"] == 2
    assert verify_manifest_hashes(manifest) == ()
    assert {row["solver"] for row in rows} == {"richardson", "chebyshev"}
    assert all(row["preconditioner"] == "jacobi" for row in rows)
    assert all(row["status"] == "success" for row in rows)
    assert all(row["final_residual_norm"] <= 1.0e-6 for row in rows)
    assert all(row["relative_error_to_true"] < 5.0e-3 for row in rows)
    assert all(row["residual_drop"] <= 1.0e-6 for row in rows)
