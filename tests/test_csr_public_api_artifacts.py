from pathlib import Path
import json

from transsolvestack.profiling.provenance import read_manifest, verify_manifest_hashes


def test_csr_public_api_smoke_artifacts_are_valid():
    root = Path("runs/phase1_csr_public_api")
    payload = json.loads((root / "csr_public_api_smoke.json").read_text(encoding="utf-8"))
    manifest = read_manifest(root / "artifact_manifest.json")
    summary = payload["summary"]

    assert manifest.artifact_kind == "csr_public_api_smoke"
    assert verify_manifest_hashes(manifest) == ()
    assert summary["status"] == "passed"
    assert summary["precision"] == "float64"
    assert summary["num_matrices"] == 3
    assert summary["num_solves"] == 5
    assert summary["num_success"] == 5
    assert summary["num_failed"] == 0
    assert len(payload["diagnostics"]) == 3
    assert len(payload["results"]) == 5
    cg_diagnostics = [
        row
        for row in payload["diagnostics"]
        if row["matrix_id"]
        in {"suitesparse:JGD_Trefethen/Trefethen_20b", "suitesparse:FIDAP/ex5"}
    ]
    gmres_diagnostics = [
        row
        for row in payload["diagnostics"]
        if row["matrix_id"] == "suitesparse:HB/curtis54"
    ]
    assert all(row["actual_symmetric"] for row in cg_diagnostics)
    assert all(row["cg_candidate"] for row in cg_diagnostics)
    assert all(not row["actual_symmetric"] for row in gmres_diagnostics)
    assert {row["solver"] for row in payload["results"]} == {"cg", "pcg", "gmres"}
    assert all(row["status"] == "success" for row in payload["results"])
    assert all(row["trace"]["backend"] == "taichi_gpu" for row in payload["results"])
    assert all(
        row["trace"]["metadata"]["operator_backend"] == "taichi_csr"
        for row in payload["results"]
    )
    assert summary["max_final_relative_residual"] <= 1.0e-5
    assert summary["max_cpu_recomputed_relative_residual"] <= 1.0e-4
    assert summary["max_solution_relative_error"] <= 5.0e-3
