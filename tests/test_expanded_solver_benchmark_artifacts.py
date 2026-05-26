from pathlib import Path

from transsolvestack.profiling.artifacts import (
    read_candidate_performance,
    read_jsonl,
)
from transsolvestack.profiling.provenance import read_manifest, verify_manifest_hashes


def test_expanded_solver_benchmark_artifacts_are_valid():
    root = Path("runs/phase1_smoke_expanded_solvers")
    rows = read_candidate_performance(root / "candidate_performance.jsonl")
    evaluations = read_jsonl(root / "candidate_evaluation.jsonl")
    manifest = read_manifest(root / "artifact_manifest.json")
    assert len(rows) == 35
    assert len(evaluations) == 35
    assert manifest.artifact_kind == "taichi_smoke_benchmark"
    assert manifest.metadata["num_performance_records"] == 35
    assert manifest.metadata["candidate_set"] == (
        "configs/candidates/phase1_taichi_gpu_expanded.yaml"
    )
    assert verify_manifest_hashes(manifest) == ()
    assert {row.candidate_id for row in rows} == {
        "taichi_pcg_jacobi_f32",
        "taichi_cg_none_f32",
        "taichi_bicgstab_jacobi_f32",
        "taichi_richardson_jacobi_f32",
        "taichi_chebyshev_jacobi_f32",
    }
    failures = [row for row in rows if row.status != "success"]
    assert len(failures) == 7
    assert {row.candidate_id for row in failures} == {
        "taichi_richardson_jacobi_f32"
    }
    successes = [row for row in rows if row.status == "success"]
    assert len(successes) == 28
    assert all(row.final_residual_norm <= 1.0e-6 for row in successes)
    assert all(row.metadata["relative_error_to_true"] < 5.0e-3 for row in successes)
    assert sum(1 for row in evaluations if row["is_oracle"]) == 7
