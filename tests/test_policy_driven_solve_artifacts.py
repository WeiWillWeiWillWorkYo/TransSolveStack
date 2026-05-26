from pathlib import Path

from transsolvestack.profiling.artifacts import read_jsonl
from transsolvestack.profiling.provenance import read_manifest, verify_manifest_hashes


def test_policy_driven_solve_artifacts_are_valid():
    root = Path("runs/phase1_policy_solve")
    results = read_jsonl(root / "policy_solve_results.jsonl")
    plans = read_jsonl(root / "policy_run_plans.jsonl")
    traces = read_jsonl(root / "policy_run_traces.jsonl")
    manifest = read_manifest(root / "artifact_manifest.json")
    assert len(results) == 7
    assert len(plans) == 7
    assert len(traces) == 7
    assert manifest.artifact_kind == "policy_driven_taichi_solve"
    assert manifest.metadata["num_results"] == 7
    assert manifest.metadata["num_success"] == 7
    assert verify_manifest_hashes(manifest) == ()
    assert all(row["status"] == "success" for row in results)
    assert all(row["selection_reason"] == "profiled_success" for row in results)
    assert all(row["audit"]["is_oracle"] for row in results)
    assert all(len(row["fallback_candidate_ids"]) == 2 for row in results)
    assert all(row["final_residual_norm"] <= 1.0e-6 for row in results)
    assert all(row["relative_error_to_true"] < 5.0e-3 for row in results)
    assert all(row["policy_plan"]["backend"] == "taichi_gpu" for row in plans)
