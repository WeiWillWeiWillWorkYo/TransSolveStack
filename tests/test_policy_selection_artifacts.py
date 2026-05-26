from pathlib import Path

from transsolvestack.profiling.artifacts import read_jsonl
from transsolvestack.profiling.provenance import read_manifest, verify_manifest_hashes


def test_policy_selection_artifacts_are_valid():
    root = Path("runs/phase1_policy_selection")
    rows = read_jsonl(root / "selected_policy_plans.jsonl")
    manifest = read_manifest(root / "artifact_manifest.json")
    assert len(rows) == 7
    assert manifest.artifact_kind == "policy_selection"
    assert manifest.metadata["num_selected_plans"] == 7
    assert verify_manifest_hashes(manifest) == ()
    assert all(row["backend"] == "taichi_gpu" for row in rows)
    assert all(row["reason"] == "profiled_success" for row in rows)
    assert all(row["audit"]["is_oracle"] for row in rows)
    assert all(row["audit"]["regret_vs_oracle"] == 0.0 for row in rows)
    assert all(len(row["fallback_candidate_ids"]) == 2 for row in rows)
    assert all(row["solver"]["name"] for row in rows)
