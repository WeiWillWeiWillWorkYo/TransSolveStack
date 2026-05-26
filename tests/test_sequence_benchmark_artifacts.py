from pathlib import Path

from transsolvestack.profiling.artifacts import read_jsonl
from transsolvestack.profiling.provenance import read_manifest, verify_manifest_hashes


def test_sequence_benchmark_artifacts_are_valid():
    root = Path("runs/phase1_sequence_taichi")
    steps = read_jsonl(root / "sequence_steps.jsonl")
    summary = read_jsonl(root / "sequence_summary.jsonl")
    manifest = read_manifest(root / "artifact_manifest.json")
    assert len(steps) == 8
    assert len(summary) == 2
    assert manifest.artifact_kind == "taichi_sequence_benchmark"
    assert manifest.metadata["num_step_records"] == 8
    assert verify_manifest_hashes(manifest) == ()
    assert all(row["status"] == "success" for row in steps)
    assert all(row["final_residual_norm"] <= 1.0e-6 for row in steps)
    assert all(row["relative_error_to_true"] < 5.0e-3 for row in steps)
    assert all(row["status"] == "success" for row in summary)

