from pathlib import Path

from transsolvestack.profiling.artifacts import read_jsonl
from transsolvestack.profiling.provenance import read_manifest, verify_manifest_hashes


def test_dataset_download_plan_artifacts_are_valid():
    root = Path("runs/phase1_dataset_plan")
    rows = read_jsonl(root / "download_plan.jsonl")
    manifest = read_manifest(root / "artifact_manifest.json")
    assert len(rows) == 3
    assert manifest.artifact_kind == "dataset_download_plan"
    assert manifest.metadata["dry_run"] is True
    assert manifest.metadata["num_matrices"] == 3
    assert manifest.metadata["total_estimated_download_gb"] < 0.01
    assert verify_manifest_hashes(manifest) == ()
    assert all(row["download_required"] for row in rows)
    assert all(row["url"].startswith("https://") for row in rows)
    assert all(row["local_path"].startswith("data/raw/") for row in rows)
