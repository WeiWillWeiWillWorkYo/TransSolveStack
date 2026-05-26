from pathlib import Path
import json

from transsolvestack.profiling.provenance import read_manifest, verify_manifest_hashes


def test_public_api_smoke_artifact_is_valid():
    root = Path("runs/phase1_public_api_smoke")
    payload = json.loads((root / "public_api_smoke.json").read_text(encoding="utf-8"))
    manifest = read_manifest(root / "artifact_manifest.json")
    assert manifest.artifact_kind == "public_api_smoke"
    assert manifest.metadata["status"] == "success"
    assert verify_manifest_hashes(manifest) == ()
    assert payload["status"] == "success"
    assert payload["trace"]["backend"] == "taichi_gpu"
    assert payload["trace"]["final_residual_norm"] <= 1.0e-6
    assert payload["trace"]["metadata"]["relative_error_to_true"] < 5.0e-3
    assert payload["num_training_rows"] == 21
    assert payload["num_dataset_plan_entries"] == 3
