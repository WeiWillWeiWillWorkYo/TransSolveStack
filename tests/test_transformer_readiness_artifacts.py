from pathlib import Path
import json

from transsolvestack.profiling.artifacts import read_jsonl
from transsolvestack.profiling.provenance import read_manifest, verify_manifest_hashes


def test_transformer_readiness_artifacts_are_valid():
    root = Path("runs/phase1_transformer_readiness")
    rows = read_jsonl(root / "policy_training_rows.jsonl")
    schema = json.loads((root / "transformer_feature_schema.json").read_text(encoding="utf-8"))
    manifest = read_manifest(root / "artifact_manifest.json")
    assert len(rows) == 21
    assert sum(1 for row in rows if row["label_is_oracle"]) == 7
    assert schema["schema_version"] == "phase1_policy_features_v1"
    assert schema["integration_boundary"]["model_required"] is False
    assert manifest.artifact_kind == "transformer_readiness_export"
    assert manifest.metadata["num_rows"] == 21
    assert manifest.metadata["num_oracle_rows"] == 7
    assert verify_manifest_hashes(manifest) == ()
