import json
from pathlib import Path

from transsolvestack.profiling.artifacts import read_jsonl
from transsolvestack.profiling.provenance import read_manifest, verify_manifest_hashes


def test_csr_policy_model_acceptance_artifacts_are_valid():
    root = Path("runs/phase1_csr_policy_model_acceptance")
    rows = read_jsonl(root / "csr_policy_model_acceptance_rows.jsonl")
    summary = json.loads(
        (root / "csr_policy_model_acceptance_summary.json").read_text(encoding="utf-8")
    )
    schema = json.loads(
        (root / "csr_policy_model_acceptance_schema.json").read_text(encoding="utf-8")
    )
    manifest = read_manifest(root / "artifact_manifest.json")

    assert manifest.artifact_kind == "csr_policy_model_acceptance"
    assert verify_manifest_hashes(manifest) == ()
    assert summary["status"] == "passed"
    assert summary["schema_version"] == "phase1_csr_policy_model_acceptance_v1"
    assert schema["schema_version"] == summary["schema_version"]
    assert summary["accepted_for_shadow"] is True
    assert summary["accepted_for_runtime_promotion"] is False
    assert summary["model_loaded"] is True
    assert summary["prediction_contract_checked"] is True
    assert summary["guard_plan_shadow_checked"] is True
    assert summary["guarded_gpu_shadow_smoke_checked"] is True
    assert summary["runtime_selector_changed"] is False
    assert summary["validation_error_count"] == 0

    row_kinds = {row["row_kind"] for row in rows}
    assert "guard_plan_shadow" in row_kinds
    assert "guarded_gpu_shadow_result" in row_kinds
    guard_rows = [row for row in rows if row["row_kind"] == "guard_plan_shadow"]
    gpu_rows = [row for row in rows if row["row_kind"] == "guarded_gpu_shadow_result"]
    assert len(guard_rows) == 2
    assert len(gpu_rows) == 2
    assert all(row["learned_policy_source"]["source_kind"] == "model_artifact" for row in guard_rows)
    assert all(row["backend"] == "taichi_gpu" for row in gpu_rows)
    assert manifest.metadata["accepted_for_shadow"] is True
    assert manifest.metadata["accepted_for_runtime_promotion"] is False
    assert manifest.metadata["runtime_selector_changed"] is False
