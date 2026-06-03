import json
from pathlib import Path

from transsolvestack.profiling.artifacts import read_jsonl
from transsolvestack.profiling.provenance import read_manifest, verify_manifest_hashes


def test_csr_external_model_intake_artifacts_are_valid():
    root = Path("runs/phase1_csr_external_model_intake")
    rows = read_jsonl(root / "csr_external_model_intake_rows.jsonl")
    summary = json.loads(
        (root / "csr_external_model_intake_summary.json").read_text(encoding="utf-8")
    )
    schema = json.loads(
        (root / "csr_external_model_intake_schema.json").read_text(encoding="utf-8")
    )
    report = (root / "csr_external_model_intake_report.md").read_text(encoding="utf-8")
    manifest = read_manifest(root / "artifact_manifest.json")

    assert manifest.artifact_kind == "csr_external_model_intake"
    assert verify_manifest_hashes(manifest) == ()
    assert summary["status"] == "passed"
    assert summary["schema_version"] == "phase1_csr_external_model_intake_v1"
    assert schema["schema_version"] == summary["schema_version"]
    assert summary["intake_ready"] is True
    assert summary["shadow_submission_ready"] is True
    assert summary["runtime_promotion_ready"] is False
    assert summary["default_runtime_mode"] == "shadow"
    assert summary["adapter_ready"] is True
    assert summary["replay_exact"] is True
    assert summary["quality_gate_runtime_eligible"] is False
    assert summary["policy_model_artifact_ready"] is True
    assert summary["accepted_for_shadow"] is True
    assert summary["accepted_for_runtime_promotion"] is False
    assert "quality_gate_not_runtime_eligible" in summary["promotion_blockers"]
    assert summary["guarded_gpu_shadow_smoke_checked"] is True
    assert summary["guarded_gpu_smoke_successes"] == 2
    assert summary["num_predictions"] == 20
    assert summary["runtime_selector_changed"] is False
    assert {row["row_kind"] for row in rows} == {"stage_status", "intake_decision"}
    assert len([row for row in rows if row["row_kind"] == "stage_status"]) == 7
    assert "Promotion Blockers" in report
    assert (root / "csr_policy_model_artifact.json").exists()
    assert (root / "csr_policy_model_submission_manifest.json").exists()
    assert manifest.metadata["intake_ready"] is True
    assert manifest.metadata["shadow_submission_ready"] is True
    assert manifest.metadata["runtime_promotion_ready"] is False
    assert manifest.metadata["runtime_selector_changed"] is False
