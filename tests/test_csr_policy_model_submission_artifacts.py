import json
from pathlib import Path

from transsolvestack.profiling.artifacts import read_jsonl
from transsolvestack.profiling.provenance import read_manifest, verify_manifest_hashes


def test_csr_policy_model_submission_artifacts_are_valid():
    root = Path("runs/phase1_csr_policy_model_submission")
    submission = json.loads(
        (root / "csr_policy_model_submission_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    rows = read_jsonl(root / "csr_policy_model_submission_rows.jsonl")
    summary = json.loads(
        (root / "csr_policy_model_submission_summary.json").read_text(
            encoding="utf-8"
        )
    )
    schema = json.loads(
        (root / "csr_policy_model_submission_schema.json").read_text(
            encoding="utf-8"
        )
    )
    model_card = (root / "MODEL_CARD.md").read_text(encoding="utf-8")
    manifest = read_manifest(root / "artifact_manifest.json")

    assert manifest.artifact_kind == "csr_policy_model_submission"
    assert verify_manifest_hashes(manifest) == ()
    assert summary["status"] == "passed"
    assert summary["schema_version"] == "phase1_csr_policy_model_submission_v1"
    assert schema["schema_version"] == summary["schema_version"]
    assert submission["schema_version"] == summary["schema_version"]
    assert summary["submission_ready"] is True
    assert summary["shadow_submission_ready"] is True
    assert summary["runtime_promotion_ready"] is False
    assert summary["default_runtime_mode"] == "shadow"
    assert summary["terms_acknowledged"] is True
    assert summary["accepted_for_shadow"] is True
    assert summary["accepted_for_runtime_promotion"] is False
    assert "quality_gate_not_runtime_eligible" in summary["promotion_blockers"]
    assert summary["guarded_gpu_shadow_smoke_checked"] is True
    assert summary["guarded_gpu_smoke_successes"] == 2
    assert summary["num_packaged_files"] == 8
    assert summary["all_file_checksums_present"] is True
    assert summary["runtime_selector_changed"] is False
    assert summary["validation_error_count"] == 0
    assert submission["runtime_boundary"]["guard_required"] is True
    assert submission["runtime_boundary"]["runtime_selector_changed"] is False

    row_kinds = {row["row_kind"] for row in rows}
    assert row_kinds == {
        "submission_identity",
        "terms",
        "model_artifact",
        "acceptance_gate",
        "submission_decision",
        "file_checksum",
    }
    checksum_rows = [row for row in rows if row["row_kind"] == "file_checksum"]
    assert len(checksum_rows) == summary["num_packaged_files"]
    assert all(row["exists"] and row["sha256"] for row in checksum_rows)
    assert "Runtime Boundary" in model_card
    assert "guarded_gpu_shadow_smoke_checked" in model_card
    assert manifest.metadata["submission_ready"] is True
    assert manifest.metadata["shadow_submission_ready"] is True
    assert manifest.metadata["runtime_promotion_ready"] is False
    assert manifest.metadata["runtime_selector_changed"] is False
