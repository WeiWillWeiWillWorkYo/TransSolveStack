import json
from pathlib import Path

from transsolvestack.profiling.artifacts import read_jsonl
from transsolvestack.profiling.provenance import read_manifest, verify_manifest_hashes


def test_csr_transformer_package_consumer_dry_run_artifacts_are_valid():
    root = Path("runs/phase1_csr_transformer_package_consumer_dry_run")
    consumer = json.loads(
        (root / "csr_transformer_package_consumer_dry_run_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    rows = read_jsonl(root / "csr_transformer_package_consumer_dry_run_rows.jsonl")
    summary = json.loads(
        (root / "csr_transformer_package_consumer_dry_run_summary.json").read_text(
            encoding="utf-8"
        )
    )
    schema = json.loads(
        (root / "csr_transformer_package_consumer_dry_run_schema.json").read_text(
            encoding="utf-8"
        )
    )
    report = (
        root / "csr_transformer_package_consumer_dry_run_report.md"
    ).read_text(encoding="utf-8")
    manifest = read_manifest(root / "artifact_manifest.json")

    assert manifest.artifact_kind == "csr_transformer_package_consumer_dry_run"
    assert verify_manifest_hashes(manifest) == ()
    assert summary["status"] == "passed"
    assert summary["schema_version"] == (
        "phase1_csr_transformer_package_consumer_dry_run_v1"
    )
    assert consumer["schema_version"] == summary["schema_version"]
    assert schema["schema_version"] == summary["schema_version"]
    assert summary["consumer_ready"] is True
    assert summary["dry_run_only"] is True
    assert summary["model_trained"] is False
    assert summary["executes_gpu"] is False
    assert summary["runtime_selector_changed"] is False
    assert summary["source_package_ready"] is True
    assert summary["input_checksum_verified"] is True
    assert summary["num_inputs_verified"] == 16
    assert summary["num_expected_outputs_resolved"] == 6
    assert summary["num_validation_steps_resolved"] == 6
    assert summary["reference_intake_ready"] is True
    assert summary["reference_shadow_submission_ready"] is True
    assert summary["reference_runtime_promotion_ready"] is False
    assert summary["reference_guarded_gpu_shadow_smoke_checked"] is True
    assert summary["reference_guarded_gpu_smoke_successes"] == 2
    assert summary["submission_shadow_ready"] is True
    assert summary["submission_runtime_promotion_ready"] is False
    assert summary["acceptance_shadow_ready"] is True
    assert summary["acceptance_runtime_promotion_ready"] is False
    assert "quality_gate_not_runtime_eligible" in summary["promotion_blockers"]
    assert summary["validation_error_count"] == 0

    row_kinds = {row["row_kind"] for row in rows}
    assert row_kinds == {
        "consumer_decision",
        "package_input_check",
        "expected_output_reference",
        "validation_step_route",
    }
    input_rows = [row for row in rows if row["row_kind"] == "package_input_check"]
    output_rows = [
        row for row in rows if row["row_kind"] == "expected_output_reference"
    ]
    step_rows = [row for row in rows if row["row_kind"] == "validation_step_route"]
    assert len(input_rows) == 16
    assert all(row["checksum_match"] is True for row in input_rows)
    assert all(row["internal_development_doc"] is False for row in input_rows)
    assert len(output_rows) == 6
    assert all(row["exists_in_reference_fixture"] is True for row in output_rows)
    assert len(step_rows) == 6
    assert all(row["script_exists"] is True for row in step_rows)
    assert len([row for row in step_rows if row["guarded_or_submission_step"]]) == 3
    assert consumer["runtime_boundary"]["guard_required"] is True
    assert consumer["runtime_boundary"]["runtime_selector_changed"] is False
    assert schema["runtime_boundary"]["dry_run_only"] is True
    assert "Promotion Blockers" in report
    assert manifest.metadata["consumer_ready"] is True
    assert manifest.metadata["reference_shadow_submission_ready"] is True
    assert manifest.metadata["reference_runtime_promotion_ready"] is False
    assert manifest.metadata["runtime_selector_changed"] is False
