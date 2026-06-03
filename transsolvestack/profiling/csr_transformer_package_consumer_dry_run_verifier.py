"""Artifact verifier for D24 CSR Transformer package consumer dry run."""

from __future__ import annotations

import json
from pathlib import Path

from transsolvestack.profiling.artifacts import read_jsonl
from transsolvestack.profiling.provenance import read_manifest, verify_manifest_hashes


def verify_csr_transformer_package_consumer_dry_run(path: Path) -> None:
    manifest = read_manifest(path / "artifact_manifest.json")
    stale = verify_manifest_hashes(manifest)
    if stale:
        raise SystemExit(f"stale CSR Transformer package consumer manifest: {stale}")
    consumer = json.loads(
        (path / "csr_transformer_package_consumer_dry_run_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    rows = read_jsonl(path / "csr_transformer_package_consumer_dry_run_rows.jsonl")
    summary = json.loads(
        (path / "csr_transformer_package_consumer_dry_run_summary.json").read_text(
            encoding="utf-8"
        )
    )
    schema = json.loads(
        (path / "csr_transformer_package_consumer_dry_run_schema.json").read_text(
            encoding="utf-8"
        )
    )
    report = (
        path / "csr_transformer_package_consumer_dry_run_report.md"
    ).read_text(encoding="utf-8")

    if manifest.artifact_kind != "csr_transformer_package_consumer_dry_run":
        raise SystemExit("unexpected CSR Transformer package consumer artifact kind")
    if summary["schema_version"] != "phase1_csr_transformer_package_consumer_dry_run_v1":
        raise SystemExit("CSR Transformer package consumer schema mismatch")
    if consumer["schema_version"] != summary["schema_version"]:
        raise SystemExit("CSR Transformer package consumer manifest schema mismatch")
    if schema["schema_version"] != summary["schema_version"]:
        raise SystemExit("CSR Transformer package consumer schema summary mismatch")
    if summary["status"] != "passed" or summary["consumer_ready"] is not True:
        raise SystemExit("CSR Transformer package consumer dry run did not pass")
    if (
        summary["dry_run_only"] is not True
        or summary["model_trained"] is not False
        or summary["executes_gpu"] is not False
        or summary["runtime_selector_changed"] is not False
    ):
        raise SystemExit("CSR Transformer package consumer runtime boundary mismatch")
    if summary["source_package_ready"] is not True:
        raise SystemExit("CSR Transformer package consumer source not ready")
    if summary["input_checksum_verified"] is not True:
        raise SystemExit("CSR Transformer package consumer checksum mismatch")
    if summary["num_inputs_verified"] != 16:
        raise SystemExit("CSR Transformer package consumer input count mismatch")
    if summary["num_expected_outputs_resolved"] != 6:
        raise SystemExit("CSR Transformer package consumer output count mismatch")
    if summary["num_validation_steps_resolved"] != 6:
        raise SystemExit("CSR Transformer package consumer validation count mismatch")
    if summary["reference_intake_ready"] is not True:
        raise SystemExit("CSR Transformer package consumer reference intake mismatch")
    if summary["reference_shadow_submission_ready"] is not True:
        raise SystemExit("CSR Transformer package consumer shadow readiness mismatch")
    if summary["reference_runtime_promotion_ready"] is not False:
        raise SystemExit("CSR Transformer package consumer promoted runtime")
    if summary["reference_guarded_gpu_shadow_smoke_checked"] is not True:
        raise SystemExit("CSR Transformer package consumer guard smoke mismatch")
    if summary["reference_guarded_gpu_smoke_successes"] != 2:
        raise SystemExit("CSR Transformer package consumer guard smoke count mismatch")
    if summary["submission_shadow_ready"] is not True:
        raise SystemExit("CSR Transformer package consumer submission mismatch")
    if summary["submission_runtime_promotion_ready"] is not False:
        raise SystemExit("CSR Transformer package consumer submission promoted")
    if summary["acceptance_shadow_ready"] is not True:
        raise SystemExit("CSR Transformer package consumer acceptance mismatch")
    if summary["acceptance_runtime_promotion_ready"] is not False:
        raise SystemExit("CSR Transformer package consumer acceptance promoted")
    if "quality_gate_not_runtime_eligible" not in summary["promotion_blockers"]:
        raise SystemExit("CSR Transformer package consumer blocker mismatch")
    if summary["validation_error_count"] != 0:
        raise SystemExit("CSR Transformer package consumer validation errors present")

    row_kinds = {row["row_kind"] for row in rows}
    if row_kinds != {
        "consumer_decision",
        "package_input_check",
        "expected_output_reference",
        "validation_step_route",
    }:
        raise SystemExit("CSR Transformer package consumer row kinds mismatch")
    input_rows = [row for row in rows if row["row_kind"] == "package_input_check"]
    output_rows = [
        row for row in rows if row["row_kind"] == "expected_output_reference"
    ]
    step_rows = [row for row in rows if row["row_kind"] == "validation_step_route"]
    if len(input_rows) != 16 or any(row["checksum_match"] is not True for row in input_rows):
        raise SystemExit("CSR Transformer package consumer input rows mismatch")
    if any(row["internal_development_doc"] is not False for row in input_rows):
        raise SystemExit("CSR Transformer package consumer references internal docs")
    if len(output_rows) != 6 or any(
        row["exists_in_reference_fixture"] is not True for row in output_rows
    ):
        raise SystemExit("CSR Transformer package consumer output rows mismatch")
    if len(step_rows) != 6 or any(row["script_exists"] is not True for row in step_rows):
        raise SystemExit("CSR Transformer package consumer step rows mismatch")
    if consumer["runtime_boundary"]["guard_required"] is not True:
        raise SystemExit("CSR Transformer package consumer guard mismatch")
    if consumer["runtime_boundary"]["runtime_selector_changed"] is not False:
        raise SystemExit("CSR Transformer package consumer runtime mismatch")
    if schema["runtime_boundary"]["dry_run_only"] is not True:
        raise SystemExit("CSR Transformer package consumer schema boundary mismatch")
    if "Promotion Blockers" not in report:
        raise SystemExit("CSR Transformer package consumer report missing blockers")
    if manifest.metadata["consumer_ready"] is not True:
        raise SystemExit("CSR Transformer package consumer manifest ready mismatch")
    if manifest.metadata["runtime_selector_changed"] is not False:
        raise SystemExit("CSR Transformer package consumer manifest runtime mismatch")
