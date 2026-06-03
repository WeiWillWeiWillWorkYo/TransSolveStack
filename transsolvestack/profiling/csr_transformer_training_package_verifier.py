"""Artifact verifier for D23 CSR Transformer external training package."""

from __future__ import annotations

import json
from pathlib import Path

from transsolvestack.profiling.artifacts import read_jsonl
from transsolvestack.profiling.provenance import read_manifest, verify_manifest_hashes


def verify_csr_transformer_training_package(path: Path) -> None:
    manifest = read_manifest(path / "artifact_manifest.json")
    stale = verify_manifest_hashes(manifest)
    if stale:
        raise SystemExit(f"stale CSR Transformer training package manifest: {stale}")
    package = json.loads(
        (path / "csr_transformer_training_package_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    rows = read_jsonl(path / "csr_transformer_training_package_rows.jsonl")
    summary = json.loads(
        (path / "csr_transformer_training_package_summary.json").read_text(
            encoding="utf-8"
        )
    )
    schema = json.loads(
        (path / "csr_transformer_training_package_schema.json").read_text(
            encoding="utf-8"
        )
    )
    readme = (path / "TRAINING_PACKAGE.md").read_text(encoding="utf-8")
    if manifest.artifact_kind != "csr_transformer_training_package":
        raise SystemExit("unexpected CSR Transformer training package artifact kind")
    if summary["status"] != "passed" or summary["package_ready"] is not True:
        raise SystemExit("CSR Transformer training package did not pass")
    if summary["schema_version"] != "phase1_csr_transformer_training_package_v1":
        raise SystemExit("CSR Transformer training package schema mismatch")
    if schema["schema_version"] != summary["schema_version"]:
        raise SystemExit("CSR Transformer training package schema/summary mismatch")
    if package["schema_version"] != summary["schema_version"]:
        raise SystemExit("CSR Transformer training package manifest schema mismatch")
    if (
        schema["runtime_boundary"]["model_trained"] is not False
        or schema["runtime_boundary"]["runtime_selector_changed"] is not False
        or schema["runtime_boundary"]["executes_gpu"] is not False
        or schema["runtime_boundary"]["guard_required"] is not True
    ):
        raise SystemExit("CSR Transformer training package runtime boundary mismatch")
    if (
        summary["model_trained"] is not False
        or summary["runtime_selector_changed"] is not False
        or summary["executes_gpu"] is not False
        or summary["model_training_required"] is not True
    ):
        raise SystemExit("CSR Transformer training package changed model/runtime boundary")
    if summary["source_handoff_ready"] is not True:
        raise SystemExit("CSR Transformer training package handoff source not ready")
    if summary["num_input_files"] != 16:
        raise SystemExit("CSR Transformer training package input count mismatch")
    if summary["num_expected_outputs"] != 6:
        raise SystemExit("CSR Transformer training package expected output count mismatch")
    if summary["num_validation_steps"] != 6:
        raise SystemExit("CSR Transformer training package validation step count mismatch")
    if summary["training_model_requests"] != 105:
        raise SystemExit("CSR Transformer training package request count mismatch")
    if summary["training_tensor_requests"] != 105:
        raise SystemExit("CSR Transformer training package tensor count mismatch")
    if summary["training_global_candidates"] != 12:
        raise SystemExit("CSR Transformer training package candidate count mismatch")
    if summary["matrix_feature_dim"] != 21 or summary["candidate_feature_dim"] != 19:
        raise SystemExit("CSR Transformer training package feature dim mismatch")
    if summary["guard_actual_quality_gate_blocks"] != 5:
        raise SystemExit("CSR Transformer training package guard block mismatch")
    if summary["guard_actual_runtime_selector_changes"] != 0:
        raise SystemExit("CSR Transformer training package guard runtime mismatch")
    if summary["guard_fallback_chain_enforced_rows"] != 5:
        raise SystemExit("CSR Transformer training package fallback mismatch")
    if summary["current_shadow_baseline_runtime_eligible"] is not False:
        raise SystemExit("CSR Transformer training package baseline eligibility mismatch")
    if summary["internal_development_docs_included"] is not False:
        raise SystemExit("CSR Transformer training package includes internal docs")
    if summary["validation_error_count"] != 0:
        raise SystemExit("CSR Transformer training package validation errors present")
    input_rows = [row for row in rows if row["row_kind"] == "input_file"]
    expected_rows = [row for row in rows if row["row_kind"] == "expected_output"]
    validation_rows = [row for row in rows if row["row_kind"] == "validation_step"]
    if len(input_rows) != summary["num_input_files"]:
        raise SystemExit("CSR Transformer training package input row mismatch")
    if len(expected_rows) != summary["num_expected_outputs"]:
        raise SystemExit("CSR Transformer training package output row mismatch")
    if len(validation_rows) != summary["num_validation_steps"]:
        raise SystemExit("CSR Transformer training package validation row mismatch")
    if any(row["exists"] is not True for row in input_rows):
        raise SystemExit("CSR Transformer training package input missing")
    if any(not row["sha256"] for row in input_rows):
        raise SystemExit("CSR Transformer training package input checksum missing")
    forbidden = ("docs/toms_paper/", "MILESTONE_LOG", "SCOPE_PLATFORM", "ADVICE")
    if any(str(row["path"]).startswith(forbidden) for row in input_rows):
        raise SystemExit("CSR Transformer training package references internal docs")
    step_ids = {row["step_id"] for row in validation_rows}
    if step_ids != {
        "saved_model_replay",
        "quality_gate",
        "policy_model_artifact",
        "policy_model_acceptance",
        "policy_model_submission",
        "guarded_shadow_smoke",
    }:
        raise SystemExit("CSR Transformer training package validation steps mismatch")
    if "scripts/tss_csr_transformer_model_replay.py" not in readme:
        raise SystemExit("CSR Transformer training package readme missing replay step")
    if package["runtime_boundary"]["guard_required"] is not True:
        raise SystemExit("CSR Transformer training package manifest guard mismatch")
    if package["contribution_terms"]["commercial_use_requires_permission"] is not True:
        raise SystemExit("CSR Transformer training package terms mismatch")
    if package["contribution_terms"]["attribution"] != "Wei CUI":
        raise SystemExit("CSR Transformer training package attribution mismatch")
    if manifest.metadata["package_ready"] is not True:
        raise SystemExit("CSR Transformer training package manifest ready mismatch")
    if manifest.metadata["runtime_selector_changed"] is not False:
        raise SystemExit("CSR Transformer training package manifest runtime mismatch")
