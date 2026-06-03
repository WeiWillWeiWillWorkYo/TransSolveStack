"""Artifact verifier for D22 CSR Transformer handoff bundle."""

from __future__ import annotations

import json
from pathlib import Path

from transsolvestack.profiling.artifacts import read_jsonl
from transsolvestack.profiling.provenance import read_manifest, verify_manifest_hashes


def verify_csr_transformer_handoff_bundle(path: Path) -> None:
    manifest = read_manifest(path / "artifact_manifest.json")
    stale = verify_manifest_hashes(manifest)
    if stale:
        raise SystemExit(f"stale CSR Transformer handoff bundle manifest: {stale}")
    rows = read_jsonl(path / "csr_transformer_handoff_bundle_rows.jsonl")
    summary = json.loads(
        (path / "csr_transformer_handoff_bundle_summary.json").read_text(
            encoding="utf-8"
        )
    )
    schema = json.loads(
        (path / "csr_transformer_handoff_bundle_schema.json").read_text(
            encoding="utf-8"
        )
    )
    training_spec = json.loads(
        (path / "csr_transformer_handoff_training_spec.json").read_text(
            encoding="utf-8"
        )
    )
    guard_contract = json.loads(
        (path / "csr_transformer_handoff_guard_contract.json").read_text(
            encoding="utf-8"
        )
    )
    if manifest.artifact_kind != "csr_transformer_handoff_bundle":
        raise SystemExit("unexpected CSR Transformer handoff artifact kind")
    if summary["status"] != "passed" or summary["handoff_ready"] is not True:
        raise SystemExit("CSR Transformer handoff bundle did not pass")
    if summary["schema_version"] != "phase1_csr_transformer_handoff_bundle_v1":
        raise SystemExit("CSR Transformer handoff schema mismatch")
    if schema["schema_version"] != summary["schema_version"]:
        raise SystemExit("CSR Transformer handoff schema/summary mismatch")
    if (
        schema["runtime_boundary"]["runtime_selector_changed"] is not False
        or schema["runtime_boundary"]["executes_gpu"] is not False
        or schema["runtime_boundary"]["model_trained"] is not False
        or schema["runtime_boundary"]["shadow_only"] is not True
    ):
        raise SystemExit("CSR Transformer handoff runtime boundary mismatch")
    if (
        summary["runtime_selector_changed"] is not False
        or summary["executes_gpu"] is not False
        or summary["model_trained"] is not False
        or summary["model_training_required"] is not True
    ):
        raise SystemExit("CSR Transformer handoff changed runtime/model boundary")
    if summary["training_model_requests"] != 105:
        raise SystemExit("CSR Transformer handoff request count mismatch")
    if summary["training_tensor_requests"] != 105:
        raise SystemExit("CSR Transformer handoff tensor count mismatch")
    if summary["training_global_candidates"] != 12:
        raise SystemExit("CSR Transformer handoff candidate count mismatch")
    if summary["positive_selector_rows"] != 8:
        raise SystemExit("CSR Transformer handoff positive selector mismatch")
    if summary["positive_success_rows"] != 8:
        raise SystemExit("CSR Transformer handoff positive success mismatch")
    if summary["ranker_eval_non_success_selection_count"] != 23:
        raise SystemExit("CSR Transformer handoff ranker safety mismatch")
    if summary["blocked_gap_success_predictions"] != 5:
        raise SystemExit("CSR Transformer handoff blocked-gap success mismatch")
    if summary["guard_actual_quality_gate_blocks"] != 5:
        raise SystemExit("CSR Transformer handoff guard block mismatch")
    if summary["guard_actual_runtime_selector_changes"] != 0:
        raise SystemExit("CSR Transformer handoff guard runtime mismatch")
    if summary["guard_fallback_chain_enforced_rows"] != 5:
        raise SystemExit("CSR Transformer handoff fallback chain mismatch")
    if summary["submission_shadow_ready"] is not True:
        raise SystemExit("CSR Transformer handoff submission shadow mismatch")
    if summary["submission_runtime_promotion_ready"] is not False:
        raise SystemExit("CSR Transformer handoff submission runtime mismatch")
    if summary["validation_error_count"] != 0:
        raise SystemExit("CSR Transformer handoff validation errors present")
    source_rows = [row for row in rows if row["row_kind"] == "source_file"]
    if len(source_rows) != summary["num_source_files"]:
        raise SystemExit("CSR Transformer handoff source row count mismatch")
    if not source_rows or any(row["exists"] is not True for row in source_rows):
        raise SystemExit("CSR Transformer handoff missing source file")
    if any(not row["sha256"] for row in source_rows):
        raise SystemExit("CSR Transformer handoff missing source checksum")
    row_kinds = {row["row_kind"] for row in rows}
    if not {
        "handoff_summary",
        "training_tensor_contract",
        "shadow_baseline_contract",
        "guard_contract",
        "submission_contract",
        "source_file",
    } <= row_kinds:
        raise SystemExit("CSR Transformer handoff rows missing required kinds")
    if training_spec["runtime_selector_changed"] is not False:
        raise SystemExit("CSR Transformer handoff training spec changed runtime")
    if training_spec["model_trained_by_this_artifact"] is not False:
        raise SystemExit("CSR Transformer handoff training spec trained model")
    if "saved_model_schema_validation" not in training_spec["required_post_training_checks"]:
        raise SystemExit("CSR Transformer handoff training checks incomplete")
    if guard_contract["guard_required"] is not True:
        raise SystemExit("CSR Transformer handoff guard not required")
    if guard_contract["current_shadow_baseline"]["runtime_eligible"] is not False:
        raise SystemExit("CSR Transformer handoff current baseline incorrectly eligible")
    if guard_contract["actual_guard_replay"]["runtime_selector_changes"] != 0:
        raise SystemExit("CSR Transformer handoff guard contract runtime mismatch")
    if manifest.metadata["runtime_selector_changed"] is not False:
        raise SystemExit("CSR Transformer handoff manifest runtime mismatch")
