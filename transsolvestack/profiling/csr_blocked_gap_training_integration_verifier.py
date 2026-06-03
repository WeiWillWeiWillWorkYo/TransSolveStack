"""Artifact verifier for D19 blocked-gap training integration."""

from __future__ import annotations

import json
from pathlib import Path

from transsolvestack.profiling.artifacts import read_jsonl
from transsolvestack.profiling.provenance import read_manifest, verify_manifest_hashes


def verify_csr_blocked_gap_training_integration(path: Path) -> None:
    manifest = read_manifest(path / "artifact_manifest.json")
    stale = verify_manifest_hashes(manifest)
    if stale:
        raise SystemExit(f"stale CSR blocked gap training integration manifest: {stale}")
    selector_rows = read_jsonl(path / "combined_csr_selector_rows.jsonl")
    requests = read_jsonl(path / "csr_blocked_gap_training_model_requests.jsonl")
    targets = read_jsonl(path / "csr_blocked_gap_training_model_targets.jsonl")
    request_index = read_jsonl(path / "csr_blocked_gap_training_request_index.jsonl")
    positive_membership = read_jsonl(
        path / "csr_blocked_gap_training_positive_membership.jsonl"
    )
    tensor_summary = json.loads(
        (path / "csr_blocked_gap_training_tensor_summary.json").read_text(
            encoding="utf-8"
        )
    )
    summary = json.loads(
        (path / "csr_blocked_gap_training_integration_summary.json").read_text(
            encoding="utf-8"
        )
    )
    schema = json.loads(
        (path / "csr_blocked_gap_training_integration_schema.json").read_text(
            encoding="utf-8"
        )
    )
    if manifest.artifact_kind != "csr_blocked_gap_training_integration":
        raise SystemExit("unexpected CSR blocked gap training integration artifact kind")
    if summary["status"] != "passed":
        raise SystemExit("CSR blocked gap training integration did not pass")
    if summary["schema_version"] != "phase1_csr_blocked_gap_training_integration_v1":
        raise SystemExit("CSR blocked gap training integration schema mismatch")
    if schema["schema_version"] != summary["schema_version"]:
        raise SystemExit("CSR blocked gap training integration schema/summary mismatch")
    boundary = schema["runtime_boundary"]
    if (
        boundary["runtime_selector_changed"] is not False
        or boundary["generic_queue_merge"] is not False
    ):
        raise SystemExit("CSR blocked gap training integration runtime boundary mismatch")
    if (
        summary["runtime_selector_changed"] is not False
        or summary["generic_queue_merge"] is not False
        or summary["executes_gpu"] is not False
        or summary["imports_matrices"] is not False
    ):
        raise SystemExit("CSR blocked gap training integration changed runtime boundary")
    if summary["training_only_integration"] is not True:
        raise SystemExit("CSR blocked gap training integration training-only mismatch")
    if summary["base_selector_rows"] != 372 or summary["positive_selector_rows"] != 8:
        raise SystemExit("CSR blocked gap training integration source row mismatch")
    if summary["integrated_selector_rows"] != len(selector_rows) or len(selector_rows) != 380:
        raise SystemExit("CSR blocked gap training integration selector count mismatch")
    if summary["integrated_model_requests"] != len(requests) or len(requests) != 105:
        raise SystemExit("CSR blocked gap training integration request count mismatch")
    if len(targets) != len(requests) or len(request_index) != len(requests):
        raise SystemExit("CSR blocked gap training integration target/index mismatch")
    if summary["model_request_delta"] != 5 or summary["duplicate_key_count"] != 0:
        raise SystemExit("CSR blocked gap training integration request/dedupe mismatch")
    if summary["positive_success_rows"] != 8 or summary["positive_oracle_rows"] != 5:
        raise SystemExit("CSR blocked gap training integration positive row mismatch")
    if summary["integrated_success_rows"] != 79 or summary["integrated_oracle_rows"] != 37:
        raise SystemExit("CSR blocked gap training integration integrated label mismatch")
    required_gaps = {
        "general_bicgstab_ilu0",
        "general_bicgstab_row_column_equilibration",
        "symmetric_chebyshev_jacobi",
        "symmetric_pcg_symmetric_equilibration",
    }
    if set(summary["positive_gap_ids"]) != required_gaps:
        raise SystemExit("CSR blocked gap training integration gap set mismatch")
    if set(summary["new_global_candidate_ids"]) != {
        "taichi_csr_bicgstab_ilu0_float64",
        "taichi_csr_bicgstab_row_column_equilibration_float64",
        "taichi_csr_pcg_symmetric_equilibration_float64",
    }:
        raise SystemExit("CSR blocked gap training integration new candidate mismatch")
    if summary["integrated_global_candidates"] != 12:
        raise SystemExit("CSR blocked gap training integration global candidate mismatch")
    positive_rows = [
        row
        for row in selector_rows
        if row["context_id"] == "phase1_csr_blocked_gap_positive_search"
    ]
    if len(positive_rows) != len(positive_membership) or len(positive_rows) != 8:
        raise SystemExit("CSR blocked gap training integration positive membership mismatch")
    if {row["target_status"] for row in positive_rows} != {"success"}:
        raise SystemExit("CSR blocked gap training integration positive rows not success")
    if (
        tensor_summary["status"] != "passed"
        or tensor_summary["runtime_selector_changed"] is not False
        or tensor_summary["num_requests"] != summary["integrated_tensor_requests"]
    ):
        raise SystemExit("CSR blocked gap training tensor summary mismatch")
    if manifest.metadata["integrated_selector_rows"] != summary["integrated_selector_rows"]:
        raise SystemExit("CSR blocked gap training integration manifest selector mismatch")
    if (
        manifest.metadata["runtime_selector_changed"] is not False
        or manifest.metadata["generic_queue_merge"] is not False
    ):
        raise SystemExit("CSR blocked gap training integration manifest boundary mismatch")
