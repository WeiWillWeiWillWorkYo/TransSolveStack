"""Artifact verifier for D21 blocked-gap guarded replay."""

from __future__ import annotations

import json
import math
from pathlib import Path

from transsolvestack.profiling.artifacts import read_jsonl
from transsolvestack.profiling.provenance import read_manifest, verify_manifest_hashes


def verify_csr_blocked_gap_guarded_replay(path: Path) -> None:
    manifest = read_manifest(path / "artifact_manifest.json")
    stale = verify_manifest_hashes(manifest)
    if stale:
        raise SystemExit(f"stale CSR blocked gap guarded replay manifest: {stale}")
    rows = read_jsonl(path / "csr_blocked_gap_guarded_replay_rows.jsonl")
    summary = json.loads(
        (path / "csr_blocked_gap_guarded_replay_summary.json").read_text(
            encoding="utf-8"
        )
    )
    schema = json.loads(
        (path / "csr_blocked_gap_guarded_replay_schema.json").read_text(
            encoding="utf-8"
        )
    )
    blocked_gate = json.loads(
        (path / "csr_blocked_gap_guarded_replay_blocked_quality_gate.json").read_text(
            encoding="utf-8"
        )
    )
    counterfactual_gate = json.loads(
        (
            path
            / "csr_blocked_gap_guarded_replay_counterfactual_quality_gate.json"
        ).read_text(encoding="utf-8")
    )
    if manifest.artifact_kind != "csr_blocked_gap_guarded_replay":
        raise SystemExit("unexpected CSR blocked gap guarded replay artifact kind")
    if summary["status"] != "passed":
        raise SystemExit("CSR blocked gap guarded replay did not pass")
    if summary["schema_version"] != "phase1_csr_blocked_gap_guarded_replay_v1":
        raise SystemExit("CSR blocked gap guarded replay schema mismatch")
    if schema["schema_version"] != summary["schema_version"]:
        raise SystemExit("CSR blocked gap guarded replay schema/summary mismatch")
    if (
        schema["runtime_boundary"]["runtime_selector_changed"] is not False
        or schema["runtime_boundary"]["shadow_only"] is not True
        or schema["runtime_boundary"]["executes_gpu"] is not False
    ):
        raise SystemExit("CSR blocked gap guarded replay runtime boundary mismatch")
    if (
        summary["runtime_selector_changed"] is not False
        or summary["shadow_only"] is not True
        or summary["executes_gpu"] is not False
    ):
        raise SystemExit("CSR blocked gap guarded replay changed runtime boundary")
    if len(rows) != summary["num_replay_rows"] or len(rows) != 5:
        raise SystemExit("CSR blocked gap guarded replay row count mismatch")
    if summary["shadow_only_decisions"] != 5:
        raise SystemExit("CSR blocked gap guarded replay shadow count mismatch")
    if summary["actual_quality_gate_blocks"] != 5:
        raise SystemExit("CSR blocked gap guarded replay quality gate block mismatch")
    if summary["actual_runtime_selector_changes"] != 0:
        raise SystemExit("CSR blocked gap guarded replay promoted actual runtime")
    if summary["actual_artifact_runtime_selections"] != 5:
        raise SystemExit("CSR blocked gap guarded replay artifact fallback mismatch")
    if summary["fallback_chain_enforced_rows"] != 5:
        raise SystemExit("CSR blocked gap guarded replay fallback enforcement mismatch")
    if summary["learned_success_predictions"] != 5:
        raise SystemExit("CSR blocked gap guarded replay learned success mismatch")
    if summary["learned_oracle_predictions"] != 4:
        raise SystemExit("CSR blocked gap guarded replay oracle count mismatch")
    if summary["learned_profiled_success_non_oracle_predictions"] != 1:
        raise SystemExit("CSR blocked gap guarded replay non-oracle count mismatch")
    if summary["learned_differs_from_artifact_rows"] != 1:
        raise SystemExit("CSR blocked gap guarded replay artifact delta mismatch")
    if summary["counterfactual_only"] is not True:
        raise SystemExit("CSR blocked gap guarded replay counterfactual flag mismatch")
    if summary["counterfactual_eligible_gate_promotions"] != 5:
        raise SystemExit("CSR blocked gap guarded replay counterfactual promotion mismatch")
    if summary["counterfactual_runtime_selector_changes"] != 5:
        raise SystemExit("CSR blocked gap guarded replay counterfactual runtime mismatch")
    if not math.isclose(
        float(summary["max_learned_regret_vs_artifact_ms"]),
        1098.03906083107,
        rel_tol=0.0,
        abs_tol=1.0e-9,
    ):
        raise SystemExit("CSR blocked gap guarded replay regret mismatch")
    if blocked_gate["challenger_runtime_eligible"] is not False:
        raise SystemExit("CSR blocked gap guarded replay blocked gate mismatch")
    if counterfactual_gate["counterfactual_only"] is not True:
        raise SystemExit("CSR blocked gap guarded replay counterfactual gate mismatch")
    if any(row["blocked_guard_status"] != "blocked_quality_gate" for row in rows):
        raise SystemExit("CSR blocked gap guarded replay row not blocked")
    if any(row["blocked_runtime_selector_changed"] is not False for row in rows):
        raise SystemExit("CSR blocked gap guarded replay row changed runtime")
    if any(row["blocked_runtime_selection_source"] != "artifact" for row in rows):
        raise SystemExit("CSR blocked gap guarded replay row not artifact-backed")
    if any(row["blocked_fallback_chain_enforced"] is not True for row in rows):
        raise SystemExit("CSR blocked gap guarded replay row fallback not enforced")
    if any(row["counterfactual_only"] is not True for row in rows):
        raise SystemExit("CSR blocked gap guarded replay row counterfactual flag mismatch")
    if manifest.metadata["runtime_selector_changed"] is not False:
        raise SystemExit("CSR blocked gap guarded replay manifest runtime mismatch")
