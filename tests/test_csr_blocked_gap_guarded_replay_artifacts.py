import json
import math
from pathlib import Path

from transsolvestack.profiling.artifacts import read_jsonl
from transsolvestack.profiling.provenance import read_manifest, verify_manifest_hashes


def test_csr_blocked_gap_guarded_replay_artifacts_are_valid():
    root = Path("runs/phase1_csr_blocked_gap_guarded_replay")
    rows = read_jsonl(root / "csr_blocked_gap_guarded_replay_rows.jsonl")
    summary = json.loads(
        (root / "csr_blocked_gap_guarded_replay_summary.json").read_text(
            encoding="utf-8"
        )
    )
    schema = json.loads(
        (root / "csr_blocked_gap_guarded_replay_schema.json").read_text(
            encoding="utf-8"
        )
    )
    blocked_gate = json.loads(
        (root / "csr_blocked_gap_guarded_replay_blocked_quality_gate.json").read_text(
            encoding="utf-8"
        )
    )
    counterfactual_gate = json.loads(
        (
            root
            / "csr_blocked_gap_guarded_replay_counterfactual_quality_gate.json"
        ).read_text(encoding="utf-8")
    )
    manifest = read_manifest(root / "artifact_manifest.json")

    assert manifest.artifact_kind == "csr_blocked_gap_guarded_replay"
    assert verify_manifest_hashes(manifest) == ()
    assert summary["status"] == "passed"
    assert summary["schema_version"] == "phase1_csr_blocked_gap_guarded_replay_v1"
    assert schema["schema_version"] == summary["schema_version"]
    assert schema["runtime_boundary"]["runtime_selector_changed"] is False
    assert schema["runtime_boundary"]["shadow_only"] is True
    assert schema["runtime_boundary"]["executes_gpu"] is False
    assert summary["runtime_selector_changed"] is False
    assert summary["shadow_only"] is True
    assert summary["executes_gpu"] is False

    assert len(rows) == summary["num_replay_rows"] == 5
    assert summary["shadow_only_decisions"] == 5
    assert summary["actual_quality_gate_blocks"] == 5
    assert summary["actual_runtime_selector_changes"] == 0
    assert summary["actual_artifact_runtime_selections"] == 5
    assert summary["fallback_chain_enforced_rows"] == 5
    assert summary["learned_success_predictions"] == 5
    assert summary["learned_oracle_predictions"] == 4
    assert summary["learned_profiled_success_non_oracle_predictions"] == 1
    assert summary["learned_differs_from_artifact_rows"] == 1
    assert summary["learned_differs_from_artifact_request_ids"] == [
        "eval:suitesparse:FIDAP/ex5:phase1_csr_blocked_gap_positive_search"
    ]
    assert math.isclose(
        summary["max_learned_regret_vs_artifact_ms"],
        1098.03906083107,
        rel_tol=0.0,
        abs_tol=1.0e-9,
    )
    assert summary["counterfactual_only"] is True
    assert summary["counterfactual_eligible_gate_promotions"] == 5
    assert summary["counterfactual_runtime_selector_changes"] == 5
    assert summary["source_ranker_model_id"] == "csr_blocked_gap_augmented_shadow_ranker_v1"
    assert summary["source_ranker_eval_non_success_selection_count"] == 23
    assert summary["blocked_guard_reason_counts"] == {
        "quality_gate:below_runtime_profiled_success_rate": 5,
        "quality_gate:no_dedicated_runtime_quality_gate": 5,
        "quality_gate:overall_eval_non_success_selections": 5,
    }

    assert blocked_gate["challenger_runtime_eligible"] is False
    assert blocked_gate["runtime_selected_model_id"] is None
    assert counterfactual_gate["counterfactual_only"] is True
    assert counterfactual_gate["challenger_runtime_eligible"] is True

    assert {row["shadow_guard_status"] for row in rows} == {"shadow_only"}
    assert {row["blocked_guard_status"] for row in rows} == {"blocked_quality_gate"}
    assert {row["blocked_runtime_selection_source"] for row in rows} == {"artifact"}
    assert {row["blocked_runtime_selector_changed"] for row in rows} == {False}
    assert {row["blocked_fallback_chain_enforced"] for row in rows} == {True}
    assert {row["counterfactual_guard_status"] for row in rows} == {"promoted"}
    assert {row["counterfactual_only"] for row in rows} == {True}
    assert all(row["learned_is_profiled_success"] for row in rows)
    assert sum(row["learned_is_oracle"] for row in rows) == 4

    fidap = [
        row
        for row in rows
        if row["matrix_id"] == "suitesparse:FIDAP/ex5"
    ][0]
    assert fidap["learned_candidate_id"] == "taichi_csr_pcg_symmetric_equilibration_float64"
    assert fidap["artifact_candidate_id"] == "taichi_csr_bicgstab_ilu0_float64"
    assert fidap["learned_matches_artifact"] is False
    assert fidap["learned_is_oracle"] is False
    assert math.isclose(
        fidap["learned_regret_vs_artifact_ms"],
        1098.03906083107,
        rel_tol=0.0,
        abs_tol=1.0e-9,
    )

    assert manifest.metadata["runtime_selector_changed"] is False
    assert manifest.metadata["actual_quality_gate_blocks"] == 5
    assert manifest.metadata["counterfactual_only"] is True
