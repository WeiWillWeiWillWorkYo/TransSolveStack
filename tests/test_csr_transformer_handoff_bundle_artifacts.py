import json
import math
from pathlib import Path

from transsolvestack.profiling.artifacts import read_jsonl
from transsolvestack.profiling.provenance import read_manifest, verify_manifest_hashes


def test_csr_transformer_handoff_bundle_artifacts_are_valid():
    root = Path("runs/phase1_csr_transformer_handoff_bundle")
    rows = read_jsonl(root / "csr_transformer_handoff_bundle_rows.jsonl")
    summary = json.loads(
        (root / "csr_transformer_handoff_bundle_summary.json").read_text(
            encoding="utf-8"
        )
    )
    schema = json.loads(
        (root / "csr_transformer_handoff_bundle_schema.json").read_text(
            encoding="utf-8"
        )
    )
    training_spec = json.loads(
        (root / "csr_transformer_handoff_training_spec.json").read_text(
            encoding="utf-8"
        )
    )
    guard_contract = json.loads(
        (root / "csr_transformer_handoff_guard_contract.json").read_text(
            encoding="utf-8"
        )
    )
    manifest = read_manifest(root / "artifact_manifest.json")

    assert manifest.artifact_kind == "csr_transformer_handoff_bundle"
    assert verify_manifest_hashes(manifest) == ()
    assert summary["status"] == "passed"
    assert summary["schema_version"] == "phase1_csr_transformer_handoff_bundle_v1"
    assert summary["handoff_ready"] is True
    assert schema["schema_version"] == summary["schema_version"]
    assert schema["handoff_ready"] is True
    assert schema["runtime_boundary"]["runtime_selector_changed"] is False
    assert schema["runtime_boundary"]["executes_gpu"] is False
    assert schema["runtime_boundary"]["model_trained"] is False
    assert schema["runtime_boundary"]["shadow_only"] is True

    assert summary["model_training_required"] is True
    assert summary["model_trained"] is False
    assert summary["runtime_selector_changed"] is False
    assert summary["executes_gpu"] is False
    assert summary["shadow_only"] is True
    assert summary["validation_error_count"] == 0
    assert summary["all_source_files_present"] is True
    assert summary["num_source_files"] == 24
    assert summary["source_file_total_size_bytes"] > 1_000_000

    assert summary["training_selector_rows"] == 380
    assert summary["training_model_requests"] == 105
    assert summary["training_tensor_requests"] == 105
    assert summary["training_train_requests"] == 79
    assert summary["training_eval_requests"] == 26
    assert summary["training_global_candidates"] == 12
    assert summary["training_success_rows"] == 79
    assert summary["training_oracle_rows"] == 37
    assert summary["positive_selector_rows"] == 8
    assert summary["positive_success_rows"] == 8
    assert summary["positive_oracle_rows"] == 5
    assert summary["matrix_feature_dim"] == 21
    assert summary["candidate_feature_dim"] == 19
    assert set(summary["new_global_candidate_ids"]) == {
        "taichi_csr_bicgstab_ilu0_float64",
        "taichi_csr_bicgstab_row_column_equilibration_float64",
        "taichi_csr_pcg_symmetric_equilibration_float64",
    }

    assert summary["ranker_model_id"] == "csr_blocked_gap_augmented_shadow_ranker_v1"
    assert summary["ranker_num_requests"] == 105
    assert summary["ranker_eval_oracle_top1_accuracy"] == 0.4
    assert summary["ranker_eval_profiled_success_selection_rate"] == (
        0.11538461538461539
    )
    assert summary["ranker_eval_non_success_selection_count"] == 23
    assert summary["blocked_gap_success_predictions"] == 5
    assert summary["blocked_gap_non_success_predictions"] == 0
    assert summary["blocked_gap_oracle_matches"] == 4
    assert summary["blocked_gap_positive_candidate_coverage_complete"] is True

    assert summary["guard_actual_quality_gate_blocks"] == 5
    assert summary["guard_actual_runtime_selector_changes"] == 0
    assert summary["guard_fallback_chain_enforced_rows"] == 5
    assert math.isclose(
        summary["guard_max_learned_regret_vs_artifact_ms"],
        1098.03906083107,
        rel_tol=0.0,
        abs_tol=1.0e-9,
    )
    assert summary["counterfactual_only"] is True
    assert summary["submission_contract_ready"] is True
    assert summary["submission_shadow_ready"] is True
    assert summary["submission_runtime_promotion_ready"] is False
    assert set(summary["terms_documents_required"]) == {
        "MODEL_CONTRIBUTION_TERMS.md",
        "CONTRIBUTOR_LICENSE_AGREEMENT.md",
    }

    row_kinds = {row["row_kind"] for row in rows}
    assert {
        "handoff_summary",
        "training_tensor_contract",
        "shadow_baseline_contract",
        "guard_contract",
        "submission_contract",
        "source_file",
    } <= row_kinds
    source_rows = [row for row in rows if row["row_kind"] == "source_file"]
    assert len(source_rows) == summary["num_source_files"]
    assert all(row["exists"] is True for row in source_rows)
    assert all(row["sha256"] for row in source_rows)
    assert {
        "training_input",
        "shadow_baseline",
        "runtime_safety",
        "contribution_contract",
        "source_metadata",
    } <= {row["source_kind"] for row in source_rows}

    assert training_spec["job_kind"] == "external_csr_transformer_policy_training"
    assert training_spec["runtime_selector_changed"] is False
    assert training_spec["model_trained_by_this_artifact"] is False
    assert "saved_model_schema_validation" in training_spec[
        "required_post_training_checks"
    ]
    assert "do_not_train_inside_handoff_artifact" in training_spec["non_goals"]
    assert guard_contract["guard_required"] is True
    assert guard_contract["default_mode"] == "shadow"
    assert guard_contract["current_shadow_baseline"]["runtime_eligible"] is False
    assert guard_contract["actual_guard_replay"]["runtime_selector_changes"] == 0
    assert guard_contract["counterfactual_rows_are_runtime_evidence"] is False

    assert manifest.metadata["handoff_ready"] is True
    assert manifest.metadata["model_trained"] is False
    assert manifest.metadata["runtime_selector_changed"] is False
    assert manifest.metadata["executes_gpu"] is False
