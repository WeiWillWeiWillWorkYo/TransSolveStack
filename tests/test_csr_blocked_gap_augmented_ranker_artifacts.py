import json
import math
from pathlib import Path

from transsolvestack.profiling.artifacts import read_jsonl
from transsolvestack.profiling.provenance import read_manifest, verify_manifest_hashes


def test_csr_blocked_gap_augmented_ranker_artifacts_are_valid():
    root = Path("runs/phase1_csr_blocked_gap_augmented_ranker")
    predictions = read_jsonl(root / "csr_blocked_gap_augmented_ranker_predictions.jsonl")
    positive_predictions = read_jsonl(
        root / "csr_blocked_gap_augmented_ranker_positive_predictions.jsonl"
    )
    summary = json.loads(
        (root / "csr_blocked_gap_augmented_ranker_summary.json").read_text(
            encoding="utf-8"
        )
    )
    schema = json.loads(
        (root / "csr_blocked_gap_augmented_ranker_schema.json").read_text(
            encoding="utf-8"
        )
    )
    model = json.loads(
        (root / "csr_blocked_gap_augmented_ranker_model.json").read_text(
            encoding="utf-8"
        )
    )
    manifest = read_manifest(root / "artifact_manifest.json")

    assert manifest.artifact_kind == "csr_blocked_gap_augmented_ranker"
    assert verify_manifest_hashes(manifest) == ()
    assert summary["status"] == "passed"
    assert summary["schema_version"] == "phase1_csr_blocked_gap_augmented_ranker_v1"
    assert schema["schema_version"] == summary["schema_version"]
    assert schema["runtime_boundary"]["runtime_selector_changed"] is False
    assert schema["runtime_boundary"]["shadow_only"] is True
    assert summary["runtime_selector_changed"] is False
    assert summary["shadow_only"] is True
    assert summary["model_trained"] is True
    assert model["runtime_integration"]["runtime_selector_changed"] is False

    assert len(predictions) == summary["ranker_num_predictions"] == 105
    assert summary["ranker_num_requests"] == 105
    assert summary["ranker_num_train_requests"] == 79
    assert summary["ranker_num_eval_requests"] == 26
    assert summary["ranker_num_global_candidates"] == 12
    assert summary["ranker_token_feature_dim"] == 40
    assert summary["ranker_scorer_feature_dim"] == len(model["scorer_head"]) == 65
    assert summary["ranker_num_epochs"] == 160
    assert summary["ranker_num_pairwise_constraints"] == 157
    assert summary["ranker_num_pairwise_updates"] == 6981
    assert math.isfinite(summary["ranker_final_train_pairwise_loss"])

    assert summary["source_positive_selector_rows"] == 8
    assert summary["source_positive_success_rows"] == 8
    assert summary["source_positive_oracle_rows"] == 5
    assert summary["source_positive_matrix_contexts"] == 5
    assert set(summary["source_positive_candidate_ids"]) == {
        "taichi_csr_bicgstab_ilu0_float64",
        "taichi_csr_bicgstab_row_column_equilibration_float64",
        "taichi_csr_chebyshev_jacobi_float64",
        "taichi_csr_pcg_symmetric_equilibration_float64",
    }
    assert set(summary["source_new_global_candidate_ids"]) == {
        "taichi_csr_bicgstab_ilu0_float64",
        "taichi_csr_bicgstab_row_column_equilibration_float64",
        "taichi_csr_pcg_symmetric_equilibration_float64",
    }

    assert len(positive_predictions) == summary["blocked_gap_prediction_rows"] == 5
    assert summary["blocked_gap_train_predictions"] == 4
    assert summary["blocked_gap_eval_predictions"] == 1
    assert summary["blocked_gap_success_predictions"] == 5
    assert summary["blocked_gap_non_success_predictions"] == 0
    assert summary["blocked_gap_oracle_matches"] == 4
    assert summary["blocked_gap_profiled_success_non_oracle"] == 1
    assert summary["blocked_gap_eval_success_selection_rate"] == 1.0
    assert summary["blocked_gap_selected_status_counts"] == {"success": 5}
    assert summary["blocked_gap_evaluation_status_counts"] == {
        "oracle_match": 4,
        "profiled_success_non_oracle": 1,
    }
    assert summary["blocked_gap_split_counts"] == {"eval": 1, "train": 4}
    assert summary["blocked_gap_positive_candidate_coverage_complete"] is True
    assert set(summary["blocked_gap_ranked_positive_candidate_ids"]) == set(
        summary["source_positive_candidate_ids"]
    )
    assert set(summary["blocked_gap_ranked_new_candidate_ids"]) == set(
        summary["source_new_global_candidate_ids"]
    )
    assert {row["selected_target_status"] for row in positive_predictions} == {
        "success"
    }
    assert all(row["selected_is_positive_candidate"] for row in positive_predictions)
    assert all(row["selected_is_profiled_success"] for row in positive_predictions)
    assert sum(row["selected_is_oracle"] for row in positive_predictions) == 4

    assert summary["overall_eval_is_not_acceptance_gate"] is True
    assert manifest.metadata["shadow_only"] is True
    assert manifest.metadata["runtime_selector_changed"] is False
    assert manifest.metadata["blocked_gap_success_predictions"] == 5
    assert manifest.metadata["blocked_gap_non_success_predictions"] == 0
