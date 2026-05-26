import math

from transsolvestack.policies.csr_linear_ranker import (
    train_csr_linear_ranker_from_tensor_file,
)


def test_csr_linear_ranker_trains_pairwise_model_from_tensors():
    export = train_csr_linear_ranker_from_tensor_file(
        "runs/phase1_csr_training_tensors/csr_training_tensors.json",
        "runs/phase1_csr_training_tensors/csr_training_request_index.jsonl",
    )

    summary = export.summary
    assert summary.status == "passed"
    assert summary.schema_version == "phase1_csr_linear_ranker_v1"
    assert summary.model_family == "pairwise_linear_ranker_v1"
    assert summary.model_trained is True
    assert summary.runtime_selector_changed is False
    assert summary.num_requests == 12
    assert summary.num_predictions == 12
    assert summary.num_train_requests == 9
    assert summary.num_eval_requests == 3
    assert summary.num_train_oracle_requests == 2
    assert summary.num_eval_oracle_requests == 2
    assert summary.num_global_candidates == 9
    assert summary.num_features == 374
    assert summary.num_pairwise_constraints == 110
    assert summary.num_pairwise_updates == 148
    assert math.isfinite(summary.final_train_pairwise_loss)
    assert summary.train_oracle_top1_accuracy == 0.5
    assert summary.eval_oracle_top1_accuracy == 0.0
    assert summary.eval_profiled_success_selection_rate == 0.0
    assert summary.eval_non_success_selection_count == 3
    assert summary.eval_mean_regret_ms is None
    assert summary.eval_max_regret_ms is None


def test_csr_linear_ranker_model_and_predictions_are_complete():
    export = train_csr_linear_ranker_from_tensor_file(
        "runs/phase1_csr_training_tensors/csr_training_tensors.json",
        "runs/phase1_csr_training_tensors/csr_training_request_index.jsonl",
    )

    assert export.schema["model_required"] is True
    assert export.schema["runtime_selector_changed"] is False
    assert (
        export.schema["training_objective"]
        == "label_utility_pairwise_hinge_with_oracle_priority"
    )
    assert len(export.model["feature_names"]) == export.summary.num_features
    assert len(export.model["weights"]) == export.summary.num_features
    assert len(export.model["normalization"]["mean"]) == export.summary.num_features
    assert len(export.model["normalization"]["scale"]) == export.summary.num_features
    assert set(export.model["weights"]) == set(export.model["feature_names"])
    assert all(math.isfinite(float(value)) for value in export.model["weights"].values())
    assert all(
        math.isfinite(float(value)) and float(value) > 0.0
        for value in export.model["normalization"]["scale"].values()
    )
    assert export.model["runtime_integration"]["runtime_selector_changed"] is False

    statuses = {row.evaluation_status for row in export.predictions}
    assert statuses == {
        "oracle_match",
        "non_success_selected",
        "no_oracle_target",
        "profiled_success_non_oracle",
    }
    assert len(export.predictions) == 12
    for row in export.predictions:
        ranked = row.ranked_candidate_ids
        assert len(ranked) == 9
        assert len(row.scores) == 9
        assert set(ranked) == set(row.scores)
        assert ranked[0] == row.selected_candidate_id
        assert row.scores[ranked[0]] == row.selected_score
        assert all(
            row.scores[left] >= row.scores[right]
            for left, right in zip(ranked, ranked[1:])
        )
