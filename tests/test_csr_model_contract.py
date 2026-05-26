import pytest

from transsolvestack.policies.csr_model_contract import (
    build_csr_model_contract_export,
    validate_csr_model_prediction,
)


def test_csr_model_contract_builds_label_free_requests():
    export = build_csr_model_contract_export(
        "runs/phase1_csr_learning_readiness/csr_learning_rows.jsonl",
        "runs/phase1_csr_learning_readiness/csr_baseline_predictions.jsonl",
    )

    assert export.summary.status == "passed"
    assert export.summary.schema_version == "phase1_csr_model_contract_v1"
    assert export.summary.num_requests == 12
    assert export.summary.num_request_candidates == 108
    assert export.summary.num_predictions == 3
    assert export.summary.model_required is False
    assert export.summary.runtime_selector_changed is False
    assert export.summary.validation_error_count == 0
    assert export.summary.eval_oracle_top1_accuracy == 1.0 / 3.0
    assert export.summary.eval_profiled_selection_rate == 2.0 / 3.0

    leakage_keys = {
        "measurement_repeats",
        "median_solve_time_ms",
        "solve_time_iqr_ms",
        "success_rate",
        "failure_reason",
        "screened_out_source",
        "skip_reason",
        "applicability_status",
        "applicability_reason",
    }
    for request in export.requests:
        assert len(request.candidate_ids) == 9
        assert set(request.candidate_ids) == set(request.candidate_features)
        for features in request.candidate_features.values():
            assert not leakage_keys & set(features)


def test_csr_model_prediction_validation_rejects_missing_score():
    export = build_csr_model_contract_export(
        "runs/phase1_csr_learning_readiness/csr_learning_rows.jsonl",
        "runs/phase1_csr_learning_readiness/csr_baseline_predictions.jsonl",
    )
    request = next(
        row for row in export.requests if row.request_id == export.predictions[0].request_id
    )
    prediction = export.predictions[0]
    scores = dict(prediction.scores)
    scores.pop(prediction.ranked_candidate_ids[-1])
    invalid = type(prediction)(**{**prediction.__dict__, "scores": scores})

    with pytest.raises(ValueError, match="missing scores"):
        validate_csr_model_prediction(request, invalid)
