import pytest

from transsolvestack.policies.transformer_contract import (
    TransformerPolicyPrediction,
    TransformerPolicyRequest,
    validate_transformer_prediction,
)


def test_transformer_prediction_contract_accepts_complete_ranking():
    request = TransformerPolicyRequest(
        schema_version="phase1_policy_features_v1",
        system_id="s",
        context_id="ctx",
        candidate_ids=("a", "b"),
    )
    prediction = TransformerPolicyPrediction(
        schema_version="phase1_policy_features_v1",
        system_id="s",
        context_id="ctx",
        ranked_candidate_ids=("b", "a"),
        scores={"a": 0.2, "b": 0.8},
        model_id="unit-test",
    )
    validate_transformer_prediction(request, prediction)


def test_transformer_prediction_contract_rejects_partial_ranking():
    request = TransformerPolicyRequest(
        schema_version="phase1_policy_features_v1",
        system_id="s",
        context_id="ctx",
        candidate_ids=("a", "b"),
    )
    prediction = TransformerPolicyPrediction(
        schema_version="phase1_policy_features_v1",
        system_id="s",
        context_id="ctx",
        ranked_candidate_ids=("a",),
        scores={"a": 1.0},
        model_id="unit-test",
    )
    with pytest.raises(ValueError):
        validate_transformer_prediction(request, prediction)
