from transsolvestack.policies.csr_transformer_model_replay import (
    build_csr_transformer_model_replay_from_files,
)


def test_csr_transformer_model_replay_loads_saved_model_without_training():
    export = build_csr_transformer_model_replay_from_files()
    summary = export.summary

    assert summary.status == "passed"
    assert summary.schema_version == "phase1_csr_transformer_model_replay_v1"
    assert summary.source_model_schema_version == "phase1_csr_transformer_ranker_v1"
    assert summary.model_family == "masked_self_attention_ranker_v1"
    assert summary.model_id == "csr_masked_self_attention_ranker_v1"
    assert summary.model_loaded is True
    assert summary.model_trained is True
    assert summary.runtime_selector_changed is False
    assert summary.num_predictions == 20
    assert summary.num_reference_predictions == 20
    assert summary.exact_replay is True
    assert summary.selected_candidate_mismatch_count == 0
    assert summary.ranked_order_mismatch_count == 0
    assert summary.evaluation_status_mismatch_count == 0
    assert summary.missing_reference_count == 0
    assert summary.max_abs_score_delta <= 1.0e-9
    assert summary.prediction_contract_valid is True

    assert len(export.predictions) == 20
    assert len(export.comparison_rows) == 20
    assert all(row["selected_candidate_match"] for row in export.comparison_rows)
    assert all(row["ranked_order_match"] for row in export.comparison_rows)
    assert all(row["evaluation_status_match"] for row in export.comparison_rows)
    assert export.schema["runtime_integration"]["guard_required_before_execution"] is True

