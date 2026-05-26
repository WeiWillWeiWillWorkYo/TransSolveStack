from transsolvestack.policies.csr_selector_model_eval import (
    evaluate_csr_selector_models_from_files,
)


def test_csr_selector_model_eval_blocks_weak_ranker_from_runtime():
    export = evaluate_csr_selector_models_from_files(
        "runs/phase1_csr_learning_readiness/csr_learning_summary.json",
        "runs/phase1_csr_learning_readiness/csr_baseline_predictions.jsonl",
        "runs/phase1_csr_linear_ranker/csr_linear_ranker_summary.json",
        "runs/phase1_csr_linear_ranker/csr_linear_ranker_predictions.jsonl",
    )

    summary = export.summary
    assert summary.status == "passed"
    assert summary.schema_version == "phase1_csr_selector_model_eval_v1"
    assert summary.evaluation_id == "csr_selector_model_quality_gate_v1"
    assert summary.baseline_model_id == "candidate_prior_success_median_v1"
    assert summary.challenger_model_id == "csr_pairwise_linear_ranker_v1"
    assert summary.best_offline_model_id == "candidate_prior_success_median_v1"
    assert summary.runtime_selected_model_id is None
    assert summary.runtime_selector_changed is False
    assert summary.num_models == 2
    assert summary.num_eval_predictions == 3
    assert summary.num_eval_oracle_requests == 2
    assert summary.min_required_eval_oracle_requests == 10
    assert summary.challenger_beats_baseline is False
    assert summary.challenger_runtime_eligible is False
    assert set(summary.challenger_gate_failures) == {
        "insufficient_eval_oracle_requests:2<10",
        "below_min_oracle_top1",
        "below_min_profiled_success_rate",
        "below_baseline_oracle_top1",
        "below_baseline_profiled_success_rate",
        "non_success_eval_selections",
    }


def test_csr_selector_model_eval_rows_compare_baseline_and_ranker():
    export = evaluate_csr_selector_models_from_files(
        "runs/phase1_csr_learning_readiness/csr_learning_summary.json",
        "runs/phase1_csr_learning_readiness/csr_baseline_predictions.jsonl",
        "runs/phase1_csr_linear_ranker/csr_linear_ranker_summary.json",
        "runs/phase1_csr_linear_ranker/csr_linear_ranker_predictions.jsonl",
    )

    rows = {row.comparison_role: row for row in export.rows}
    assert set(rows) == {"baseline", "challenger"}
    baseline = rows["baseline"]
    challenger = rows["challenger"]
    assert baseline.model_trained is False
    assert challenger.model_trained is True
    assert baseline.runtime_eligible is False
    assert challenger.runtime_eligible is False
    assert baseline.eval_oracle_top1_accuracy == 1.0 / 3.0
    assert baseline.eval_profiled_success_selection_rate == 2.0 / 3.0
    assert baseline.eval_non_success_selection_count == 1
    assert challenger.eval_oracle_top1_accuracy == 0.0
    assert challenger.eval_profiled_success_selection_rate == 0.0
    assert challenger.eval_non_success_selection_count == 3
    assert export.schema["integration_boundary"]["status"] == "offline_quality_gate_only"
