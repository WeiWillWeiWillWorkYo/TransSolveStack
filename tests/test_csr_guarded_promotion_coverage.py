from transsolvestack.profiling.csr_guarded_promotion_coverage import (
    build_csr_guarded_promotion_coverage_plan_from_files,
)


def test_csr_guarded_promotion_coverage_plan_selects_safety_matrix():
    plan = build_csr_guarded_promotion_coverage_plan_from_files()
    summary = plan.summary

    assert summary.status == "passed"
    assert summary.schema_version == "phase1_csr_guarded_promotion_coverage_plan_v1"
    assert summary.runtime_selector_changed is False
    assert summary.executes_gpu is False
    assert summary.quality_gate_runtime_eligible is False
    assert set(summary.quality_gate_failures) == {
        "below_min_profiled_success_rate",
        "below_baseline_oracle_top1",
        "below_baseline_profiled_success_rate",
        "non_success_eval_selections",
    }
    assert summary.num_predictions == 20
    assert summary.profiled_success_predictions == 7
    assert summary.non_success_predictions == 13
    assert summary.exact_profiled_success_matrices == 8
    assert summary.planned_scenarios == 15
    assert summary.planned_gpu_final_solves == 11
    assert summary.planned_guard_only_scenarios == 4
    assert summary.current_blocked_scenarios == 4
    assert summary.fixture_promotion_scenarios == 4
    assert summary.fixture_promotions_changing_candidate == 3
    assert summary.non_success_block_scenarios == 4
    assert summary.runtime_fallback_scenarios == 3
    assert summary.by_scenario_kind == {
        "current_ranker_quality_gate_block": 4,
        "fixture_non_success_candidate_block": 4,
        "fixture_profiled_success_promotion": 4,
        "runtime_exception_fallback": 3,
    }
    assert summary.by_selected_target_status == {
        "not_applicable": 1,
        "not_profiled": 1,
        "screened_out": 2,
        "success": 8,
    }
    assert summary.by_runtime_solver == {
        "bicgstab": 3,
        "chebyshev": 3,
        "gmres": 4,
        "pcg": 1,
    }


def test_csr_guarded_promotion_coverage_rows_are_safe_to_execute_later():
    plan = build_csr_guarded_promotion_coverage_plan_from_files()
    rows_by_kind = {}
    for row in plan.rows:
        rows_by_kind.setdefault(row.scenario_kind, []).append(row)

    assert all(row.matrix_source_path for row in plan.rows)
    assert all(
        row.artifact_candidate_id is not None
        for row in plan.rows
        if row.requires_gpu_execution
    )
    assert all(
        row.selected_candidate_is_exact_profiled_success
        for row in rows_by_kind["fixture_profiled_success_promotion"]
    )
    assert all(
        row.requires_fixture_quality_gate and row.requires_fixture_prediction
        for row in rows_by_kind["fixture_profiled_success_promotion"]
    )
    assert all(
        row.expected_guard_status == "blocked_non_success_candidate"
        and not row.requires_gpu_execution
        for row in rows_by_kind["fixture_non_success_candidate_block"]
    )
    assert all(
        row.requires_runtime_exception_injection
        and row.expected_final_result_status == "fallback_success"
        and row.requires_gpu_execution
        for row in rows_by_kind["runtime_exception_fallback"]
    )
    assert plan.schema["integration_boundary"]["status"] == "plan_only_no_gpu_execution"
