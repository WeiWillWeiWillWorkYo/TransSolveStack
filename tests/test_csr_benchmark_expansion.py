from transsolvestack.profiling.csr_benchmark_expansion import (
    build_csr_benchmark_expansion_plan_from_files,
)


def test_csr_benchmark_expansion_plan_selects_bounded_queue():
    plan = build_csr_benchmark_expansion_plan_from_files(
        "runs/phase1_suitesparse_selection/selected_matrices.jsonl",
        "runs/phase1_csr_selector_readiness/csr_selector_rows.jsonl",
        "configs/runtime/resource_limits.yaml",
    )

    summary = plan.summary
    assert summary.status == "passed"
    assert summary.schema_version == "phase1_csr_benchmark_expansion_plan_v1"
    assert summary.runtime_selector_changed is False
    assert summary.executes_gpu is False
    assert summary.selected_source_matrices == 64
    assert summary.already_profiled_matrices == 12
    assert summary.eligible_unprofiled_matrices == 36
    assert summary.planned_matrices == 8
    assert summary.planned_candidate_jobs == 24
    assert summary.planned_gpu_solve_attempts == 24
    assert summary.measurement_repeats == 1
    assert summary.total_planned_nnz == 82789
    assert summary.estimated_total_nnz_visits == 92343696
    assert summary.by_candidate_profile == {"general": 4, "symmetric": 4}
    assert summary.by_solver == {
        "bicgstab": 8,
        "cg": 4,
        "gmres": 4,
        "pcg": 4,
        "richardson": 4,
    }


def test_csr_benchmark_expansion_candidate_queue_is_cpu_screened():
    plan = build_csr_benchmark_expansion_plan_from_files(
        "runs/phase1_suitesparse_selection/selected_matrices.jsonl",
        "runs/phase1_csr_selector_readiness/csr_selector_rows.jsonl",
        "configs/runtime/resource_limits.yaml",
    )

    assert len(plan.matrix_rows) == 8
    assert len(plan.candidate_rows) == 24
    assert {row.matrix_id for row in plan.matrix_rows} == {
        "suitesparse:MathWorks/tomography",
        "suitesparse:HB/jgl009",
        "suitesparse:HB/ibm32",
        "suitesparse:SNAP/email-Eu-core",
        "suitesparse:Oberwolfach/t2dal_e",
        "suitesparse:HB/bcsstk07",
        "suitesparse:HB/lshp1009",
        "suitesparse:Gset/G17",
    }
    assert all(row.planned_import for row in plan.matrix_rows)
    assert all(row.planned_gpu_benchmark for row in plan.matrix_rows)
    assert all(row.skipped_reason is None for row in plan.matrix_rows)
    assert all(row.requires_cpu_screen for row in plan.candidate_rows)
    assert all(row.planned_status == "queued_cpu_screen_required" for row in plan.candidate_rows)
    assert all(row.measurement_repeats == 1 for row in plan.candidate_rows)
    assert all(row.max_iter == 256 for row in plan.candidate_rows)
    assert all(row.precision == "float64" for row in plan.candidate_rows)
    assert sum(row.estimated_nnz_visits for row in plan.candidate_rows) == (
        plan.summary.estimated_total_nnz_visits
    )
    assert plan.schema["integration_boundary"]["status"] == "plan_only_no_benchmark_execution"
