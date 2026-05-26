import math
from dataclasses import asdict

from transsolvestack.policies.csr_selector_data import (
    CSR_SELECTOR_SCHEMA_VERSION,
    build_csr_selector_export_from_files,
)


def test_csr_selector_export_joins_diagnostics_with_real_solve_rows():
    export = build_csr_selector_export_from_files(
        "runs/phase1_suitesparse_csr_import/csr_matrices.jsonl",
        (
            "runs/phase1_taichi_csr_solve/csr_solve_results.jsonl",
            "runs/phase1_taichi_csr_bicgstab/csr_bicgstab_results.jsonl",
            "runs/phase1_taichi_csr_gmres/csr_gmres_results.jsonl",
            "runs/phase1_taichi_csr_richardson/csr_richardson_results.jsonl",
            "runs/phase1_taichi_csr_chebyshev/csr_chebyshev_results.jsonl",
        ),
        screened_results_path=(
            "runs/phase1_taichi_csr_solve/csr_solve_summary.json",
            "runs/phase1_taichi_csr_bicgstab/csr_bicgstab_summary.json",
            "runs/phase1_taichi_csr_gmres/csr_gmres_summary.json",
            "runs/phase1_taichi_csr_richardson/csr_richardson_summary.json",
            "runs/phase1_taichi_csr_chebyshev/csr_chebyshev_summary.json",
        ),
    )

    assert export.summary.status == "passed"
    assert export.summary.schema_version == CSR_SELECTOR_SCHEMA_VERSION
    assert export.summary.num_diagnostic_rows == 12
    assert export.summary.num_selector_rows == 108
    assert export.summary.num_success_rows == 15
    assert export.summary.num_failed_rows == 15
    assert export.summary.num_applicability_rows == 78
    assert export.summary.num_oracle_rows == 4
    assert export.summary.num_matrices_with_selector_rows == 12
    assert export.summary.max_final_relative_residual <= 1.0e-5
    assert export.summary.max_cpu_recomputed_relative_residual <= 1.0e-4
    assert export.summary.max_solution_relative_error <= 5.0e-3
    assert export.summary.min_success_rate == 0.0
    assert export.summary.max_solve_time_iqr_ms >= 0.0
    assert math.isinf(export.summary.max_failed_final_relative_residual)
    assert export.summary.failure_reason_counts == {
        "cpu_reference_bicgstab_screen_failed": 4,
        "cpu_reference_cg_screen_failed": 2,
        "cpu_reference_chebyshev_screen_failed": 1,
        "cpu_reference_gmres_screen_failed": 7,
        "cpu_reference_richardson_screen_failed": 1,
    }
    assert export.summary.applicability_status_counts == {
        "not_applicable": 34,
        "not_profiled": 44,
    }
    assert export.summary.applicability_reason_counts == {
        "above_smoke_size_limit": 2,
        "after_selection_limit": 42,
        "not_symmetric": 34,
    }

    selector_rows = tuple(asdict(row) for row in export.selector_rows)
    assert {row["candidate_id"] for row in selector_rows} == {
        "taichi_csr_cg_none_float64",
        "taichi_csr_pcg_jacobi_float64",
        "taichi_csr_bicgstab_none_float64",
        "taichi_csr_bicgstab_jacobi_float64",
        "taichi_csr_gmres_jacobi_restart8_float64",
        "taichi_csr_gmres_jacobi_restart16_float64",
        "taichi_csr_gmres_jacobi_restart32_float64",
        "taichi_csr_richardson_jacobi_float64",
        "taichi_csr_chebyshev_jacobi_float64",
    }
    success_rows = tuple(row for row in selector_rows if row["target_status"] == "success")
    failed_rows = tuple(row for row in selector_rows if row["target_status"] == "screened_out")
    applicability_rows = tuple(
        row
        for row in selector_rows
        if row["target_status"] in {"not_applicable", "not_profiled"}
    )
    assert len(success_rows) == 15
    assert len(failed_rows) == 15
    assert len(applicability_rows) == 78
    assert {row["target_failure_reason"] for row in failed_rows} == {
        "cpu_reference_cg_screen_failed",
        "cpu_reference_bicgstab_screen_failed",
        "cpu_reference_gmres_screen_failed",
        "cpu_reference_richardson_screen_failed",
        "cpu_reference_chebyshev_screen_failed",
    }
    assert all(row["target_success_rate"] == 0.0 for row in failed_rows)
    assert all(row["target_measurement_repeats"] == 1 for row in failed_rows)
    assert all(row["target_median_solve_time_ms"] is None for row in failed_rows)
    assert all(row["target_failure_reason"] for row in failed_rows)
    assert all(row["target_applicability_status"] == "applicable" for row in success_rows)
    assert all(row["target_applicability_status"] == "applicable" for row in failed_rows)
    assert all(row["target_failure_reason"] is None for row in applicability_rows)
    assert all(row["target_success_rate"] == 0.0 for row in applicability_rows)
    assert all(row["target_measurement_repeats"] == 0 for row in applicability_rows)
    assert all(row["target_applicability_reason"] for row in applicability_rows)
    assert {row["target_status"] for row in applicability_rows} == {
        "not_applicable",
        "not_profiled",
    }
    assert {row["target_applicability_reason"] for row in applicability_rows} == {
        "not_symmetric",
        "after_selection_limit",
        "above_smoke_size_limit",
    }
    assert all(row["features"]["screened_out_source"] for row in failed_rows)
    assert all(row["features"]["screened_out_source"] for row in applicability_rows)
    assert any(
        row["matrix_id"] == "suitesparse:HB/curtis54"
        and row["solver"] == "gmres"
        and row["solver_parameters"]["restart"] == 8
        and row["target_final_relative_residual"] > 1.0e-5
        for row in failed_rows
    )
    assert any(math.isinf(row["target_final_relative_residual"]) for row in failed_rows)
    assert any(row["features"]["actual_symmetric"] for row in selector_rows)
    assert any(not row["features"]["actual_symmetric"] for row in selector_rows)
    assert all(
        row["features"]["recommended_precision"] in {"float32", "float64"}
        for row in selector_rows
    )
    assert all(row["features"]["precision"] == "float64" for row in selector_rows)
    assert all(row["target_success_rate"] == 1.0 for row in success_rows)
    assert all(row["target_measurement_repeats"] == 3 for row in success_rows)
    assert all(
        row["target_solve_time_ms"] == row["target_median_solve_time_ms"]
        for row in success_rows
    )
    assert all(row["target_solve_time_iqr_ms"] >= 0.0 for row in success_rows)
    gmres = tuple(row for row in selector_rows if row["solver"] == "gmres")
    assert len(gmres) == 36
    assert {row["solver_parameters"]["restart"] for row in gmres} == {8, 16, 32}
    chebyshev = next(row for row in selector_rows if row["solver"] == "chebyshev")
    assert chebyshev["solver_parameters"]["lambda_min"] > 0.0
    assert (
        chebyshev["solver_parameters"]["lambda_min"]
        < chebyshev["solver_parameters"]["lambda_max"]
    )
    assert (
        chebyshev["solver_parameters"]["spectral_bounds_source"]
        == "dense_eigvalsh_jacobi_preconditioned_padded"
    )
