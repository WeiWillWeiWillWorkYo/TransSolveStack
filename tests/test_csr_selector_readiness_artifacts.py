import json
import math
from pathlib import Path

from transsolvestack.profiling.artifacts import read_jsonl
from transsolvestack.profiling.provenance import read_manifest, verify_manifest_hashes


def test_csr_selector_readiness_artifact_is_current_and_numeric():
    root = Path("runs/phase1_csr_selector_readiness")
    manifest = read_manifest(root / "artifact_manifest.json")
    diagnostics = read_jsonl(root / "csr_diagnostic_rows.jsonl")
    selector_rows = read_jsonl(root / "csr_selector_rows.jsonl")
    summary = json.loads((root / "csr_selector_summary.json").read_text(encoding="utf-8"))
    schema = json.loads((root / "csr_selector_schema.json").read_text(encoding="utf-8"))

    assert manifest.artifact_kind == "csr_selector_readiness_export"
    assert verify_manifest_hashes(manifest) == ()
    assert summary["status"] == "passed"
    assert summary["schema_version"] == "phase1_csr_selector_features_v8"
    assert schema["schema_version"] == summary["schema_version"]
    assert schema["integration_boundary"]["status"] == "ready_for_real_csr_scaleout"
    assert len(diagnostics) == 12
    assert len(selector_rows) == 108
    assert summary["num_success_rows"] == 15
    assert summary["num_failed_rows"] == 15
    assert summary["num_applicability_rows"] == 78
    assert summary["num_oracle_rows"] == 4
    assert summary["num_matrices_with_selector_rows"] == 12
    assert float(summary["max_final_relative_residual"]) <= 1.0e-5
    assert float(summary["max_cpu_recomputed_relative_residual"]) <= 1.0e-4
    assert float(summary["max_solution_relative_error"]) <= 5.0e-3
    assert float(summary["min_success_rate"]) == 0.0
    assert float(summary["max_solve_time_iqr_ms"]) >= 0.0
    assert math.isinf(float(summary["max_failed_final_relative_residual"]))
    assert summary["failure_reason_counts"] == {
        "cpu_reference_bicgstab_screen_failed": 4,
        "cpu_reference_cg_screen_failed": 2,
        "cpu_reference_chebyshev_screen_failed": 1,
        "cpu_reference_gmres_screen_failed": 7,
        "cpu_reference_richardson_screen_failed": 1,
    }
    assert summary["applicability_status_counts"] == {
        "not_applicable": 34,
        "not_profiled": 44,
    }
    assert summary["applicability_reason_counts"] == {
        "above_smoke_size_limit": 2,
        "after_selection_limit": 42,
        "not_symmetric": 34,
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
    assert all(row["target_status"] == "screened_out" for row in failed_rows)
    assert all(row["target_success_rate"] == 0.0 for row in failed_rows)
    assert all(row["target_measurement_repeats"] == 1 for row in failed_rows)
    assert all(row["target_median_solve_time_ms"] is None for row in failed_rows)
    assert all(row["target_applicability_status"] == "applicable" for row in success_rows)
    assert all(row["target_applicability_status"] == "applicable" for row in failed_rows)
    assert all(row["target_failure_reason"] is None for row in applicability_rows)
    assert all(row["target_success_rate"] == 0.0 for row in applicability_rows)
    assert all(row["target_measurement_repeats"] == 0 for row in applicability_rows)
    assert {row["target_status"] for row in applicability_rows} == {
        "not_applicable",
        "not_profiled",
    }
    assert {row["target_applicability_reason"] for row in applicability_rows} == {
        "not_symmetric",
        "after_selection_limit",
        "above_smoke_size_limit",
    }
    assert all(row["features"]["screened_out_source"] for row in applicability_rows)
    assert all(row["features"]["screened_out_source"] for row in failed_rows)
    assert any(
        row["matrix_id"] == "suitesparse:HB/curtis54"
        and row["solver"] == "gmres"
        and row["solver_parameters"]["restart"] == 8
        and row["target_final_relative_residual"] > 1.0e-5
        for row in failed_rows
    )
    assert any(math.isinf(row["target_final_relative_residual"]) for row in failed_rows)
    assert all(row["target_success_rate"] == 1.0 for row in success_rows)
    assert all(row["target_measurement_repeats"] == 3 for row in success_rows)
    assert all(row["target_solve_time_ms"] == row["target_median_solve_time_ms"] for row in success_rows)
    assert all(row["target_solve_time_iqr_ms"] >= 0.0 for row in success_rows)
    assert sum(1 for row in selector_rows if row["label_is_oracle"]) == 4
    assert {row["solver"] for row in selector_rows} == {
        "cg",
        "pcg",
        "bicgstab",
        "gmres",
        "richardson",
        "chebyshev",
    }
    gmres = tuple(row for row in selector_rows if row["solver"] == "gmres")
    assert len(gmres) == 36
    assert {row["solver_parameters"]["restart"] for row in gmres} == {8, 16, 32}
    assert {row["candidate_id"] for row in gmres} == {
        "taichi_csr_gmres_jacobi_restart8_float64",
        "taichi_csr_gmres_jacobi_restart16_float64",
        "taichi_csr_gmres_jacobi_restart32_float64",
    }
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
