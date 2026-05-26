import json
from pathlib import Path

from transsolvestack.profiling.artifacts import read_jsonl
from transsolvestack.profiling.provenance import read_manifest, verify_manifest_hashes


def test_csr_ilu0_coverage_expansion_artifacts_are_valid():
    root = Path("runs/phase1_csr_ilu0_coverage_expansion")
    rows = read_jsonl(root / "csr_ilu0_coverage_results.jsonl")
    selector_rows = read_jsonl(root / "csr_ilu0_coverage_selector_rows.jsonl")
    summary = json.loads(
        (root / "csr_ilu0_coverage_summary.json").read_text(encoding="utf-8")
    )
    schema = json.loads(
        (root / "csr_ilu0_coverage_schema.json").read_text(encoding="utf-8")
    )
    manifest = read_manifest(root / "artifact_manifest.json")

    assert manifest.artifact_kind == "csr_ilu0_coverage_expansion"
    assert verify_manifest_hashes(manifest) == ()
    assert summary["status"] == "passed"
    assert summary["schema_version"] == "phase1_csr_ilu0_coverage_expansion_v1"
    assert schema["schema_version"] == summary["schema_version"]
    assert schema["integration_boundary"]["production_runtime_selector_changed"] is False
    assert schema["integration_boundary"]["merged_into_main_transformer_ready"] is False

    assert len(rows) == 8
    assert len(selector_rows) == 8
    assert summary["candidate_rows"] == 8
    assert summary["selector_rows"] == 8
    assert summary["numeric_success_rows"] >= 3
    assert summary["failed_numeric_gate_rows"] >= 1
    assert summary["setup_failed_rows"] >= 1
    assert summary["merge_ready_success_rows"] == summary["numeric_success_rows"]
    assert summary["gpu_solve_rows"] == (
        summary["numeric_success_rows"] + summary["failed_numeric_gate_rows"]
    )
    assert summary["gpu_attempted_rows"] == summary["candidate_rows"]
    assert summary["runtime_selector_changed"] is False
    assert summary["merged_into_main_transformer_ready"] is False
    assert float(summary["max_success_final_relative_residual"]) <= float(
        summary["tolerance_rel"]
    )
    assert float(summary["max_success_cpu_recomputed_relative_residual"]) <= 1.0e-4
    assert float(summary["max_success_solution_relative_error"]) <= 5.0e-3

    required_successes = {
        "suitesparse:JGD_Trefethen/Trefethen_20b",
        "suitesparse:FIDAP/ex5",
        "suitesparse:Bai/cdde1",
    }
    assert required_successes.issubset(set(summary["success_matrix_ids"]))

    rows_by_matrix = {row["matrix_id"]: row for row in rows}
    selector_by_matrix = {row["matrix_id"]: row for row in selector_rows}
    assert rows_by_matrix["suitesparse:HB/young3c"]["numeric_status"] == (
        "failed_numeric_gate"
    )
    assert rows_by_matrix["suitesparse:Grund/b1_ss"]["numeric_status"] == "setup_failed"

    for matrix_id in required_successes:
        row = rows_by_matrix[matrix_id]
        selector_row = selector_by_matrix[matrix_id]
        assert row["gpu_executed"] is True
        assert row["candidate_promoted"] is True
        assert row["runtime_selector_changed"] is False
        assert selector_row["target_status"] == "success"
        assert float(selector_row["target_success_rate"]) == 1.0

    for row in rows:
        assert row["solver"] == "bicgstab"
        assert row["preconditioner"] == "ilu0"
        assert row["production_runtime_selector_changed"] is False
        if row["numeric_status"] != "success":
            assert row["candidate_promoted"] is False

    assert manifest.metadata["numeric_success_rows"] == summary["numeric_success_rows"]
    assert manifest.metadata["setup_failed_rows"] == summary["setup_failed_rows"]
    assert manifest.metadata["runtime_selector_changed"] is False
