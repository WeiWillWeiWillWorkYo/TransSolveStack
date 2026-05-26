import json
from pathlib import Path

from transsolvestack.profiling.artifacts import read_jsonl
from transsolvestack.profiling.provenance import read_manifest, verify_manifest_hashes


def test_csr_ilu0_guarded_integration_artifacts_are_valid():
    root = Path("runs/phase1_csr_ilu0_guarded_integration")
    rows = read_jsonl(root / "csr_ilu0_guarded_integration_results.jsonl")
    selector_rows = read_jsonl(root / "csr_ilu0_guarded_selector_rows.jsonl")
    predictions = read_jsonl(root / "fixture_ilu0_predictions.jsonl")
    summary = json.loads(
        (root / "csr_ilu0_guarded_integration_summary.json").read_text(
            encoding="utf-8"
        )
    )
    schema = json.loads(
        (root / "csr_ilu0_guarded_integration_schema.json").read_text(
            encoding="utf-8"
        )
    )
    quality_gate = json.loads(
        (root / "fixture_runtime_eligible_quality_gate_summary.json").read_text(
            encoding="utf-8"
        )
    )
    manifest = read_manifest(root / "artifact_manifest.json")

    assert manifest.artifact_kind == "csr_ilu0_guarded_integration"
    assert verify_manifest_hashes(manifest) == ()
    assert summary["status"] == "passed"
    assert summary["schema_version"] == "phase1_csr_ilu0_guarded_integration_v1"
    assert schema["schema_version"] == summary["schema_version"]
    assert schema["safety_policy"]["production_runtime_selector_changed"] is False
    assert quality_gate["challenger_runtime_eligible"] is True

    assert len(rows) == 2
    assert len(selector_rows) == 2
    assert len(predictions) == 2
    assert summary["ilu0_success_rows_imported"] == 1
    assert summary["ilu0_failed_numeric_gate_rows_imported"] == 1
    assert summary["promoted_success_rows"] == 1
    assert summary["failed_numeric_gate_rows_blocked"] == 1
    assert summary["executed_gpu_scenarios"] == 1
    assert summary["guard_only_scenarios"] == 1
    assert summary["production_runtime_selector_changed"] is False
    assert summary["fixture_runtime_selector_changed_count"] == 1
    assert summary["selected_solver_set"] == ["bicgstab"]
    assert summary["selected_preconditioner_set"] == ["ilu0"]
    assert float(summary["max_final_relative_residual"]) <= 1.0e-5
    assert float(summary["max_cpu_recomputed_relative_residual"]) <= 1.0e-4
    assert float(summary["max_solution_relative_error"]) <= 5.0e-3

    selector_by_matrix = {row["matrix_id"]: row for row in selector_rows}
    assert selector_by_matrix["suitesparse:Bai/cdde1"]["target_status"] == "success"
    assert (
        selector_by_matrix["suitesparse:MathWorks/tomography"]["target_status"]
        == "failed_numeric_gate"
    )

    rows_by_matrix = {row["matrix_id"]: row for row in rows}
    bai = rows_by_matrix["suitesparse:Bai/cdde1"]
    assert bai["learned_guard_status"] == "promoted"
    assert bai["runtime_selection_source"] == "learned"
    assert bai["learned_runtime_selector_changed"] is True
    assert bai["executed_gpu"] is True
    assert bai["trace"]["backend"] == "taichi_gpu"
    assert bai["trace"]["metadata"]["solver"] == "bicgstab"
    assert bai["trace"]["metadata"]["preconditioner"] == "ilu0"
    assert bai["failure_reasons"] == []

    tomography = rows_by_matrix["suitesparse:MathWorks/tomography"]
    assert tomography["learned_guard_status"] == "blocked_non_success_candidate"
    assert tomography["learned_selected_target_status"] == "failed_numeric_gate"
    assert tomography["learned_runtime_selector_changed"] is False
    assert tomography["executed_gpu"] is False
    assert tomography["trace"] is None
    assert tomography["failure_reasons"] == []

    assert manifest.metadata["ilu0_success_rows_imported"] == 1
    assert manifest.metadata["failed_numeric_gate_rows_blocked"] == 1
    assert manifest.metadata["executed_gpu_scenarios"] == 1
    assert manifest.metadata["production_runtime_selector_changed"] is False
