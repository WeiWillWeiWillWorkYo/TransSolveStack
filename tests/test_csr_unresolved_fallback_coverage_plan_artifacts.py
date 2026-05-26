import json
from pathlib import Path

from transsolvestack.profiling.artifacts import read_jsonl
from transsolvestack.profiling.provenance import read_manifest, verify_manifest_hashes


def test_csr_unresolved_fallback_coverage_plan_artifacts_are_valid():
    root = Path("runs/phase1_csr_unresolved_fallback_coverage_plan")
    rows = read_jsonl(root / "csr_unresolved_fallback_candidate_plan.jsonl")
    diagnostics = read_jsonl(root / "csr_unresolved_fallback_matrix_diagnostics.jsonl")
    summary = json.loads(
        (root / "csr_unresolved_fallback_coverage_plan_summary.json").read_text(
            encoding="utf-8"
        )
    )
    schema = json.loads(
        (root / "csr_unresolved_fallback_coverage_plan_schema.json").read_text(
            encoding="utf-8"
        )
    )
    manifest = read_manifest(root / "artifact_manifest.json")

    assert manifest.artifact_kind == "csr_unresolved_fallback_coverage_plan"
    assert verify_manifest_hashes(manifest) == ()
    assert summary["status"] == "passed"
    assert summary["schema_version"] == "phase1_csr_unresolved_fallback_coverage_plan_v1"
    assert schema["schema_version"] == summary["schema_version"]
    assert schema["integration_boundary"]["plan_only"] is True
    assert summary["executes_gpu"] is False
    assert summary["runtime_selector_changed"] is False

    assert len(diagnostics) == 2
    assert len(rows) == 6
    assert summary["unresolved_matrix_count"] == 2
    assert set(summary["unresolved_matrices"]) == {
        "suitesparse:Gset/G17",
        "suitesparse:Zitney/extr1b",
    }
    assert summary["candidate_plan_rows"] == 6
    assert summary["cpu_screen_ready_candidates"] == 4
    assert summary["future_dependency_candidates"] == 2
    assert summary["by_plan_kind"] == {
        "existing_solver_parameter_sweep": 4,
        "future_preconditioner_required": 1,
        "matrix_formulation_diagnostic": 1,
    }
    assert summary["by_execution_stage"] == {
        "blocked_until_preconditioner_exists": 1,
        "cpu_screen_ready": 4,
        "diagnostic_only_no_gpu": 1,
    }

    assert all(row["executes_gpu"] is False for row in rows)
    assert all(row["runtime_selector_changed"] is False for row in rows)
    assert sum(1 for row in rows if row["cpu_screen_first"]) == 4
    assert sum(1 for row in rows if row["gpu_allowed_after_cpu_screen"]) == 4
    assert all(row["source_evidence"]["probe_attempt_count"] >= 3 for row in rows)
    assert any(
        row["matrix_id"] == "suitesparse:Gset/G17"
        and row["plan_kind"] == "matrix_formulation_diagnostic"
        and row["execution_stage"] == "diagnostic_only_no_gpu"
            and row["source_evidence"]["zero_diagonal_observed"] is True
            for row in rows
    )
    assert any(
        row["matrix_id"] == "suitesparse:Zitney/extr1b"
        and row["plan_kind"] == "future_preconditioner_required"
        and row["execution_stage"] == "blocked_until_preconditioner_exists"
        for row in rows
    )

    assert manifest.metadata["candidate_plan_rows"] == 6
    assert manifest.metadata["cpu_screen_ready_candidates"] == 4
    assert manifest.metadata["future_dependency_candidates"] == 2
    assert manifest.metadata["executes_gpu"] is False
    assert manifest.metadata["runtime_selector_changed"] is False
