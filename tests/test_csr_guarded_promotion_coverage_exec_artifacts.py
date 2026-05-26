import json
from pathlib import Path

from transsolvestack.profiling.artifacts import read_jsonl
from transsolvestack.profiling.provenance import read_manifest, verify_manifest_hashes


def test_csr_guarded_promotion_coverage_exec_artifacts_are_valid():
    root = Path("runs/phase1_csr_guarded_promotion_coverage_exec")
    rows = read_jsonl(root / "csr_guarded_promotion_coverage_exec_results.jsonl")
    summary = json.loads(
        (root / "csr_guarded_promotion_coverage_exec_summary.json").read_text(
            encoding="utf-8"
        )
    )
    schema = json.loads(
        (root / "csr_guarded_promotion_coverage_exec_schema.json").read_text(
            encoding="utf-8"
        )
    )
    manifest = read_manifest(root / "artifact_manifest.json")

    assert manifest.artifact_kind == "csr_guarded_promotion_coverage_exec"
    assert verify_manifest_hashes(manifest) == ()
    assert summary["status"] == "passed"
    assert summary["schema_version"] == "phase1_csr_guarded_promotion_coverage_exec_v1"
    assert schema["schema_version"] == summary["schema_version"]
    assert schema["source_plan_schema"] == "phase1_csr_guarded_promotion_coverage_plan_v1"
    assert len(rows) == 12
    assert summary["num_scenarios"] == 12
    assert summary["num_success"] == 12
    assert summary["num_failed"] == 0
    assert summary["executed_gpu_scenarios"] == 8
    assert summary["guard_only_scenarios"] == 4
    assert summary["current_quality_gate_blocks"] == 4
    assert summary["fixture_learned_promotions"] == 2
    assert summary["fixture_runtime_selector_changed_count"] == 2
    assert summary["current_runtime_selector_changed"] is False
    assert summary["non_success_blocks"] == 4
    assert summary["runtime_exception_fallbacks"] == 2
    assert summary["runtime_fallback_used_count"] == 2
    assert summary["runtime_guard_success_count"] == 8
    assert set(summary["selected_solver_set"]) == {
        "bicgstab",
        "chebyshev",
        "gmres",
        "pcg",
    }
    assert float(summary["max_final_relative_residual"]) <= 1.0e-5
    assert float(summary["max_cpu_recomputed_relative_residual"]) <= 1.0e-4
    assert float(summary["max_solution_relative_error"]) <= 5.0e-3

    rows_by_kind = {}
    for row in rows:
        rows_by_kind.setdefault(row["scenario_kind"], []).append(row)
        assert row["status"] == "success"
        assert row["failure_reasons"] == []
        if row["requires_gpu_execution"]:
            assert row["trace"]["backend"] == "taichi_gpu"
            assert row["trace"]["metadata"]["operator_backend"] == "taichi_csr"
        else:
            assert row["trace"] is None

    assert {kind: len(values) for kind, values in rows_by_kind.items()} == {
        "current_ranker_quality_gate_block": 4,
        "fixture_profiled_success_promotion": 2,
        "fixture_non_success_candidate_block": 4,
        "runtime_exception_fallback": 2,
    }
    assert all(
        row["learned_guard_status"] == "blocked_quality_gate"
        and row["runtime_selection_source"] == "artifact"
        and row["learned_runtime_selector_changed"] is False
        for row in rows_by_kind["current_ranker_quality_gate_block"]
    )
    assert all(
        row["learned_guard_status"] == "promoted"
        and row["runtime_selection_source"] == "learned"
        and row["learned_runtime_selector_changed"] is True
        for row in rows_by_kind["fixture_profiled_success_promotion"]
    )
    assert all(
        row["final_result_status"] == "guard_plan_only"
        and row["runtime_guard_attempt_count"] == 0
        and any("not a profiled success" in reason for reason in row["learned_guard_reasons"])
        for row in rows_by_kind["fixture_non_success_candidate_block"]
    )
    assert all(
        row["final_result_status"] == "fallback_success"
        and row["runtime_guard_used_fallback"] is True
        and row["runtime_guard_attempt_count"] == 2
        and row["runtime_guard_attempts"][0]["status"] == "exception"
        for row in rows_by_kind["runtime_exception_fallback"]
    )

    assert manifest.metadata["executed_gpu_scenarios"] == 8
    assert manifest.metadata["current_runtime_selector_changed"] is False
