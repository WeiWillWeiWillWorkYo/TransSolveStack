import json
from pathlib import Path

from transsolvestack.profiling.artifacts import read_jsonl
from transsolvestack.profiling.provenance import read_manifest, verify_manifest_hashes


def test_csr_guarded_promotion_readiness_artifacts_are_valid():
    root = Path("runs/phase1_csr_guarded_promotion_readiness")
    rows = read_jsonl(root / "csr_guarded_promotion_readiness_results.jsonl")
    summary = json.loads(
        (root / "csr_guarded_promotion_readiness_summary.json").read_text(
            encoding="utf-8"
        )
    )
    schema = json.loads(
        (root / "csr_guarded_promotion_readiness_schema.json").read_text(
            encoding="utf-8"
        )
    )
    manifest = read_manifest(root / "artifact_manifest.json")

    assert manifest.artifact_kind == "csr_guarded_promotion_readiness"
    assert verify_manifest_hashes(manifest) == ()
    assert summary["status"] == "passed"
    assert summary["schema_version"] == "phase1_csr_guarded_promotion_readiness_v1"
    assert schema["schema_version"] == summary["schema_version"]
    assert schema["runtime"]["backend"] == "taichi_gpu"
    assert schema["runtime"]["production_runtime_selector_changed"] is False
    assert summary["num_scenarios"] == 3
    assert summary["num_success"] == 3
    assert summary["num_failed"] == 0
    assert summary["current_quality_gate_blocks"] == 1
    assert summary["fixture_learned_promotions"] == 1
    assert summary["fixture_runtime_selector_changed_count"] == 1
    assert summary["current_runtime_selector_changed"] is False
    assert summary["runtime_exception_fallbacks"] == 1
    assert summary["runtime_fallback_used_count"] == 1
    assert summary["runtime_guard_success_count"] == 3
    assert summary["real_gpu_final_result_count"] == 3
    assert set(summary["selected_solver_set"]) == {"bicgstab", "pcg"}
    assert float(summary["max_final_relative_residual"]) <= 1.0e-5
    assert float(summary["max_cpu_recomputed_relative_residual"]) <= 1.0e-4
    assert float(summary["max_solution_relative_error"]) <= 5.0e-3

    rows_by_kind = {row["scenario_kind"]: row for row in rows}
    assert set(rows_by_kind) == {
        "current_ranker_blocked",
        "fixture_learned_promotion",
        "runtime_exception_fallback",
    }
    blocked = rows_by_kind["current_ranker_blocked"]
    assert blocked["learned_guard_status"] == "blocked_quality_gate"
    assert blocked["runtime_selection_source"] == "artifact"
    assert blocked["learned_runtime_selector_changed"] is False

    promoted = rows_by_kind["fixture_learned_promotion"]
    assert promoted["learned_guard_status"] == "promoted"
    assert promoted["runtime_selection_source"] == "learned"
    assert promoted["candidate_id"] == "taichi_csr_pcg_jacobi_float64"
    assert promoted["learned_runtime_selector_changed"] is True

    fallback = rows_by_kind["runtime_exception_fallback"]
    assert fallback["runtime_guard_used_fallback"] is True
    assert fallback["final_result_status"] == "fallback_success"
    assert fallback["runtime_guard_attempt_count"] == 2
    assert fallback["runtime_guard_attempts"][0]["status"] == "exception"
    assert fallback["runtime_guard_attempts"][0]["exception_type"] == "ValueError"

    for row in rows:
        assert row["status"] == "success"
        assert row["failure_reasons"] == []
        assert row["trace"]["backend"] == "taichi_gpu"
        assert row["trace"]["metadata"]["operator_backend"] == "taichi_csr"

    assert manifest.metadata["current_runtime_selector_changed"] is False
    assert manifest.metadata["fixture_learned_promotions"] == 1
