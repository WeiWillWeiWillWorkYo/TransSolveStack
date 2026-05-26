import json
from pathlib import Path

from transsolvestack.profiling.artifacts import read_jsonl
from transsolvestack.profiling.provenance import read_manifest, verify_manifest_hashes


def test_csr_guarded_promotion_coverage_artifacts_are_valid():
    root = Path("runs/phase1_csr_guarded_promotion_coverage_plan")
    rows = read_jsonl(root / "csr_guarded_promotion_coverage_scenarios.jsonl")
    summary = json.loads(
        (root / "csr_guarded_promotion_coverage_summary.json").read_text(
            encoding="utf-8"
        )
    )
    schema = json.loads(
        (root / "csr_guarded_promotion_coverage_schema.json").read_text(
            encoding="utf-8"
        )
    )
    manifest = read_manifest(root / "artifact_manifest.json")

    assert manifest.artifact_kind == "csr_guarded_promotion_coverage_plan"
    assert verify_manifest_hashes(manifest) == ()
    assert summary["status"] == "passed"
    assert summary["schema_version"] == "phase1_csr_guarded_promotion_coverage_plan_v1"
    assert schema["schema_version"] == summary["schema_version"]
    assert schema["integration_boundary"]["status"] == "plan_only_no_gpu_execution"
    assert summary["runtime_selector_changed"] is False
    assert summary["executes_gpu"] is False
    assert summary["planned_scenarios"] == 15
    assert len(rows) == 15
    assert summary["planned_gpu_final_solves"] == 11
    assert summary["planned_guard_only_scenarios"] == 4
    assert summary["current_blocked_scenarios"] == 4
    assert summary["fixture_promotion_scenarios"] == 4
    assert summary["fixture_promotions_changing_candidate"] == 3
    assert summary["non_success_block_scenarios"] == 4
    assert summary["runtime_fallback_scenarios"] == 3
    assert summary["quality_gate_runtime_eligible"] is False
    assert set(summary["quality_gate_failures"]) == {
        "below_min_profiled_success_rate",
        "below_baseline_oracle_top1",
        "below_baseline_profiled_success_rate",
        "non_success_eval_selections",
    }

    rows_by_kind = {}
    for row in rows:
        rows_by_kind.setdefault(row["scenario_kind"], []).append(row)
        assert row["schema_version"] == summary["schema_version"]
        assert row["matrix_source_path"]
        if row["requires_gpu_execution"]:
            assert row["artifact_candidate_id"] is not None

    assert set(rows_by_kind) == {
        "current_ranker_quality_gate_block",
        "fixture_profiled_success_promotion",
        "fixture_non_success_candidate_block",
        "runtime_exception_fallback",
    }
    assert all(
        row["expected_guard_status"] == "blocked_quality_gate"
        and row["expected_runtime_selection_source"] == "artifact"
        for row in rows_by_kind["current_ranker_quality_gate_block"]
    )
    assert all(
        row["expected_guard_status"] == "promoted"
        and row["expected_runtime_selection_source"] == "learned"
        and row["selected_candidate_is_exact_profiled_success"] is True
        for row in rows_by_kind["fixture_profiled_success_promotion"]
    )
    assert all(
        row["expected_guard_status"] == "blocked_non_success_candidate"
        and row["requires_gpu_execution"] is False
        for row in rows_by_kind["fixture_non_success_candidate_block"]
    )
    assert all(
        row["requires_runtime_exception_injection"] is True
        and row["expected_final_result_status"] == "fallback_success"
        for row in rows_by_kind["runtime_exception_fallback"]
    )
    assert manifest.metadata["planned_scenarios"] == 15
    assert manifest.metadata["planned_gpu_final_solves"] == 11
    assert manifest.metadata["runtime_selector_changed"] is False
