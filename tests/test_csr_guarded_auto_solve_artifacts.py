import json
from pathlib import Path

from transsolvestack.profiling.artifacts import read_jsonl
from transsolvestack.profiling.provenance import read_manifest, verify_manifest_hashes


def test_csr_guarded_auto_solve_artifacts_are_valid():
    root = Path("runs/phase1_csr_guarded_auto_solve")
    rows = read_jsonl(root / "csr_guarded_auto_solve_results.jsonl")
    summary = json.loads(
        (root / "csr_guarded_auto_solve_summary.json").read_text(encoding="utf-8")
    )
    schema = json.loads(
        (root / "csr_guarded_auto_solve_schema.json").read_text(encoding="utf-8")
    )
    manifest = read_manifest(root / "artifact_manifest.json")

    assert manifest.artifact_kind == "csr_guarded_auto_solve_smoke"
    assert verify_manifest_hashes(manifest) == ()
    assert summary["status"] == "passed"
    assert summary["schema_version"] == "phase1_csr_guarded_auto_solve_v1"
    assert schema["schema_version"] == summary["schema_version"]
    assert schema["runtime"]["backend"] == "taichi_gpu"
    assert schema["runtime"]["learned_policy_mode"] == "promote_if_safe"
    assert summary["num_solves"] == 2
    assert summary["num_success"] == 2
    assert summary["num_failed"] == 0
    assert summary["quality_gate_blocks"] == 2
    assert summary["learned_runtime_promotions"] == 0
    assert summary["runtime_selector_changed"] is False
    assert summary["fallback_chain_enforced_count"] == 2
    assert summary["runtime_guard_success_count"] == 2
    assert summary["runtime_fallback_used_count"] == 0
    assert float(summary["max_final_relative_residual"]) <= 1.0e-5
    assert float(summary["max_cpu_recomputed_relative_residual"]) <= 1.0e-4
    assert float(summary["max_solution_relative_error"]) <= 5.0e-3

    assert {row["matrix_id"] for row in rows} == {
        "suitesparse:FIDAP/ex5",
        "suitesparse:HB/curtis54",
    }
    for row in rows:
        assert row["status"] == "success"
        assert row["runtime_selection_source"] == "artifact"
        assert row["learned_guard_status"] == "blocked_quality_gate"
        assert "quality_gate:non_success_eval_selections" in row["learned_guard_reasons"]
        assert row["fallback_chain_enforced"] is True
        assert row["runtime_guard_status"] == "success"
        assert row["runtime_guard_used_fallback"] is False
        assert row["runtime_guard_attempt_count"] == 1
        assert row["trace"]["backend"] == "taichi_gpu"
        assert "learned_policy_guard" in row["trace"]["metadata"]
        assert "runtime_guard" in row["trace"]["metadata"]
        assert row["trace"]["metadata"]["learned_policy_guard"]["runtime_selection_source"] == "artifact"
        assert row["trace"]["metadata"]["runtime_guard"]["guard_status"] == "success"

    assert manifest.metadata["quality_gate_blocks"] == 2
    assert manifest.metadata["learned_runtime_promotions"] == 0
    assert manifest.metadata["runtime_selector_changed"] is False
