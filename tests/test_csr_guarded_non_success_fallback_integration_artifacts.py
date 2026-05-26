import json
from pathlib import Path

from transsolvestack.profiling.artifacts import read_jsonl
from transsolvestack.profiling.provenance import read_manifest, verify_manifest_hashes


def test_csr_guarded_non_success_fallback_integration_artifacts_are_valid():
    root = Path("runs/phase1_csr_guarded_non_success_fallback_integration")
    rows = read_jsonl(root / "csr_guarded_non_success_fallback_integration_results.jsonl")
    augmented_rows = read_jsonl(root / "augmented_csr_selector_rows.jsonl")
    predictions = read_jsonl(root / "fixture_non_success_predictions.jsonl")
    summary = json.loads(
        (root / "csr_guarded_non_success_fallback_integration_summary.json").read_text(
            encoding="utf-8"
        )
    )
    schema = json.loads(
        (root / "csr_guarded_non_success_fallback_integration_schema.json").read_text(
            encoding="utf-8"
        )
    )
    quality_gate = json.loads(
        (root / "fixture_runtime_eligible_quality_gate_summary.json").read_text(
            encoding="utf-8"
        )
    )
    manifest = read_manifest(root / "artifact_manifest.json")

    assert manifest.artifact_kind == "csr_guarded_non_success_fallback_integration"
    assert verify_manifest_hashes(manifest) == ()
    assert summary["status"] == "passed"
    assert summary["schema_version"] == (
        "phase1_csr_guarded_non_success_fallback_integration_v1"
    )
    assert schema["schema_version"] == summary["schema_version"]
    assert schema["source_probe_schema"] == "phase1_csr_non_success_fallback_probe_v1"
    assert schema["safety_policy"]["runtime_selector_changed"] is False
    assert summary["runtime_selector_changed"] is False
    assert quality_gate["challenger_runtime_eligible"] is True

    assert len(rows) == 4
    assert len(predictions) == 4
    assert len(augmented_rows) == summary["augmented_selector_rows"]
    assert summary["base_selector_rows"] == 132
    assert summary["fallback_probe_selector_rows"] == 15
    assert summary["fallback_probe_success_rows_imported"] == 4
    assert summary["augmented_selector_rows"] == 136
    assert summary["resolved_exact_fallback_scenarios"] == 2
    assert set(summary["resolved_matrices"]) == {
        "suitesparse:Bai/cdde1",
        "suitesparse:FIDAP/ex5",
    }
    assert summary["unresolved_exact_fallback_scenarios"] == 2
    assert set(summary["unresolved_matrices"]) == {
        "suitesparse:Gset/G17",
        "suitesparse:Zitney/extr1b",
    }
    assert summary["executed_gpu_scenarios"] == 2
    assert summary["guard_only_scenarios"] == 2
    assert summary["learned_non_success_blocks"] == 4
    assert summary["artifact_fallback_solves"] == 2
    assert summary["runtime_guard_success_count"] == 2
    assert set(summary["selected_solver_set"]) == {"bicgstab", "pcg"}
    assert float(summary["max_final_relative_residual"]) <= 1.0e-5
    assert float(summary["max_cpu_recomputed_relative_residual"]) <= 1.0e-4
    assert float(summary["max_solution_relative_error"]) <= 5.0e-3

    gpu_rows = [row for row in rows if row["executed_gpu"]]
    guard_rows = [row for row in rows if not row["executed_gpu"]]
    assert len(gpu_rows) == 2
    assert len(guard_rows) == 2
    for gpu_row in gpu_rows:
        assert gpu_row["matrix_id"] in {
            "suitesparse:Bai/cdde1",
            "suitesparse:FIDAP/ex5",
        }
        assert gpu_row["runtime_selection_source"] == "artifact"
        assert gpu_row["learned_guard_status"] == "blocked_non_success_candidate"
        assert gpu_row["learned_runtime_selector_changed"] is False
        assert gpu_row["runtime_guard_status"] == "success"
        assert gpu_row["runtime_guard_used_fallback"] is False
        assert gpu_row["trace"]["backend"] == "taichi_gpu"
    cdde1 = [row for row in gpu_rows if row["matrix_id"] == "suitesparse:Bai/cdde1"][0]
    assert set(cdde1["exact_success_candidate_ids"]) == {
        "taichi_csr_bicgstab_none_float64",
        "taichi_csr_bicgstab_jacobi_float64",
        "taichi_csr_gmres_jacobi_restart32_float64",
    }
    assert all(row["trace"] is None for row in guard_rows)
    assert all(row["exact_success_candidate_count"] == 0 for row in guard_rows)
    assert all(_guard_blocks_non_success_candidate(row) for row in rows)
    assert all(row["failure_reasons"] == [] for row in rows)

    assert manifest.metadata["resolved_exact_fallback_scenarios"] == 2
    assert manifest.metadata["unresolved_exact_fallback_scenarios"] == 2
    assert manifest.metadata["executed_gpu_scenarios"] == 2
    assert manifest.metadata["runtime_selector_changed"] is False


def _guard_blocks_non_success_candidate(row):
    if row["learned_guard_status"] == "blocked_non_success_candidate":
        return True
    return row["learned_guard_status"] == "blocked_fallback_chain" and any(
        "not a profiled success" in reason for reason in row["learned_guard_reasons"]
    )
