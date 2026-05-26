import json
from pathlib import Path

from transsolvestack.profiling.artifacts import read_jsonl
from transsolvestack.profiling.provenance import read_manifest, verify_manifest_hashes


def test_csr_non_success_fallback_probe_artifacts_are_valid():
    root = Path("runs/phase1_csr_non_success_fallback_probe")
    rows = read_jsonl(root / "csr_non_success_fallback_probe_results.jsonl")
    selector_rows = read_jsonl(root / "csr_non_success_fallback_selector_rows.jsonl")
    summary = json.loads(
        (root / "csr_non_success_fallback_probe_summary.json").read_text(
            encoding="utf-8"
        )
    )
    schema = json.loads(
        (root / "csr_non_success_fallback_probe_schema.json").read_text(
            encoding="utf-8"
        )
    )
    manifest = read_manifest(root / "artifact_manifest.json")

    assert manifest.artifact_kind == "csr_non_success_fallback_probe"
    assert verify_manifest_hashes(manifest) == ()
    assert summary["status"] == "passed"
    assert summary["schema_version"] == "phase1_csr_non_success_fallback_probe_v1"
    assert schema["schema_version"] == summary["schema_version"]
    assert schema["source_plan_schema"] == "phase1_csr_guarded_promotion_coverage_plan_v1"
    assert summary["runtime_selector_changed"] is False
    assert summary["executes_gpu"] is True

    assert len(rows) == 15
    assert len(selector_rows) == summary["selector_rows"]
    assert summary["target_scenarios"] == 4
    assert summary["candidate_attempts"] == 15
    assert summary["gpu_success_rows"] == 4
    assert summary["cpu_screened_out_rows"] == 11
    assert summary["gpu_failed_rows"] == 0
    assert set(summary["resolved_matrices"]) == {
        "suitesparse:Bai/cdde1",
        "suitesparse:FIDAP/ex5",
    }
    assert set(summary["unresolved_matrices"]) == {
        "suitesparse:Gset/G17",
        "suitesparse:Zitney/extr1b",
    }
    assert summary["by_status"] == {"screened_out": 11, "success": 4}
    assert float(summary["max_final_relative_residual"]) <= 1.0e-5
    assert float(summary["max_cpu_recomputed_relative_residual"]) <= 1.0e-4
    assert float(summary["max_solution_relative_error"]) <= 5.0e-3

    success_rows = [row for row in rows if row["status"] == "success"]
    screen_rows = [row for row in rows if row["status"] == "screened_out"]
    assert len(success_rows) == 4
    assert len(screen_rows) == 11
    assert {row["matrix_id"] for row in success_rows} == {
        "suitesparse:Bai/cdde1",
        "suitesparse:FIDAP/ex5",
    }
    assert {row["backend"] for row in success_rows} == {"taichi_gpu"}
    assert {row["backend"] for row in screen_rows} == {"cpu_reference_screen"}
    assert all(row["gpu_executed"] is True for row in success_rows)
    assert all(row["gpu_executed"] is False for row in screen_rows)
    assert all(row["cpu_screen_success"] is True for row in success_rows)
    assert all(row["cpu_screen_success"] is False for row in screen_rows)
    assert all(not row["failure_reasons"] for row in success_rows)
    assert all(row["failure_reasons"] for row in screen_rows)

    assert {row["target_status"] for row in selector_rows} == {
        "screened_out",
        "success",
    }
    assert sum(1 for row in selector_rows if row["label_is_oracle"]) == 2
    assert manifest.metadata["gpu_success_rows"] == 4
    assert manifest.metadata["cpu_screened_out_rows"] == 11
    assert manifest.metadata["resolved_matrix_count"] == 2
    assert manifest.metadata["unresolved_matrix_count"] == 2
