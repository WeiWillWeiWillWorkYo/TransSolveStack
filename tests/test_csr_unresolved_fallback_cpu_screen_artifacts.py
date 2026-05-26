import json
from pathlib import Path

from transsolvestack.profiling.artifacts import read_jsonl
from transsolvestack.profiling.provenance import read_manifest, verify_manifest_hashes


def test_csr_unresolved_fallback_cpu_screen_artifacts_are_valid():
    root = Path("runs/phase1_csr_unresolved_fallback_cpu_screen")
    rows = read_jsonl(root / "csr_unresolved_fallback_cpu_screen_results.jsonl")
    summary = json.loads(
        (root / "csr_unresolved_fallback_cpu_screen_summary.json").read_text(
            encoding="utf-8"
        )
    )
    schema = json.loads(
        (root / "csr_unresolved_fallback_cpu_screen_schema.json").read_text(
            encoding="utf-8"
        )
    )
    manifest = read_manifest(root / "artifact_manifest.json")

    assert manifest.artifact_kind == "csr_unresolved_fallback_cpu_screen"
    assert verify_manifest_hashes(manifest) == ()
    assert summary["status"] == "passed"
    assert summary["schema_version"] == "phase1_csr_unresolved_fallback_cpu_screen_v1"
    assert schema["schema_version"] == summary["schema_version"]
    assert schema["integration_boundary"]["cpu_only"] is True
    assert summary["executes_gpu"] is False
    assert summary["runtime_selector_changed"] is False

    assert len(rows) == 4
    assert summary["attempted_cpu_screens"] == 4
    assert summary["screen_success_rows"] == 0
    assert summary["cpu_screened_out_rows"] == 4
    assert summary["gpu_probe_ready_candidates"] == 0
    assert summary["unresolved_after_cpu_screen"] == 4
    assert summary["by_status"] == {"screened_out": 4}
    assert summary["by_solver_status"] == {
        "bicgstab:screened_out": 1,
        "gmres:screened_out": 3,
    }
    assert summary["by_matrix_status"] == {
        "suitesparse:Gset/G17:screened_out": 2,
        "suitesparse:Zitney/extr1b:screened_out": 2,
    }
    assert all(row["status"] == "screened_out" for row in rows)
    assert all(row["gpu_probe_ready"] is False for row in rows)
    assert all(row["executes_gpu"] is False for row in rows)
    assert all(row["runtime_selector_changed"] is False for row in rows)
    assert all(row["failure_reason"].startswith("cpu_screen_") for row in rows)
    assert any(
        row["matrix_id"] == "suitesparse:Gset/G17"
        and row["candidate_id"] == "taichi_csr_bicgstab_none_maxiter2048_float64"
        and row["failure_reason"] == "cpu_screen_solution_error_above_tolerance"
        and row["solution_relative_error"] > 5.0e-3
        for row in rows
    )
    assert any(
        row["matrix_id"] == "suitesparse:Zitney/extr1b"
        and row["candidate_id"] == "taichi_csr_gmres_none_restart64_float64"
        and row["failure_reason"] == "cpu_screen_residual_above_tolerance"
        for row in rows
    )
    assert (
        summary["next_step"]
        == "defer_unresolved_to_future_preconditioners_or_formulation_diagnostics"
    )

    assert manifest.metadata["attempted_cpu_screens"] == 4
    assert manifest.metadata["screen_success_rows"] == 0
    assert manifest.metadata["cpu_screened_out_rows"] == 4
    assert manifest.metadata["gpu_probe_ready_candidates"] == 0
    assert manifest.metadata["executes_gpu"] is False
    assert manifest.metadata["runtime_selector_changed"] is False
