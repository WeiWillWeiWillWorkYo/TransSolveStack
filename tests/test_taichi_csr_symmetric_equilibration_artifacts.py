import json
from pathlib import Path

from transsolvestack.profiling.artifacts import read_jsonl
from transsolvestack.profiling.provenance import read_manifest, verify_manifest_hashes


def test_taichi_csr_symmetric_equilibration_artifacts_are_valid():
    root = Path("runs/phase1_taichi_csr_symmetric_equilibration")
    rows = read_jsonl(root / "csr_symmetric_equilibration_results.jsonl")
    summary = json.loads(
        (root / "csr_symmetric_equilibration_summary.json").read_text(
            encoding="utf-8"
        )
    )
    schema = json.loads(
        (root / "csr_symmetric_equilibration_schema.json").read_text(encoding="utf-8")
    )
    manifest = read_manifest(root / "artifact_manifest.json")

    assert manifest.artifact_kind == "taichi_csr_symmetric_equilibration_smoke"
    assert verify_manifest_hashes(manifest) == ()
    assert summary["status"] == "passed"
    assert summary["schema_version"] == "phase1_taichi_csr_symmetric_equilibration_v1"
    assert schema["schema_version"] == summary["schema_version"]
    assert schema["integration_boundary"]["executes_gpu"] is True
    assert summary["executes_gpu"] is True
    assert summary["runtime_selector_changed"] is False

    assert len(rows) == 1
    assert summary["candidate_rows"] == 1
    assert summary["gpu_executed_rows"] == 1
    assert summary["candidate_promoted"] is False
    assert summary["failed_numeric_gate_rows"] == 1
    assert summary["by_numeric_status"] == {"failed_numeric_gate": 1}

    row = rows[0]
    assert row["matrix_id"] == "suitesparse:HB/bcsstk07"
    assert row["backend"] == "taichi_gpu"
    assert row["gpu_executed"] is True
    assert row["solver"] == "pcg"
    assert row["preconditioner"] == "symmetric_equilibration"
    assert row["solver_status"] == "success"
    assert row["numeric_status"] == "failed_numeric_gate"
    assert row["candidate_promoted"] is False
    assert "solution_error_above_tolerance" in row["failure_reasons"]
    assert row["final_relative_residual"] <= summary["tolerance_rel"]
    assert row["cpu_recomputed_relative_residual"] <= 1.0e-4
    assert row["solution_relative_error"] > 5.0e-3
    assert row["trace_metadata"]["equilibration"] == "symmetric_diagonal"

    assert manifest.metadata["gpu_executed_rows"] == 1
    assert manifest.metadata["candidate_promoted"] is False
    assert manifest.metadata["executes_gpu"] is True
    assert manifest.metadata["runtime_selector_changed"] is False
