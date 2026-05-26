import json
import math
from pathlib import Path

from transsolvestack.profiling.artifacts import read_jsonl
from transsolvestack.profiling.provenance import read_manifest, verify_manifest_hashes


def test_taichi_csr_ilu0_artifacts_are_valid():
    root = Path("runs/phase1_taichi_csr_ilu0")
    rows = read_jsonl(root / "csr_ilu0_results.jsonl")
    summary = json.loads((root / "csr_ilu0_summary.json").read_text(encoding="utf-8"))
    schema = json.loads((root / "csr_ilu0_schema.json").read_text(encoding="utf-8"))
    manifest = read_manifest(root / "artifact_manifest.json")

    assert manifest.artifact_kind == "taichi_csr_ilu0_smoke"
    assert verify_manifest_hashes(manifest) == ()
    assert summary["status"] == "passed"
    assert summary["schema_version"] == "phase1_taichi_csr_ilu0_v1"
    assert schema["schema_version"] == summary["schema_version"]
    assert schema["integration_boundary"]["executes_gpu"] is True
    assert summary["executes_gpu"] is True
    assert summary["runtime_selector_changed"] is False

    assert len(rows) == summary["candidate_rows"]
    assert len(rows) >= 2
    assert summary["gpu_executed_rows"] == len(rows)
    assert summary["numeric_success_rows"] >= 1
    assert "suitesparse:Bai/cdde1" in {row["matrix_id"] for row in rows}

    for row in rows:
        assert row["backend"] == "taichi_gpu"
        assert row["gpu_executed"] is True
        assert row["solver"] == "bicgstab"
        assert row["preconditioner"] == "ilu0"
        assert row["trace_metadata"]["ilu0_valid"] is True
        assert row["trace_metadata"]["ilu0_factorization"] == (
            "doolittle_level_zero_csr_pattern"
        )
        assert row["runtime_selector_changed"] is False
        assert math.isfinite(float(row["final_relative_residual"]))
        assert math.isfinite(float(row["solution_relative_error"]))
        if row["numeric_status"] == "success":
            assert row["solver_status"] == "success"
            assert row["candidate_promoted"] is True
            assert row["final_relative_residual"] <= summary["tolerance_rel"]
            assert row["cpu_recomputed_relative_residual"] <= 1.0e-4
            assert row["solution_relative_error"] <= 5.0e-3

    assert manifest.metadata["gpu_executed_rows"] == len(rows)
    assert manifest.metadata["numeric_success_rows"] == summary["numeric_success_rows"]
    assert manifest.metadata["executes_gpu"] is True
    assert manifest.metadata["runtime_selector_changed"] is False
