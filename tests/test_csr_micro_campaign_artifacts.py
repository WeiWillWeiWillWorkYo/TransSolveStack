import json
from pathlib import Path

from transsolvestack.profiling.artifacts import read_jsonl
from transsolvestack.profiling.provenance import read_manifest, verify_manifest_hashes


def test_csr_micro_campaign_artifacts_are_valid():
    root = Path("runs/phase1_csr_micro_campaign")
    results = read_jsonl(root / "csr_micro_campaign_results.jsonl")
    selector_rows = read_jsonl(root / "csr_micro_selector_rows.jsonl")
    summary = json.loads((root / "csr_micro_campaign_summary.json").read_text(encoding="utf-8"))
    selector_summary = json.loads((root / "csr_micro_selector_summary.json").read_text(encoding="utf-8"))
    manifest = read_manifest(root / "artifact_manifest.json")

    assert manifest.artifact_kind == "csr_micro_campaign"
    assert verify_manifest_hashes(manifest) == ()
    assert summary["status"] == "passed"
    assert selector_summary["status"] == "passed"
    assert summary["schema_version"] == "phase1_csr_micro_campaign_v1"
    assert summary["executes_gpu"] is True
    assert summary["runtime_selector_changed"] is False
    assert summary["imported_matrices"] == 8
    assert summary["candidate_jobs"] == 24
    assert len(results) == 24
    assert summary["gpu_success_rows"] == 8
    assert summary["cpu_screened_out_rows"] == 16
    assert summary["gpu_failed_rows"] == 0
    assert summary["selector_rows"] == 24
    assert len(selector_rows) == 24
    assert summary["selector_oracle_rows"] == 4
    assert summary["by_status"] == {"screened_out": 16, "success": 8}
    assert summary["by_solver_status"] == {
        "bicgstab:screened_out": 4,
        "bicgstab:success": 4,
        "cg:screened_out": 4,
        "gmres:screened_out": 2,
        "gmres:success": 2,
        "pcg:screened_out": 3,
        "pcg:success": 1,
        "richardson:screened_out": 3,
        "richardson:success": 1,
    }
    assert summary["max_final_relative_residual"] <= 1.0e-5
    assert summary["max_cpu_recomputed_relative_residual"] <= 1.0e-4
    assert summary["max_solution_relative_error"] <= 5.0e-3
    assert all(row["status"] != "failed" for row in results)
    assert sum(1 for row in results if row["backend"] == "taichi_gpu") == 8
    assert sum(1 for row in results if row["backend"] == "cpu_reference_screen") == 16

