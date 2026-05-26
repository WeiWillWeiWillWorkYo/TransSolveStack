from pathlib import Path
import json

from transsolvestack.profiling.artifacts import read_jsonl
from transsolvestack.profiling.provenance import read_manifest, verify_manifest_hashes


def test_taichi_csr_matvec_artifacts_are_valid():
    root = Path("runs/phase1_taichi_csr_matvec")
    rows = read_jsonl(root / "csr_matvec_results.jsonl")
    summary = json.loads((root / "csr_matvec_summary.json").read_text(encoding="utf-8"))
    manifest = read_manifest(root / "artifact_manifest.json")

    assert manifest.artifact_kind == "taichi_csr_matvec_smoke"
    assert verify_manifest_hashes(manifest) == ()
    assert summary["status"] == "passed"
    assert summary["num_matrices"] == 4
    assert summary["num_success"] == 4
    assert summary["num_failed"] == 0
    assert len(rows) == 4
    assert sum(row["csr_nnz"] for row in rows) == summary["total_csr_nnz"]
    assert all(row["status"] == "success" for row in rows)
    assert all(row["backend"] == "taichi_gpu" for row in rows)
    assert all(row["relative_l2_error"] <= 1.0e-5 for row in rows)
    assert all(row["max_abs_error"] <= row["tolerance_abs"] for row in rows)
