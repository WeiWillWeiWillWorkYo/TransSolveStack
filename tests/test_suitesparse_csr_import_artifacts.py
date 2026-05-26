from pathlib import Path
import json
import math

from transsolvestack.profiling.artifacts import read_jsonl
from transsolvestack.profiling.provenance import read_manifest, verify_manifest_hashes


def test_suitesparse_csr_import_artifacts_are_valid():
    root = Path("runs/phase1_suitesparse_csr_import")
    rows = read_jsonl(root / "csr_matrices.jsonl")
    summary = json.loads((root / "csr_import_summary.json").read_text(encoding="utf-8"))
    manifest = read_manifest(root / "artifact_manifest.json")

    assert manifest.artifact_kind == "suitesparse_csr_import_boundary"
    assert verify_manifest_hashes(manifest) == ()
    assert summary["status"] == "passed"
    assert summary["eligible_rows"] >= summary["attempted_imports"]
    assert summary["attempted_imports"] == 12
    assert summary["imported_matrices"] == 12
    assert summary["failed_imports"] == 0
    assert len(rows) == 12
    assert sum(row["csr_nnz"] for row in rows) == summary["total_csr_nnz"]
    for row in rows:
        assert row["status"] == "success"
        assert len(row["row_ptr"]) == row["n_rows"] + 1
        assert row["row_ptr"][0] == 0
        assert row["row_ptr"][-1] == row["csr_nnz"]
        assert len(row["col_ind"]) == row["csr_nnz"]
        assert len(row["values"]) == row["csr_nnz"]
        assert all(
            row["row_ptr"][i] <= row["row_ptr"][i + 1]
            for i in range(row["n_rows"])
        )
        assert all(0 <= col < row["n_cols"] for col in row["col_ind"])
        assert all(math.isfinite(float(value)) for value in row["values"])
