from pathlib import Path
import json

from transsolvestack.profiling.artifacts import read_jsonl
from transsolvestack.profiling.provenance import read_manifest, verify_manifest_hashes


def test_suitesparse_header_probe_artifacts_are_valid():
    root = Path("runs/phase1_suitesparse_header_probe")
    rows = read_jsonl(root / "archive_header_probe.jsonl")
    summary = json.loads((root / "archive_header_summary.json").read_text(encoding="utf-8"))
    manifest = read_manifest(root / "artifact_manifest.json")

    assert manifest.artifact_kind == "suitesparse_archive_header_probe"
    assert verify_manifest_hashes(manifest) == ()
    assert summary["status"] == "passed"
    assert summary["selected_matrices"] == 64
    assert summary["probed_archives"] == 64
    assert summary["success_count"] == 64
    assert summary["failure_count"] == 0
    assert summary["shape_mismatch_count"] == 0
    assert len(rows) == 64
    assert all(row["status"] == "success" for row in rows)
    assert all(row["shape_matches"] is True for row in rows)
    assert all(row["storage_format"] == "coordinate" for row in rows)
    assert summary["stored_nnz_mismatch_count"] >= 0
    assert summary["field_mismatch_count"] >= 0
