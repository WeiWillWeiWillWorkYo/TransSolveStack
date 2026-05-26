from pathlib import Path
import json

from transsolvestack.profiling.artifacts import read_jsonl
from transsolvestack.profiling.provenance import read_manifest, verify_manifest_hashes


def test_suitesparse_selection_artifacts_are_valid():
    root = Path("runs/phase1_suitesparse_selection")
    selected = read_jsonl(root / "selected_matrices.jsonl")
    systems = read_jsonl(root / "selected_systems.jsonl")
    summary = json.loads((root / "selection_summary.json").read_text(encoding="utf-8"))
    manifest = read_manifest(root / "artifact_manifest.json")

    assert manifest.artifact_kind == "suitesparse_matrix_subset_selection"
    assert verify_manifest_hashes(manifest) == ()
    assert summary["status"] == "ready"
    assert summary["candidates_considered"] == 2904
    assert summary["candidates_after_filter"] > summary["selected_matrices"]
    assert summary["selected_matrices"] == 64
    assert len(selected) == 64
    assert len(systems) == 64
    assert all(row["source"] == "suitesparse" for row in selected)
    assert all(row["n_rows"] == row["n_cols"] for row in selected)
    assert all(row["is_real"] for row in selected)
    assert all(row["operator"]["kind"] == "assembled_sparse" for row in systems)
    assert all(row["operator"]["device_resident"] is False for row in systems)
    assert all(
        row["operator"]["metadata"]["import_status"] == "archive_indexed_not_loaded"
        for row in systems
    )
