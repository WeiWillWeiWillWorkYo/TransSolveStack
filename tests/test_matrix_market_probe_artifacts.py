from pathlib import Path

from transsolvestack.profiling.artifacts import read_jsonl
from transsolvestack.profiling.provenance import read_manifest, verify_manifest_hashes


def test_matrix_market_probe_artifacts_are_valid():
    root = Path("runs/phase1_matrix_market_probe")
    rows = read_jsonl(root / "matrix_metadata.jsonl")
    manifest = read_manifest(root / "artifact_manifest.json")
    assert len(rows) == 1
    assert manifest.artifact_kind == "matrix_market_metadata_probe"
    assert manifest.metadata["num_matrices"] == 1
    assert verify_manifest_hashes(manifest) == ()
    assert rows[0]["path"] == "tests/fixtures/tiny_spd.mtx"
    assert rows[0]["storage_format"] == "coordinate"
    assert rows[0]["n_rows"] == 3
    assert rows[0]["n_cols"] == 3
    assert rows[0]["nnz"] == 5
    assert rows[0]["symmetry"] == "symmetric"
