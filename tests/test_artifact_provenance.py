from pathlib import Path

from transsolvestack.profiling.provenance import read_manifest, verify_manifest_hashes


def test_benchmark_artifact_manifest_is_current():
    manifest = read_manifest("runs/phase1_smoke_taichi/artifact_manifest.json")
    assert manifest.artifact_kind == "taichi_smoke_benchmark"
    assert manifest.metadata["num_performance_records"] == 21
    assert verify_manifest_hashes(manifest) == ()


def test_regression_artifact_manifest_is_current():
    manifest = read_manifest("runs/phase1_numerical_regression/artifact_manifest.json")
    assert manifest.artifact_kind == "taichi_numerical_regression"
    assert manifest.metadata["num_records"] == 21
    assert manifest.metadata["num_passed"] == 21
    assert verify_manifest_hashes(manifest) == ()
