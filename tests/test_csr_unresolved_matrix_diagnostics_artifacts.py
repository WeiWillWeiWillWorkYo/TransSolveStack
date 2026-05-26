import json
from pathlib import Path

from transsolvestack.profiling.artifacts import read_jsonl
from transsolvestack.profiling.provenance import read_manifest, verify_manifest_hashes


def test_csr_unresolved_matrix_diagnostics_artifacts_are_valid():
    root = Path("runs/phase1_csr_unresolved_matrix_diagnostics")
    rows = read_jsonl(root / "csr_unresolved_matrix_diagnostics.jsonl")
    summary = json.loads(
        (root / "csr_unresolved_matrix_diagnostics_summary.json").read_text(
            encoding="utf-8"
        )
    )
    schema = json.loads(
        (root / "csr_unresolved_matrix_diagnostics_schema.json").read_text(
            encoding="utf-8"
        )
    )
    manifest = read_manifest(root / "artifact_manifest.json")

    assert manifest.artifact_kind == "csr_unresolved_matrix_diagnostics"
    assert verify_manifest_hashes(manifest) == ()
    assert summary["status"] == "passed"
    assert summary["schema_version"] == "phase1_csr_unresolved_matrix_diagnostics_v1"
    assert schema["schema_version"] == summary["schema_version"]
    assert schema["integration_boundary"]["diagnostic_only"] is True
    assert summary["executes_gpu"] is False
    assert summary["runtime_selector_changed"] is False

    assert len(rows) == 2
    assert summary["diagnosed_matrices"] == 2
    assert summary["source_cpu_screen_rows"] == 4
    assert summary["source_gpu_probe_ready_candidates"] == 0
    assert summary["by_recommended_route"] == {
        "formulation_diagnostic_required": 1,
        "ilu_or_nonsymmetric_scaling_preconditioner": 1,
    }
    assert summary["matrix_routes"] == {
        "suitesparse:Gset/G17": "formulation_diagnostic_required",
        "suitesparse:Zitney/extr1b": "ilu_or_nonsymmetric_scaling_preconditioner",
    }
    assert all(row["executes_gpu"] is False for row in rows)
    assert all(row["runtime_selector_changed"] is False for row in rows)

    gset = _matrix_row(rows, "suitesparse:Gset/G17")
    assert gset["recommended_route"] == "formulation_diagnostic_required"
    assert gset["diagnostics"]["diagonal"]["zero_count"] == 800
    assert gset["diagnostics"]["dense_cholesky_probe"]["status"] == "failed"
    assert gset["diagnostics"]["symmetric_spectrum_probe"]["ritz_min"] < 0.0

    zitney = _matrix_row(rows, "suitesparse:Zitney/extr1b")
    assert zitney["recommended_route"] == "ilu_or_nonsymmetric_scaling_preconditioner"
    assert zitney["diagnostics"]["diagonal"]["zero_count"] == 2834
    assert zitney["diagnostics"]["dense_cholesky_probe"]["status"] == "skipped_nonsymmetric"

    assert (
        summary["next_step"]
        == "implement_and_test_preconditioner_or_formulation_candidates_before_gpu_probe"
    )
    assert manifest.metadata["diagnosed_matrices"] == 2
    assert manifest.metadata["executes_gpu"] is False
    assert manifest.metadata["runtime_selector_changed"] is False


def _matrix_row(rows, matrix_id):
    return next(row for row in rows if row["matrix_id"] == matrix_id)
