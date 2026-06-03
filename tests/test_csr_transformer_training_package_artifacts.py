import json
from pathlib import Path

from transsolvestack.profiling.artifacts import read_jsonl
from transsolvestack.profiling.provenance import read_manifest, verify_manifest_hashes


def test_csr_transformer_training_package_artifacts_are_valid():
    root = Path("runs/phase1_csr_transformer_training_package")
    package = json.loads(
        (root / "csr_transformer_training_package_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    rows = read_jsonl(root / "csr_transformer_training_package_rows.jsonl")
    summary = json.loads(
        (root / "csr_transformer_training_package_summary.json").read_text(
            encoding="utf-8"
        )
    )
    schema = json.loads(
        (root / "csr_transformer_training_package_schema.json").read_text(
            encoding="utf-8"
        )
    )
    readme = (root / "TRAINING_PACKAGE.md").read_text(encoding="utf-8")
    manifest = read_manifest(root / "artifact_manifest.json")

    assert manifest.artifact_kind == "csr_transformer_training_package"
    assert verify_manifest_hashes(manifest) == ()
    assert summary["status"] == "passed"
    assert summary["schema_version"] == "phase1_csr_transformer_training_package_v1"
    assert package["schema_version"] == summary["schema_version"]
    assert schema["schema_version"] == summary["schema_version"]
    assert summary["package_kind"] == "csr_transformer_external_training_package"
    assert summary["package_ready"] is True
    assert summary["source_handoff_ready"] is True
    assert summary["model_training_required"] is True
    assert summary["model_trained"] is False
    assert summary["runtime_selector_changed"] is False
    assert summary["executes_gpu"] is False
    assert summary["copies_large_files"] is False
    assert summary["input_reference_mode"] == "path_reference_with_sha256"
    assert summary["validation_error_count"] == 0
    assert summary["internal_development_docs_included"] is False

    assert summary["num_input_files"] == 16
    assert summary["input_total_size_bytes"] > 1_000_000
    assert summary["num_expected_outputs"] == 6
    assert summary["num_validation_steps"] == 6
    assert summary["training_model_requests"] == 105
    assert summary["training_tensor_requests"] == 105
    assert summary["training_train_requests"] == 79
    assert summary["training_eval_requests"] == 26
    assert summary["training_global_candidates"] == 12
    assert summary["matrix_feature_dim"] == 21
    assert summary["candidate_feature_dim"] == 19
    assert summary["current_shadow_baseline_model_id"] == (
        "csr_blocked_gap_augmented_shadow_ranker_v1"
    )
    assert summary["current_shadow_baseline_runtime_eligible"] is False
    assert summary["guard_actual_quality_gate_blocks"] == 5
    assert summary["guard_actual_runtime_selector_changes"] == 0
    assert summary["guard_fallback_chain_enforced_rows"] == 5
    assert set(summary["terms_documents_required"]) == {
        "MODEL_CONTRIBUTION_TERMS.md",
        "CONTRIBUTOR_LICENSE_AGREEMENT.md",
    }

    assert schema["runtime_boundary"]["model_trained"] is False
    assert schema["runtime_boundary"]["runtime_selector_changed"] is False
    assert schema["runtime_boundary"]["executes_gpu"] is False
    assert schema["runtime_boundary"]["guard_required"] is True
    assert package["runtime_boundary"]["default_mode"] == "shadow"
    assert package["runtime_boundary"]["guard_required"] is True
    assert package["contribution_terms"]["commercial_use_requires_permission"] is True
    assert package["contribution_terms"]["attribution"] == "Wei CUI"

    input_rows = [row for row in rows if row["row_kind"] == "input_file"]
    output_rows = [row for row in rows if row["row_kind"] == "expected_output"]
    step_rows = [row for row in rows if row["row_kind"] == "validation_step"]
    assert len(input_rows) == summary["num_input_files"]
    assert len(output_rows) == summary["num_expected_outputs"]
    assert len(step_rows) == summary["num_validation_steps"]
    assert all(row["exists"] is True for row in input_rows)
    assert all(row["sha256"] for row in input_rows)
    forbidden = ("docs/toms_paper/", "MILESTONE_LOG", "SCOPE_PLATFORM", "ADVICE")
    assert not any(str(row["path"]).startswith(forbidden) for row in input_rows)
    assert {row["step_id"] for row in step_rows} == {
        "saved_model_replay",
        "quality_gate",
        "policy_model_artifact",
        "policy_model_acceptance",
        "policy_model_submission",
        "guarded_shadow_smoke",
    }
    assert "scripts/tss_csr_transformer_model_replay.py" in readme
    assert "This package does not train a model." in readme

    assert manifest.metadata["package_ready"] is True
    assert manifest.metadata["model_trained"] is False
    assert manifest.metadata["runtime_selector_changed"] is False
    assert manifest.metadata["executes_gpu"] is False
    assert manifest.metadata["internal_development_docs_included"] is False
