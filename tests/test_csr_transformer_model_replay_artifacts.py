import json
from pathlib import Path

from transsolvestack.profiling.artifacts import read_jsonl
from transsolvestack.profiling.provenance import read_manifest, verify_manifest_hashes


def test_csr_transformer_model_replay_artifacts_are_valid():
    root = Path("runs/phase1_csr_transformer_model_replay")
    predictions = read_jsonl(root / "csr_transformer_model_replay_predictions.jsonl")
    comparison = read_jsonl(root / "csr_transformer_model_replay_comparison.jsonl")
    summary = json.loads(
        (root / "csr_transformer_model_replay_summary.json").read_text(encoding="utf-8")
    )
    schema = json.loads(
        (root / "csr_transformer_model_replay_schema.json").read_text(encoding="utf-8")
    )
    manifest = read_manifest(root / "artifact_manifest.json")

    assert manifest.artifact_kind == "csr_transformer_model_replay"
    assert verify_manifest_hashes(manifest) == ()
    assert summary["status"] == "passed"
    assert summary["schema_version"] == "phase1_csr_transformer_model_replay_v1"
    assert schema["schema_version"] == summary["schema_version"]
    assert summary["source_model_schema_version"] == "phase1_csr_transformer_ranker_v1"
    assert summary["model_family"] == "masked_self_attention_ranker_v1"
    assert summary["model_id"] == "csr_masked_self_attention_ranker_v1"
    assert summary["model_loaded"] is True
    assert summary["model_trained"] is True
    assert summary["runtime_selector_changed"] is False
    assert summary["num_predictions"] == 20
    assert summary["num_reference_predictions"] == 20
    assert summary["exact_replay"] is True
    assert summary["selected_candidate_mismatch_count"] == 0
    assert summary["ranked_order_mismatch_count"] == 0
    assert summary["evaluation_status_mismatch_count"] == 0
    assert summary["missing_reference_count"] == 0
    assert summary["max_abs_score_delta"] <= 1.0e-9
    assert summary["prediction_contract_valid"] is True
    assert len(predictions) == 20
    assert len(comparison) == 20
    assert all(row["selected_candidate_match"] for row in comparison)
    assert all(row["ranked_order_match"] for row in comparison)
    assert all(row["evaluation_status_match"] for row in comparison)
    assert manifest.metadata["model_loaded"] is True
    assert manifest.metadata["exact_replay"] is True
