import json
from pathlib import Path

from transsolvestack.profiling.artifacts import read_jsonl
from transsolvestack.profiling.provenance import read_manifest, verify_manifest_hashes


def test_csr_queue_batch_model_replay_artifacts_are_valid():
    root = Path("runs/phase1_csr_queue_batch_model_replay")
    predictions = read_jsonl(root / "csr_queue_batch_model_replay_predictions.jsonl")
    comparison = read_jsonl(root / "csr_queue_batch_model_replay_comparison.jsonl")
    summary = json.loads(
        (root / "csr_queue_batch_model_replay_summary.json").read_text(
            encoding="utf-8"
        )
    )
    manifest = read_manifest(root / "artifact_manifest.json")

    assert manifest.artifact_kind == "csr_queue_batch_model_replay"
    assert verify_manifest_hashes(manifest) == ()
    assert summary["status"] == "passed"
    assert summary["model_family"] == "masked_self_attention_ranker_v1"
    assert summary["model_id"] == "csr_queue_batch_shadow_ranker_v1"
    assert summary["model_loaded"] is True
    assert summary["model_trained"] is True
    assert summary["runtime_selector_changed"] is False
    assert summary["num_predictions"] == len(predictions)
    assert summary["num_reference_predictions"] == len(comparison)
    assert len(comparison) == len(predictions)
    assert summary["exact_replay"] is True
    assert summary["selected_candidate_mismatch_count"] == 0
    assert summary["ranked_order_mismatch_count"] == 0
    assert summary["evaluation_status_mismatch_count"] == 0
    assert summary["missing_reference_count"] == 0
    assert summary["max_abs_score_delta"] == 0.0
    assert summary["prediction_contract_valid"] is True
    assert manifest.metadata["shadow_only"] is True
    assert manifest.metadata["exact_replay"] is True
    assert manifest.metadata["runtime_selector_changed"] is False
    assert all(row["selected_candidate_match"] is True for row in comparison)
    assert all(row["ranked_order_match"] is True for row in comparison)
    assert all(row["evaluation_status_match"] is True for row in comparison)
