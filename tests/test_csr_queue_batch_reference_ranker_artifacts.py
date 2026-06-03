import json
import math
from pathlib import Path

from transsolvestack.profiling.artifacts import read_jsonl
from transsolvestack.profiling.provenance import read_manifest, verify_manifest_hashes


def test_csr_queue_batch_reference_ranker_artifacts_are_valid():
    root = Path("runs/phase1_csr_queue_batch_reference_ranker")
    model = json.loads(
        (root / "csr_queue_batch_reference_ranker_model.json").read_text(
            encoding="utf-8"
        )
    )
    predictions = read_jsonl(root / "csr_queue_batch_reference_ranker_predictions.jsonl")
    summary = json.loads(
        (root / "csr_queue_batch_reference_ranker_summary.json").read_text(
            encoding="utf-8"
        )
    )
    manifest = read_manifest(root / "artifact_manifest.json")

    assert manifest.artifact_kind == "csr_queue_batch_reference_ranker"
    assert verify_manifest_hashes(manifest) == ()
    assert summary["status"] == "passed"
    assert summary["model_family"] == "masked_self_attention_ranker_v1"
    assert summary["model_id"] == "csr_queue_batch_shadow_ranker_v1"
    assert summary["model_trained"] is True
    assert summary["runtime_selector_changed"] is False
    train_rows = [row for row in predictions if row["split"] == "train"]
    eval_rows = [row for row in predictions if row["split"] == "eval"]
    train_oracle_rows = [row for row in train_rows if row["oracle_candidate_id"]]
    eval_oracle_rows = [row for row in eval_rows if row["oracle_candidate_id"]]
    eval_non_success_rows = [
        row for row in eval_rows if row["selected_target_status"] != "success"
    ]
    eval_profiled_success_rows = [
        row
        for row in eval_rows
        if row["evaluation_status"]
        in {"oracle_match", "profiled_success_non_oracle"}
    ]
    assert summary["num_requests"] == len(predictions)
    assert summary["num_predictions"] == len(predictions)
    assert summary["num_train_requests"] == len(train_rows)
    assert summary["num_eval_requests"] == len(eval_rows)
    assert summary["num_train_oracle_requests"] == len(train_oracle_rows)
    assert summary["num_eval_oracle_requests"] == len(eval_oracle_rows)
    assert summary["num_global_candidates"] == 9
    assert summary["token_feature_dim"] == 37
    assert summary["d_model"] == 24
    assert summary["num_attention_heads"] == 4
    assert summary["feedforward_dim"] == 48
    assert summary["scorer_feature_dim"] == 62
    assert summary["num_epochs"] == 160
    assert summary["num_pairwise_constraints"] > 0
    assert summary["num_pairwise_updates"] > 0
    assert math.isfinite(summary["final_train_pairwise_loss"])
    assert 0.0 <= summary["train_oracle_top1_accuracy"] <= 1.0
    assert 0.0 <= summary["eval_oracle_top1_accuracy"] <= 1.0
    assert 0.0 <= summary["eval_oracle_top1_accuracy_all_requests"] <= 1.0
    assert summary["eval_profiled_success_selection_rate"] == (
        len(eval_profiled_success_rows) / len(eval_rows)
    )
    assert summary["eval_non_success_selection_count"] == len(eval_non_success_rows)
    assert math.isfinite(summary["eval_mean_regret_ms"])
    assert math.isfinite(summary["eval_max_regret_ms"])
    assert model["runtime_integration"]["runtime_selector_changed"] is False
    assert len(model["scorer_head"]) == summary["scorer_feature_dim"]
    assert len(model["token_feature_names"]) == summary["token_feature_dim"]
    assert manifest.metadata["shadow_only"] is True
    assert manifest.metadata["runtime_selector_changed"] is False
    assert all(row["selected_candidate_id"] == row["ranked_candidate_ids"][0] for row in predictions)
    assert all(set(row["ranked_candidate_ids"]) == set(row["scores"]) for row in predictions)
