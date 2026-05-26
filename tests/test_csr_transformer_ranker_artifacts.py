import json
import math
from pathlib import Path

from transsolvestack.profiling.artifacts import read_jsonl
from transsolvestack.profiling.provenance import read_manifest, verify_manifest_hashes


def test_csr_transformer_ranker_artifacts_are_valid():
    root = Path("runs/phase1_csr_transformer_ranker")
    predictions = read_jsonl(root / "csr_transformer_ranker_predictions.jsonl")
    summary = json.loads((root / "csr_transformer_ranker_summary.json").read_text(encoding="utf-8"))
    schema = json.loads((root / "csr_transformer_ranker_schema.json").read_text(encoding="utf-8"))
    model = json.loads((root / "csr_transformer_ranker_model.json").read_text(encoding="utf-8"))
    manifest = read_manifest(root / "artifact_manifest.json")

    assert manifest.artifact_kind == "csr_transformer_ranker"
    assert verify_manifest_hashes(manifest) == ()
    assert summary["status"] == "passed"
    assert summary["schema_version"] == "phase1_csr_transformer_ranker_v1"
    assert summary["model_family"] == "masked_self_attention_ranker_v1"
    assert summary["model_id"] == "csr_masked_self_attention_ranker_v1"
    assert summary["model_trained"] is True
    assert summary["runtime_selector_changed"] is False
    assert schema["schema_version"] == summary["schema_version"]
    assert model["schema_version"] == summary["schema_version"]

    assert summary["num_requests"] == 20
    assert summary["num_predictions"] == 20
    assert len(predictions) == 20
    assert summary["num_train_requests"] == 15
    assert summary["num_eval_requests"] == 5
    assert summary["num_train_oracle_requests"] == 4
    assert summary["num_eval_oracle_requests"] == 4
    assert summary["num_global_candidates"] == 9
    assert summary["token_feature_dim"] == 37
    assert summary["d_model"] == 24
    assert summary["num_attention_heads"] == 4
    assert summary["feedforward_dim"] == 48
    assert summary["scorer_feature_dim"] == 62
    assert summary["num_epochs"] == 160
    assert summary["num_pairwise_constraints"] == 116
    assert summary["num_pairwise_updates"] == 3264

    assert summary["train_oracle_top1_accuracy"] == 0.5
    assert summary["eval_oracle_top1_accuracy"] == 0.5
    assert summary["eval_profiled_success_selection_rate"] == 0.6
    assert summary["eval_non_success_selection_count"] == 2
    assert summary["eval_mean_regret_ms"] == 36.56358985851208
    assert summary["eval_max_regret_ms"] == 109.69076957553625

    eval_rows = [row for row in predictions if row["split"] == "eval"]
    assert len(eval_rows) == 5
    assert sum(row["evaluation_status"] == "oracle_match" for row in eval_rows) == 2
    assert sum(row["selected_target_status"] == "success" for row in eval_rows) == 3
    for row in predictions:
        ranked = row["ranked_candidate_ids"]
        assert ranked
        assert row["selected_candidate_id"] == ranked[0]
        assert len(ranked) == len(set(ranked))
        assert set(ranked) == set(row["scores"])
        assert all(math.isfinite(float(value)) for value in row["scores"].values())

    assert len(model["scorer_head"]) == summary["scorer_feature_dim"]
    assert manifest.metadata["model_trained"] is True
    assert manifest.metadata["runtime_selector_changed"] is False
    assert manifest.metadata["eval_oracle_top1_accuracy"] == 0.5
    assert manifest.metadata["eval_non_success_selection_count"] == 2
