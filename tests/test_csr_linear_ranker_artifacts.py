import json
import math
from pathlib import Path

from transsolvestack.profiling.artifacts import read_jsonl
from transsolvestack.profiling.provenance import read_manifest, verify_manifest_hashes


def test_csr_linear_ranker_artifacts_are_valid():
    root = Path("runs/phase1_csr_linear_ranker")
    model = json.loads((root / "csr_linear_ranker_model.json").read_text(encoding="utf-8"))
    predictions = read_jsonl(root / "csr_linear_ranker_predictions.jsonl")
    summary = json.loads((root / "csr_linear_ranker_summary.json").read_text(encoding="utf-8"))
    schema = json.loads((root / "csr_linear_ranker_schema.json").read_text(encoding="utf-8"))
    manifest = read_manifest(root / "artifact_manifest.json")

    assert manifest.artifact_kind == "csr_linear_ranker_baseline"
    assert verify_manifest_hashes(manifest) == ()
    assert summary["status"] == "passed"
    assert summary["schema_version"] == "phase1_csr_linear_ranker_v1"
    assert summary["model_family"] == "pairwise_linear_ranker_v1"
    assert summary["model_trained"] is True
    assert summary["runtime_selector_changed"] is False
    assert schema["schema_version"] == summary["schema_version"]
    assert schema["model_required"] is True
    assert schema["runtime_selector_changed"] is False
    assert model["schema_version"] == summary["schema_version"]
    assert model["model_family"] == summary["model_family"]
    assert model["model_id"] == summary["model_id"]
    assert model["training"]["objective"] == (
        "label_utility_pairwise_hinge_with_oracle_priority"
    )
    assert model["runtime_integration"]["runtime_selector_changed"] is False

    assert summary["num_requests"] == 12
    assert summary["num_predictions"] == 12
    assert len(predictions) == 12
    assert summary["num_train_requests"] == 9
    assert summary["num_eval_requests"] == 3
    assert summary["num_train_oracle_requests"] == 2
    assert summary["num_eval_oracle_requests"] == 2
    assert summary["num_global_candidates"] == 9
    assert summary["num_features"] == 374
    assert summary["num_pairwise_constraints"] == 110
    assert summary["num_pairwise_updates"] == 148
    assert math.isfinite(float(summary["final_train_pairwise_loss"]))
    assert summary["train_oracle_top1_accuracy"] == 0.5
    assert summary["eval_oracle_top1_accuracy"] == 0.0
    assert summary["eval_profiled_success_selection_rate"] == 0.0
    assert summary["eval_non_success_selection_count"] == 3
    assert summary["eval_mean_regret_ms"] is None
    assert summary["eval_max_regret_ms"] is None

    assert len(model["feature_names"]) == 374
    assert len(model["weights"]) == 374
    assert len(model["normalization"]["mean"]) == 374
    assert len(model["normalization"]["scale"]) == 374
    assert set(model["weights"]) == set(model["feature_names"])
    assert all(math.isfinite(float(value)) for value in model["weights"].values())
    assert all(
        math.isfinite(float(value)) and float(value) > 0.0
        for value in model["normalization"]["scale"].values()
    )

    assert {row["evaluation_status"] for row in predictions} == {
        "oracle_match",
        "non_success_selected",
        "no_oracle_target",
        "profiled_success_non_oracle",
    }
    for row in predictions:
        ranked = row["ranked_candidate_ids"]
        scores = row["scores"]
        assert len(ranked) == 9
        assert len(scores) == 9
        assert len(set(ranked)) == 9
        assert set(ranked) == set(scores)
        assert ranked[0] == row["selected_candidate_id"]
        assert scores[ranked[0]] == row["selected_score"]
        assert all(scores[left] >= scores[right] for left, right in zip(ranked, ranked[1:]))
