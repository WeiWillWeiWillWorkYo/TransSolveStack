import json
from pathlib import Path

from transsolvestack.profiling.artifacts import read_jsonl
from transsolvestack.profiling.provenance import read_manifest, verify_manifest_hashes


def test_csr_learning_readiness_artifacts_are_valid():
    root = Path("runs/phase1_csr_learning_readiness")
    rows = read_jsonl(root / "csr_learning_rows.jsonl")
    predictions = read_jsonl(root / "csr_baseline_predictions.jsonl")
    summary = json.loads((root / "csr_learning_summary.json").read_text(encoding="utf-8"))
    schema = json.loads((root / "csr_learning_schema.json").read_text(encoding="utf-8"))
    manifest = read_manifest(root / "artifact_manifest.json")

    assert manifest.artifact_kind == "csr_learning_readiness_export"
    assert verify_manifest_hashes(manifest) == ()
    assert summary["status"] == "passed"
    assert summary["schema_version"] == "phase1_csr_learning_features_v1"
    assert schema["schema_version"] == summary["schema_version"]
    assert schema["model_required"] is False
    assert len(rows) == 108
    assert len(predictions) == 3
    assert summary["num_train_rows"] == 81
    assert summary["num_eval_rows"] == 27
    assert summary["num_train_matrices"] == 9
    assert summary["num_eval_matrices"] == 3
    assert summary["label_class_counts"] == {
        "not_applicable": 34,
        "not_profiled": 44,
        "screened_out": 15,
        "success_non_oracle": 11,
        "success_oracle": 4,
    }
    assert summary["num_eval_matrices_with_success"] == 2
    assert summary["num_eval_profiled_success_predictions"] == 2
    assert summary["eval_oracle_top1_accuracy"] == 1.0 / 3.0
    assert summary["eval_profiled_success_rate"] == 2.0 / 3.0
    assert summary["eval_max_regret_ms"] > 0.0
    assert {row["split"] for row in rows} == {"train", "eval"}
    assert {row["label_class"] for row in rows} == {
        "success_oracle",
        "success_non_oracle",
        "screened_out",
        "not_applicable",
        "not_profiled",
    }
    assert {row["status"] for row in predictions} == {
        "oracle_match",
        "profiled_success_non_oracle",
        "no_profiled_success_candidate",
    }
