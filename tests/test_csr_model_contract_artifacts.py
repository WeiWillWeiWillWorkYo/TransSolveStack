import json
from pathlib import Path

from transsolvestack.profiling.artifacts import read_jsonl
from transsolvestack.profiling.provenance import read_manifest, verify_manifest_hashes


def test_csr_model_contract_artifacts_are_valid():
    root = Path("runs/phase1_csr_model_contract")
    requests = read_jsonl(root / "csr_model_requests.jsonl")
    targets = read_jsonl(root / "csr_model_targets.jsonl")
    predictions = read_jsonl(root / "csr_model_predictions.jsonl")
    summary = json.loads((root / "csr_model_contract_summary.json").read_text(encoding="utf-8"))
    schema = json.loads((root / "csr_model_contract_schema.json").read_text(encoding="utf-8"))
    manifest = read_manifest(root / "artifact_manifest.json")

    assert manifest.artifact_kind == "csr_model_contract_export"
    assert verify_manifest_hashes(manifest) == ()
    assert summary["status"] == "passed"
    assert summary["schema_version"] == "phase1_csr_model_contract_v1"
    assert schema["schema_version"] == summary["schema_version"]
    assert schema["model_required"] is False
    assert schema["runtime_integration"]["runtime_selector_changed"] is False
    assert summary["model_required"] is False
    assert summary["runtime_selector_changed"] is False
    assert len(requests) == 12
    assert len(targets) == 12
    assert len(predictions) == 3
    assert summary["num_request_candidates"] == 108
    assert summary["min_candidates_per_request"] == 9
    assert summary["max_candidates_per_request"] == 9
    assert summary["validation_error_count"] == 0
    assert summary["eval_oracle_top1_accuracy"] == 1.0 / 3.0
    assert summary["eval_profiled_selection_rate"] == 2.0 / 3.0
    assert summary["target_label_class_counts"] == {
        "not_applicable": 34,
        "not_profiled": 44,
        "screened_out": 15,
        "success_non_oracle": 11,
        "success_oracle": 4,
    }
    assert {row["evaluation_status"] for row in predictions} == {
        "oracle_match",
        "profiled_success_non_oracle",
        "no_profiled_success_candidate",
    }
