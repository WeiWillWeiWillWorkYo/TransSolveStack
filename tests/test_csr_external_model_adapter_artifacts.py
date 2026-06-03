import json
from pathlib import Path

from transsolvestack.profiling.artifacts import read_jsonl
from transsolvestack.profiling.provenance import read_manifest, verify_manifest_hashes


def test_csr_external_model_adapter_artifacts_are_valid():
    root = Path("runs/phase1_csr_external_model_adapter")
    checkpoint = json.loads(
        (root / "external_csr_ranker_checkpoint.json").read_text(encoding="utf-8")
    )
    adapted_model = json.loads(
        (root / "adapted_csr_transformer_ranker_model.json").read_text(
            encoding="utf-8"
        )
    )
    predictions = read_jsonl(root / "adapted_csr_transformer_ranker_predictions.jsonl")
    ranker_summary = json.loads(
        (root / "adapted_csr_transformer_ranker_summary.json").read_text(
            encoding="utf-8"
        )
    )
    rows = read_jsonl(root / "csr_external_model_adapter_rows.jsonl")
    summary = json.loads(
        (root / "csr_external_model_adapter_summary.json").read_text(encoding="utf-8")
    )
    schema = json.loads(
        (root / "csr_external_model_adapter_schema.json").read_text(encoding="utf-8")
    )
    manifest = read_manifest(root / "artifact_manifest.json")

    assert manifest.artifact_kind == "csr_external_model_adapter"
    assert verify_manifest_hashes(manifest) == ()
    assert summary["status"] == "passed"
    assert summary["schema_version"] == "phase1_csr_external_model_adapter_v1"
    assert schema["schema_version"] == summary["schema_version"]
    assert checkpoint["schema_version"] == "phase1_csr_external_ranker_checkpoint_v1"
    assert checkpoint["runtime_contract"]["guard_required"] is True
    assert checkpoint["runtime_contract"]["runtime_selector_changed"] is False
    assert adapted_model["schema_version"] == "phase1_csr_transformer_ranker_v1"
    assert adapted_model["model_family"] == "masked_self_attention_ranker_v1"
    assert ranker_summary["status"] == "passed"
    assert ranker_summary["schema_version"] == adapted_model["schema_version"]
    assert summary["adapter_ready"] is True
    assert summary["quality_gate_input_ready"] is True
    assert summary["policy_model_artifact_input_ready"] is True
    assert summary["prediction_contract_valid"] is True
    assert summary["num_predictions"] == 20
    assert len(predictions) == 20
    assert summary["num_eval_predictions"] == 5
    assert summary["runtime_selector_changed"] is False
    assert summary["validation_error_count"] == 0
    assert {row["row_kind"] for row in rows} == {
        "checkpoint_contract",
        "tensor_contract",
        "adapted_model",
        "prediction_contract",
        "runtime_boundary",
        "adapter_decision",
    }
    assert all(row["selected_candidate_id"] == row["ranked_candidate_ids"][0] for row in predictions)
    assert manifest.metadata["adapter_ready"] is True
    assert manifest.metadata["quality_gate_input_ready"] is True
    assert manifest.metadata["policy_model_artifact_input_ready"] is True
    assert manifest.metadata["runtime_selector_changed"] is False
