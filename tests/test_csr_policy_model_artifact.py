import json

import transsolvestack as tss
from transsolvestack.policies.csr_policy_model_artifact import (
    load_csr_policy_model_artifact,
)


def test_csr_policy_model_artifact_formalizes_runtime_loader_contract(tmp_path):
    export = tss.build_csr_policy_model_artifact()
    summary = export["summary"]
    artifact = export["artifact"]

    assert summary["status"] == "passed"
    assert summary["schema_version"] == "phase1_csr_policy_model_artifact_v1"
    assert summary["artifact_ready"] is True
    assert summary["adapter"] == "csr_transformer_ranker_saved_model_v1"
    assert summary["model_id"] == "csr_masked_self_attention_ranker_v1"
    assert summary["model_family"] == "masked_self_attention_ranker_v1"
    assert summary["source_model_schema_version"] == "phase1_csr_transformer_ranker_v1"
    assert summary["model_loaded"] is True
    assert summary["prediction_contract_checked"] is True
    assert summary["num_request_index_rows"] == 20
    assert summary["num_predictions"] == 20
    assert summary["replay_exact"] is True
    assert summary["guard_required"] is True
    assert summary["runtime_selector_changed"] is False
    assert summary["validation_error_count"] == 0

    assert artifact["runtime_contract"]["guard_required"] is True
    assert artifact["runtime_contract"]["default_mode"] == "shadow"
    assert artifact["runtime_contract"]["promotion_mode"] == "promote_if_safe"
    assert artifact["runtime_contract"]["runtime_selector_changed"] is False
    assert artifact["inputs"]["model_path"].endswith("csr_transformer_ranker_model.json")
    assert {row["row_kind"] for row in export["rows"]} == {
        "model",
        "tensor_contract",
        "request_index",
        "quality_gate",
        "runtime_contract",
    }

    artifact_path = tmp_path / "csr_policy_model_artifact.json"
    artifact_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    loaded = load_csr_policy_model_artifact(artifact_path)
    assert loaded["adapter"] == "csr_transformer_ranker_saved_model_v1"
