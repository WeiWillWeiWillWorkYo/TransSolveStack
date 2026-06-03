import json

import transsolvestack as tss
from transsolvestack.policies.csr_external_model_adapter import (
    adapt_csr_external_ranker_checkpoint_from_files,
)


def test_csr_external_model_adapter_converts_checkpoint_to_tss_ranker(tmp_path):
    checkpoint = tss.build_reference_csr_external_ranker_checkpoint()
    checkpoint_path = tmp_path / "external_checkpoint.json"
    checkpoint_path.write_text(
        json.dumps(checkpoint, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    export = adapt_csr_external_ranker_checkpoint_from_files(checkpoint_path)
    summary = export["summary"]
    ranker_summary = export["adapted_ranker_summary"]

    assert summary["status"] == "passed"
    assert summary["schema_version"] == "phase1_csr_external_model_adapter_v1"
    assert summary["checkpoint_schema_version"] == "phase1_csr_external_ranker_checkpoint_v1"
    assert summary["adapter"] == "csr_external_json_ranker_checkpoint_v1"
    assert summary["adapter_ready"] is True
    assert summary["quality_gate_input_ready"] is True
    assert summary["policy_model_artifact_input_ready"] is True
    assert summary["prediction_contract_valid"] is True
    assert summary["num_predictions"] == 20
    assert summary["runtime_selector_changed"] is False
    assert summary["validation_error_count"] == 0
    assert ranker_summary["schema_version"] == "phase1_csr_transformer_ranker_v1"
    assert ranker_summary["status"] == "passed"
    assert ranker_summary["num_eval_requests"] == 5
    assert ranker_summary["eval_oracle_top1_accuracy"] == 0.5
    assert ranker_summary["eval_profiled_success_selection_rate"] == 0.6


def test_csr_external_model_adapter_rejects_runtime_selector_changes(tmp_path):
    checkpoint = tss.build_reference_csr_external_ranker_checkpoint()
    checkpoint["runtime_contract"]["runtime_selector_changed"] = True
    checkpoint_path = tmp_path / "bad_external_checkpoint.json"
    checkpoint_path.write_text(
        json.dumps(checkpoint, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    try:
        adapt_csr_external_ranker_checkpoint_from_files(checkpoint_path)
    except ValueError as exc:
        assert "runtime selector" in str(exc)
    else:
        raise AssertionError("expected runtime selector mutation to be rejected")
