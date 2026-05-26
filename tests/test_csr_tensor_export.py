from transsolvestack.policies.csr_tensor_export import build_csr_tensor_export


def test_csr_tensor_export_builds_numeric_arrays():
    export = build_csr_tensor_export(
        "runs/phase1_csr_model_contract/csr_model_requests.jsonl",
        "runs/phase1_csr_model_contract/csr_model_targets.jsonl",
    )

    assert export.summary.status == "passed"
    assert export.summary.schema_version == "phase1_csr_training_tensors_v1"
    assert export.summary.num_requests == 12
    assert export.summary.num_train_requests == 9
    assert export.summary.num_eval_requests == 3
    assert export.summary.num_global_candidates == 9
    assert export.summary.num_active_candidate_slots == 108
    assert export.summary.num_oracle_targets == 4
    assert export.summary.num_requests_without_oracle == 8
    assert export.summary.matrix_feature_dim == 21
    assert export.summary.candidate_feature_dim == 16
    assert export.summary.has_missing_candidate_slots is False
    assert export.summary.model_required is False
    assert export.summary.runtime_selector_changed is False
    assert len(export.arrays["matrix_features"]) == 12
    assert len(export.arrays["matrix_features"][0]) == export.summary.matrix_feature_dim
    assert len(export.arrays["candidate_features"]) == 12
    assert len(export.arrays["candidate_features"][0]) == 9
    assert (
        len(export.arrays["candidate_features"][0][0])
        == export.summary.candidate_feature_dim
    )
    assert sum(sum(row) for row in export.arrays["candidate_mask"]) == 108
    assert sum(1 for value in export.arrays["oracle_index"] if value >= 0) == 4
    assert set(export.arrays["split_ids"]) == {0, 1}


def test_csr_tensor_export_schema_records_label_mappings():
    export = build_csr_tensor_export(
        "runs/phase1_csr_model_contract/csr_model_requests.jsonl",
        "runs/phase1_csr_model_contract/csr_model_targets.jsonl",
    )

    assert export.schema["storage_format"] == "json_numeric_arrays_v1"
    assert len(export.schema["axes"]["matrix_feature"]) == 21
    assert len(export.schema["axes"]["candidate_feature"]) == 16
    assert export.schema["label_class_to_id"]["success_oracle"] == 4
    assert export.schema["target_status_to_id"]["success"] == 3
    assert "matrix_features" in export.schema["arrays"]
    assert "candidate_features" in export.schema["arrays"]
    assert "oracle_index" in export.schema["arrays"]
