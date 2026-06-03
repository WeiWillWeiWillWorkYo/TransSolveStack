import transsolvestack as tss


def test_csr_policy_model_submission_package_is_shadow_ready():
    export = tss.prepare_csr_policy_model_submission()
    summary = export["summary"]
    submission = export["submission"]

    assert summary["status"] == "passed"
    assert summary["schema_version"] == "phase1_csr_policy_model_submission_v1"
    assert summary["submission_ready"] is True
    assert summary["shadow_submission_ready"] is True
    assert summary["runtime_promotion_ready"] is False
    assert summary["default_runtime_mode"] == "shadow"
    assert summary["terms_acknowledged"] is True
    assert summary["accepted_for_shadow"] is True
    assert summary["accepted_for_runtime_promotion"] is False
    assert summary["guarded_gpu_shadow_smoke_checked"] is True
    assert summary["guarded_gpu_smoke_successes"] == 2
    assert summary["num_packaged_files"] == 8
    assert summary["all_file_checksums_present"] is True
    assert summary["runtime_selector_changed"] is False
    assert summary["validation_error_count"] == 0

    assert submission["runtime_boundary"]["guard_required"] is True
    assert submission["runtime_boundary"]["runtime_selector_changed"] is False
    assert len(submission["files"]) == 8
    assert all(item["exists"] and item["sha256"] for item in submission["files"])
    assert "Runtime Boundary" in export["model_card"]
    assert "guarded_gpu_shadow_smoke_checked" in export["model_card"]


def test_csr_policy_model_submission_requires_terms_acknowledgement():
    export = tss.prepare_csr_policy_model_submission(
        contributor_terms_acknowledged=False,
    )
    summary = export["summary"]

    assert summary["status"] == "failed"
    assert summary["submission_ready"] is False
    assert summary["shadow_submission_ready"] is False
    assert summary["runtime_promotion_ready"] is False
    assert summary["runtime_selector_changed"] is False
    assert "contributor_terms_not_acknowledged" in summary["validation_errors"]
