import transsolvestack as tss


def test_csr_full_dataset_queue_plans_resumable_batches():
    queue = tss.plan_csr_full_dataset_queue(max_queue_matrices=16, batch_matrix_count=4)
    summary = queue.summary

    assert summary.status == "passed"
    assert summary.schema_version == "phase1_csr_full_dataset_queue_v1"
    assert summary.runtime_selector_changed is False
    assert summary.executes_gpu is False
    assert summary.imports_matrices is False
    assert summary.resumable is True
    assert summary.external_drive_required is True
    assert summary.index_matrices == 2904
    assert summary.index_present_archives == 2904
    assert summary.eligible_matrices > summary.already_profiled_matrices
    assert summary.queued_matrices == 16
    assert len(queue.batch_rows) == 4
    assert len(queue.job_rows) == summary.queued_jobs
    assert summary.planned_gpu_solve_attempts == summary.queued_jobs
    assert summary.first_pending_batch_id == "batch_00001"
    assert queue.state["resume_policy"]["first_pending_batch_id"] == "batch_00001"
    assert queue.schema["execution_boundary"]["automatic_execution"] is False
    assert all(row.requires_import for row in queue.job_rows)
    assert all(row.requires_cpu_screen for row in queue.job_rows)
    assert all(row.queue_status == "pending_import" for row in queue.job_rows)
    assert all(row.queue_status == "pending" for row in queue.batch_rows)
