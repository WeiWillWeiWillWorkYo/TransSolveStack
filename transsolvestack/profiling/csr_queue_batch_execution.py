"""Execute one resumable CSR full-dataset queue batch."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from transsolvestack.profiling.artifacts import read_jsonl, write_jsonl
from transsolvestack.profiling.csr_micro_campaign import (
    run_csr_micro_campaign,
    write_csr_micro_campaign_artifacts,
)


CSR_QUEUE_BATCH_EXECUTION_SCHEMA_VERSION = "phase1_csr_queue_batch_execution_v1"


def execute_csr_queue_batch_from_files(
    matrix_queue_path: str | Path = (
        "runs/phase1_csr_full_dataset_queue/csr_full_dataset_queue_matrices.jsonl"
    ),
    job_queue_path: str | Path = (
        "runs/phase1_csr_full_dataset_queue/csr_full_dataset_queue_jobs.jsonl"
    ),
    batch_queue_path: str | Path = (
        "runs/phase1_csr_full_dataset_queue/csr_full_dataset_queue_batches.jsonl"
    ),
    output_dir: str | Path = "runs/phase1_csr_queue_batch_00001",
    *,
    batch_id: str = "batch_00001",
    device_memory_gb: float = 0.20,
    max_rows: int = 65_536,
    max_cols: int = 65_536,
    max_stored_entries: int = 1_000_000,
    max_archive_size_bytes: int = 256_000_000,
) -> dict[str, Any]:
    """Execute one queue batch through import, CPU screen, and Taichi GPU solves."""

    matrix_rows = tuple(read_jsonl(matrix_queue_path))
    job_rows = tuple(read_jsonl(job_queue_path))
    batch_rows = tuple(read_jsonl(batch_queue_path))
    batch = _find_batch(batch_rows, batch_id)
    batch_matrices = tuple(
        _micro_matrix_row(row)
        for row in matrix_rows
        if row.get("batch_id") == batch_id and row.get("queue_status") == "queued"
    )
    batch_jobs = tuple(row for row in job_rows if row.get("batch_id") == batch_id)
    if not batch_matrices:
        raise ValueError(f"batch has no queued matrices: {batch_id}")
    if not batch_jobs:
        raise ValueError(f"batch has no queued jobs: {batch_id}")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "batch_matrix_queue": output / "csr_queue_batch_matrix_queue.jsonl",
        "batch_candidate_queue": output / "csr_queue_batch_candidate_queue.jsonl",
    }
    write_jsonl(batch_matrices, paths["batch_matrix_queue"])
    write_jsonl(batch_jobs, paths["batch_candidate_queue"])
    micro = run_csr_micro_campaign(
        batch_matrices,
        batch_jobs,
        matrix_queue_path=str(paths["batch_matrix_queue"]),
        candidate_queue_path=str(paths["batch_candidate_queue"]),
        device_memory_gb=device_memory_gb,
        max_matrices=len(batch_matrices),
        max_rows=max_rows,
        max_cols=max_cols,
        max_stored_entries=max_stored_entries,
        max_archive_size_bytes=max_archive_size_bytes,
        context_id=f"phase1_csr_queue_{batch_id}",
    )
    micro_paths = write_csr_micro_campaign_artifacts(micro, output)
    summary = _summary(
        batch=batch,
        batch_matrices=batch_matrices,
        batch_jobs=batch_jobs,
        micro_summary=micro["summary"],
        output_dir=output,
        device_memory_gb=device_memory_gb,
    )
    schema = _schema(summary)
    report = _report(summary)
    summary_path = output / "csr_queue_batch_execution_summary.json"
    schema_path = output / "csr_queue_batch_execution_schema.json"
    report_path = output / "csr_queue_batch_execution_report.md"
    state_path = output / "csr_queue_batch_execution_state.json"
    _write_json(summary, summary_path)
    _write_json(schema, schema_path)
    _write_json(_state(summary), state_path)
    report_path.write_text(report, encoding="utf-8")
    return {
        "paths": {
            **{key: str(value) for key, value in paths.items()},
            **{key: str(value) for key, value in micro_paths.items()},
            "batch_summary": str(summary_path),
            "batch_schema": str(schema_path),
            "batch_report": str(report_path),
            "batch_state": str(state_path),
        },
        "summary": summary,
        "schema": schema,
        "report": report,
        "micro": micro,
    }


def _find_batch(batch_rows: tuple[dict[str, Any], ...], batch_id: str) -> dict[str, Any]:
    for row in batch_rows:
        if row.get("batch_id") == batch_id:
            return row
    raise ValueError(f"unknown queue batch: {batch_id}")


def _micro_matrix_row(row: dict[str, Any]) -> dict[str, Any]:
    result = dict(row)
    result["selection_rank"] = int(row["queue_rank"])
    result["metadata"] = {
        "source_queue_id": row["queue_id"],
        "source_batch_id": row["batch_id"],
        "source_queue_rank": row["queue_rank"],
    }
    return result


def _summary(
    *,
    batch: dict[str, Any],
    batch_matrices: tuple[dict[str, Any], ...],
    batch_jobs: tuple[dict[str, Any], ...],
    micro_summary: dict[str, Any],
    output_dir: Path,
    device_memory_gb: float,
) -> dict[str, Any]:
    imported_matrices = int(micro_summary["imported_matrices"])
    candidate_jobs = int(micro_summary["candidate_jobs"])
    gpu_success_rows = int(micro_summary["gpu_success_rows"])
    cpu_screened_out_rows = int(micro_summary["cpu_screened_out_rows"])
    gpu_failed_rows = int(micro_summary["gpu_failed_rows"])
    selector_rows = int(micro_summary["selector_rows"])
    selector_oracle_rows = int(micro_summary["selector_oracle_rows"])
    all_imported = imported_matrices == len(batch_matrices) == int(batch["matrix_count"])
    all_candidates_accounted = (
        gpu_success_rows + cpu_screened_out_rows + gpu_failed_rows == candidate_jobs
    )
    completed_with_success = (
        micro_summary["status"] == "passed"
        and all_imported
        and candidate_jobs == len(batch_jobs) == int(batch["job_count"])
        and all_candidates_accounted
        and gpu_success_rows > 0
        and gpu_failed_rows == 0
        and selector_rows == candidate_jobs
        and selector_oracle_rows > 0
    )
    screen_only_no_oracle = (
        all_imported
        and candidate_jobs == len(batch_jobs) == int(batch["job_count"])
        and all_candidates_accounted
        and gpu_success_rows == 0
        and cpu_screened_out_rows == candidate_jobs
        and gpu_failed_rows == 0
        and selector_rows == candidate_jobs
        and selector_oracle_rows == 0
    )
    completed = completed_with_success or screen_only_no_oracle
    batch_outcome = "failed"
    resume_status = "failed"
    if completed_with_success:
        batch_outcome = "profiled_with_gpu_success"
        resume_status = "completed"
    elif screen_only_no_oracle:
        batch_outcome = "screen_only_no_oracle"
        resume_status = "completed_no_oracle"
    return {
        "status": "passed" if completed else "failed",
        "schema_version": CSR_QUEUE_BATCH_EXECUTION_SCHEMA_VERSION,
        "batch_id": batch["batch_id"],
        "batch_rank": int(batch["batch_rank"]),
        "queue_id": batch["queue_id"],
        "runtime_selector_changed": False,
        "executes_gpu": True,
        "imports_matrices": True,
        "cpu_screen_required": True,
        "append_only_selector_rows": True,
        "batch_outcome": batch_outcome,
        "micro_campaign_status": str(micro_summary["status"]),
        "completed_without_oracle": screen_only_no_oracle,
        "has_oracle_rows": selector_oracle_rows > 0,
        "training_pool_role": (
            "oracle_and_negative_selector_rows"
            if completed_with_success
            else "negative_no_oracle_selector_rows"
            if screen_only_no_oracle
            else "failed_not_appendable"
        ),
        "device_memory_gb": device_memory_gb,
        "output_dir": str(output_dir),
        "planned_matrices": int(batch["matrix_count"]),
        "planned_jobs": int(batch["job_count"]),
        "planned_gpu_solve_attempts": int(batch["planned_gpu_solve_attempts"]),
        "planned_total_nnz": int(batch["total_nnz"]),
        "planned_max_matrix_nnz": int(batch["max_matrix_nnz"]),
        "executed_matrices": len(batch_matrices),
        "executed_jobs": len(batch_jobs),
        "imported_matrices": imported_matrices,
        "candidate_jobs": candidate_jobs,
        "gpu_success_rows": gpu_success_rows,
        "cpu_screened_out_rows": cpu_screened_out_rows,
        "gpu_failed_rows": gpu_failed_rows,
        "selector_rows": selector_rows,
        "selector_oracle_rows": selector_oracle_rows,
        "matrices_with_selector_rows": int(micro_summary["matrices_with_selector_rows"]),
        "max_final_relative_residual": float(
            micro_summary["max_final_relative_residual"]
        ),
        "max_cpu_recomputed_relative_residual": float(
            micro_summary["max_cpu_recomputed_relative_residual"]
        ),
        "max_solution_relative_error": float(
            micro_summary["max_solution_relative_error"]
        ),
        "by_status": dict(micro_summary["by_status"]),
        "by_solver_status": dict(micro_summary["by_solver_status"]),
        "resume_status": resume_status,
        "next_step": "append_selector_rows_to_training_pool_then_rebuild_learning_bundle",
    }


def _schema(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": CSR_QUEUE_BATCH_EXECUTION_SCHEMA_VERSION,
        "task": "execute_one_full_dataset_queue_batch",
        "runtime_selector_changed": False,
        "executes_gpu": True,
        "imports_matrices": True,
        "screening_policy": {
            "cpu_screen_required": True,
            "gpu_execution": "only_after_candidate_cpu_screen_success",
        },
        "outputs": {
            "csr_imports": "csr_matrices.jsonl",
            "gpu_results": "csr_micro_campaign_results.jsonl",
            "selector_rows": "csr_micro_selector_rows.jsonl",
            "batch_state": "csr_queue_batch_execution_state.json",
        },
        "resume_policy": {
            "batch_id": summary["batch_id"],
            "status": summary["resume_status"],
            "batch_outcome": summary["batch_outcome"],
        },
    }


def _state(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": summary["schema_version"],
        "state_kind": "csr_queue_batch_execution_state",
        "queue_id": summary["queue_id"],
        "batch_id": summary["batch_id"],
        "batch_outcome": summary["batch_outcome"],
        "resume_status": summary["resume_status"],
        "runtime_selector_changed": False,
        "completed_outputs": {
            "csr_imports": "csr_matrices.jsonl",
            "gpu_results": "csr_micro_campaign_results.jsonl",
            "selector_rows": "csr_micro_selector_rows.jsonl",
        },
    }


def _report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# CSR Queue Batch Execution",
            "",
            f"- status: `{summary['status']}`",
            f"- schema_version: `{summary['schema_version']}`",
            f"- queue_id: `{summary['queue_id']}`",
            f"- batch_id: `{summary['batch_id']}`",
            f"- batch_outcome: `{summary['batch_outcome']}`",
            f"- micro_campaign_status: `{summary['micro_campaign_status']}`",
            f"- runtime_selector_changed: `{summary['runtime_selector_changed']}`",
            f"- executes_gpu: `{summary['executes_gpu']}`",
            f"- imports_matrices: `{summary['imports_matrices']}`",
            f"- cpu_screen_required: `{summary['cpu_screen_required']}`",
            f"- completed_without_oracle: `{summary['completed_without_oracle']}`",
            f"- training_pool_role: `{summary['training_pool_role']}`",
            f"- planned/executed matrices: `{summary['planned_matrices']}` / `{summary['executed_matrices']}`",
            f"- planned/executed jobs: `{summary['planned_jobs']}` / `{summary['executed_jobs']}`",
            f"- imported_matrices: `{summary['imported_matrices']}`",
            f"- gpu_success_rows: `{summary['gpu_success_rows']}`",
            f"- cpu_screened_out_rows: `{summary['cpu_screened_out_rows']}`",
            f"- gpu_failed_rows: `{summary['gpu_failed_rows']}`",
            f"- selector_rows: `{summary['selector_rows']}`",
            f"- selector_oracle_rows: `{summary['selector_oracle_rows']}`",
            f"- max_final_relative_residual: `{summary['max_final_relative_residual']:.6g}`",
            f"- max_cpu_recomputed_relative_residual: `{summary['max_cpu_recomputed_relative_residual']:.6g}`",
            f"- max_solution_relative_error: `{summary['max_solution_relative_error']:.6g}`",
            f"- resume_status: `{summary['resume_status']}`",
            f"- next_step: `{summary['next_step']}`",
            "",
        ]
    )


def _write_json(data: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
