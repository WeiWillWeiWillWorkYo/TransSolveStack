"""Probe GMRES restart candidates from queue-ready coverage gaps."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from transsolvestack.profiling.artifacts import read_jsonl, write_jsonl
from transsolvestack.profiling.csr_micro_campaign import (
    run_csr_micro_campaign,
    write_csr_micro_campaign_artifacts,
)


CSR_GMRES_RESTART_COVERAGE_SCHEMA_VERSION = "phase1_csr_gmres_restart_coverage_v1"


def run_csr_gmres_restart_coverage_from_files(
    *,
    matrix_queue_path: str | Path = (
        "runs/phase1_csr_queue_batch_00010/csr_queue_batch_matrix_queue.jsonl"
    ),
    coverage_summary_path: str | Path = (
        "runs/phase1_csr_queue_candidate_coverage/"
        "csr_queue_candidate_coverage_summary.json"
    ),
    output_dir: str | Path = "runs/phase1_csr_gmres_restart_coverage",
    restarts: tuple[int, ...] = (32, 64),
    max_matrices: int = 8,
    device_memory_gb: float = 0.20,
    max_rows: int = 65_536,
    max_cols: int = 65_536,
    max_stored_entries: int = 1_000_000,
    max_archive_size_bytes: int = 256_000_000,
    max_iter: int = 300,
    tolerance_rel: float = 1.0e-5,
) -> dict[str, Any]:
    source_matrix_rows = tuple(read_jsonl(matrix_queue_path))
    coverage_summary = json.loads(Path(coverage_summary_path).read_text(encoding="utf-8"))
    selected_matrix_rows = tuple(
        row for row in source_matrix_rows if str(row.get("candidate_profile")) == "general"
    )[:max_matrices]
    if not selected_matrix_rows:
        raise ValueError("GMRES restart coverage found no general-profile matrices")
    candidate_rows = _candidate_rows(
        selected_matrix_rows,
        restarts=restarts,
        max_iter=max_iter,
        tolerance_rel=tolerance_rel,
    )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    probe_matrix_queue = output / "csr_gmres_restart_matrix_queue.jsonl"
    probe_candidate_queue = output / "csr_gmres_restart_candidate_queue.jsonl"
    write_jsonl(selected_matrix_rows, probe_matrix_queue)
    write_jsonl(candidate_rows, probe_candidate_queue)

    micro = run_csr_micro_campaign(
        selected_matrix_rows,
        candidate_rows,
        matrix_queue_path=str(probe_matrix_queue),
        candidate_queue_path=str(probe_candidate_queue),
        device_memory_gb=device_memory_gb,
        max_matrices=len(selected_matrix_rows),
        max_rows=max_rows,
        max_cols=max_cols,
        max_stored_entries=max_stored_entries,
        max_archive_size_bytes=max_archive_size_bytes,
        context_id="phase1_csr_gmres_restart_coverage",
    )
    micro_paths = write_csr_micro_campaign_artifacts(micro, output)
    summary = _summary(
        source_matrix_rows=source_matrix_rows,
        selected_matrix_rows=selected_matrix_rows,
        candidate_rows=candidate_rows,
        result_rows=micro["result_rows"],
        micro_summary=micro["summary"],
        coverage_summary=coverage_summary,
        restarts=restarts,
        device_memory_gb=device_memory_gb,
        max_iter=max_iter,
        tolerance_rel=tolerance_rel,
    )
    schema = _schema()
    report = _report(summary)
    summary_path = output / "csr_gmres_restart_coverage_summary.json"
    schema_path = output / "csr_gmres_restart_coverage_schema.json"
    report_path = output / "csr_gmres_restart_coverage_report.md"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    schema_path.write_text(
        json.dumps(schema, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(report, encoding="utf-8")
    return {
        "paths": {
            "probe_matrix_queue": str(probe_matrix_queue),
            "probe_candidate_queue": str(probe_candidate_queue),
            **{key: str(value) for key, value in micro_paths.items()},
            "summary": str(summary_path),
            "schema": str(schema_path),
            "report": str(report_path),
        },
        "summary": summary,
        "schema": schema,
        "micro": micro,
    }


def _candidate_rows(
    matrix_rows: tuple[dict[str, Any], ...],
    *,
    restarts: tuple[int, ...],
    max_iter: int,
    tolerance_rel: float,
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for matrix in matrix_rows:
        matrix_id = str(matrix["matrix_id"])
        safe_matrix = matrix_id.replace(":", "_").replace("/", "_")
        for restart in restarts:
            candidate_id = f"taichi_csr_gmres_jacobi_restart{restart}_float64"
            rows.append(
                {
                    "schema_version": CSR_GMRES_RESTART_COVERAGE_SCHEMA_VERSION,
                    "queue_id": "phase1_csr_gmres_restart_coverage",
                    "job_id": f"gmres_restart_coverage:{safe_matrix}:restart{restart}",
                    "batch_id": "gmres_restart_coverage",
                    "matrix_id": matrix_id,
                    "candidate_id": candidate_id,
                    "solver": "gmres",
                    "preconditioner": "jacobi",
                    "precision": "float64",
                    "solver_parameters": {"restart": restart},
                    "measurement_repeats": 1,
                    "max_iter": max_iter,
                    "tolerance_rel": tolerance_rel,
                    "requires_import": True,
                    "requires_cpu_screen": True,
                    "planned_gpu_solve_attempts": 1,
                    "estimated_matvecs": _estimated_gmres_matvecs(max_iter, restart),
                    "estimated_nnz_visits": (
                        int(matrix["nnz"]) * _estimated_gmres_matvecs(max_iter, restart)
                    ),
                    "queue_status": "coverage_probe_pending",
                    "coverage_gap_id": f"general_gmres_jacobi_restart{restart}",
                }
            )
    return tuple(rows)


def _summary(
    *,
    source_matrix_rows: tuple[dict[str, Any], ...],
    selected_matrix_rows: tuple[dict[str, Any], ...],
    candidate_rows: tuple[dict[str, Any], ...],
    result_rows: tuple[dict[str, Any], ...],
    micro_summary: dict[str, Any],
    coverage_summary: dict[str, Any],
    restarts: tuple[int, ...],
    device_memory_gb: float,
    max_iter: int,
    tolerance_rel: float,
) -> dict[str, Any]:
    candidate_jobs = int(micro_summary["candidate_jobs"])
    gpu_success_rows = int(micro_summary["gpu_success_rows"])
    cpu_screened_out_rows = int(micro_summary["cpu_screened_out_rows"])
    gpu_failed_rows = int(micro_summary["gpu_failed_rows"])
    selector_oracle_rows = int(micro_summary["selector_oracle_rows"])
    all_accounted = (
        gpu_success_rows + cpu_screened_out_rows + gpu_failed_rows == candidate_jobs
    )
    completed = (
        int(micro_summary["imported_matrices"]) == len(selected_matrix_rows)
        and candidate_jobs == len(candidate_rows)
        and all_accounted
        and gpu_failed_rows == 0
        and int(micro_summary["selector_rows"]) == candidate_jobs
    )
    outcome = (
        "profiled_with_gpu_success"
        if gpu_success_rows > 0 and selector_oracle_rows > 0
        else "screen_only_no_oracle"
        if gpu_success_rows == 0 and cpu_screened_out_rows == candidate_jobs
        else "failed"
    )
    by_restart = _by_restart(result_rows)
    return {
        "status": "passed" if completed and outcome != "failed" else "failed",
        "schema_version": CSR_GMRES_RESTART_COVERAGE_SCHEMA_VERSION,
        "runtime_selector_changed": False,
        "executes_gpu": True,
        "imports_matrices": True,
        "cpu_screen_required": True,
        "coverage_source_latest_batch_id": coverage_summary["latest_batch_id"],
        "coverage_trigger_candidate_coverage": coverage_summary[
            "trigger_candidate_coverage"
        ],
        "coverage_trigger_reason": coverage_summary["trigger_reason"],
        "coverage_gap_ids": tuple(
            f"general_gmres_jacobi_restart{restart}" for restart in restarts
        ),
        "coverage_outcome": outcome,
        "source_matrices": len(source_matrix_rows),
        "selected_matrices": len(selected_matrix_rows),
        "candidate_jobs": candidate_jobs,
        "restarts": tuple(restarts),
        "device_memory_gb": device_memory_gb,
        "max_iter": max_iter,
        "tolerance_rel": tolerance_rel,
        "micro_campaign_status": micro_summary["status"],
        "gpu_success_rows": gpu_success_rows,
        "cpu_screened_out_rows": cpu_screened_out_rows,
        "gpu_failed_rows": gpu_failed_rows,
        "selector_rows": int(micro_summary["selector_rows"]),
        "selector_oracle_rows": selector_oracle_rows,
        "success_rate": gpu_success_rows / candidate_jobs if candidate_jobs else 0.0,
        "by_status": dict(micro_summary["by_status"]),
        "by_solver_status": dict(micro_summary["by_solver_status"]),
        "by_restart": by_restart,
        "max_final_relative_residual": float(
            micro_summary["max_final_relative_residual"]
        ),
        "max_cpu_recomputed_relative_residual": float(
            micro_summary["max_cpu_recomputed_relative_residual"]
        ),
        "max_solution_relative_error": float(micro_summary["max_solution_relative_error"]),
        "queue_merge_ready": gpu_success_rows > 0 and selector_oracle_rows > 0,
        "next_step": (
            "compare_restart_probe_against_restart16_before_queue_template_merge"
            if gpu_success_rows > 0
            else "do_not_merge_gmres_restart_candidates_continue_with_blocked_gap_cpu_screen_integration"
        ),
    }


def _by_restart(result_rows: tuple[dict[str, Any], ...]) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = {}
    for row in result_rows:
        restart = int(row["restart"])
        status = str(row["status"])
        counts.setdefault(str(restart), Counter())[status] += 1
    return {restart: dict(counter) for restart, counter in sorted(counts.items())}


def _estimated_gmres_matvecs(max_iter: int, restart: int) -> int:
    return max_iter + max_iter // max(restart, 1)


def _schema() -> dict[str, Any]:
    return {
        "schema_version": CSR_GMRES_RESTART_COVERAGE_SCHEMA_VERSION,
        "task": "bounded_gmres_restart_coverage_probe",
        "runtime_selector_changed": False,
        "executes_gpu": True,
        "imports_matrices": True,
        "coverage_boundary": {
            "does_not_modify_full_dataset_queue": True,
            "does_not_promote_runtime_selector": True,
            "requires_cpu_screen_before_gpu_solve": True,
        },
    }


def _report(summary: dict[str, Any]) -> str:
    lines = [
        "# CSR GMRES Restart Coverage",
        "",
        f"- status: `{summary['status']}`",
        f"- schema_version: `{summary['schema_version']}`",
        f"- runtime_selector_changed: `{summary['runtime_selector_changed']}`",
        f"- executes_gpu: `{summary['executes_gpu']}`",
        f"- coverage_source_latest_batch_id: `{summary['coverage_source_latest_batch_id']}`",
        f"- coverage_trigger_reason: `{summary['coverage_trigger_reason']}`",
        f"- coverage_outcome: `{summary['coverage_outcome']}`",
        f"- selected_matrices: `{summary['selected_matrices']}`",
        f"- candidate_jobs: `{summary['candidate_jobs']}`",
        f"- gpu_success_rows: `{summary['gpu_success_rows']}`",
        f"- cpu_screened_out_rows: `{summary['cpu_screened_out_rows']}`",
        f"- gpu_failed_rows: `{summary['gpu_failed_rows']}`",
        f"- selector_oracle_rows: `{summary['selector_oracle_rows']}`",
        f"- success_rate: `{summary['success_rate']}`",
        f"- queue_merge_ready: `{summary['queue_merge_ready']}`",
        f"- next_step: `{summary['next_step']}`",
        "",
        "## By Restart",
        "",
        "| restart | success | screened_out | failed |",
        "|---:|---:|---:|---:|",
    ]
    for restart, counts in summary["by_restart"].items():
        lines.append(
            "| "
            f"{restart} | "
            f"{counts.get('success', 0)} | "
            f"{counts.get('screened_out', 0)} | "
            f"{counts.get('failed', 0)} |"
        )
    lines.append("")
    return "\n".join(lines)
