"""Probe CPU-screen-blocked CSR coverage gaps without queue promotion."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from transsolvestack.profiling.artifacts import read_jsonl, write_jsonl
from transsolvestack.profiling.csr_blocked_gap_runtime import (
    run_blocked_gap_probe,
    write_blocked_gap_probe_artifacts,
)


CSR_BLOCKED_GAP_PROBE_SCHEMA_VERSION = "phase1_csr_blocked_gap_probe_v1"


def run_csr_blocked_gap_probe_from_files(
    *,
    matrix_queue_path: str | Path = (
        "runs/phase1_csr_queue_batch_00010/csr_queue_batch_matrix_queue.jsonl"
    ),
    coverage_summary_path: str | Path = (
        "runs/phase1_csr_queue_candidate_coverage/"
        "csr_queue_candidate_coverage_summary.json"
    ),
    output_dir: str | Path = "runs/phase1_csr_blocked_gap_probe",
    max_general_matrices: int = 7,
    max_symmetric_matrices: int = 1,
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
    general_rows = tuple(
        row for row in source_matrix_rows if str(row.get("candidate_profile")) == "general"
    )[:max_general_matrices]
    symmetric_rows = tuple(
        row for row in source_matrix_rows if str(row.get("candidate_profile")) == "symmetric"
    )[:max_symmetric_matrices]
    selected_matrix_rows = (*general_rows, *symmetric_rows)
    if not selected_matrix_rows:
        raise ValueError("blocked-gap probe found no matrix rows")
    candidate_rows = _candidate_rows(
        general_rows=general_rows,
        symmetric_rows=symmetric_rows,
        max_iter=max_iter,
        tolerance_rel=tolerance_rel,
    )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    probe_matrix_queue = output / "csr_blocked_gap_matrix_queue.jsonl"
    probe_candidate_queue = output / "csr_blocked_gap_candidate_queue.jsonl"
    write_jsonl(selected_matrix_rows, probe_matrix_queue)
    write_jsonl(candidate_rows, probe_candidate_queue)

    probe = run_blocked_gap_probe(
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
        context_id="phase1_csr_blocked_gap_probe",
    )
    probe_paths = write_blocked_gap_probe_artifacts(probe, output)
    summary = _summary(
        source_matrix_rows=source_matrix_rows,
        general_rows=general_rows,
        symmetric_rows=symmetric_rows,
        candidate_rows=candidate_rows,
        result_rows=probe["result_rows"],
        probe_summary=probe["summary"],
        coverage_summary=coverage_summary,
        device_memory_gb=device_memory_gb,
        max_iter=max_iter,
        tolerance_rel=tolerance_rel,
    )
    schema = _schema()
    report = _report(summary)
    summary_path = output / "csr_blocked_gap_probe_summary.json"
    schema_path = output / "csr_blocked_gap_probe_schema.json"
    report_path = output / "csr_blocked_gap_probe_report.md"
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
            **{key: str(value) for key, value in probe_paths.items()},
            "summary": str(summary_path),
            "schema": str(schema_path),
            "report": str(report_path),
        },
        "summary": summary,
        "schema": schema,
        "probe": probe,
    }


def _candidate_rows(
    *,
    general_rows: tuple[dict[str, Any], ...],
    symmetric_rows: tuple[dict[str, Any], ...],
    max_iter: int,
    tolerance_rel: float,
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for matrix in general_rows:
        rows.extend(
            (
                _candidate(
                    matrix,
                    gap_id="general_bicgstab_ilu0",
                    solver="bicgstab",
                    preconditioner="ilu0",
                    candidate_id="taichi_csr_bicgstab_ilu0_float64",
                    solver_parameters={},
                    preconditioner_parameters={"pivot_tolerance": 1.0e-12},
                    max_iter=max_iter,
                    tolerance_rel=tolerance_rel,
                ),
                _candidate(
                    matrix,
                    gap_id="general_bicgstab_row_column_equilibration",
                    solver="bicgstab",
                    preconditioner="row_column_equilibration",
                    candidate_id=(
                        "taichi_csr_bicgstab_row_column_equilibration_float64"
                    ),
                    solver_parameters={},
                    preconditioner_parameters={"passes": 4},
                    max_iter=max_iter,
                    tolerance_rel=tolerance_rel,
                ),
            )
        )
    for matrix in symmetric_rows:
        rows.extend(
            (
                _candidate(
                    matrix,
                    gap_id="symmetric_chebyshev_jacobi",
                    solver="chebyshev",
                    preconditioner="jacobi",
                    candidate_id="taichi_csr_chebyshev_jacobi_float64",
                    solver_parameters={},
                    preconditioner_parameters={},
                    max_iter=max_iter,
                    tolerance_rel=tolerance_rel,
                ),
                _candidate(
                    matrix,
                    gap_id="symmetric_pcg_symmetric_equilibration",
                    solver="pcg",
                    preconditioner="symmetric_equilibration",
                    candidate_id=(
                        "taichi_csr_pcg_symmetric_equilibration_float64"
                    ),
                    solver_parameters={},
                    preconditioner_parameters={},
                    max_iter=max_iter,
                    tolerance_rel=tolerance_rel,
                ),
            )
        )
    return tuple(rows)


def _candidate(
    matrix: dict[str, Any],
    *,
    gap_id: str,
    solver: str,
    preconditioner: str,
    candidate_id: str,
    solver_parameters: dict[str, Any],
    preconditioner_parameters: dict[str, Any],
    max_iter: int,
    tolerance_rel: float,
) -> dict[str, Any]:
    matrix_id = str(matrix["matrix_id"])
    safe_matrix = matrix_id.replace(":", "_").replace("/", "_")
    return {
        "schema_version": CSR_BLOCKED_GAP_PROBE_SCHEMA_VERSION,
        "queue_id": "phase1_csr_blocked_gap_probe",
        "job_id": f"blocked_gap_probe:{safe_matrix}:{gap_id}",
        "batch_id": "blocked_gap_probe",
        "matrix_id": matrix_id,
        "candidate_id": candidate_id,
        "solver": solver,
        "preconditioner": preconditioner,
        "precision": "float64",
        "solver_parameters": solver_parameters,
        "preconditioner_parameters": preconditioner_parameters,
        "measurement_repeats": 1,
        "max_iter": max_iter,
        "tolerance_rel": tolerance_rel,
        "requires_import": True,
        "requires_cpu_screen": True,
        "planned_gpu_solve_attempts": 1,
        "estimated_matvecs": max_iter,
        "estimated_nnz_visits": int(matrix["nnz"]) * max_iter,
        "queue_status": "blocked_gap_probe_pending",
        "coverage_gap_id": gap_id,
    }


def _summary(
    *,
    source_matrix_rows: tuple[dict[str, Any], ...],
    general_rows: tuple[dict[str, Any], ...],
    symmetric_rows: tuple[dict[str, Any], ...],
    candidate_rows: tuple[dict[str, Any], ...],
    result_rows: tuple[dict[str, Any], ...],
    probe_summary: dict[str, Any],
    coverage_summary: dict[str, Any],
    device_memory_gb: float,
    max_iter: int,
    tolerance_rel: float,
) -> dict[str, Any]:
    candidate_jobs = int(probe_summary["candidate_jobs"])
    gpu_success_rows = int(probe_summary["gpu_success_rows"])
    cpu_screened_out_rows = int(probe_summary["cpu_screened_out_rows"])
    gpu_failed_rows = int(probe_summary["gpu_failed_rows"])
    selector_oracle_rows = int(probe_summary["selector_oracle_rows"])
    by_gap_status = _by_gap_status(candidate_rows, result_rows)
    successful_gap_ids = tuple(
        sorted(
            gap_id
            for gap_id, counts in by_gap_status.items()
            if int(counts.get("success", 0)) > 0
        )
    )
    all_accounted = (
        gpu_success_rows + cpu_screened_out_rows + gpu_failed_rows == candidate_jobs
    )
    completed = (
        int(probe_summary["imported_matrices"]) == len(general_rows) + len(symmetric_rows)
        and candidate_jobs == len(candidate_rows)
        and len(result_rows) == len(candidate_rows)
        and all_accounted
        and gpu_failed_rows == 0
        and int(probe_summary["selector_rows"]) == candidate_jobs
    )
    outcome = (
        "profiled_with_gpu_success"
        if gpu_success_rows > 0 and selector_oracle_rows > 0
        else "screen_only_no_oracle"
        if gpu_success_rows == 0 and cpu_screened_out_rows == candidate_jobs
        else "failed"
    )
    return {
        "status": "passed" if completed and outcome != "failed" else "failed",
        "schema_version": CSR_BLOCKED_GAP_PROBE_SCHEMA_VERSION,
        "runtime_selector_changed": False,
        "executes_gpu": True,
        "imports_matrices": True,
        "cpu_screen_required": True,
        "coverage_source_latest_batch_id": coverage_summary["latest_batch_id"],
        "coverage_trigger_candidate_coverage": coverage_summary[
            "trigger_candidate_coverage"
        ],
        "coverage_trigger_reason": coverage_summary["trigger_reason"],
        "coverage_outcome": outcome,
        "source_matrices": len(source_matrix_rows),
        "selected_general_matrices": len(general_rows),
        "selected_symmetric_matrices": len(symmetric_rows),
        "selected_matrices": len(general_rows) + len(symmetric_rows),
        "candidate_jobs": candidate_jobs,
        "blocked_gap_ids": tuple(sorted(by_gap_status)),
        "newly_cpu_screen_integrated_gap_count": len(by_gap_status),
        "device_memory_gb": device_memory_gb,
        "max_iter": max_iter,
        "tolerance_rel": tolerance_rel,
        "probe_internal_status": probe_summary["status"],
        "gpu_success_rows": gpu_success_rows,
        "cpu_screened_out_rows": cpu_screened_out_rows,
        "gpu_failed_rows": gpu_failed_rows,
        "selector_rows": int(probe_summary["selector_rows"]),
        "selector_oracle_rows": selector_oracle_rows,
        "success_rate": gpu_success_rows / candidate_jobs if candidate_jobs else 0.0,
        "by_status": dict(probe_summary["by_status"]),
        "by_solver_status": dict(probe_summary["by_solver_status"]),
        "by_gap_status": by_gap_status,
        "queue_merge_ready_gap_ids": successful_gap_ids,
        "queue_merge_ready": bool(successful_gap_ids),
        "max_final_relative_residual": _max_float(
            result_rows,
            "final_relative_residual",
        ),
        "max_cpu_recomputed_relative_residual": _max_float(
            result_rows,
            "cpu_recomputed_relative_residual",
        ),
        "max_solution_relative_error": _max_float(result_rows, "solution_relative_error"),
        "next_step": (
            "run_bounded_gpu_probe_for_successful_blocked_gap_ids_before_queue_merge"
            if successful_gap_ids
            else "keep_blocked_gaps_out_of_generic_queue_and_search_better_formulations_or_matrices"
        ),
    }


def _by_gap_status(
    candidate_rows: tuple[dict[str, Any], ...],
    result_rows: tuple[dict[str, Any], ...],
) -> dict[str, dict[str, int]]:
    gap_by_pair = {
        (str(row["matrix_id"]), str(row["candidate_id"])): str(row["coverage_gap_id"])
        for row in candidate_rows
    }
    counts: dict[str, Counter[str]] = {}
    for row in result_rows:
        gap_id = gap_by_pair[(str(row["matrix_id"]), str(row["candidate_id"]))]
        counts.setdefault(gap_id, Counter())[str(row["status"])] += 1
    return {gap_id: dict(counter) for gap_id, counter in sorted(counts.items())}


def _max_float(rows: tuple[dict[str, Any], ...], key: str) -> float:
    values: list[float] = []
    for row in rows:
        try:
            value = float(row.get(key) or 0.0)
        except (TypeError, ValueError):
            continue
        if value != float("inf") and value != float("-inf"):
            values.append(value)
    return max(values, default=0.0)


def _schema() -> dict[str, Any]:
    return {
        "schema_version": CSR_BLOCKED_GAP_PROBE_SCHEMA_VERSION,
        "task": "integrate_and_probe_cpu_screens_for_blocked_csr_candidate_gaps",
        "runtime_selector_changed": False,
        "executes_gpu": True,
        "imports_matrices": True,
        "coverage_boundary": {
            "does_not_modify_full_dataset_queue": True,
            "does_not_promote_runtime_selector": True,
            "requires_cpu_screen_before_gpu_solve": True,
            "gpu_solve_only_after_screen_success": True,
        },
    }


def _report(summary: dict[str, Any]) -> str:
    lines = [
        "# CSR Blocked Gap Probe",
        "",
        f"- status: `{summary['status']}`",
        f"- schema_version: `{summary['schema_version']}`",
        f"- runtime_selector_changed: `{summary['runtime_selector_changed']}`",
        f"- executes_gpu: `{summary['executes_gpu']}`",
        f"- coverage_source_latest_batch_id: `{summary['coverage_source_latest_batch_id']}`",
        f"- coverage_outcome: `{summary['coverage_outcome']}`",
        f"- selected_matrices: `{summary['selected_matrices']}`",
        f"- candidate_jobs: `{summary['candidate_jobs']}`",
        f"- gpu_success_rows: `{summary['gpu_success_rows']}`",
        f"- cpu_screened_out_rows: `{summary['cpu_screened_out_rows']}`",
        f"- gpu_failed_rows: `{summary['gpu_failed_rows']}`",
        f"- selector_oracle_rows: `{summary['selector_oracle_rows']}`",
        f"- queue_merge_ready: `{summary['queue_merge_ready']}`",
        f"- queue_merge_ready_gap_ids: `{list(summary['queue_merge_ready_gap_ids'])}`",
        f"- next_step: `{summary['next_step']}`",
        "",
        "## By Gap",
        "",
        "| gap | success | screened_out | failed |",
        "|---|---:|---:|---:|",
    ]
    for gap_id, counts in summary["by_gap_status"].items():
        lines.append(
            "| "
            f"{gap_id} | "
            f"{counts.get('success', 0)} | "
            f"{counts.get('screened_out', 0)} | "
            f"{counts.get('failed', 0)} |"
        )
    lines.append("")
    return "\n".join(lines)
