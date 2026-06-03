"""Isolated runtime for blocked CSR coverage-gap probes."""

from __future__ import annotations

import json
import math
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from transsolvestack.core.result import PolicyPlan
from transsolvestack.core.types import SolveContext
from transsolvestack.datasets.csr import (
    CsrImportBatch,
    CsrMatrix,
    csr_matrix_from_record,
    import_selected_archives_to_csr,
    write_csr_import_records,
    write_csr_import_report,
    write_csr_import_summary,
)
from transsolvestack.operators.taichi_csr import TaichiCsrMatrixOperator
from transsolvestack.operators.taichi_diffusion2d import ensure_taichi_cuda
from transsolvestack.policies.csr_selector_data import (
    CsrSelectorExport,
    build_csr_selector_export,
    write_csr_diagnostic_rows,
    write_csr_selector_report,
    write_csr_selector_rows,
    write_csr_selector_schema,
    write_csr_selector_summary,
)
from transsolvestack.profiling.artifacts import write_jsonl
from transsolvestack.profiling.csr_blocked_gap_screens import (
    screen_blocked_gap_candidate,
)
from transsolvestack.runtime.engine import TaichiExecutionEngine


CSR_BLOCKED_GAP_RUNTIME_SCHEMA_VERSION = "phase1_csr_blocked_gap_probe_v1"


def run_blocked_gap_probe(
    matrix_queue: tuple[dict[str, Any], ...],
    candidate_queue: tuple[dict[str, Any], ...],
    *,
    matrix_queue_path: str,
    candidate_queue_path: str,
    device_memory_gb: float,
    max_matrices: int,
    max_rows: int,
    max_cols: int,
    max_stored_entries: int,
    max_archive_size_bytes: int,
    context_id: str,
) -> dict[str, Any]:
    import_batch = import_selected_archives_to_csr(
        matrix_queue,
        source_selection_path=matrix_queue_path,
        max_matrices=max_matrices,
        max_rows=max_rows,
        max_cols=max_cols,
        max_stored_entries=max_stored_entries,
        max_archive_size_bytes=max_archive_size_bytes,
    )
    imported_records = tuple(
        asdict(record) for record in import_batch.records if record.status == "success"
    )
    csr_by_id = {
        str(record["matrix_id"]): csr_matrix_from_record(record)
        for record in imported_records
    }

    engine: TaichiExecutionEngine | None = None
    result_rows: list[dict[str, Any]] = []
    for candidate in candidate_queue:
        csr = csr_by_id.get(str(candidate["matrix_id"]))
        if csr is None:
            continue
        screen = screen_blocked_gap_candidate(csr, candidate)
        if screen is None:
            result_rows.append(
                _screened_out_row(
                    csr,
                    candidate,
                    _unsupported_screen(candidate),
                    source="phase1_csr_blocked_gap_probe_cpu_screen",
                )
            )
            continue
        if not screen["success"]:
            result_rows.append(
                _screened_out_row(
                    csr,
                    candidate,
                    screen,
                    source="phase1_csr_blocked_gap_probe_cpu_screen",
                )
            )
            continue
        if engine is None:
            ensure_taichi_cuda(device_memory_gb=device_memory_gb)
            engine = TaichiExecutionEngine()
        try:
            result_rows.append(
                _run_gpu_once(
                    csr,
                    candidate,
                    screen=screen,
                    engine=engine,
                    context_id=context_id,
                )
            )
        except Exception as exc:
            result_rows.append(_gpu_failed_row(csr, candidate, screen=screen, exc=exc))

    result_tuple = tuple(result_rows)
    selector_export = build_csr_selector_export(
        imported_records,
        result_tuple,
        context_id=context_id,
    )
    summary = _runtime_summary(
        import_batch=import_batch,
        candidate_rows=candidate_queue,
        result_rows=result_tuple,
        selector_export=selector_export,
        matrix_queue_path=matrix_queue_path,
        candidate_queue_path=candidate_queue_path,
        device_memory_gb=device_memory_gb,
        max_rows=max_rows,
        max_cols=max_cols,
        max_stored_entries=max_stored_entries,
        max_archive_size_bytes=max_archive_size_bytes,
    )
    return {
        "import_batch": import_batch,
        "csr_records": imported_records,
        "result_rows": result_tuple,
        "selector_export": selector_export,
        "summary": summary,
        "schema": _runtime_schema(),
    }


def write_blocked_gap_probe_artifacts(
    export: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    import_batch: CsrImportBatch = export["import_batch"]
    selector_export: CsrSelectorExport = export["selector_export"]
    paths = {
        "csr_records": output / "csr_matrices.jsonl",
        "csr_import_summary": output / "csr_import_summary.json",
        "csr_import_report": output / "csr_import_report.md",
        "results": output / "csr_micro_campaign_results.jsonl",
        "internal_summary": output / "csr_micro_campaign_summary.json",
        "internal_schema": output / "csr_micro_campaign_schema.json",
        "internal_report": output / "csr_micro_campaign_report.md",
        "diagnostics": output / "csr_micro_diagnostic_rows.jsonl",
        "selector_rows": output / "csr_micro_selector_rows.jsonl",
        "selector_summary": output / "csr_micro_selector_summary.json",
        "selector_schema": output / "csr_micro_selector_schema.json",
        "selector_report": output / "csr_micro_selector_report.md",
    }
    write_csr_import_records(import_batch, paths["csr_records"])
    write_csr_import_summary(import_batch.summary, paths["csr_import_summary"])
    write_csr_import_report(import_batch, paths["csr_import_report"])
    write_jsonl(export["result_rows"], paths["results"])
    paths["internal_summary"].write_text(
        json.dumps(export["summary"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths["internal_schema"].write_text(
        json.dumps(export["schema"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_internal_report(export["result_rows"], export["summary"], paths["internal_report"])
    write_csr_diagnostic_rows(selector_export.diagnostics, paths["diagnostics"])
    write_csr_selector_rows(selector_export.selector_rows, paths["selector_rows"])
    write_csr_selector_summary(selector_export.summary, paths["selector_summary"])
    write_csr_selector_schema(paths["selector_schema"])
    write_csr_selector_report(selector_export, paths["selector_report"])
    return paths


def _run_gpu_once(
    csr: CsrMatrix,
    candidate: dict[str, Any],
    *,
    screen: dict[str, Any],
    engine: TaichiExecutionEngine,
    context_id: str,
) -> dict[str, Any]:
    solver = str(candidate["solver"])
    preconditioner = str(candidate["preconditioner"])
    precision = str(candidate["precision"])
    context = SolveContext(
        context_id=context_id,
        tolerance_abs=0.0,
        tolerance_rel=float(candidate["tolerance_rel"]),
        max_iter=int(candidate["max_iter"]),
        precision=precision,
        required_backend="taichi_gpu",
    )
    solver_parameters = _runtime_solver_parameters(candidate, screen)
    preconditioner_parameters = _runtime_preconditioner_parameters(candidate)
    rhs = csr.matvec(tuple(1.0 for _ in range(csr.n_cols)))
    operator = TaichiCsrMatrixOperator.from_csr_matrix(csr, dtype=context.precision)
    plan = PolicyPlan(
        plan_id=f"{candidate['candidate_id']}:{csr.matrix_id}",
        backend="taichi_gpu",
        solver={"name": solver, **solver_parameters},
        preconditioner={"name": preconditioner, **preconditioner_parameters},
    )
    start = time.perf_counter()
    result = engine.solve(operator, rhs=rhs, context=context, plan=plan)
    wall_ms = (time.perf_counter() - start) * 1000.0
    solution = tuple(float(value) for value in result.solution)
    solution_relative_error = _relative_error_to_ones(solution)
    cpu_relative_residual = _relative_residual(csr, solution, rhs)
    final_relative_residual = (
        float(result.trace.final_residual_norm)
        if result.trace.final_residual_norm is not None
        else math.inf
    )
    status = "success"
    failure_reasons: list[str] = []
    if result.status != "success":
        status = "failed"
        failure_reasons.append(result.trace.failure_class or "solver_failed")
    if not math.isfinite(final_relative_residual) or (
        final_relative_residual > context.tolerance_rel
    ):
        status = "failed"
        failure_reasons.append("trace_residual_above_tolerance")
    if not math.isfinite(cpu_relative_residual) or cpu_relative_residual > 1.0e-4:
        status = "failed"
        failure_reasons.append("cpu_recomputed_residual_above_tolerance")
    if not math.isfinite(solution_relative_error) or solution_relative_error > 5.0e-3:
        status = "failed"
        failure_reasons.append("solution_error_above_tolerance")
    residual_history = tuple(result.trace.residual_history)
    if residual_history and residual_history[-1] > residual_history[0]:
        status = "failed"
        failure_reasons.append("residual_did_not_drop")
    row = {
        "schema_version": CSR_BLOCKED_GAP_RUNTIME_SCHEMA_VERSION,
        "matrix_id": csr.matrix_id,
        "candidate_id": str(candidate["candidate_id"]),
        "coverage_gap_id": str(candidate["coverage_gap_id"]),
        "solver": solver,
        "preconditioner": preconditioner,
        "status": status,
        "failure_reasons": tuple(failure_reasons),
        "failure_reason": failure_reasons[0] if failure_reasons else None,
        "n_rows": csr.n_rows,
        "n_cols": csr.n_cols,
        "csr_nnz": csr.nnz,
        "field": csr.field,
        "symmetry": csr.symmetry,
        "num_iterations": result.trace.num_iterations,
        "final_relative_residual": final_relative_residual,
        "cpu_recomputed_relative_residual": cpu_relative_residual,
        "solution_relative_error": solution_relative_error,
        "residual_history": residual_history,
        "wall_time_ms": wall_ms,
        "solve_time_ms": result.trace.solve_time_ms,
        "backend": result.trace.backend,
        "precision": precision,
        "input": "rhs=A@ones",
        "cpu_screen": screen,
        "target_screen_max_iter": int(candidate["max_iter"]),
        "measurement_repeats": 1,
        "success_count": 1 if status == "success" else 0,
        "success_rate": 1.0 if status == "success" else 0.0,
        "transfer_time_ms": result.trace.transfer_time_ms,
        "preconditioner_parameters": preconditioner_parameters,
        "applicability_status": "applicable",
    }
    row.update(solver_parameters)
    return row


def _screened_out_row(
    csr: CsrMatrix,
    candidate: dict[str, Any],
    screen: dict[str, Any],
    *,
    source: str,
) -> dict[str, Any]:
    solver = str(candidate["solver"])
    solver_parameters = _runtime_solver_parameters(candidate, screen)
    failure_reason = f"cpu_reference_{solver}_screen_failed"
    row = {
        "schema_version": CSR_BLOCKED_GAP_RUNTIME_SCHEMA_VERSION,
        "matrix_id": csr.matrix_id,
        "candidate_id": str(candidate["candidate_id"]),
        "coverage_gap_id": str(candidate["coverage_gap_id"]),
        "solver": solver,
        "preconditioner": str(candidate["preconditioner"]),
        "status": "screened_out",
        "failure_reason": failure_reason,
        "failure_reasons": (failure_reason,),
        "backend": "cpu_reference_screen",
        "n_rows": csr.n_rows,
        "n_cols": csr.n_cols,
        "csr_nnz": csr.nnz,
        "field": csr.field,
        "symmetry": csr.symmetry,
        "num_iterations": screen.get("iterations"),
        "final_relative_residual": screen.get("relative_residual"),
        "cpu_recomputed_relative_residual": screen.get("relative_residual"),
        "solution_relative_error": screen.get("solution_relative_error"),
        "precision": str(candidate["precision"]),
        "input": "rhs=A@ones",
        "cpu_screen": screen,
        "screened_out_source": source,
        "target_screen_max_iter": int(candidate["max_iter"]),
        "measurement_repeats": 1,
        "success_count": 0,
        "success_rate": 0.0,
        "preconditioner_parameters": _runtime_preconditioner_parameters(candidate),
        "applicability_status": "applicable",
    }
    row.update(solver_parameters)
    return row


def _gpu_failed_row(
    csr: CsrMatrix,
    candidate: dict[str, Any],
    *,
    screen: dict[str, Any],
    exc: Exception,
) -> dict[str, Any]:
    row = _screened_out_row(
        csr,
        candidate,
        screen,
        source="phase1_csr_blocked_gap_probe_gpu_exception",
    )
    failure_reason = "gpu_execution_exception"
    row.update(
        {
            "status": "failed",
            "backend": "taichi_gpu",
            "failure_reason": failure_reason,
            "failure_reasons": (failure_reason,),
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
        }
    )
    return row


def _unsupported_screen(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "success": False,
        "iterations": 0,
        "relative_residual": math.inf,
        "solution_relative_error": math.inf,
        "breakdown": (
            "unsupported_blocked_gap_screen:"
            f"{candidate['solver']}:{candidate['preconditioner']}"
        ),
        "residual_history_length": 0,
        "residual_history_head": (),
        "residual_history_tail": (),
    }


def _runtime_solver_parameters(
    candidate: dict[str, Any],
    screen: dict[str, Any],
) -> dict[str, Any]:
    solver = str(candidate["solver"])
    params = dict(candidate.get("solver_parameters", {}))
    if solver == "gmres":
        params["restart"] = int(params.get("restart", 16))
    if solver == "richardson" and "omega" in screen:
        params["omega"] = float(screen["omega"])
    if solver == "chebyshev":
        if screen.get("lambda_min") is not None:
            params["lambda_min"] = float(screen["lambda_min"])
        if screen.get("lambda_max") is not None:
            params["lambda_max"] = float(screen["lambda_max"])
        if screen.get("spectral_bounds_source") is not None:
            params["spectral_bounds_source"] = str(screen["spectral_bounds_source"])
    return params


def _runtime_preconditioner_parameters(candidate: dict[str, Any]) -> dict[str, Any]:
    return dict(candidate.get("preconditioner_parameters", {}))


def _runtime_summary(
    *,
    import_batch: CsrImportBatch,
    candidate_rows: tuple[dict[str, Any], ...],
    result_rows: tuple[dict[str, Any], ...],
    selector_export: CsrSelectorExport,
    matrix_queue_path: str,
    candidate_queue_path: str,
    device_memory_gb: float,
    max_rows: int,
    max_cols: int,
    max_stored_entries: int,
    max_archive_size_bytes: int,
) -> dict[str, Any]:
    by_status = _counts(str(row["status"]) for row in result_rows)
    by_solver_status = _counts(
        f"{row['solver']}:{row['status']}" for row in result_rows
    )
    gpu_success_rows = int(by_status.get("success", 0))
    cpu_screened_out_rows = int(by_status.get("screened_out", 0))
    gpu_failed_rows = int(by_status.get("failed", 0))
    selector_summary = selector_export.summary
    accounted = gpu_success_rows + cpu_screened_out_rows + gpu_failed_rows
    imported_matrix_count = int(import_batch.summary.imported_matrices)
    candidate_matrix_count = len({str(row["matrix_id"]) for row in candidate_rows})
    status = (
        "passed"
        if import_batch.summary.status == "passed"
        and imported_matrix_count == candidate_matrix_count
        and len(result_rows) == len(candidate_rows)
        and accounted == len(candidate_rows)
        and gpu_failed_rows == 0
        and len(selector_export.selector_rows) == len(candidate_rows)
        and (selector_summary.status == "passed" or gpu_success_rows == 0)
        else "failed"
    )
    return {
        "status": status,
        "schema_version": CSR_BLOCKED_GAP_RUNTIME_SCHEMA_VERSION,
        "matrix_queue_path": matrix_queue_path,
        "candidate_queue_path": candidate_queue_path,
        "imported_matrices": imported_matrix_count,
        "candidate_jobs": len(candidate_rows),
        "gpu_success_rows": gpu_success_rows,
        "cpu_screened_out_rows": cpu_screened_out_rows,
        "gpu_failed_rows": gpu_failed_rows,
        "selector_rows": len(selector_export.selector_rows),
        "selector_oracle_rows": int(selector_summary.num_oracle_rows),
        "selector_summary_status": selector_summary.status,
        "by_status": by_status,
        "by_solver_status": by_solver_status,
        "device_memory_gb": device_memory_gb,
        "max_rows": max_rows,
        "max_cols": max_cols,
        "max_stored_entries": max_stored_entries,
        "max_archive_size_bytes": max_archive_size_bytes,
        "max_final_relative_residual": _max_float(
            result_rows,
            "final_relative_residual",
        ),
        "max_cpu_recomputed_relative_residual": _max_float(
            result_rows,
            "cpu_recomputed_relative_residual",
        ),
        "max_solution_relative_error": _max_float(
            result_rows,
            "solution_relative_error",
        ),
    }


def _runtime_schema() -> dict[str, Any]:
    return {
        "schema_version": CSR_BLOCKED_GAP_RUNTIME_SCHEMA_VERSION,
        "task": "execute_blocked_gap_cpu_screens_before_any_gpu_admission",
        "runtime_selector_changed": False,
        "executes_gpu": True,
        "gpu_solve_only_after_screen_success": True,
        "screen_only_artifacts_may_have_no_oracle_rows": True,
    }


def _write_internal_report(
    result_rows: tuple[dict[str, Any], ...],
    summary: dict[str, Any],
    path: Path,
) -> None:
    lines = [
        "# CSR Blocked Gap Internal Execution",
        "",
        f"- status: `{summary['status']}`",
        f"- imported_matrices: `{summary['imported_matrices']}`",
        f"- candidate_jobs: `{summary['candidate_jobs']}`",
        f"- gpu_success_rows: `{summary['gpu_success_rows']}`",
        f"- cpu_screened_out_rows: `{summary['cpu_screened_out_rows']}`",
        f"- gpu_failed_rows: `{summary['gpu_failed_rows']}`",
        "",
        "| matrix | candidate | status | backend | failure | rel_res | sol_err |",
        "|---|---|---|---|---|---:|---:|",
    ]
    for row in result_rows:
        lines.append(
            "| "
            f"{row['matrix_id']} | "
            f"{row['candidate_id']} | "
            f"{row['status']} | "
            f"{row['backend']} | "
            f"{row.get('failure_reason') or ''} | "
            f"{_fmt(row.get('final_relative_residual'))} | "
            f"{_fmt(row.get('solution_relative_error'))} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _relative_residual(
    csr: CsrMatrix,
    solution: tuple[float, ...],
    rhs: tuple[float, ...],
) -> float:
    actual = csr.matvec(solution)
    numerator = math.sqrt(sum((left - right) ** 2 for left, right in zip(actual, rhs)))
    denominator = max(math.sqrt(sum(value * value for value in rhs)), 1.0e-30)
    return numerator / denominator


def _relative_error_to_ones(solution: tuple[float, ...]) -> float:
    return math.sqrt(sum((value - 1.0) ** 2 for value in solution)) / max(
        math.sqrt(float(len(solution))),
        1.0e-30,
    )


def _counts(values) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def _max_float(rows: tuple[dict[str, Any], ...], key: str) -> float:
    values: list[float] = []
    for row in rows:
        try:
            value = float(row.get(key) or 0.0)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return max(values, default=0.0)


def _fmt(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(number):
        return str(value)
    return f"{number:.6g}"
