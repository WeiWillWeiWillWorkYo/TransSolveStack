"""Execute a bounded CPU-screened Taichi CSR micro-campaign."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

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
from transsolvestack.profiling.artifacts import read_jsonl, write_jsonl
from transsolvestack.profiling.repeats import aggregate_repeated_rows
from transsolvestack.runtime.engine import TaichiExecutionEngine


CSR_MICRO_CAMPAIGN_SCHEMA_VERSION = "phase1_csr_micro_campaign_v1"

RICHARDSON_OMEGA_CANDIDATES = (2.0 / 3.0, 0.5, 0.25, 0.125, 0.0625, 0.03125)


def run_csr_micro_campaign_from_files(
    matrix_queue_path: str | Path,
    candidate_queue_path: str | Path,
    *,
    device_memory_gb: float = 0.25,
    max_matrices: int = 8,
    max_rows: int = 10_000,
    max_cols: int = 10_000,
    max_stored_entries: int = 100_000,
    max_archive_size_bytes: int = 8_000_000,
    context_id: str = "phase1_csr_micro_campaign",
) -> dict[str, Any]:
    matrix_queue = tuple(read_jsonl(matrix_queue_path))
    candidate_queue = tuple(read_jsonl(candidate_queue_path))
    return run_csr_micro_campaign(
        matrix_queue,
        candidate_queue,
        matrix_queue_path=str(matrix_queue_path),
        candidate_queue_path=str(candidate_queue_path),
        device_memory_gb=device_memory_gb,
        max_matrices=max_matrices,
        max_rows=max_rows,
        max_cols=max_cols,
        max_stored_entries=max_stored_entries,
        max_archive_size_bytes=max_archive_size_bytes,
        context_id=context_id,
    )


def run_csr_micro_campaign(
    matrix_queue: Iterable[dict[str, Any]],
    candidate_queue: Iterable[dict[str, Any]],
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
    matrix_rows = tuple(matrix_queue)
    candidate_rows = tuple(candidate_queue)
    import_batch = import_selected_archives_to_csr(
        matrix_rows,
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
    ensure_taichi_cuda(device_memory_gb=device_memory_gb)
    engine = TaichiExecutionEngine()
    result_rows = tuple(
        _evaluate_candidate(
            csr_by_id[str(candidate["matrix_id"])],
            candidate,
            engine=engine,
            context_id=context_id,
        )
        for candidate in candidate_rows
        if str(candidate["matrix_id"]) in csr_by_id
    )
    selector_export = build_csr_selector_export(
        imported_records,
        result_rows,
        context_id=context_id,
    )
    summary = _summary(
        import_batch,
        result_rows,
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
        "result_rows": result_rows,
        "selector_export": selector_export,
        "summary": summary,
        "schema": _schema(),
    }


def write_csr_micro_campaign_artifacts(
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
        "summary": output / "csr_micro_campaign_summary.json",
        "schema": output / "csr_micro_campaign_schema.json",
        "report": output / "csr_micro_campaign_report.md",
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
    paths["summary"].write_text(
        json.dumps(export["summary"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths["schema"].write_text(
        json.dumps(export["schema"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_report(export["result_rows"], export["summary"], paths["report"])
    write_csr_diagnostic_rows(selector_export.diagnostics, paths["diagnostics"])
    write_csr_selector_rows(selector_export.selector_rows, paths["selector_rows"])
    write_csr_selector_summary(selector_export.summary, paths["selector_summary"])
    write_csr_selector_schema(paths["selector_schema"])
    write_csr_selector_report(selector_export, paths["selector_report"])
    return paths


def _evaluate_candidate(
    csr: CsrMatrix,
    candidate: dict[str, Any],
    *,
    engine: TaichiExecutionEngine,
    context_id: str,
) -> dict[str, Any]:
    screen = _screen_candidate(csr, candidate)
    if not screen["success"]:
        return _screened_out_row(csr, candidate, screen)
    repeats = int(candidate["measurement_repeats"])
    return aggregate_repeated_rows(
        _run_gpu_once(
            csr,
            candidate,
            screen=screen,
            engine=engine,
            context_id=context_id,
            repeat_index=repeat_index,
        )
        for repeat_index in range(repeats)
    )


def _screen_candidate(csr: CsrMatrix, candidate: dict[str, Any]) -> dict[str, Any]:
    solver = str(candidate["solver"])
    preconditioner = str(candidate["preconditioner"])
    max_iter = int(candidate["max_iter"])
    tolerance_rel = float(candidate["tolerance_rel"])
    if solver in {"cg", "pcg"}:
        return _cpu_cg_screen(
            csr,
            use_jacobi=preconditioner == "jacobi",
            tolerance_rel=tolerance_rel,
            max_iter=max_iter,
        )
    if solver == "bicgstab":
        return _cpu_bicgstab_screen(
            csr,
            use_jacobi=preconditioner == "jacobi",
            tolerance_rel=tolerance_rel,
            max_iter=max_iter,
        )
    if solver == "gmres":
        restart = int(candidate.get("solver_parameters", {}).get("restart", 16))
        return _cpu_gmres_screen(
            csr,
            use_jacobi=preconditioner == "jacobi",
            tolerance_rel=tolerance_rel,
            max_iter=max_iter,
            restart=restart,
        )
    if solver == "richardson":
        return _cpu_richardson_screen(
            csr,
            tolerance_rel=tolerance_rel,
            max_iter=max_iter,
        )
    return {"success": False, "breakdown": f"unsupported_solver:{solver}"}


def _run_gpu_once(
    csr: CsrMatrix,
    candidate: dict[str, Any],
    *,
    screen: dict[str, Any],
    engine: TaichiExecutionEngine,
    context_id: str,
    repeat_index: int,
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
    rhs = csr.matvec(tuple(1.0 for _ in range(csr.n_cols)))
    operator = TaichiCsrMatrixOperator.from_csr_matrix(csr, dtype=context.precision)
    plan = PolicyPlan(
        plan_id=f"{candidate['candidate_id']}:{csr.matrix_id}",
        backend="taichi_gpu",
        solver={"name": solver, **solver_parameters},
        preconditioner={"name": preconditioner},
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
    if final_relative_residual > context.tolerance_rel:
        status = "failed"
        failure_reasons.append("trace_residual_above_tolerance")
    if cpu_relative_residual > 1.0e-4:
        status = "failed"
        failure_reasons.append("cpu_recomputed_residual_above_tolerance")
    if solution_relative_error > 5.0e-3:
        status = "failed"
        failure_reasons.append("solution_error_above_tolerance")
    residual_history = tuple(result.trace.residual_history)
    if residual_history and residual_history[-1] > residual_history[0]:
        status = "failed"
        failure_reasons.append("residual_did_not_drop")
    row = {
        "schema_version": CSR_MICRO_CAMPAIGN_SCHEMA_VERSION,
        "matrix_id": csr.matrix_id,
        "candidate_id": str(candidate["candidate_id"]),
        "solver": solver,
        "preconditioner": preconditioner,
        "status": status,
        "failure_reasons": tuple(failure_reasons),
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
        "repeat_index": repeat_index,
        "target_screen_max_iter": int(candidate["max_iter"]),
        "transfer_time_ms": result.trace.transfer_time_ms,
    }
    row.update(solver_parameters)
    return row


def _screened_out_row(
    csr: CsrMatrix,
    candidate: dict[str, Any],
    screen: dict[str, Any],
) -> dict[str, Any]:
    solver = str(candidate["solver"])
    solver_parameters = _runtime_solver_parameters(candidate, screen)
    failure_reason = f"cpu_reference_{solver}_screen_failed"
    row = {
        "schema_version": CSR_MICRO_CAMPAIGN_SCHEMA_VERSION,
        "matrix_id": csr.matrix_id,
        "candidate_id": str(candidate["candidate_id"]),
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
        "screened_out_source": "phase1_csr_micro_campaign_cpu_screen",
        "target_screen_max_iter": int(candidate["max_iter"]),
        "measurement_repeats": 1,
        "success_count": 0,
        "success_rate": 0.0,
    }
    row.update(solver_parameters)
    return row


def _runtime_solver_parameters(
    candidate: dict[str, Any],
    screen: dict[str, Any],
) -> dict[str, Any]:
    solver = str(candidate["solver"])
    params = dict(candidate.get("solver_parameters", {}))
    params.pop("omega_source", None)
    if solver == "gmres":
        params["restart"] = int(params.get("restart", 16))
    if solver == "richardson":
        params["omega"] = float(screen.get("omega", RICHARDSON_OMEGA_CANDIDATES[0]))
    return params


def _cpu_cg_screen(
    csr: CsrMatrix,
    *,
    use_jacobi: bool,
    tolerance_rel: float,
    max_iter: int,
) -> dict[str, Any]:
    data = _CsrNp.from_csr(csr)
    x = np.zeros(csr.n_cols, dtype=np.float64)
    rhs = data.matvec(np.ones(csr.n_cols, dtype=np.float64))
    r = rhs.copy()
    z = data.apply_jacobi(r) if use_jacobi else r.copy()
    p = z.copy()
    rz_old = float(np.dot(r, z))
    b_norm = max(float(np.linalg.norm(rhs)), 1.0e-30)
    rel = float(np.linalg.norm(r)) / b_norm
    history = [rel]
    breakdown = None
    iterations = 0
    for iteration in range(1, max_iter + 1):
        ap = data.matvec(p)
        denom = float(np.dot(p, ap))
        if not math.isfinite(denom) or abs(denom) < 1.0e-30:
            breakdown = "zero_or_nonfinite_denominator"
            break
        if denom <= 0.0:
            breakdown = "non_positive_curvature"
            break
        alpha = rz_old / denom
        x = x + alpha * p
        r = r - alpha * ap
        rel = float(np.linalg.norm(r)) / b_norm
        history.append(rel)
        iterations = iteration
        if rel <= tolerance_rel:
            break
        z = data.apply_jacobi(r) if use_jacobi else r.copy()
        rz_new = float(np.dot(r, z))
        if not math.isfinite(rz_new) or abs(rz_old) < 1.0e-30:
            breakdown = "zero_or_nonfinite_rz"
            break
        beta = rz_new / rz_old
        if not math.isfinite(beta):
            breakdown = "nonfinite_beta"
            break
        p = z + beta * p
        rz_old = rz_new
    solution_error = _relative_error_to_ones(tuple(float(value) for value in x))
    return _screen_result(
        success=breakdown is None and rel <= tolerance_rel and solution_error <= 5.0e-3,
        iterations=iterations,
        rel=rel,
        solution_error=solution_error,
        breakdown=breakdown,
        history=history,
    )


def _cpu_bicgstab_screen(
    csr: CsrMatrix,
    *,
    use_jacobi: bool,
    tolerance_rel: float,
    max_iter: int,
) -> dict[str, Any]:
    data = _CsrNp.from_csr(csr)
    x = np.zeros(csr.n_cols, dtype=np.float64)
    rhs = data.matvec(np.ones(csr.n_cols, dtype=np.float64))
    r = rhs.copy()
    r_hat = r.copy()
    p = np.zeros(csr.n_rows, dtype=np.float64)
    v = np.zeros(csr.n_rows, dtype=np.float64)
    b_norm = max(float(np.linalg.norm(rhs)), 1.0e-30)
    rel = float(np.linalg.norm(r)) / b_norm
    rho_old = 1.0
    alpha = 1.0
    omega = 1.0
    history = [rel]
    breakdown = None
    iterations = 0
    for iteration in range(1, max_iter + 1):
        rho_new = float(np.dot(r_hat, r))
        if not math.isfinite(rho_new) or abs(rho_new) < 1.0e-30:
            breakdown = "zero_or_nonfinite_rho"
            break
        if iteration == 1:
            p = r.copy()
        else:
            if abs(omega) < 1.0e-30:
                breakdown = "zero_omega"
                break
            beta = (rho_new / rho_old) * (alpha / omega)
            if not math.isfinite(beta):
                breakdown = "nonfinite_beta"
                break
            p = r + beta * (p - omega * v)
        phat = data.apply_jacobi(p) if use_jacobi else p.copy()
        v = data.matvec(phat)
        denom = float(np.dot(r_hat, v))
        if not math.isfinite(denom) or abs(denom) < 1.0e-30:
            breakdown = "zero_or_nonfinite_alpha_denominator"
            break
        alpha = rho_new / denom
        if not math.isfinite(alpha):
            breakdown = "nonfinite_alpha"
            break
        x_alpha = x + alpha * phat
        s = r - alpha * v
        rel_s = float(np.linalg.norm(s)) / b_norm
        if rel_s <= tolerance_rel:
            x = x_alpha
            r = s
            rel = rel_s
            history.append(rel)
            iterations = iteration
            break
        shat = data.apply_jacobi(s) if use_jacobi else s.copy()
        t = data.matvec(shat)
        tt = float(np.dot(t, t))
        if not math.isfinite(tt) or abs(tt) < 1.0e-30:
            breakdown = "zero_or_nonfinite_omega_denominator"
            break
        omega = float(np.dot(t, s)) / tt
        if not math.isfinite(omega) or abs(omega) < 1.0e-30:
            breakdown = "zero_or_nonfinite_omega"
            break
        x = x_alpha + omega * shat
        r = s - omega * t
        rel = float(np.linalg.norm(r)) / b_norm
        history.append(rel)
        iterations = iteration
        if rel <= tolerance_rel:
            break
        rho_old = rho_new
    solution_error = _relative_error_to_ones(tuple(float(value) for value in x))
    return _screen_result(
        success=breakdown is None and rel <= tolerance_rel and solution_error <= 5.0e-3,
        iterations=iterations,
        rel=rel,
        solution_error=solution_error,
        breakdown=breakdown,
        history=history,
    )


def _cpu_gmres_screen(
    csr: CsrMatrix,
    *,
    use_jacobi: bool,
    tolerance_rel: float,
    max_iter: int,
    restart: int,
) -> dict[str, Any]:
    data = _CsrNp.from_csr(csr)
    x = np.zeros(csr.n_cols, dtype=np.float64)
    rhs = data.matvec(np.ones(csr.n_cols, dtype=np.float64))
    b_norm = max(float(np.linalg.norm(rhs)), 1.0e-30)
    r = rhs - data.matvec(x)
    rel = float(np.linalg.norm(r)) / b_norm
    history = [rel]
    breakdown = None
    iterations = 0
    while rel > tolerance_rel and breakdown is None and iterations < max_iter:
        z = data.apply_jacobi(r) if use_jacobi else r.copy()
        beta = float(np.linalg.norm(z))
        if not math.isfinite(beta) or beta <= 1.0e-30:
            breakdown = "zero_or_nonfinite_preconditioned_residual"
            break
        basis = [z / beta]
        hessenberg = np.zeros((restart + 1, restart), dtype=np.float64)
        target = np.zeros(restart + 1, dtype=np.float64)
        target[0] = beta
        x_base = x.copy()
        cycle_candidate = x.copy()
        inner_limit = min(restart, max_iter - iterations, csr.n_rows)
        for inner in range(inner_limit):
            w = data.matvec(basis[inner])
            if use_jacobi:
                w = data.apply_jacobi(w)
            for basis_index in range(inner + 1):
                hessenberg[basis_index, inner] = float(np.dot(basis[basis_index], w))
                w = w - hessenberg[basis_index, inner] * basis[basis_index]
            next_norm = float(np.linalg.norm(w))
            hessenberg[inner + 1, inner] = next_norm
            coefficients = np.linalg.lstsq(
                hessenberg[: inner + 2, : inner + 1],
                target[: inner + 2],
                rcond=None,
            )[0]
            cycle_candidate = x_base.copy()
            for coefficient, vector in zip(coefficients, basis):
                cycle_candidate = cycle_candidate + float(coefficient) * vector
            candidate_r = rhs - data.matvec(cycle_candidate)
            rel = float(np.linalg.norm(candidate_r)) / b_norm
            history.append(rel)
            iterations += 1
            if rel <= tolerance_rel:
                x = cycle_candidate
                r = candidate_r
                break
            if not math.isfinite(next_norm) or next_norm <= 1.0e-30:
                breakdown = "arnoldi_breakdown_without_convergence"
                break
            basis.append(w / next_norm)
        if rel <= tolerance_rel or breakdown is not None:
            break
        x = cycle_candidate
        r = rhs - data.matvec(x)
        rel = float(np.linalg.norm(r)) / b_norm
    solution_error = _relative_error_to_ones(tuple(float(value) for value in x))
    result = _screen_result(
        success=breakdown is None and rel <= tolerance_rel and solution_error <= 5.0e-3,
        iterations=iterations,
        rel=rel,
        solution_error=solution_error,
        breakdown=breakdown,
        history=history,
    )
    result["restart"] = restart
    return result


def _cpu_richardson_screen(
    csr: CsrMatrix,
    *,
    tolerance_rel: float,
    max_iter: int,
) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for omega in RICHARDSON_OMEGA_CANDIDATES:
        screen = _cpu_richardson_screen_one(
            csr,
            tolerance_rel=tolerance_rel,
            max_iter=max_iter,
            omega=omega,
        )
        if best is None or float(screen["relative_residual"]) < float(
            best["relative_residual"]
        ):
            best = screen
        if screen["success"]:
            return screen
    assert best is not None
    best = dict(best)
    best["tried_omega"] = tuple(RICHARDSON_OMEGA_CANDIDATES)
    return best


def _cpu_richardson_screen_one(
    csr: CsrMatrix,
    *,
    tolerance_rel: float,
    max_iter: int,
    omega: float,
) -> dict[str, Any]:
    data = _CsrNp.from_csr(csr)
    if np.any(np.abs(data.diagonal) <= 1.0e-30):
        return {
            "success": False,
            "breakdown": "zero_diagonal",
            "iterations": 0,
            "relative_residual": math.inf,
            "solution_relative_error": math.inf,
            "omega": omega,
        }
    x = np.zeros(csr.n_cols, dtype=np.float64)
    rhs = data.matvec(np.ones(csr.n_cols, dtype=np.float64))
    r = rhs.copy()
    b_norm = max(float(np.linalg.norm(rhs)), 1.0e-30)
    rel = float(np.linalg.norm(r)) / b_norm
    history = [rel]
    breakdown = None
    iterations = 0
    for iteration in range(1, max_iter + 1):
        z = data.apply_jacobi(r)
        az = data.matvec(z)
        x = x + omega * z
        r = r - omega * az
        rel = float(np.linalg.norm(r)) / b_norm
        history.append(rel)
        iterations = iteration
        if not math.isfinite(rel):
            breakdown = "nonfinite_residual"
            break
        if rel <= tolerance_rel:
            break
    solution_error = _relative_error_to_ones(tuple(float(value) for value in x))
    result = _screen_result(
        success=(
            breakdown is None
            and rel <= tolerance_rel
            and solution_error <= 5.0e-3
            and history[-1] <= history[0]
        ),
        iterations=iterations,
        rel=rel,
        solution_error=solution_error,
        breakdown=breakdown,
        history=history,
    )
    result["omega"] = omega
    return result


def _screen_result(
    *,
    success: bool,
    iterations: int,
    rel: float,
    solution_error: float,
    breakdown: str | None,
    history: list[float],
) -> dict[str, Any]:
    return {
        "success": success,
        "iterations": iterations,
        "relative_residual": rel,
        "solution_relative_error": solution_error,
        "breakdown": breakdown,
        "residual_history_length": len(history),
        "residual_history_head": tuple(history[:8]),
        "residual_history_tail": tuple(history[-8:]),
    }


class _CsrNp:
    def __init__(
        self,
        *,
        n_rows: int,
        row_ptr: np.ndarray,
        col_ind: np.ndarray,
        values: np.ndarray,
        diagonal: np.ndarray,
    ) -> None:
        self.n_rows = n_rows
        self.row_ptr = row_ptr
        self.col_ind = col_ind
        self.values = values
        self.diagonal = diagonal

    @classmethod
    def from_csr(cls, csr: CsrMatrix) -> "_CsrNp":
        row_ptr = np.asarray(csr.row_ptr, dtype=np.int64)
        col_ind = np.asarray(csr.col_ind, dtype=np.int64)
        values = np.asarray(csr.values, dtype=np.float64)
        diagonal = np.zeros(csr.n_rows, dtype=np.float64)
        for row in range(csr.n_rows):
            start = int(row_ptr[row])
            end = int(row_ptr[row + 1])
            columns = col_ind[start:end]
            hits = np.where(columns == row)[0]
            if hits.size:
                diagonal[row] = float(np.sum(values[start:end][hits]))
        return cls(
            n_rows=csr.n_rows,
            row_ptr=row_ptr,
            col_ind=col_ind,
            values=values,
            diagonal=diagonal,
        )

    def matvec(self, vector: np.ndarray) -> np.ndarray:
        out = np.zeros(self.n_rows, dtype=np.float64)
        for row in range(self.n_rows):
            start = int(self.row_ptr[row])
            end = int(self.row_ptr[row + 1])
            out[row] = float(np.dot(self.values[start:end], vector[self.col_ind[start:end]]))
        return out

    def apply_jacobi(self, values: np.ndarray) -> np.ndarray:
        return np.divide(
            values,
            self.diagonal,
            out=np.asarray(values, dtype=np.float64).copy(),
            where=np.abs(self.diagonal) > 1.0e-30,
        )


def _summary(
    import_batch: CsrImportBatch,
    result_rows: tuple[dict[str, Any], ...],
    *,
    selector_export: CsrSelectorExport,
    matrix_queue_path: str,
    candidate_queue_path: str,
    device_memory_gb: float,
    max_rows: int,
    max_cols: int,
    max_stored_entries: int,
    max_archive_size_bytes: int,
) -> dict[str, Any]:
    success_rows = tuple(row for row in result_rows if row["status"] == "success")
    screened_rows = tuple(row for row in result_rows if row["status"] == "screened_out")
    failed_rows = tuple(row for row in result_rows if row["status"] == "failed")
    max_final = max(
        (float(row.get("final_relative_residual") or 0.0) for row in success_rows),
        default=0.0,
    )
    max_cpu = max(
        (
            float(row.get("cpu_recomputed_relative_residual") or 0.0)
            for row in success_rows
        ),
        default=0.0,
    )
    max_solution = max(
        (float(row.get("solution_relative_error") or 0.0) for row in success_rows),
        default=0.0,
    )
    return {
        "status": (
            "passed"
            if import_batch.summary.status == "passed"
            and result_rows
            and success_rows
            and not failed_rows
            and selector_export.summary.status == "passed"
            else "failed"
        ),
        "schema_version": CSR_MICRO_CAMPAIGN_SCHEMA_VERSION,
        "matrix_queue_path": matrix_queue_path,
        "candidate_queue_path": candidate_queue_path,
        "device_memory_gb": device_memory_gb,
        "executes_gpu": True,
        "runtime_selector_changed": False,
        "imported_matrices": import_batch.summary.imported_matrices,
        "candidate_jobs": len(result_rows),
        "gpu_success_rows": len(success_rows),
        "cpu_screened_out_rows": len(screened_rows),
        "gpu_failed_rows": len(failed_rows),
        "selector_rows": selector_export.summary.num_selector_rows,
        "selector_oracle_rows": selector_export.summary.num_oracle_rows,
        "matrices_with_selector_rows": selector_export.summary.num_matrices_with_selector_rows,
        "max_final_relative_residual": max_final,
        "max_cpu_recomputed_relative_residual": max_cpu,
        "max_solution_relative_error": max_solution,
        "max_rows": max_rows,
        "max_cols": max_cols,
        "max_stored_entries": max_stored_entries,
        "max_archive_size_bytes": max_archive_size_bytes,
        "by_status": _counts(row["status"] for row in result_rows),
        "by_solver_status": _counts(
            f"{row['solver']}:{row['status']}" for row in result_rows
        ),
    }


def _schema() -> dict[str, Any]:
    return {
        "schema_version": CSR_MICRO_CAMPAIGN_SCHEMA_VERSION,
        "task": "execute_resource_bounded_real_csr_benchmark_queue",
        "runtime_selector_changed": False,
        "executes_gpu": True,
        "screening_policy": {
            "cpu_screen_required": True,
            "gpu_execution": "only_after_candidate_cpu_screen_success",
            "screened_out_status": "screened_out",
        },
        "integration_boundary": {
            "status": "real_selector_rows_ready_for_transformer_contract_export",
            "next_step": "merge_with_existing_selector_rows_and_export_model_contract_tensors",
        },
    }


def _write_report(
    result_rows: tuple[dict[str, Any], ...],
    summary: dict[str, Any],
    path: Path,
) -> Path:
    lines = [
        "# CSR Micro-Campaign",
        "",
        f"- status: `{summary['status']}`",
        f"- imported_matrices: `{summary['imported_matrices']}`",
        f"- candidate_jobs: `{summary['candidate_jobs']}`",
        f"- gpu_success_rows: `{summary['gpu_success_rows']}`",
        f"- cpu_screened_out_rows: `{summary['cpu_screened_out_rows']}`",
        f"- gpu_failed_rows: `{summary['gpu_failed_rows']}`",
        f"- selector_rows: `{summary['selector_rows']}`",
        f"- selector_oracle_rows: `{summary['selector_oracle_rows']}`",
        f"- max_final_relative_residual: `{summary['max_final_relative_residual']:.6g}`",
        f"- max_cpu_recomputed_relative_residual: "
        f"`{summary['max_cpu_recomputed_relative_residual']:.6g}`",
        f"- max_solution_relative_error: `{summary['max_solution_relative_error']:.6g}`",
        "",
        "| matrix | candidate | status | solver | preconditioner | repeats | median_ms | iters | rel_res | sol_err |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in result_rows:
        lines.append(
            "| "
            f"{row['matrix_id']} | "
            f"{row['candidate_id']} | "
            f"{row['status']} | "
            f"{row['solver']} | "
            f"{row['preconditioner']} | "
            f"{row.get('measurement_repeats', 1)} | "
            f"{_fmt(row.get('median_solve_time_ms'))} | "
            f"{row.get('num_iterations') or ''} | "
            f"{_fmt(row.get('final_relative_residual'))} | "
            f"{_fmt(row.get('solution_relative_error'))} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _counts(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _relative_error_to_ones(solution: tuple[float, ...]) -> float:
    diff_norm = math.sqrt(sum((value - 1.0) ** 2 for value in solution))
    true_norm = math.sqrt(max(float(len(solution)), 1.0e-30))
    return diff_norm / true_norm


def _relative_residual(
    csr: CsrMatrix,
    solution: tuple[float, ...],
    rhs: tuple[float, ...],
) -> float:
    actual = csr.matvec(solution)
    diff = [a - b for a, b in zip(actual, rhs)]
    return _norm(diff) / max(_norm(list(rhs)), 1.0e-30)


def _norm(values: list[float]) -> float:
    return math.sqrt(max(sum(value * value for value in values), 0.0))


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):.6g}"
