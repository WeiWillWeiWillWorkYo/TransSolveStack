"""Run Taichi GPU CSR CG/PCG smoke solves on imported real CSR fixtures."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transsolvestack.core.result import PolicyPlan
from transsolvestack.core.types import SolveContext
from transsolvestack.datasets.csr import CsrMatrix
from transsolvestack.operators.taichi_csr import TaichiCsrMatrixOperator
from transsolvestack.operators.taichi_diffusion2d import ensure_taichi_cuda
from transsolvestack.profiling.artifacts import read_jsonl, write_jsonl
from transsolvestack.profiling.provenance import (
    CORE_TAICHI_CSR_SOLVE_PROVENANCE_FILES,
    build_artifact_manifest,
    git_commit_or_unknown,
    write_manifest,
)
from transsolvestack.profiling.repeats import aggregate_repeated_rows
from transsolvestack.runtime.engine import TaichiExecutionEngine


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csr",
        default="runs/phase1_suitesparse_csr_import/csr_matrices.jsonl",
    )
    parser.add_argument("--out", default="runs/phase1_taichi_csr_solve")
    parser.add_argument("--max-matrices", type=int, default=2)
    parser.add_argument("--device-memory-gb", type=float, default=0.25)
    parser.add_argument("--precision", choices=("float32", "float64"), default="float64")
    parser.add_argument("--tolerance-rel", type=float, default=1.0e-5)
    parser.add_argument("--max-iter", type=int, default=256)
    parser.add_argument("--measurement-repeats", type=int, default=3)
    args = parser.parse_args()
    if args.measurement_repeats < 1:
        raise SystemExit("--measurement-repeats must be >= 1")

    rows = read_jsonl(args.csr)
    selected, skipped = _select_cg_eligible_rows(rows, max_matrices=args.max_matrices)
    if not selected:
        raise SystemExit("no CPU-screened symmetric CSR rows available for Taichi CSR solve")

    ensure_taichi_cuda(device_memory_gb=args.device_memory_gb)
    context = SolveContext(
        context_id="phase1_taichi_csr_solve_smoke",
        tolerance_abs=1.0e-7,
        tolerance_rel=args.tolerance_rel,
        max_iter=args.max_iter,
        precision=args.precision,
        required_backend="taichi_gpu",
    )
    engine = TaichiExecutionEngine()
    results: list[dict] = []
    for row in selected:
        csr = _csr_from_record(row)
        for solver_name, preconditioner in (
            ("cg", "none"),
            ("pcg", "jacobi"),
        ):
            results.append(
                _run_one(
                    csr,
                    solver_name=solver_name,
                    preconditioner=preconditioner,
                    context=context,
                    engine=engine,
                    measurement_repeats=args.measurement_repeats,
                )
            )

    summary = _build_summary(
        tuple(results),
        skipped=skipped,
        source_csr_path=args.csr,
        device_memory_gb=args.device_memory_gb,
        tolerance_rel=args.tolerance_rel,
        max_iter=args.max_iter,
        precision=args.precision,
        measurement_repeats=args.measurement_repeats,
    )
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "results": output / "csr_solve_results.jsonl",
        "summary": output / "csr_solve_summary.json",
        "report": output / "csr_solve_report.md",
        "manifest": output / "artifact_manifest.json",
    }
    write_jsonl(results, paths["results"])
    paths["summary"].write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_report(tuple(results), summary, paths["report"])
    write_manifest(
        build_artifact_manifest(
            artifact_kind="taichi_csr_solve_smoke",
            command="scripts/tss_taichi_csr_solve_smoke.py",
            tracked_files=CORE_TAICHI_CSR_SOLVE_PROVENANCE_FILES,
            metadata={
                "git_commit": git_commit_or_unknown(),
                "status": summary["status"],
                "num_solves": summary["num_solves"],
                "num_success": summary["num_success"],
                "num_failed": summary["num_failed"],
                "num_selected_matrices": summary["num_selected_matrices"],
                "max_final_relative_residual": summary[
                    "max_final_relative_residual"
                ],
                "max_solution_relative_error": summary[
                    "max_solution_relative_error"
                ],
                "precision": summary["precision"],
                "measurement_repeats": summary["measurement_repeats"],
            },
        ),
        paths["manifest"],
    )
    for key, value in paths.items():
        print(f"{key}: {value}")
    print(f"status: {summary['status']}")
    if summary["status"] != "passed":
        raise SystemExit(2)


def _select_cg_eligible_rows(
    rows: tuple[dict, ...],
    *,
    max_matrices: int,
) -> tuple[tuple[dict, ...], tuple[dict, ...]]:
    selected: list[dict] = []
    skipped: list[dict] = []
    for row in rows:
        if len(selected) >= max_matrices:
            break
        matrix_id = str(row["matrix_id"])
        if row["symmetry"] != "symmetric":
            skipped.append({"matrix_id": matrix_id, "reason": "not_symmetric"})
            continue
        csr = _csr_from_record(row)
        cg_screen = _cpu_cg_screen(csr, use_jacobi=False)
        pcg_screen = _cpu_cg_screen(csr, use_jacobi=True)
        if not cg_screen["success"] or not pcg_screen["success"]:
            skipped.append(
                {
                    "matrix_id": matrix_id,
                    "reason": "cpu_reference_cg_screen_failed",
                    "cg_screen": cg_screen,
                    "pcg_screen": pcg_screen,
                }
            )
            continue
        selected.append(row)
    for row in rows:
        matrix_id = str(row["matrix_id"])
        if len(selected) >= max_matrices and all(
            matrix_id != str(item["matrix_id"]) for item in selected
        ):
            if not any(matrix_id == item["matrix_id"] for item in skipped):
                skipped.append({"matrix_id": matrix_id, "reason": "after_selection_limit"})
    return tuple(selected), tuple(skipped)


def _run_one(
    csr: CsrMatrix,
    *,
    solver_name: str,
    preconditioner: str,
    context: SolveContext,
    engine: TaichiExecutionEngine,
    measurement_repeats: int,
) -> dict:
    return aggregate_repeated_rows(
        _run_single(
            csr,
            solver_name=solver_name,
            preconditioner=preconditioner,
            context=context,
            engine=engine,
            repeat_index=repeat_index,
        )
        for repeat_index in range(measurement_repeats)
    )


def _run_single(
    csr: CsrMatrix,
    *,
    solver_name: str,
    preconditioner: str,
    context: SolveContext,
    engine: TaichiExecutionEngine,
    repeat_index: int,
) -> dict:
    rhs = csr.matvec(tuple(1.0 for _ in range(csr.n_cols)))
    operator = TaichiCsrMatrixOperator.from_csr_matrix(csr, dtype=context.precision)
    plan = PolicyPlan(
        plan_id=f"taichi_csr_{solver_name}:{csr.matrix_id}",
        backend="taichi_gpu",
        solver={"name": solver_name},
        preconditioner={"name": preconditioner},
    )
    start = time.perf_counter()
    result = engine.solve(operator, rhs=rhs, context=context, plan=plan)
    wall_ms = (time.perf_counter() - start) * 1000.0
    solution = tuple(float(value) for value in result.solution)
    solution_relative_error = _relative_error_to_ones(solution)
    cpu_relative_residual = _relative_residual(csr, solution, rhs)
    final_relative_residual = float(result.trace.final_residual_norm or math.inf)
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
    if result.trace.residual_history[-1] > result.trace.residual_history[0]:
        status = "failed"
        failure_reasons.append("residual_did_not_drop")
    return {
        "matrix_id": csr.matrix_id,
        "solver": solver_name,
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
        "residual_history": tuple(result.trace.residual_history),
        "wall_time_ms": wall_ms,
        "solve_time_ms": result.trace.solve_time_ms,
        "backend": result.trace.backend,
        "precision": context.precision,
        "input": "rhs=A@ones",
        "repeat_index": repeat_index,
    }


def _csr_from_record(row: dict) -> CsrMatrix:
    csr = CsrMatrix(
        matrix_id=str(row["matrix_id"]),
        n_rows=int(row["n_rows"]),
        n_cols=int(row["n_cols"]),
        row_ptr=tuple(int(value) for value in row["row_ptr"]),
        col_ind=tuple(int(value) for value in row["col_ind"]),
        values=tuple(float(value) for value in row["values"]),
        field=str(row["field"]),
        symmetry=str(row["symmetry"]),
        source_path=str(row["source_archive_path"]),
        metadata=dict(row.get("metadata", {})),
    )
    csr.validate()
    return csr


def _cpu_cg_screen(csr: CsrMatrix, *, use_jacobi: bool) -> dict:
    x = [0.0 for _ in range(csr.n_cols)]
    true_x = [1.0 for _ in range(csr.n_cols)]
    rhs = list(csr.matvec(true_x))
    diag = _diagonal(csr)
    r = rhs[:]
    z = _apply_jacobi(r, diag) if use_jacobi else r[:]
    p = z[:]
    rz_old = _dot(r, z)
    b_norm = max(_norm(rhs), 1.0e-30)
    rel = _norm(r) / b_norm
    breakdown = None
    iterations = 0
    for iteration in range(1, 1000 + 1):
        ap = list(csr.matvec(p))
        denom = _dot(p, ap)
        if not math.isfinite(denom) or abs(denom) < 1.0e-30:
            breakdown = "zero_or_nonfinite_denominator"
            break
        if denom <= 0.0:
            breakdown = "non_positive_curvature"
            break
        alpha = rz_old / denom
        x = [x_i + alpha * p_i for x_i, p_i in zip(x, p)]
        r = [r_i - alpha * ap_i for r_i, ap_i in zip(r, ap)]
        rel = _norm(r) / b_norm
        iterations = iteration
        if rel <= 1.0e-6:
            break
        z = _apply_jacobi(r, diag) if use_jacobi else r[:]
        rz_new = _dot(r, z)
        if not math.isfinite(rz_new) or abs(rz_old) < 1.0e-30:
            breakdown = "zero_or_nonfinite_rz"
            break
        beta = rz_new / rz_old
        p = [z_i + beta * p_i for z_i, p_i in zip(z, p)]
        rz_old = rz_new
    solution_error = _relative_error_to_ones(tuple(x))
    return {
        "success": breakdown is None and rel <= 1.0e-6 and solution_error <= 5.0e-3,
        "iterations": iterations,
        "relative_residual": rel,
        "solution_relative_error": solution_error,
        "breakdown": breakdown,
    }


def _diagonal(csr: CsrMatrix) -> tuple[float, ...]:
    diag = [0.0 for _ in range(csr.n_rows)]
    for row in range(csr.n_rows):
        for offset in range(csr.row_ptr[row], csr.row_ptr[row + 1]):
            if csr.col_ind[offset] == row:
                diag[row] += csr.values[offset]
    return tuple(diag)


def _apply_jacobi(values: list[float], diag: tuple[float, ...]) -> list[float]:
    return [
        value / diag_value if abs(diag_value) > 1.0e-30 else value
        for value, diag_value in zip(values, diag)
    ]


def _dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _norm(values: list[float]) -> float:
    return math.sqrt(max(_dot(values, values), 0.0))


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


def _build_summary(
    results: tuple[dict, ...],
    *,
    skipped: tuple[dict, ...],
    source_csr_path: str,
    device_memory_gb: float,
    tolerance_rel: float,
    max_iter: int,
    precision: str,
    measurement_repeats: int,
) -> dict:
    num_success = sum(1 for row in results if row["status"] == "success")
    num_failed = len(results) - num_success
    selected_matrices = sorted({row["matrix_id"] for row in results})
    return {
        "status": "passed" if results and num_failed == 0 else "failed",
        "source_csr_path": source_csr_path,
        "device_memory_gb": device_memory_gb,
        "tolerance_rel": tolerance_rel,
        "max_iter": max_iter,
        "precision": precision,
        "measurement_repeats": measurement_repeats,
        "num_selected_matrices": len(selected_matrices),
        "selected_matrices": selected_matrices,
        "num_solves": len(results),
        "num_success": num_success,
        "num_failed": num_failed,
        "total_csr_nnz": sum(int(row["csr_nnz"]) for row in results),
        "max_final_relative_residual": max(
            (float(row["final_relative_residual"]) for row in results),
            default=0.0,
        ),
        "max_cpu_recomputed_relative_residual": max(
            (float(row["cpu_recomputed_relative_residual"]) for row in results),
            default=0.0,
        ),
        "max_solution_relative_error": max(
            (float(row["solution_relative_error"]) for row in results),
            default=0.0,
        ),
        "skipped_candidates": skipped,
    }


def _write_report(results: tuple[dict, ...], summary: dict, path: Path) -> Path:
    lines = [
        "# Taichi CSR Solve Smoke",
        "",
        f"- status: `{summary['status']}`",
        f"- selected_matrices: `{summary['num_selected_matrices']}`",
        f"- solves: `{summary['num_solves']}`",
        f"- success: `{summary['num_success']}`",
        f"- failed: `{summary['num_failed']}`",
        f"- measurement_repeats: `{summary['measurement_repeats']}`",
        f"- total_csr_nnz: `{summary['total_csr_nnz']}`",
        f"- max_final_relative_residual: "
        f"`{summary['max_final_relative_residual']:.6g}`",
        f"- max_cpu_recomputed_relative_residual: "
        f"`{summary['max_cpu_recomputed_relative_residual']:.6g}`",
        f"- max_solution_relative_error: "
        f"`{summary['max_solution_relative_error']:.6g}`",
        "",
        "| matrix | solver | preconditioner | status | repeats | median_solve_ms | iters | rel_res | cpu_rel_res | sol_rel_err |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        lines.append(
            "| "
            f"{row['matrix_id']} | "
            f"{row['solver']} | "
            f"{row['preconditioner']} | "
            f"{row['status']} | "
            f"{row['measurement_repeats']} | "
            f"{row['median_solve_time_ms']:.6g} | "
            f"{row['num_iterations']} | "
            f"{row['final_relative_residual']:.6g} | "
            f"{row['cpu_recomputed_relative_residual']:.6g} | "
            f"{row['solution_relative_error']:.6g} |"
        )
    lines.append("")
    lines.append("## Skipped Candidates")
    lines.append("")
    for row in summary["skipped_candidates"]:
        lines.append(f"- `{row['matrix_id']}`: `{row['reason']}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


if __name__ == "__main__":
    main()
