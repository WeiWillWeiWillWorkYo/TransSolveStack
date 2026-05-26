"""Run Taichi GPU CSR restarted GMRES smoke solves on nonsymmetric CSR fixtures."""

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
from transsolvestack.datasets.csr import CsrMatrix, csr_matrix_from_record
from transsolvestack.operators.taichi_csr import TaichiCsrMatrixOperator
from transsolvestack.operators.taichi_diffusion2d import ensure_taichi_cuda
from transsolvestack.profiling.artifacts import read_jsonl, write_jsonl
from transsolvestack.profiling.provenance import (
    CORE_TAICHI_CSR_GMRES_PROVENANCE_FILES,
    build_artifact_manifest,
    git_commit_or_unknown,
    write_manifest,
)
from transsolvestack.profiling.repeats import aggregate_repeated_rows
from transsolvestack.runtime.engine import TaichiExecutionEngine


PRECONDITIONERS = ("jacobi",)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csr",
        default="runs/phase1_suitesparse_csr_import/csr_matrices.jsonl",
    )
    parser.add_argument("--out", default="runs/phase1_taichi_csr_gmres")
    parser.add_argument("--max-matrices", type=int, default=2)
    parser.add_argument("--device-memory-gb", type=float, default=0.25)
    parser.add_argument("--precision", choices=("float32", "float64"), default="float64")
    parser.add_argument("--tolerance-rel", type=float, default=1.0e-5)
    parser.add_argument("--max-iter", type=int, default=512)
    parser.add_argument("--restart", type=int, default=None)
    parser.add_argument("--restarts", default="8,16,32")
    parser.add_argument("--measurement-repeats", type=int, default=3)
    args = parser.parse_args()
    if args.measurement_repeats < 1:
        raise SystemExit("--measurement-repeats must be >= 1")
    restarts = _parse_restarts(args.restarts, single_restart=args.restart)

    rows = read_jsonl(args.csr)
    selected, skipped = _select_gmres_rows(
        rows,
        max_matrices=args.max_matrices,
        tolerance_rel=args.tolerance_rel,
        max_iter=args.max_iter,
        restarts=restarts,
    )
    if not selected:
        raise SystemExit("no CPU-screened nonsymmetric CSR rows available for GMRES")

    ensure_taichi_cuda(device_memory_gb=args.device_memory_gb)
    context = SolveContext(
        context_id="phase1_taichi_csr_gmres_smoke",
        tolerance_abs=1.0e-7,
        tolerance_rel=args.tolerance_rel,
        max_iter=args.max_iter,
        precision=args.precision,
        required_backend="taichi_gpu",
    )
    engine = TaichiExecutionEngine()
    results: list[dict] = []
    screened_out_restart_configs = tuple(
        config for item in selected for config in item["screened_out_restart_configs"]
    )
    for item in selected:
        csr = csr_matrix_from_record(item["row"])
        for restart in restarts:
            for preconditioner in PRECONDITIONERS:
                cpu_screen = item["screens"][preconditioner][str(restart)]
                if not cpu_screen["success"]:
                    continue
                results.append(
                    _run_one(
                        csr,
                        preconditioner=preconditioner,
                        context=context,
                        engine=engine,
                        cpu_screen=cpu_screen,
                        restart=restart,
                        measurement_repeats=args.measurement_repeats,
                    )
                )

    summary = _build_summary(
        tuple(results),
        skipped=skipped,
        screened_out_restart_configs=screened_out_restart_configs,
        source_csr_path=args.csr,
        device_memory_gb=args.device_memory_gb,
        tolerance_rel=args.tolerance_rel,
        max_iter=args.max_iter,
        restarts=restarts,
        precision=args.precision,
        measurement_repeats=args.measurement_repeats,
    )
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "results": output / "csr_gmres_results.jsonl",
        "summary": output / "csr_gmres_summary.json",
        "report": output / "csr_gmres_report.md",
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
            artifact_kind="taichi_csr_gmres_smoke",
            command="scripts/tss_taichi_csr_gmres_smoke.py",
            tracked_files=CORE_TAICHI_CSR_GMRES_PROVENANCE_FILES,
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
                "restarts": summary["restarts"],
                "num_restarts": summary["num_restarts"],
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


def _select_gmres_rows(
    rows: tuple[dict, ...],
    *,
    max_matrices: int,
    tolerance_rel: float,
    max_iter: int,
    restarts: tuple[int, ...],
) -> tuple[tuple[dict, ...], tuple[dict, ...]]:
    selected: list[dict] = []
    skipped: list[dict] = []
    for row in rows:
        matrix_id = str(row["matrix_id"])
        if len(selected) >= max_matrices:
            skipped.append({"matrix_id": matrix_id, "reason": "after_selection_limit"})
            continue
        if row["symmetry"] == "symmetric":
            skipped.append({"matrix_id": matrix_id, "reason": "symmetric_matrix"})
            continue
        if int(row["n_rows"]) != int(row["n_cols"]):
            skipped.append({"matrix_id": matrix_id, "reason": "not_square"})
            continue
        csr = csr_matrix_from_record(row)
        screens = {
            preconditioner: {
                str(restart): _cpu_gmres_screen(
                    csr,
                    use_jacobi=preconditioner == "jacobi",
                    tolerance_rel=tolerance_rel,
                    max_iter=max_iter,
                    restart=restart,
                )
                for restart in restarts
            }
            for preconditioner in PRECONDITIONERS
        }
        failed_configs = tuple(
            {
                "matrix_id": matrix_id,
                "preconditioner": preconditioner,
                "restart": int(restart),
                "screen": screen,
            }
            for preconditioner, preconditioner_screens in screens.items()
            for restart, screen in preconditioner_screens.items()
            if not screen["success"]
        )
        if len(failed_configs) == len(PRECONDITIONERS) * len(restarts):
            skipped.append(
                {
                    "matrix_id": matrix_id,
                    "reason": "cpu_reference_gmres_screen_failed",
                    "screens": screens,
                }
            )
            continue
        selected.append(
            {
                "row": row,
                "screens": screens,
                "screened_out_restart_configs": failed_configs,
            }
        )
    return tuple(selected), tuple(skipped)


def _run_one(
    csr: CsrMatrix,
    *,
    preconditioner: str,
    context: SolveContext,
    engine: TaichiExecutionEngine,
    cpu_screen: dict,
    restart: int,
    measurement_repeats: int,
) -> dict:
    return aggregate_repeated_rows(
        _run_single(
            csr,
            preconditioner=preconditioner,
            context=context,
            engine=engine,
            cpu_screen=cpu_screen,
            restart=restart,
            repeat_index=repeat_index,
        )
        for repeat_index in range(measurement_repeats)
    )


def _run_single(
    csr: CsrMatrix,
    *,
    preconditioner: str,
    context: SolveContext,
    engine: TaichiExecutionEngine,
    cpu_screen: dict,
    restart: int,
    repeat_index: int,
) -> dict:
    rhs = csr.matvec(tuple(1.0 for _ in range(csr.n_cols)))
    operator = TaichiCsrMatrixOperator.from_csr_matrix(csr, dtype=context.precision)
    plan = PolicyPlan(
        plan_id=f"taichi_csr_gmres_{preconditioner}_restart{restart}:{csr.matrix_id}",
        backend="taichi_gpu",
        solver={"name": "gmres", "restart": restart},
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
        "solver": "gmres",
        "preconditioner": preconditioner,
        "restart": restart,
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
        "orthogonalization_backend": result.trace.metadata.get("orthogonalization_backend"),
        "basis_storage": result.trace.metadata.get("basis_storage"),
        "least_squares_backend": result.trace.metadata.get("least_squares_backend"),
        "host_control_time_ms": result.trace.metadata.get("host_control_time_ms"),
        "transfer_time_ms": result.trace.transfer_time_ms,
        "cpu_screen": cpu_screen,
        "repeat_index": repeat_index,
    }


def _cpu_gmres_screen(
    csr: CsrMatrix,
    *,
    use_jacobi: bool,
    tolerance_rel: float,
    max_iter: int,
    restart: int,
) -> dict:
    import numpy as np

    x = np.zeros(csr.n_cols, dtype=np.float64)
    true_x = np.ones(csr.n_cols, dtype=np.float64)
    rhs = np.asarray(csr.matvec(tuple(float(value) for value in true_x)), dtype=np.float64)
    diag = _diagonal(csr)
    b_norm = max(float(np.linalg.norm(rhs)), 1.0e-30)
    r = rhs - _matvec_np(csr, x)
    rel = float(np.linalg.norm(r)) / b_norm
    breakdown = None
    iterations = 0
    history = [rel]

    while rel > tolerance_rel and breakdown is None and iterations < max_iter:
        z = _apply_jacobi_np(r, diag) if use_jacobi else r.copy()
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
            w = _matvec_np(csr, basis[inner])
            if use_jacobi:
                w = _apply_jacobi_np(w, diag)
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
            candidate_r = rhs - _matvec_np(csr, cycle_candidate)
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
        r = rhs - _matvec_np(csr, x)
        rel = float(np.linalg.norm(r)) / b_norm
    solution_error = _relative_error_to_ones(tuple(float(value) for value in x))
    return {
        "success": breakdown is None and rel <= tolerance_rel and solution_error <= 5.0e-3,
        "iterations": iterations,
        "relative_residual": rel,
        "solution_relative_error": solution_error,
        "breakdown": breakdown,
        "restart": restart,
        "residual_history_length": len(history),
        "residual_history_head": tuple(history[:8]),
        "residual_history_tail": tuple(history[-8:]),
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


def _apply_jacobi_np(values, diag: tuple[float, ...]):
    import numpy as np

    diagonal = np.asarray(diag, dtype=np.float64)
    return np.divide(
        values,
        diagonal,
        out=np.asarray(values, dtype=np.float64).copy(),
        where=np.abs(diagonal) > 1.0e-30,
    )


def _matvec_np(csr: CsrMatrix, vector):
    import numpy as np

    return np.asarray(csr.matvec(tuple(float(value) for value in vector)), dtype=np.float64)


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
    screened_out_restart_configs: tuple[dict, ...],
    source_csr_path: str,
    device_memory_gb: float,
    tolerance_rel: float,
    max_iter: int,
    restarts: tuple[int, ...],
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
        "restarts": restarts,
        "num_restarts": len(restarts),
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
        "screened_out_restart_configs": screened_out_restart_configs,
        "skipped_candidates": skipped,
    }


def _parse_restarts(raw: str, *, single_restart: int | None) -> tuple[int, ...]:
    if single_restart is not None:
        values = (single_restart,)
    else:
        values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise SystemExit("--restarts must contain at least one value")
    if any(value < 1 for value in values):
        raise SystemExit("GMRES restart values must be >= 1")
    return tuple(dict.fromkeys(values))


def _write_report(results: tuple[dict, ...], summary: dict, path: Path) -> Path:
    lines = [
        "# Taichi CSR GMRES Smoke",
        "",
        f"- status: `{summary['status']}`",
        f"- selected_matrices: `{summary['num_selected_matrices']}`",
        f"- solves: `{summary['num_solves']}`",
        f"- restarts: `{', '.join(str(value) for value in summary['restarts'])}`",
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
        "| matrix | preconditioner | restart | status | repeats | median_solve_ms | iters | rel_res | cpu_rel_res | sol_rel_err |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        lines.append(
            "| "
            f"{row['matrix_id']} | "
            f"{row['preconditioner']} | "
            f"{row['restart']} | "
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
    if summary["screened_out_restart_configs"]:
        lines.extend(["", "## Screened-Out Restart Configs", ""])
        for row in summary["screened_out_restart_configs"]:
            lines.append(
                f"- `{row['matrix_id']}` `{row['preconditioner']}` "
                f"`restart={row['restart']}`"
            )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


if __name__ == "__main__":
    main()
