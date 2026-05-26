"""Run Taichi GPU CSR row/column-equilibration smoke probes."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

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
    CORE_TAICHI_CSR_ROW_COLUMN_EQUILIBRATION_PROVENANCE_FILES,
    build_artifact_manifest,
    git_commit_or_unknown,
    write_manifest,
)
from transsolvestack.runtime.engine import TaichiExecutionEngine


SCHEMA_VERSION = "phase1_taichi_csr_row_column_equilibration_v1"
DEFAULT_CSR_RECORD_PATHS = (
    "runs/phase1_suitesparse_csr_import/csr_matrices.jsonl",
    "runs/phase1_csr_micro_campaign/csr_matrices.jsonl",
)
DEFAULT_MATRIX_IDS = (
    "suitesparse:HB/curtis54",
    "suitesparse:Zitney/extr1b",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matrix-id",
        action="append",
        default=list(DEFAULT_MATRIX_IDS),
        help="Matrix id to probe. May be provided more than once.",
    )
    parser.add_argument(
        "--csr-records",
        action="append",
        default=list(DEFAULT_CSR_RECORD_PATHS),
        help="CSR matrix JSONL records. May be provided more than once.",
    )
    parser.add_argument("--solver", choices=("bicgstab", "gmres"), default="bicgstab")
    parser.add_argument("--gmres-restart", type=int, default=32)
    parser.add_argument("--scaling-passes", type=int, default=4)
    parser.add_argument("--device-memory-gb", type=float, default=0.25)
    parser.add_argument("--max-iter", type=int, default=512)
    parser.add_argument("--tolerance-rel", type=float, default=1.0e-5)
    parser.add_argument("--out", default="runs/phase1_taichi_csr_row_column_equilibration")
    args = parser.parse_args()

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    csr_by_id = _load_csr_records(tuple(args.csr_records))
    missing = tuple(matrix_id for matrix_id in args.matrix_id if matrix_id not in csr_by_id)
    if missing:
        raise SystemExit(f"missing CSR records: {missing}")
    ensure_taichi_cuda(device_memory_gb=float(args.device_memory_gb))
    rows = tuple(
        _run_gpu_probe(
            csr_by_id[matrix_id],
            solver=str(args.solver),
            gmres_restart=int(args.gmres_restart),
            scaling_passes=int(args.scaling_passes),
            max_iter=int(args.max_iter),
            tolerance_rel=float(args.tolerance_rel),
        )
        for matrix_id in args.matrix_id
    )
    summary = _summary(
        rows,
        csr_record_paths=tuple(args.csr_records),
        device_memory_gb=float(args.device_memory_gb),
        solver=str(args.solver),
        gmres_restart=int(args.gmres_restart),
        scaling_passes=int(args.scaling_passes),
        max_iter=int(args.max_iter),
        tolerance_rel=float(args.tolerance_rel),
    )
    paths = {
        "results": output / "csr_row_column_equilibration_results.jsonl",
        "summary": output / "csr_row_column_equilibration_summary.json",
        "schema": output / "csr_row_column_equilibration_schema.json",
        "report": output / "csr_row_column_equilibration_report.md",
        "manifest": output / "artifact_manifest.json",
    }
    write_jsonl(rows, paths["results"])
    paths["summary"].write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths["schema"].write_text(
        json.dumps(_schema(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_report(rows, summary, paths["report"])
    write_manifest(
        build_artifact_manifest(
            artifact_kind="taichi_csr_row_column_equilibration_smoke",
            command="scripts/tss_taichi_csr_row_column_equilibration_smoke.py",
            tracked_files=CORE_TAICHI_CSR_ROW_COLUMN_EQUILIBRATION_PROVENANCE_FILES,
            metadata={
                "git_commit": git_commit_or_unknown(),
                "status": summary["status"],
                "schema_version": summary["schema_version"],
                "gpu_executed_rows": summary["gpu_executed_rows"],
                "numeric_success_rows": summary["numeric_success_rows"],
                "failed_numeric_gate_rows": summary["failed_numeric_gate_rows"],
                "candidate_promoted_rows": summary["candidate_promoted_rows"],
                "executes_gpu": summary["executes_gpu"],
                "runtime_selector_changed": summary["runtime_selector_changed"],
            },
        ),
        paths["manifest"],
    )
    for key, value in paths.items():
        print(f"{key}: {value}")
    print(f"status: {summary['status']}")
    if summary["status"] != "passed":
        raise SystemExit(2)


def _load_csr_records(paths: tuple[str, ...]) -> dict[str, CsrMatrix]:
    csr_by_id: dict[str, CsrMatrix] = {}
    for path in paths:
        for record in read_jsonl(path):
            if record.get("status") != "success":
                continue
            matrix_id = str(record["matrix_id"])
            if matrix_id not in csr_by_id:
                csr_by_id[matrix_id] = csr_matrix_from_record(record)
    return csr_by_id


def _run_gpu_probe(
    csr: CsrMatrix,
    *,
    solver: str,
    gmres_restart: int,
    scaling_passes: int,
    max_iter: int,
    tolerance_rel: float,
) -> dict[str, Any]:
    rhs = csr.matvec((1.0 for _ in range(csr.n_cols)))
    context = SolveContext(
        context_id="phase1_taichi_csr_row_column_equilibration",
        tolerance_abs=0.0,
        tolerance_rel=tolerance_rel,
        max_iter=max_iter,
        precision="float64",
        required_backend="taichi_gpu",
    )
    solver_config: dict[str, Any] = {"name": solver}
    if solver == "gmres":
        solver_config["restart"] = gmres_restart
    candidate_id = _candidate_id(solver, gmres_restart)
    plan = PolicyPlan(
        plan_id=f"{candidate_id}:{csr.matrix_id}",
        backend="taichi_gpu",
        solver=solver_config,
        preconditioner={
            "name": "row_column_equilibration",
            "passes": scaling_passes,
        },
    )
    start = time.perf_counter()
    result = TaichiExecutionEngine().solve(
        TaichiCsrMatrixOperator.from_csr_matrix(csr, dtype="float64"),
        rhs=rhs,
        context=context,
        plan=plan,
    )
    wall_ms = (time.perf_counter() - start) * 1000.0
    solution = tuple(float(value) for value in result.solution)
    cpu_relative_residual = _relative_residual(csr, solution, rhs)
    solution_relative_error = _relative_error_to_ones(solution)
    final_relative_residual = float(result.trace.final_residual_norm or math.inf)
    failure_reasons: list[str] = []
    if result.status != "success":
        failure_reasons.append(result.trace.failure_class or "solver_failed")
    if final_relative_residual > tolerance_rel:
        failure_reasons.append("trace_residual_above_tolerance")
    if cpu_relative_residual > 1.0e-4:
        failure_reasons.append("cpu_recomputed_residual_above_tolerance")
    if solution_relative_error > 5.0e-3:
        failure_reasons.append("solution_error_above_tolerance")
    if result.trace.residual_history[-1] > result.trace.residual_history[0]:
        failure_reasons.append("scaled_residual_did_not_drop")
    numeric_status = "success" if not failure_reasons else "failed_numeric_gate"
    return {
        "schema_version": SCHEMA_VERSION,
        "matrix_id": csr.matrix_id,
        "candidate_id": candidate_id,
        "solver": solver,
        "preconditioner": "row_column_equilibration",
        "scaling_passes": scaling_passes,
        "precision": "float64",
        "backend": result.trace.backend,
        "gpu_executed": result.trace.backend == "taichi_gpu",
        "solver_status": result.status,
        "numeric_status": numeric_status,
        "candidate_promoted": numeric_status == "success",
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
        "scaled_relative_residual": result.trace.metadata.get(
            "scaled_relative_residual_norm"
        ),
        "solve_time_ms": result.trace.solve_time_ms,
        "wall_time_ms": wall_ms,
        "trace_metadata": result.trace.metadata,
        "runtime_selector_changed": False,
        "executes_gpu": True,
        "input": "rhs=A@ones",
    }


def _summary(
    rows: tuple[dict[str, Any], ...],
    *,
    csr_record_paths: tuple[str, ...],
    device_memory_gb: float,
    solver: str,
    gmres_restart: int,
    scaling_passes: int,
    max_iter: int,
    tolerance_rel: float,
) -> dict[str, Any]:
    gpu_rows = tuple(row for row in rows if row["gpu_executed"])
    numeric_success_rows = tuple(row for row in rows if row["numeric_status"] == "success")
    failed_numeric_rows = tuple(
        row for row in rows if row["numeric_status"] == "failed_numeric_gate"
    )
    promoted_rows = tuple(row for row in rows if row["candidate_promoted"])
    matrix_ids = tuple(row["matrix_id"] for row in rows)
    status = (
        "passed"
        if len(rows) >= 2
        and len(gpu_rows) == len(rows)
        and len(numeric_success_rows) >= 1
        and "suitesparse:Zitney/extr1b" in matrix_ids
        and all(row["executes_gpu"] is True for row in rows)
        and all(row["runtime_selector_changed"] is False for row in rows)
        and all(math.isfinite(float(row["final_relative_residual"])) for row in rows)
        and all(math.isfinite(float(row["solution_relative_error"])) for row in rows)
        else "failed"
    )
    return {
        "status": status,
        "schema_version": SCHEMA_VERSION,
        "csr_record_paths": csr_record_paths,
        "device_memory_gb": device_memory_gb,
        "solver": solver,
        "gmres_restart": gmres_restart if solver == "gmres" else None,
        "scaling_passes": scaling_passes,
        "max_iter": max_iter,
        "tolerance_rel": tolerance_rel,
        "executes_gpu": True,
        "runtime_selector_changed": False,
        "gpu_executed_rows": len(gpu_rows),
        "candidate_rows": len(rows),
        "numeric_success_rows": len(numeric_success_rows),
        "failed_numeric_gate_rows": len(failed_numeric_rows),
        "candidate_promoted_rows": len(promoted_rows),
        "by_numeric_status": _counts(row["numeric_status"] for row in rows),
        "by_matrix_numeric_status": {
            row["matrix_id"]: row["numeric_status"] for row in rows
        },
        "max_final_relative_residual": max(
            float(row["final_relative_residual"]) for row in rows
        ),
        "max_cpu_recomputed_relative_residual": max(
            float(row["cpu_recomputed_relative_residual"]) for row in rows
        ),
        "max_solution_relative_error": max(
            float(row["solution_relative_error"]) for row in rows
        ),
        "next_step": (
            "merge_successful_row_column_scaled_candidates_into_guarded_fallback "
            "only after exact-matrix selector rows and fallback-chain gates are updated"
        ),
    }


def _schema() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "task": "execute_taichi_gpu_row_column_equilibration_probe",
        "algorithm": [
            "initialize Dr and Dc to identity",
            "repeat fixed L1 equilibration passes on the scaled matrix Dr A Dc",
            "solve Dr A Dc y = Dr b on Taichi GPU",
            "recover original solution with x = Dc y",
            "gate promotion on original residual and rhs=A@ones solution error",
        ],
        "integration_boundary": {
            "candidate_promotion_requires_numeric_success": True,
            "executes_gpu": True,
            "runtime_selector_changed": False,
        },
        "numeric_gate": [
            "solver status must be success",
            "trace and CPU recomputed residuals must pass",
            "solution relative error must be <= 5e-3 for rhs=A@ones",
        ],
    }


def _write_report(
    rows: tuple[dict[str, Any], ...],
    summary: dict[str, Any],
    path: Path,
) -> None:
    lines = [
        "# Taichi CSR Row/Column Equilibration",
        "",
        f"- status: `{summary['status']}`",
        f"- solver: `{summary['solver']}`",
        f"- gpu_executed_rows: `{summary['gpu_executed_rows']}`",
        f"- numeric_success_rows: `{summary['numeric_success_rows']}`",
        f"- failed_numeric_gate_rows: `{summary['failed_numeric_gate_rows']}`",
        f"- candidate_promoted_rows: `{summary['candidate_promoted_rows']}`",
        f"- executes_gpu: `{summary['executes_gpu']}`",
        f"- runtime_selector_changed: `{summary['runtime_selector_changed']}`",
        f"- next_step: `{summary['next_step']}`",
        "",
        "| matrix | candidate | solver_status | numeric_status | iters | rel_res | cpu_rel_res | sol_err | failures |",
        "|---|---|---|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['matrix_id']} | "
            f"{row['candidate_id']} | "
            f"{row['solver_status']} | "
            f"{row['numeric_status']} | "
            f"{row['num_iterations']} | "
            f"{_fmt(row['final_relative_residual'])} | "
            f"{_fmt(row['cpu_recomputed_relative_residual'])} | "
            f"{_fmt(row['solution_relative_error'])} | "
            f"{','.join(row['failure_reasons']) or 'none'} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _candidate_id(solver: str, restart: int) -> str:
    if solver == "gmres":
        return f"taichi_csr_gmres_row_column_equilibration_restart{restart}_float64"
    return "taichi_csr_bicgstab_row_column_equilibration_float64"


def _relative_residual(csr: CsrMatrix, solution: tuple[float, ...], rhs: tuple[float, ...]) -> float:
    ax = csr.matvec(solution)
    residual_sq = sum((float(b) - float(a)) ** 2 for a, b in zip(ax, rhs))
    rhs_sq = sum(float(value) ** 2 for value in rhs)
    return math.sqrt(residual_sq) / max(math.sqrt(rhs_sq), 1.0e-30)


def _relative_error_to_ones(solution: tuple[float, ...]) -> float:
    error_sq = sum((float(value) - 1.0) ** 2 for value in solution)
    truth_sq = float(len(solution))
    return math.sqrt(error_sq) / max(math.sqrt(truth_sq), 1.0e-30)


def _counts(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _fmt(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "nan"
    if not math.isfinite(numeric):
        return str(numeric)
    return f"{numeric:.6g}"


if __name__ == "__main__":
    main()
