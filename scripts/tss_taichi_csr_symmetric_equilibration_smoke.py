"""Run a Taichi GPU CSR symmetric-equilibration PCG smoke probe."""

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
    CORE_TAICHI_CSR_SYMMETRIC_EQUILIBRATION_PROVENANCE_FILES,
    build_artifact_manifest,
    git_commit_or_unknown,
    write_manifest,
)
from transsolvestack.runtime.engine import TaichiExecutionEngine


SCHEMA_VERSION = "phase1_taichi_csr_symmetric_equilibration_v1"
DEFAULT_CSR_RECORD_PATHS = (
    "runs/phase1_csr_micro_campaign/csr_matrices.jsonl",
    "runs/phase1_suitesparse_csr_import/csr_matrices.jsonl",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-id", default="suitesparse:HB/bcsstk07")
    parser.add_argument(
        "--csr-records",
        action="append",
        default=list(DEFAULT_CSR_RECORD_PATHS),
        help="CSR matrix JSONL records. May be provided more than once.",
    )
    parser.add_argument("--device-memory-gb", type=float, default=0.25)
    parser.add_argument("--max-iter", type=int, default=2048)
    parser.add_argument("--tolerance-rel", type=float, default=1.0e-5)
    parser.add_argument("--out", default="runs/phase1_taichi_csr_symmetric_equilibration")
    args = parser.parse_args()

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    csr = _load_csr(args.matrix_id, tuple(args.csr_records))
    ensure_taichi_cuda(device_memory_gb=float(args.device_memory_gb))
    row = _run_gpu_probe(
        csr,
        max_iter=int(args.max_iter),
        tolerance_rel=float(args.tolerance_rel),
    )
    rows = (row,)
    summary = _summary(
        rows,
        csr_record_paths=tuple(args.csr_records),
        device_memory_gb=float(args.device_memory_gb),
        max_iter=int(args.max_iter),
        tolerance_rel=float(args.tolerance_rel),
    )
    paths = {
        "results": output / "csr_symmetric_equilibration_results.jsonl",
        "summary": output / "csr_symmetric_equilibration_summary.json",
        "schema": output / "csr_symmetric_equilibration_schema.json",
        "report": output / "csr_symmetric_equilibration_report.md",
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
            artifact_kind="taichi_csr_symmetric_equilibration_smoke",
            command="scripts/tss_taichi_csr_symmetric_equilibration_smoke.py",
            tracked_files=CORE_TAICHI_CSR_SYMMETRIC_EQUILIBRATION_PROVENANCE_FILES,
            metadata={
                "git_commit": git_commit_or_unknown(),
                "status": summary["status"],
                "schema_version": summary["schema_version"],
                "gpu_executed_rows": summary["gpu_executed_rows"],
                "candidate_promoted": summary["candidate_promoted"],
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


def _load_csr(matrix_id: str, paths: tuple[str, ...]) -> CsrMatrix:
    for path in paths:
        for record in read_jsonl(path):
            if record.get("status") == "success" and record["matrix_id"] == matrix_id:
                return csr_matrix_from_record(record)
    raise SystemExit(f"missing CSR record for {matrix_id}")


def _run_gpu_probe(
    csr: CsrMatrix,
    *,
    max_iter: int,
    tolerance_rel: float,
) -> dict[str, Any]:
    rhs = csr.matvec((1.0 for _ in range(csr.n_cols)))
    context = SolveContext(
        context_id="phase1_taichi_csr_symmetric_equilibration",
        tolerance_abs=0.0,
        tolerance_rel=tolerance_rel,
        max_iter=max_iter,
        precision="float64",
        required_backend="taichi_gpu",
    )
    plan = PolicyPlan(
        plan_id=f"taichi_csr_pcg_symmetric_equilibration_float64:{csr.matrix_id}",
        backend="taichi_gpu",
        solver={"name": "pcg"},
        preconditioner={"name": "symmetric_equilibration"},
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
    failure_reasons: list[str] = []
    if result.status != "success":
        failure_reasons.append(result.trace.failure_class or "solver_failed")
    if float(result.trace.final_residual_norm or math.inf) > tolerance_rel:
        failure_reasons.append("trace_residual_above_tolerance")
    if cpu_relative_residual > 1.0e-4:
        failure_reasons.append("cpu_recomputed_residual_above_tolerance")
    if solution_relative_error > 5.0e-3:
        failure_reasons.append("solution_error_above_tolerance")
    numeric_status = "success" if not failure_reasons else "failed_numeric_gate"
    return {
        "schema_version": SCHEMA_VERSION,
        "matrix_id": csr.matrix_id,
        "candidate_id": "taichi_csr_pcg_symmetric_equilibration_float64",
        "solver": "pcg",
        "preconditioner": "symmetric_equilibration",
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
        "final_relative_residual": result.trace.final_residual_norm,
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
    max_iter: int,
    tolerance_rel: float,
) -> dict[str, Any]:
    gpu_rows = tuple(row for row in rows if row["gpu_executed"])
    promoted_rows = tuple(row for row in rows if row["candidate_promoted"])
    failed_numeric_rows = tuple(
        row for row in rows if row["numeric_status"] == "failed_numeric_gate"
    )
    expected_gate = any(
        "solution_error_above_tolerance" in row["failure_reasons"]
        for row in failed_numeric_rows
    )
    status = (
        "passed"
        if len(rows) == 1
        and len(gpu_rows) == 1
        and not promoted_rows
        and len(failed_numeric_rows) == 1
        and expected_gate
        and all(row["executes_gpu"] is True for row in rows)
        and all(row["runtime_selector_changed"] is False for row in rows)
        else "failed"
    )
    return {
        "status": status,
        "schema_version": SCHEMA_VERSION,
        "csr_record_paths": csr_record_paths,
        "device_memory_gb": device_memory_gb,
        "max_iter": max_iter,
        "tolerance_rel": tolerance_rel,
        "executes_gpu": True,
        "runtime_selector_changed": False,
        "gpu_executed_rows": len(gpu_rows),
        "candidate_rows": len(rows),
        "candidate_promoted": bool(promoted_rows),
        "failed_numeric_gate_rows": len(failed_numeric_rows),
        "by_numeric_status": _counts(row["numeric_status"] for row in rows),
        "max_final_relative_residual": max(
            float(row["final_relative_residual"]) for row in rows
        ),
        "max_cpu_recomputed_relative_residual": max(
            float(row["cpu_recomputed_relative_residual"]) for row in rows
        ),
        "max_solution_relative_error": max(
            float(row["solution_relative_error"]) for row in rows
        ),
        "next_step": "do_not_promote_symmetric_equilibration_for_bcsstk07_without_ic0_or_better_gate",
    }


def _schema() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "task": "execute_taichi_gpu_symmetric_equilibration_pcg_probe",
        "integration_boundary": {
            "executes_gpu": True,
            "runtime_selector_changed": False,
            "candidate_promotion_requires_numeric_success": True,
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
        "# Taichi CSR Symmetric Equilibration",
        "",
        f"- status: `{summary['status']}`",
        f"- gpu_executed_rows: `{summary['gpu_executed_rows']}`",
        f"- candidate_promoted: `{summary['candidate_promoted']}`",
        f"- failed_numeric_gate_rows: `{summary['failed_numeric_gate_rows']}`",
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
            f"{float(row['final_relative_residual']):.6g} | "
            f"{float(row['cpu_recomputed_relative_residual']):.6g} | "
            f"{float(row['solution_relative_error']):.6g} | "
            f"{', '.join(row['failure_reasons'])} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _relative_error_to_ones(solution: tuple[float, ...]) -> float:
    diff_norm = math.sqrt(sum((value - 1.0) ** 2 for value in solution))
    true_norm = math.sqrt(max(float(len(solution)), 1.0e-30))
    return diff_norm / true_norm


def _relative_residual(
    csr: CsrMatrix,
    solution: tuple[float, ...],
    rhs: tuple[float, ...],
) -> float:
    ax = csr.matvec(solution)
    residual_norm = math.sqrt(sum((a - b) ** 2 for a, b in zip(ax, rhs)))
    rhs_norm = math.sqrt(max(sum(value * value for value in rhs), 1.0e-30))
    return residual_norm / rhs_norm


def _counts(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


if __name__ == "__main__":
    main()
