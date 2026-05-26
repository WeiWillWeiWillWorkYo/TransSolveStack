"""Exercise public auto_solve_csr on real CSR selector fixtures."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import transsolvestack as tss
from transsolvestack.core.types import SolveContext
from transsolvestack.datasets.csr import CsrMatrix, csr_matrix_from_record
from transsolvestack.profiling.artifacts import read_jsonl, write_jsonl
from transsolvestack.profiling.provenance import (
    CORE_CSR_AUTO_SOLVE_PROVENANCE_FILES,
    build_artifact_manifest,
    git_commit_or_unknown,
    write_manifest,
)
from transsolvestack.profiling.trace import trace_to_record


SELECTED_MATRIX_IDS = (
    "suitesparse:JGD_Trefethen/Trefethen_20b",
    "suitesparse:HB/curtis54",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csr",
        default="runs/phase1_suitesparse_csr_import/csr_matrices.jsonl",
    )
    parser.add_argument(
        "--selector-rows",
        default="runs/phase1_csr_selector_readiness/csr_selector_rows.jsonl",
    )
    parser.add_argument("--out", default="runs/phase1_csr_auto_solve")
    parser.add_argument("--device-memory-gb", type=float, default=0.25)
    args = parser.parse_args()

    rows = {str(row["matrix_id"]): row for row in read_jsonl(args.csr)}
    missing = [matrix_id for matrix_id in SELECTED_MATRIX_IDS if matrix_id not in rows]
    if missing:
        raise SystemExit(f"missing selected CSR rows: {missing}")

    context = SolveContext(
        context_id="phase1_csr_selector",
        tolerance_abs=1.0e-7,
        tolerance_rel=1.0e-5,
        max_iter=512,
        precision="float64",
        required_backend="taichi_gpu",
    )
    result_records = []
    for matrix_id in SELECTED_MATRIX_IDS:
        csr = csr_matrix_from_record(rows[matrix_id])
        rhs = csr.matvec(tuple(1.0 for _ in range(csr.n_cols)))
        result = tss.auto_solve_csr(
            csr,
            rhs,
            selector_path=args.selector_rows,
            context=context,
            device_memory_gb=args.device_memory_gb,
        )
        result_records.append(
            _result_record(
                csr,
                result,
                rhs,
                context=context,
            )
        )

    summary = _build_summary(
        result_records,
        selector_rows_path=args.selector_rows,
        source_csr_path=args.csr,
        device_memory_gb=args.device_memory_gb,
    )
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "results": output / "csr_auto_solve_results.jsonl",
        "summary": output / "csr_auto_solve_summary.json",
        "report": output / "csr_auto_solve_report.md",
        "manifest": output / "artifact_manifest.json",
    }
    write_jsonl(result_records, paths["results"])
    paths["summary"].write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_report(result_records, summary, paths["report"])
    write_manifest(
        build_artifact_manifest(
            artifact_kind="csr_auto_solve_smoke",
            command="scripts/tss_csr_auto_solve_smoke.py",
            tracked_files=CORE_CSR_AUTO_SOLVE_PROVENANCE_FILES,
            metadata={
                "git_commit": git_commit_or_unknown(),
                "status": summary["status"],
                "num_solves": summary["num_solves"],
                "num_success": summary["num_success"],
                "num_failed": summary["num_failed"],
                "selected_solver_set": summary["selected_solver_set"],
            },
        ),
        paths["manifest"],
    )
    for key, value in paths.items():
        print(f"{key}: {value}")
    print(f"status: {summary['status']}")
    if summary["status"] != "passed":
        raise SystemExit(2)


def _result_record(
    csr: CsrMatrix,
    result,
    rhs: tuple[float, ...],
    *,
    context: SolveContext,
) -> dict:
    trace = trace_to_record(result.trace)
    auto_metadata = dict(result.metadata["auto_solve_csr"])
    plan = auto_metadata["plan"]
    solution = tuple(float(value) for value in result.solution)
    cpu_residual = _relative_residual(csr, solution, rhs)
    solution_error = _relative_error_to_ones(solution)
    failure_reasons: list[str] = []
    if result.status != "success":
        failure_reasons.append(result.trace.failure_class or "solver_failed")
    if float(result.trace.final_residual_norm or math.inf) > context.tolerance_rel:
        failure_reasons.append("trace_residual_above_tolerance")
    if cpu_residual > 1.0e-4:
        failure_reasons.append("cpu_recomputed_residual_above_tolerance")
    if solution_error > 5.0e-3:
        failure_reasons.append("solution_error_above_tolerance")
    if plan["audit"]["is_oracle"] is not True:
        failure_reasons.append("selected_non_oracle_candidate")
    return {
        "matrix_id": csr.matrix_id,
        "candidate_id": auto_metadata["candidate_id"],
        "selection_reason": auto_metadata["reason"],
        "selected_is_oracle": plan["audit"]["is_oracle"],
        "solver": plan["solver"]["name"],
        "preconditioner": plan["preconditioner"]["name"],
        "precision": plan["solver"].get("precision", context.precision),
        "status": "success" if not failure_reasons else "failed",
        "failure_reasons": tuple(failure_reasons),
        "trace": trace,
        "cpu_recomputed_relative_residual": cpu_residual,
        "solution_relative_error": solution_error,
    }


def _build_summary(
    records: list[dict],
    *,
    selector_rows_path: str,
    source_csr_path: str,
    device_memory_gb: float,
) -> dict:
    num_success = sum(1 for row in records if row["status"] == "success")
    num_failed = len(records) - num_success
    selected_solver_set = sorted({row["solver"] for row in records})
    max_final_residual = max(
        (float(row["trace"]["final_residual_norm"]) for row in records),
        default=math.inf,
    )
    max_cpu_residual = max(
        (float(row["cpu_recomputed_relative_residual"]) for row in records),
        default=math.inf,
    )
    max_solution_error = max(
        (float(row["solution_relative_error"]) for row in records),
        default=math.inf,
    )
    status = (
        "passed"
        if len(records) == 2
        and num_failed == 0
        and selected_solver_set
        and all(row["selected_is_oracle"] is True for row in records)
        and all(row["trace"]["backend"] == "taichi_gpu" for row in records)
        and all(
            row["trace"]["metadata"]["auto_solve_csr"]["selector"] == "csr_artifact"
            for row in records
        )
        and max_final_residual <= 1.0e-5
        and max_cpu_residual <= 1.0e-4
        and max_solution_error <= 5.0e-3
        else "failed"
    )
    return {
        "status": status,
        "selector_rows_path": selector_rows_path,
        "source_csr_path": source_csr_path,
        "device_memory_gb": device_memory_gb,
        "num_solves": len(records),
        "num_success": num_success,
        "num_failed": num_failed,
        "selected_solver_set": selected_solver_set,
        "num_oracle_selected": sum(1 for row in records if row["selected_is_oracle"]),
        "max_final_relative_residual": max_final_residual,
        "max_cpu_recomputed_relative_residual": max_cpu_residual,
        "max_solution_relative_error": max_solution_error,
    }


def _write_report(records: list[dict], summary: dict, path: Path) -> Path:
    lines = [
        "# CSR Auto Solve Smoke",
        "",
        f"- status: `{summary['status']}`",
        f"- solves: `{summary['num_solves']}`",
        f"- successes: `{summary['num_success']}`",
        f"- selected_solver_set: `{', '.join(summary['selected_solver_set'])}`",
        f"- oracle_selected: `{summary['num_oracle_selected']}`",
        f"- max_final_relative_residual: `{summary['max_final_relative_residual']:.6g}`",
        f"- max_cpu_recomputed_relative_residual: "
        f"`{summary['max_cpu_recomputed_relative_residual']:.6g}`",
        f"- max_solution_relative_error: `{summary['max_solution_relative_error']:.6g}`",
        "",
        "| matrix | candidate | solver | preconditioner | status | rel_res | sol_err |",
        "|---|---|---|---|---|---:|---:|",
    ]
    for row in records:
        lines.append(
            "| "
            f"{row['matrix_id']} | "
            f"{row['candidate_id']} | "
            f"{row['solver']} | "
            f"{row['preconditioner']} | "
            f"{row['status']} | "
            f"{float(row['trace']['final_residual_norm']):.6g} | "
            f"{float(row['solution_relative_error']):.6g} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


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
    diff_norm = math.sqrt(sum((a - b) ** 2 for a, b in zip(actual, rhs)))
    rhs_norm = math.sqrt(max(sum(value * value for value in rhs), 1.0e-30))
    return diff_norm / rhs_norm


if __name__ == "__main__":
    main()
