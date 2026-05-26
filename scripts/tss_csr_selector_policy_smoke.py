"""Exercise CSR artifact selector plans on real CSR fixtures."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import transsolvestack as tss
from transsolvestack.core.types import SolveContext
from transsolvestack.datasets.csr import CsrMatrix, csr_matrix_from_record
from transsolvestack.profiling.artifacts import read_jsonl, write_jsonl
from transsolvestack.profiling.provenance import (
    CORE_CSR_SELECTOR_POLICY_PROVENANCE_FILES,
    build_artifact_manifest,
    git_commit_or_unknown,
    write_manifest,
)
from transsolvestack.profiling.trace import trace_to_record


SOLVE_CHECK_MATRIX_IDS = (
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
    parser.add_argument("--out", default="runs/phase1_csr_selector_policy")
    parser.add_argument("--device-memory-gb", type=float, default=0.25)
    args = parser.parse_args()

    csr_rows = {str(row["matrix_id"]): row for row in read_jsonl(args.csr)}
    selector_rows = read_jsonl(args.selector_rows)
    covered_matrix_ids = sorted(
        {
            str(row["matrix_id"])
            for row in selector_rows
            if row["target_status"] == "success"
        }
    )
    context = SolveContext(
        context_id="phase1_csr_selector",
        tolerance_abs=1.0e-7,
        tolerance_rel=1.0e-5,
        max_iter=512,
        precision="float64",
        required_backend="taichi_gpu",
    )
    selected_records = []
    solve_records = []
    for matrix_id in covered_matrix_ids:
        if matrix_id not in csr_rows:
            raise SystemExit(f"selector-covered matrix missing from CSR import: {matrix_id}")
        row = csr_rows[matrix_id]
        csr = csr_matrix_from_record(row)
        selection = tss.plan_csr(
            row,
            selector_path=args.selector_rows,
            context=context,
        )
        selected_records.append(_selection_record(selection, matrix_id))
        if matrix_id in SOLVE_CHECK_MATRIX_IDS:
            solve_records.append(
                _solve_check(
                    csr,
                    selection,
                    context=context,
                    device_memory_gb=args.device_memory_gb,
                )
            )

    summary = _build_summary(
        selected_records,
        solve_records,
        selector_rows_path=args.selector_rows,
        source_csr_path=args.csr,
        device_memory_gb=args.device_memory_gb,
    )
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "selected_plans": output / "csr_selected_policy_plans.jsonl",
        "solve_checks": output / "csr_selector_policy_solve_checks.jsonl",
        "summary": output / "csr_selector_policy_summary.json",
        "report": output / "csr_selector_policy_report.md",
        "manifest": output / "artifact_manifest.json",
    }
    write_jsonl(selected_records, paths["selected_plans"])
    write_jsonl(solve_records, paths["solve_checks"])
    paths["summary"].write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_report(selected_records, solve_records, summary, paths["report"])
    write_manifest(
        build_artifact_manifest(
            artifact_kind="csr_selector_policy_smoke",
            command="scripts/tss_csr_selector_policy_smoke.py",
            tracked_files=CORE_CSR_SELECTOR_POLICY_PROVENANCE_FILES,
            metadata={
                "git_commit": git_commit_or_unknown(),
                "status": summary["status"],
                "num_selected_plans": summary["num_selected_plans"],
                "num_solve_checks": summary["num_solve_checks"],
                "num_successful_solve_checks": summary[
                    "num_successful_solve_checks"
                ],
            },
        ),
        paths["manifest"],
    )
    for key, value in paths.items():
        print(f"{key}: {value}")
    print(f"status: {summary['status']}")
    if summary["status"] != "passed":
        raise SystemExit(2)


def _selection_record(selection, matrix_id: str) -> dict:
    plan = selection.plan
    return {
        "matrix_id": matrix_id,
        "candidate_id": selection.candidate_id,
        "reason": selection.reason,
        "fallback_candidate_ids": selection.fallback_candidate_ids,
        "plan": asdict(plan),
        "audit": dict(plan.audit),
        "profile": asdict(selection.profile) if selection.profile is not None else None,
    }


def _solve_check(
    csr: CsrMatrix,
    selection,
    *,
    context: SolveContext,
    device_memory_gb: float,
) -> dict:
    rhs = csr.matvec(tuple(1.0 for _ in range(csr.n_cols)))
    result = tss.solve_csr(
        csr,
        rhs,
        context=context,
        solver=selection.plan.solver["name"],
        preconditioner=selection.plan.preconditioner["name"],
        precision=selection.plan.solver.get("precision", context.precision),
        policy_plan=selection.plan,
        device_memory_gb=device_memory_gb,
    )
    solution = tuple(float(value) for value in result.solution)
    cpu_residual = _relative_residual(csr, solution, rhs)
    solution_error = _relative_error_to_ones(solution)
    status = "success"
    failure_reasons: list[str] = []
    if result.status != "success":
        status = "failed"
        failure_reasons.append(result.trace.failure_class or "solver_failed")
    if float(result.trace.final_residual_norm or math.inf) > context.tolerance_rel:
        status = "failed"
        failure_reasons.append("trace_residual_above_tolerance")
    if cpu_residual > 1.0e-4:
        status = "failed"
        failure_reasons.append("cpu_recomputed_residual_above_tolerance")
    if solution_error > 5.0e-3:
        status = "failed"
        failure_reasons.append("solution_error_above_tolerance")
    return {
        "matrix_id": csr.matrix_id,
        "candidate_id": selection.candidate_id,
        "solver": selection.plan.solver["name"],
        "preconditioner": selection.plan.preconditioner["name"],
        "precision": selection.plan.solver.get("precision", context.precision),
        "status": status,
        "failure_reasons": tuple(failure_reasons),
        "trace": trace_to_record(result.trace),
        "cpu_recomputed_relative_residual": cpu_residual,
        "solution_relative_error": solution_error,
    }


def _build_summary(
    selected_records: list[dict],
    solve_records: list[dict],
    *,
    selector_rows_path: str,
    source_csr_path: str,
    device_memory_gb: float,
) -> dict:
    selected_oracle = sum(1 for row in selected_records if row["audit"]["is_oracle"])
    solve_success = sum(1 for row in solve_records if row["status"] == "success")
    max_final = max(
        (float(row["trace"]["final_residual_norm"]) for row in solve_records),
        default=0.0,
    )
    max_cpu = max(
        (float(row["cpu_recomputed_relative_residual"]) for row in solve_records),
        default=0.0,
    )
    max_solution = max(
        (float(row["solution_relative_error"]) for row in solve_records),
        default=0.0,
    )
    status = (
        "passed"
        if selected_records
        and solve_records
        and selected_oracle == len(selected_records)
        and solve_success == len(solve_records)
        and max_final <= 1.0e-5
        and max_cpu <= 1.0e-4
        and max_solution <= 5.0e-3
        else "failed"
    )
    return {
        "status": status,
        "selector_rows_path": selector_rows_path,
        "source_csr_path": source_csr_path,
        "device_memory_gb": device_memory_gb,
        "num_selected_plans": len(selected_records),
        "num_oracle_selected_plans": selected_oracle,
        "num_solve_checks": len(solve_records),
        "num_successful_solve_checks": solve_success,
        "selected_solver_set": sorted({row["plan"]["solver"]["name"] for row in selected_records}),
        "max_final_relative_residual": max_final,
        "max_cpu_recomputed_relative_residual": max_cpu,
        "max_solution_relative_error": max_solution,
    }


def _write_report(
    selected_records: list[dict],
    solve_records: list[dict],
    summary: dict,
    path: Path,
) -> Path:
    lines = [
        "# CSR Selector Policy Smoke",
        "",
        f"- status: `{summary['status']}`",
        f"- selected_plans: `{summary['num_selected_plans']}`",
        f"- oracle_selected_plans: `{summary['num_oracle_selected_plans']}`",
        f"- solve_checks: `{summary['num_solve_checks']}`",
        f"- successful_solve_checks: `{summary['num_successful_solve_checks']}`",
        f"- max_final_relative_residual: `{summary['max_final_relative_residual']:.6g}`",
        f"- max_cpu_recomputed_relative_residual: "
        f"`{summary['max_cpu_recomputed_relative_residual']:.6g}`",
        f"- max_solution_relative_error: `{summary['max_solution_relative_error']:.6g}`",
        "",
        "| matrix | candidate | oracle | reason | fallbacks |",
        "|---|---|---:|---|---:|",
    ]
    for row in selected_records:
        lines.append(
            "| "
            f"{row['matrix_id']} | "
            f"{row['candidate_id']} | "
            f"{'yes' if row['audit']['is_oracle'] else 'no'} | "
            f"{row['reason']} | "
            f"{len(row['fallback_candidate_ids'])} |"
        )
    lines.extend(
        [
            "",
            "## Solve Checks",
            "",
            "| matrix | candidate | status | rel_res | cpu_rel_res | sol_rel_err |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for row in solve_records:
        lines.append(
            "| "
            f"{row['matrix_id']} | "
            f"{row['candidate_id']} | "
            f"{row['status']} | "
            f"{float(row['trace']['final_residual_norm']):.6g} | "
            f"{float(row['cpu_recomputed_relative_residual']):.6g} | "
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
