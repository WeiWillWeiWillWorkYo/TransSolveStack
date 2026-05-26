"""Exercise public CSR diagnostics and solve API on real CSR fixtures."""

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
from transsolvestack.profiling.artifacts import read_jsonl
from transsolvestack.profiling.provenance import (
    CORE_CSR_PUBLIC_API_PROVENANCE_FILES,
    build_artifact_manifest,
    git_commit_or_unknown,
    write_manifest,
)
from transsolvestack.profiling.trace import trace_to_record


CG_MATRIX_IDS = (
    "suitesparse:JGD_Trefethen/Trefethen_20b",
    "suitesparse:FIDAP/ex5",
)
GMRES_MATRIX_IDS = ("suitesparse:HB/curtis54",)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csr",
        default="runs/phase1_suitesparse_csr_import/csr_matrices.jsonl",
    )
    parser.add_argument("--out", default="runs/phase1_csr_public_api")
    parser.add_argument("--device-memory-gb", type=float, default=0.25)
    args = parser.parse_args()

    rows = {str(row["matrix_id"]): row for row in read_jsonl(args.csr)}
    selected_matrix_ids = (*CG_MATRIX_IDS, *GMRES_MATRIX_IDS)
    missing = [matrix_id for matrix_id in selected_matrix_ids if matrix_id not in rows]
    if missing:
        raise SystemExit(f"missing selected CSR rows: {missing}")

    context = SolveContext(
        context_id="csr_public_api_smoke",
        tolerance_abs=1.0e-7,
        tolerance_rel=1.0e-5,
        max_iter=512,
        precision="float64",
        required_backend="taichi_gpu",
    )
    diagnostics = []
    results = []
    for matrix_id in CG_MATRIX_IDS:
        row = rows[matrix_id]
        csr = csr_matrix_from_record(row)
        diagnostics.append(asdict(tss.diagnose_csr_matrix(row)))
        rhs = csr.matvec(tuple(1.0 for _ in range(csr.n_cols)))
        for solver, preconditioner in (("cg", "none"), ("pcg", "jacobi")):
            result = tss.solve_csr(
                row,
                rhs,
                context=context,
                solver=solver,
                preconditioner=preconditioner,
                device_memory_gb=args.device_memory_gb,
            )
            solution = tuple(float(value) for value in result.solution)
            results.append(
                {
                    "matrix_id": matrix_id,
                    "solver": solver,
                    "preconditioner": preconditioner,
                    "status": result.status,
                    "trace": trace_to_record(result.trace),
                    "solution_relative_error": _relative_error_to_ones(solution),
                    "cpu_recomputed_relative_residual": _relative_residual(csr, solution, rhs),
                }
            )
    for matrix_id in GMRES_MATRIX_IDS:
        row = rows[matrix_id]
        csr = csr_matrix_from_record(row)
        diagnostics.append(asdict(tss.diagnose_csr_matrix(row)))
        rhs = csr.matvec(tuple(1.0 for _ in range(csr.n_cols)))
        result = tss.solve_csr(
            row,
            rhs,
            context=context,
            solver="gmres",
            preconditioner="jacobi",
            solver_parameters={"restart": 16},
            device_memory_gb=args.device_memory_gb,
        )
        solution = tuple(float(value) for value in result.solution)
        results.append(
            {
                "matrix_id": matrix_id,
                "solver": "gmres",
                "preconditioner": "jacobi",
                "status": result.status,
                "trace": trace_to_record(result.trace),
                "solution_relative_error": _relative_error_to_ones(solution),
                "cpu_recomputed_relative_residual": _relative_residual(csr, solution, rhs),
            }
        )
    summary = _build_summary(
        results,
        diagnostics,
        source_csr_path=args.csr,
        device_memory_gb=args.device_memory_gb,
    )
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "result": output / "csr_public_api_smoke.json",
        "manifest": output / "artifact_manifest.json",
    }
    payload = {
        "summary": summary,
        "diagnostics": diagnostics,
        "results": results,
    }
    paths["result"].write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_manifest(
        build_artifact_manifest(
            artifact_kind="csr_public_api_smoke",
            command="scripts/tss_csr_public_api_smoke.py",
            tracked_files=CORE_CSR_PUBLIC_API_PROVENANCE_FILES,
            metadata={
                "git_commit": git_commit_or_unknown(),
                "status": summary["status"],
                "num_solves": summary["num_solves"],
                "num_success": summary["num_success"],
                "num_failed": summary["num_failed"],
                "precision": summary["precision"],
            },
        ),
        paths["manifest"],
    )
    for key, value in paths.items():
        print(f"{key}: {value}")
    print(f"status: {summary['status']}")
    if summary["status"] != "passed":
        raise SystemExit(2)


def _build_summary(
    results: list[dict],
    diagnostics: list[dict],
    *,
    source_csr_path: str,
    device_memory_gb: float,
) -> dict:
    num_success = sum(1 for row in results if row["status"] == "success")
    num_failed = len(results) - num_success
    max_final_residual = max(
        (float(row["trace"]["final_residual_norm"]) for row in results),
        default=math.inf,
    )
    max_cpu_residual = max(
        (float(row["cpu_recomputed_relative_residual"]) for row in results),
        default=math.inf,
    )
    max_solution_error = max(
        (float(row["solution_relative_error"]) for row in results),
        default=math.inf,
    )
    status = (
        "passed"
        if results
        and num_failed == 0
        and all(
            row["actual_symmetric"] and row["cg_candidate"]
            for row in diagnostics
            if row["matrix_id"] in CG_MATRIX_IDS
        )
        and all(
            not row["actual_symmetric"]
            for row in diagnostics
            if row["matrix_id"] in GMRES_MATRIX_IDS
        )
        and max_final_residual <= 1.0e-5
        and max_cpu_residual <= 1.0e-4
        and max_solution_error <= 5.0e-3
        else "failed"
    )
    return {
        "status": status,
        "source_csr_path": source_csr_path,
        "device_memory_gb": device_memory_gb,
        "precision": "float64",
        "num_matrices": len(diagnostics),
        "num_solves": len(results),
        "num_success": num_success,
        "num_failed": num_failed,
        "max_final_relative_residual": max_final_residual,
        "max_cpu_recomputed_relative_residual": max_cpu_residual,
        "max_solution_relative_error": max_solution_error,
    }


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
