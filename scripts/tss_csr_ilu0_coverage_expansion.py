"""Expand Taichi GPU CSR ILU0 coverage without changing runtime selection."""

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

from scripts.tss_taichi_csr_ilu0_smoke import _load_csr_records, _run_gpu_probe
from transsolvestack.datasets.csr import CsrMatrix
from transsolvestack.operators.taichi_diffusion2d import ensure_taichi_cuda
from transsolvestack.profiling.artifacts import read_jsonl, write_jsonl
from transsolvestack.profiling.provenance import (
    CORE_CSR_ILU0_COVERAGE_EXPANSION_PROVENANCE_FILES,
    build_artifact_manifest,
    git_commit_or_unknown,
    write_manifest,
)


SCHEMA_VERSION = "phase1_csr_ilu0_coverage_expansion_v1"
SELECTOR_SCHEMA_VERSION = "phase1_csr_selector_features_v8"
SOURCE_PROBE_SCHEMA_VERSION = "phase1_taichi_csr_ilu0_v1"
CONTEXT_ID = "phase1_taichi_csr_ilu0"
DEFAULT_CSR_RECORD_PATHS = (
    "runs/phase1_suitesparse_csr_import/csr_matrices.jsonl",
    "runs/phase1_csr_micro_campaign/csr_matrices.jsonl",
)
DEFAULT_MATRIX_IDS = (
    "suitesparse:Grund/b1_ss",
    "suitesparse:JGD_Trefethen/Trefethen_20b",
    "suitesparse:FIDAP/ex5",
    "suitesparse:Hamrle/Hamrle1",
    "suitesparse:HB/curtis54",
    "suitesparse:Sandia/oscil_dcop_01",
    "suitesparse:HB/young3c",
    "suitesparse:Bai/cdde1",
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
    parser.add_argument(
        "--base-selector-rows",
        default="runs/phase1_csr_transformer_ready/combined_csr_selector_rows.jsonl",
    )
    parser.add_argument("--device-memory-gb", type=float, default=0.25)
    parser.add_argument("--max-iter", type=int, default=384)
    parser.add_argument("--tolerance-rel", type=float, default=1.0e-5)
    parser.add_argument("--pivot-tolerance", type=float, default=1.0e-12)
    parser.add_argument("--out", default="runs/phase1_csr_ilu0_coverage_expansion")
    args = parser.parse_args()

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    csr_by_id = _load_csr_records(tuple(args.csr_records))
    missing = tuple(matrix_id for matrix_id in args.matrix_id if matrix_id not in csr_by_id)
    if missing:
        raise SystemExit(f"missing CSR records: {missing}")

    ensure_taichi_cuda(device_memory_gb=float(args.device_memory_gb))
    rows = [
        _probe_or_setup_failure(
            csr_by_id[matrix_id],
            max_iter=int(args.max_iter),
            tolerance_rel=float(args.tolerance_rel),
            pivot_tolerance=float(args.pivot_tolerance),
        )
        for matrix_id in args.matrix_id
    ]
    base_selector_rows = list(read_jsonl(args.base_selector_rows))
    selector_rows = _selector_rows(rows, base_selector_rows)
    summary = _summary(
        rows,
        selector_rows=selector_rows,
        matrix_ids=tuple(args.matrix_id),
        csr_record_paths=tuple(args.csr_records),
        base_selector_rows_path=str(args.base_selector_rows),
        device_memory_gb=float(args.device_memory_gb),
        max_iter=int(args.max_iter),
        tolerance_rel=float(args.tolerance_rel),
        pivot_tolerance=float(args.pivot_tolerance),
    )

    paths = {
        "results": output / "csr_ilu0_coverage_results.jsonl",
        "selector_rows": output / "csr_ilu0_coverage_selector_rows.jsonl",
        "summary": output / "csr_ilu0_coverage_summary.json",
        "schema": output / "csr_ilu0_coverage_schema.json",
        "report": output / "csr_ilu0_coverage_report.md",
        "manifest": output / "artifact_manifest.json",
    }
    write_jsonl(rows, paths["results"])
    write_jsonl(selector_rows, paths["selector_rows"])
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
            artifact_kind="csr_ilu0_coverage_expansion",
            command="scripts/tss_csr_ilu0_coverage_expansion.py",
            tracked_files=CORE_CSR_ILU0_COVERAGE_EXPANSION_PROVENANCE_FILES,
            metadata={
                "git_commit": git_commit_or_unknown(),
                "status": summary["status"],
                "schema_version": summary["schema_version"],
                "candidate_rows": summary["candidate_rows"],
                "numeric_success_rows": summary["numeric_success_rows"],
                "failed_numeric_gate_rows": summary["failed_numeric_gate_rows"],
                "setup_failed_rows": summary["setup_failed_rows"],
                "merge_ready_success_rows": summary["merge_ready_success_rows"],
                "gpu_solve_rows": summary["gpu_solve_rows"],
                "runtime_selector_changed": summary["runtime_selector_changed"],
                "merged_into_main_transformer_ready": summary[
                    "merged_into_main_transformer_ready"
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


def _probe_or_setup_failure(
    csr: CsrMatrix,
    *,
    max_iter: int,
    tolerance_rel: float,
    pivot_tolerance: float,
) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        row = dict(
            _run_gpu_probe(
                csr,
                max_iter=max_iter,
                tolerance_rel=tolerance_rel,
                pivot_tolerance=pivot_tolerance,
            )
        )
    except Exception as exc:
        return _setup_failure_row(
            csr,
            exc=exc,
            pivot_tolerance=pivot_tolerance,
            wall_time_ms=(time.perf_counter() - start) * 1000.0,
        )
    row["schema_version"] = SCHEMA_VERSION
    row["source_probe_schema_version"] = SOURCE_PROBE_SCHEMA_VERSION
    row["context_id"] = CONTEXT_ID
    row["gpu_attempted"] = True
    row["setup_error_type"] = None
    row["setup_error_message"] = None
    row["selector_merge_ready"] = row["numeric_status"] == "success"
    row["merged_into_main_transformer_ready"] = False
    row["production_runtime_selector_changed"] = False
    return row


def _setup_failure_row(
    csr: CsrMatrix,
    *,
    exc: Exception,
    pivot_tolerance: float,
    wall_time_ms: float,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "source_probe_schema_version": SOURCE_PROBE_SCHEMA_VERSION,
        "context_id": CONTEXT_ID,
        "matrix_id": csr.matrix_id,
        "candidate_id": "taichi_csr_bicgstab_ilu0_float64",
        "solver": "bicgstab",
        "preconditioner": "ilu0",
        "pivot_tolerance": pivot_tolerance,
        "precision": "float64",
        "backend": "taichi_gpu",
        "gpu_attempted": True,
        "gpu_executed": False,
        "solver_status": "setup_failed",
        "numeric_status": "setup_failed",
        "candidate_promoted": False,
        "failure_reasons": ("ilu0_setup_failed",),
        "setup_error_type": type(exc).__name__,
        "setup_error_message": str(exc),
        "n_rows": csr.n_rows,
        "n_cols": csr.n_cols,
        "csr_nnz": csr.nnz,
        "field": csr.field,
        "symmetry": csr.symmetry,
        "num_iterations": None,
        "final_relative_residual": None,
        "cpu_recomputed_relative_residual": None,
        "solution_relative_error": None,
        "ilu0_min_abs_pivot": None,
        "solve_time_ms": None,
        "wall_time_ms": wall_time_ms,
        "trace_metadata": {},
        "runtime_selector_changed": False,
        "production_runtime_selector_changed": False,
        "executes_gpu": True,
        "input": "rhs=A@ones",
        "selector_merge_ready": False,
        "merged_into_main_transformer_ready": False,
    }


def _selector_rows(
    rows: list[dict[str, Any]],
    base_selector_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    base_features_by_matrix: dict[str, dict[str, Any]] = {}
    for row in base_selector_rows:
        matrix_id = str(row["matrix_id"])
        base_features_by_matrix.setdefault(matrix_id, dict(row.get("features", {})))
    selector_rows = []
    for row in rows:
        success = row["numeric_status"] == "success"
        features = dict(base_features_by_matrix.get(str(row["matrix_id"]), {}))
        features.update(
            {
                "matrix_id": row["matrix_id"],
                "candidate_id": row["candidate_id"],
                "solver": row["solver"],
                "preconditioner": row["preconditioner"],
                "precision": row["precision"],
                "solver_parameters": {},
                "applicability_status": "applicable",
                "applicability_reason": None,
                "skip_reason": None,
                "screened_out_source": None,
                "success_rate": 1.0 if success else 0.0,
                "measurement_repeats": 1,
                "median_solve_time_ms": row["solve_time_ms"] if success else None,
                "solve_time_iqr_ms": 0.0,
                "failure_reason": None if success else _failure_reason(row),
                "n_rows": row["n_rows"],
                "n_cols": row["n_cols"],
                "csr_nnz": row["csr_nnz"],
                "field": row["field"],
                "declared_symmetry": row["symmetry"],
                "source_augmented_from": "phase1_csr_ilu0_coverage_expansion",
                "ilu0_numeric_status": row["numeric_status"],
                "ilu0_setup_error_type": row["setup_error_type"],
                "ilu0_min_abs_pivot": row["ilu0_min_abs_pivot"],
                "ilu0_pivot_tolerance": row["pivot_tolerance"],
                "ilu0_candidate_promoted": bool(row["candidate_promoted"]),
            }
        )
        selector_rows.append(
            {
                "schema_version": SELECTOR_SCHEMA_VERSION,
                "matrix_id": row["matrix_id"],
                "context_id": CONTEXT_ID,
                "candidate_id": row["candidate_id"],
                "solver": row["solver"],
                "preconditioner": row["preconditioner"],
                "precision": row["precision"],
                "solver_parameters": {},
                "label_is_oracle": False,
                "target_status": row["numeric_status"],
                "target_success_rate": 1.0 if success else 0.0,
                "target_measurement_repeats": 1,
                "target_solve_time_ms": row["solve_time_ms"] if success else None,
                "target_median_solve_time_ms": row["solve_time_ms"] if success else None,
                "target_solve_time_iqr_ms": 0.0,
                "target_wall_time_ms": row["wall_time_ms"] if success else None,
                "target_regret_vs_oracle_ms": None,
                "target_num_iterations": row["num_iterations"],
                "target_final_relative_residual": row["final_relative_residual"],
                "target_cpu_recomputed_relative_residual": row[
                    "cpu_recomputed_relative_residual"
                ],
                "target_solution_relative_error": row["solution_relative_error"],
                "target_failure_reason": None if success else _failure_reason(row),
                "target_applicability_status": "applicable",
                "target_applicability_reason": None,
                "features": features,
            }
        )
    return selector_rows


def _summary(
    rows: list[dict[str, Any]],
    *,
    selector_rows: list[dict[str, Any]],
    matrix_ids: tuple[str, ...],
    csr_record_paths: tuple[str, ...],
    base_selector_rows_path: str,
    device_memory_gb: float,
    max_iter: int,
    tolerance_rel: float,
    pivot_tolerance: float,
) -> dict[str, Any]:
    success_rows = [row for row in rows if row["numeric_status"] == "success"]
    failed_numeric_rows = [
        row for row in rows if row["numeric_status"] == "failed_numeric_gate"
    ]
    setup_failed_rows = [row for row in rows if row["numeric_status"] == "setup_failed"]
    gpu_solve_rows = [row for row in rows if row["gpu_executed"] is True]
    residuals = [float(row["final_relative_residual"]) for row in success_rows]
    cpu_residuals = [
        float(row["cpu_recomputed_relative_residual"]) for row in success_rows
    ]
    solution_errors = [float(row["solution_relative_error"]) for row in success_rows]
    success_matrix_ids = sorted(row["matrix_id"] for row in success_rows)
    status = (
        "passed"
        if len(rows) == len(matrix_ids)
        and len(selector_rows) == len(rows)
        and len(success_rows) >= 3
        and len(failed_numeric_rows) >= 1
        and len(setup_failed_rows) >= 1
        and len(gpu_solve_rows) == len(success_rows) + len(failed_numeric_rows)
        and all(row["candidate_promoted"] is True for row in success_rows)
        and all(row["candidate_promoted"] is False for row in failed_numeric_rows)
        and all(row["candidate_promoted"] is False for row in setup_failed_rows)
        and all(row["runtime_selector_changed"] is False for row in rows)
        and max(residuals, default=math.inf) <= tolerance_rel
        and max(cpu_residuals, default=math.inf) <= 1.0e-4
        and max(solution_errors, default=math.inf) <= 5.0e-3
        and {
            "suitesparse:JGD_Trefethen/Trefethen_20b",
            "suitesparse:FIDAP/ex5",
            "suitesparse:Bai/cdde1",
        }.issubset(set(success_matrix_ids))
        else "failed"
    )
    return {
        "status": status,
        "schema_version": SCHEMA_VERSION,
        "source_probe_schema_version": SOURCE_PROBE_SCHEMA_VERSION,
        "selector_schema_version": SELECTOR_SCHEMA_VERSION,
        "context_id": CONTEXT_ID,
        "csr_record_paths": csr_record_paths,
        "base_selector_rows_path": base_selector_rows_path,
        "device_memory_gb": device_memory_gb,
        "max_iter": max_iter,
        "tolerance_rel": tolerance_rel,
        "pivot_tolerance": pivot_tolerance,
        "solver": "bicgstab",
        "preconditioner": "ilu0",
        "candidate_rows": len(rows),
        "selector_rows": len(selector_rows),
        "numeric_success_rows": len(success_rows),
        "failed_numeric_gate_rows": len(failed_numeric_rows),
        "setup_failed_rows": len(setup_failed_rows),
        "merge_ready_success_rows": sum(
            1 for row in rows if row["selector_merge_ready"] is True
        ),
        "gpu_solve_rows": len(gpu_solve_rows),
        "gpu_attempted_rows": sum(1 for row in rows if row["gpu_attempted"] is True),
        "runtime_selector_changed": False,
        "production_runtime_selector_changed": False,
        "merged_into_main_transformer_ready": False,
        "by_numeric_status": _counts(row["numeric_status"] for row in rows),
        "by_matrix_numeric_status": {
            row["matrix_id"]: row["numeric_status"] for row in rows
        },
        "success_matrix_ids": success_matrix_ids,
        "setup_failed_matrix_ids": sorted(row["matrix_id"] for row in setup_failed_rows),
        "failed_numeric_gate_matrix_ids": sorted(
            row["matrix_id"] for row in failed_numeric_rows
        ),
        "max_success_final_relative_residual": max(residuals, default=0.0),
        "max_success_cpu_recomputed_relative_residual": max(cpu_residuals, default=0.0),
        "max_success_solution_relative_error": max(solution_errors, default=0.0),
        "next_step": (
            "merge ILU0 success rows into the main Transformer-ready bundle only "
            "after adding enough exact context successes and a dedicated quality gate"
        ),
    }


def _schema() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "task": "expand_ilu0_success_and_negative_coverage_for_selector_training",
        "source_probe_schema_version": SOURCE_PROBE_SCHEMA_VERSION,
        "selector_schema_version": SELECTOR_SCHEMA_VERSION,
        "integration_boundary": {
            "executes_gpu": True,
            "production_runtime_selector_changed": False,
            "merged_into_main_transformer_ready": False,
            "success_rows_are_merge_candidates": True,
            "non_success_rows_remain_negative_evidence": True,
        },
        "numeric_gate": [
            "success rows must have solver status success",
            "success rows must satisfy trace residual <= tolerance_rel",
            "success rows must satisfy CPU recomputed residual <= 1e-4",
            "success rows must satisfy rhs=A@ones solution error <= 5e-3",
            "setup failures and numeric failures must not be promoted",
        ],
    }


def _write_report(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    path: Path,
) -> None:
    lines = [
        "# CSR ILU0 Coverage Expansion",
        "",
        f"- status: `{summary['status']}`",
        f"- candidate_rows: `{summary['candidate_rows']}`",
        f"- numeric_success_rows: `{summary['numeric_success_rows']}`",
        f"- failed_numeric_gate_rows: `{summary['failed_numeric_gate_rows']}`",
        f"- setup_failed_rows: `{summary['setup_failed_rows']}`",
        f"- merge_ready_success_rows: `{summary['merge_ready_success_rows']}`",
        f"- gpu_solve_rows: `{summary['gpu_solve_rows']}`",
        f"- runtime_selector_changed: `{summary['runtime_selector_changed']}`",
        f"- merged_into_main_transformer_ready: `{summary['merged_into_main_transformer_ready']}`",
        f"- next_step: `{summary['next_step']}`",
        "",
        "| matrix | status | solver_status | gpu_solve | iters | rel_res | cpu_rel_res | sol_err | failure |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['matrix_id']} | "
            f"{row['numeric_status']} | "
            f"{row['solver_status']} | "
            f"{row['gpu_executed']} | "
            f"{row['num_iterations']} | "
            f"{_fmt(row['final_relative_residual'])} | "
            f"{_fmt(row['cpu_recomputed_relative_residual'])} | "
            f"{_fmt(row['solution_relative_error'])} | "
            f"{_failure_reason(row)} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _failure_reason(row: dict[str, Any]) -> str:
    reasons = tuple(str(reason) for reason in row.get("failure_reasons", ()))
    if reasons:
        return ",".join(reasons)
    if row.get("setup_error_type"):
        return str(row["setup_error_type"])
    return "none"


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
