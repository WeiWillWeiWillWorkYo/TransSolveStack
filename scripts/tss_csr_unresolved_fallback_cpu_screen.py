"""CPU-screen unresolved guarded CSR fallback candidates before any GPU probe."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transsolvestack.datasets.csr import CsrMatrix, csr_matrix_from_record
from transsolvestack.profiling.artifacts import read_jsonl, write_jsonl
from transsolvestack.profiling.csr_micro_campaign import _screen_candidate
from transsolvestack.profiling.provenance import (
    CORE_CSR_UNRESOLVED_FALLBACK_CPU_SCREEN_PROVENANCE_FILES,
    build_artifact_manifest,
    git_commit_or_unknown,
    write_manifest,
)


SCHEMA_VERSION = "phase1_csr_unresolved_fallback_cpu_screen_v1"
SCREEN_ID = "phase1_csr_unresolved_fallback_cpu_screen_m62"
DEFAULT_CSR_RECORD_PATHS = (
    "runs/phase1_suitesparse_csr_import/csr_matrices.jsonl",
    "runs/phase1_csr_micro_campaign/csr_matrices.jsonl",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate-plan",
        default="runs/phase1_csr_unresolved_fallback_coverage_plan/csr_unresolved_fallback_candidate_plan.jsonl",
    )
    parser.add_argument(
        "--plan-summary",
        default="runs/phase1_csr_unresolved_fallback_coverage_plan/csr_unresolved_fallback_coverage_plan_summary.json",
    )
    parser.add_argument(
        "--csr-records",
        action="append",
        default=list(DEFAULT_CSR_RECORD_PATHS),
        help="CSR matrix JSONL records. May be provided more than once.",
    )
    parser.add_argument("--tolerance-rel", type=float, default=1.0e-5)
    parser.add_argument(
        "--out",
        default="runs/phase1_csr_unresolved_fallback_cpu_screen",
    )
    args = parser.parse_args()

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    plan_rows = tuple(read_jsonl(args.candidate_plan))
    plan_summary = json.loads(Path(args.plan_summary).read_text(encoding="utf-8"))
    screen_rows = tuple(
        row for row in plan_rows if row["execution_stage"] == "cpu_screen_ready"
    )
    expected_screen_rows = int(plan_summary["cpu_screen_ready_candidates"])
    if len(screen_rows) != expected_screen_rows:
        raise SystemExit(
            f"expected {expected_screen_rows} CPU-screen-ready rows, "
            f"got {len(screen_rows)}"
        )

    csr_by_id = _load_csr_records(tuple(args.csr_records))
    missing = sorted({row["matrix_id"] for row in screen_rows} - set(csr_by_id))
    if missing:
        raise SystemExit(f"missing CSR records for CPU screen: {missing}")

    rows = tuple(
        _screen_row(
            plan_row,
            csr_by_id[str(plan_row["matrix_id"])],
            tolerance_rel=float(args.tolerance_rel),
        )
        for plan_row in screen_rows
    )
    summary = _summary(
        rows,
        plan_rows,
        plan_summary=plan_summary,
        candidate_plan_path=args.candidate_plan,
        plan_summary_path=args.plan_summary,
        csr_record_paths=tuple(args.csr_records),
        tolerance_rel=float(args.tolerance_rel),
    )
    paths = {
        "results": output / "csr_unresolved_fallback_cpu_screen_results.jsonl",
        "summary": output / "csr_unresolved_fallback_cpu_screen_summary.json",
        "schema": output / "csr_unresolved_fallback_cpu_screen_schema.json",
        "report": output / "csr_unresolved_fallback_cpu_screen_report.md",
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
            artifact_kind="csr_unresolved_fallback_cpu_screen",
            command="scripts/tss_csr_unresolved_fallback_cpu_screen.py",
            tracked_files=CORE_CSR_UNRESOLVED_FALLBACK_CPU_SCREEN_PROVENANCE_FILES,
            metadata={
                "git_commit": git_commit_or_unknown(),
                "status": summary["status"],
                "schema_version": summary["schema_version"],
                "attempted_cpu_screens": summary["attempted_cpu_screens"],
                "screen_success_rows": summary["screen_success_rows"],
                "cpu_screened_out_rows": summary["cpu_screened_out_rows"],
                "gpu_probe_ready_candidates": summary["gpu_probe_ready_candidates"],
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


def _screen_row(
    plan_row: dict[str, Any],
    csr: CsrMatrix,
    *,
    tolerance_rel: float,
) -> dict[str, Any]:
    candidate = _candidate_from_plan(plan_row, tolerance_rel=tolerance_rel)
    start = time.perf_counter()
    screen = _screen_candidate(csr, candidate)
    wall_ms = (time.perf_counter() - start) * 1000.0
    success = bool(screen["success"])
    status = "success" if success else "screened_out"
    failure_reason = None if success else _failure_reason(screen, tolerance_rel)
    return {
        "schema_version": SCHEMA_VERSION,
        "screen_id": SCREEN_ID,
        "source_plan_id": str(plan_row["plan_id"]),
        "source_row_id": str(plan_row["row_id"]),
        "matrix_id": str(plan_row["matrix_id"]),
        "context_id": str(plan_row["context_id"]),
        "candidate_id": str(plan_row["candidate_id"]),
        "solver": str(plan_row["solver"]),
        "preconditioner": str(plan_row["preconditioner"]),
        "precision": "float64",
        "status": status,
        "failure_reason": failure_reason,
        "gpu_probe_ready": success,
        "executes_gpu": False,
        "runtime_selector_changed": False,
        "n_rows": csr.n_rows,
        "n_cols": csr.n_cols,
        "csr_nnz": csr.nnz,
        "field": csr.field,
        "symmetry": csr.symmetry,
        "tolerance_rel": tolerance_rel,
        "target_screen_max_iter": int(candidate["max_iter"]),
        "solver_parameters": candidate["solver_parameters"],
        "num_iterations": screen.get("iterations"),
        "final_relative_residual": screen.get("relative_residual"),
        "solution_relative_error": screen.get("solution_relative_error"),
        "cpu_screen_wall_time_ms": wall_ms,
        "cpu_screen": screen,
        "source_blocker_class": str(plan_row["blocker_class"]),
        "source_blocker_addressed": str(plan_row["blocker_addressed"]),
        "source_expected_value": str(plan_row["expected_value"]),
    }


def _candidate_from_plan(
    plan_row: dict[str, Any],
    *,
    tolerance_rel: float,
) -> dict[str, Any]:
    params = dict(plan_row.get("solver_parameters", {}))
    max_iter = int(params.pop("max_iter", 512))
    return {
        "matrix_id": str(plan_row["matrix_id"]),
        "candidate_id": str(plan_row["candidate_id"]),
        "solver": str(plan_row["solver"]),
        "preconditioner": str(plan_row["preconditioner"]),
        "precision": "float64",
        "solver_parameters": params,
        "max_iter": max_iter,
        "tolerance_rel": tolerance_rel,
        "measurement_repeats": 1,
    }


def _failure_reason(screen: dict[str, Any], tolerance_rel: float) -> str:
    breakdown = screen.get("breakdown")
    if breakdown:
        return f"cpu_screen_breakdown:{breakdown}"
    residual = float(screen.get("relative_residual", math.inf))
    solution_error = float(screen.get("solution_relative_error", math.inf))
    if not math.isfinite(residual):
        return "cpu_screen_nonfinite_residual"
    if residual > tolerance_rel:
        return "cpu_screen_residual_above_tolerance"
    if not math.isfinite(solution_error):
        return "cpu_screen_nonfinite_solution_error"
    if solution_error > 5.0e-3:
        return "cpu_screen_solution_error_above_tolerance"
    return "cpu_screen_failed_unknown"


def _summary(
    rows: tuple[dict[str, Any], ...],
    plan_rows: tuple[dict[str, Any], ...],
    *,
    plan_summary: dict[str, Any],
    candidate_plan_path: str,
    plan_summary_path: str,
    csr_record_paths: tuple[str, ...],
    tolerance_rel: float,
) -> dict[str, Any]:
    success_rows = tuple(row for row in rows if row["status"] == "success")
    screened_rows = tuple(row for row in rows if row["status"] == "screened_out")
    accepted_statuses = {"success", "screened_out"}
    status = (
        "passed"
        if plan_summary["status"] == "passed"
        and len(plan_rows) == int(plan_summary["candidate_plan_rows"])
        and len(rows) == int(plan_summary["cpu_screen_ready_candidates"])
        and all(row["status"] in accepted_statuses for row in rows)
        and all(row["executes_gpu"] is False for row in rows)
        and all(row["runtime_selector_changed"] is False for row in rows)
        else "failed"
    )
    return {
        "status": status,
        "schema_version": SCHEMA_VERSION,
        "screen_id": SCREEN_ID,
        "candidate_plan_path": candidate_plan_path,
        "plan_summary_path": plan_summary_path,
        "csr_record_paths": csr_record_paths,
        "source_plan_status": plan_summary["status"],
        "source_candidate_plan_rows": len(plan_rows),
        "source_cpu_screen_ready_candidates": plan_summary[
            "cpu_screen_ready_candidates"
        ],
        "runtime_selector_changed": False,
        "executes_gpu": False,
        "tolerance_rel": tolerance_rel,
        "attempted_cpu_screens": len(rows),
        "screen_success_rows": len(success_rows),
        "cpu_screened_out_rows": len(screened_rows),
        "gpu_probe_ready_candidates": len(success_rows),
        "unresolved_after_cpu_screen": len(screened_rows),
        "by_status": _counts(row["status"] for row in rows),
        "by_matrix_status": _counts(
            f"{row['matrix_id']}:{row['status']}" for row in rows
        ),
        "by_solver_status": _counts(
            f"{row['solver']}:{row['status']}" for row in rows
        ),
        "success_candidate_keys": tuple(
            f"{row['matrix_id']}:{row['candidate_id']}" for row in success_rows
        ),
        "screened_out_candidate_keys": tuple(
            f"{row['matrix_id']}:{row['candidate_id']}" for row in screened_rows
        ),
        "max_success_relative_residual": _max_finite(
            row["final_relative_residual"] for row in success_rows
        ),
        "max_success_solution_relative_error": _max_finite(
            row["solution_relative_error"] for row in success_rows
        ),
        "min_screened_out_relative_residual": _min_finite(
            row["final_relative_residual"] for row in screened_rows
        ),
        "min_screened_out_solution_relative_error": _min_finite(
            row["solution_relative_error"] for row in screened_rows
        ),
        "next_step": (
            "run_gpu_probe_for_cpu_successes"
            if success_rows
            else "defer_unresolved_to_future_preconditioners_or_formulation_diagnostics"
        ),
    }


def _schema() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "task": "cpu_screen_unresolved_guarded_fallback_candidates_before_gpu_probe",
        "integration_boundary": {
            "cpu_only": True,
            "executes_gpu": False,
            "runtime_selector_changed": False,
        },
        "screening_policy": {
            "input": "rhs=A@ones",
            "success_requires_relative_residual_le_tolerance": True,
            "success_requires_solution_error_le_5e_3": True,
            "gpu_probe_allowed_only_for_status_success": True,
        },
    }


def _write_report(
    rows: tuple[dict[str, Any], ...],
    summary: dict[str, Any],
    path: Path,
) -> None:
    lines = [
        "# CSR Unresolved Fallback CPU Screen",
        "",
        f"- status: `{summary['status']}`",
        f"- attempted_cpu_screens: `{summary['attempted_cpu_screens']}`",
        f"- screen_success_rows: `{summary['screen_success_rows']}`",
        f"- cpu_screened_out_rows: `{summary['cpu_screened_out_rows']}`",
        f"- gpu_probe_ready_candidates: `{summary['gpu_probe_ready_candidates']}`",
        f"- executes_gpu: `{summary['executes_gpu']}`",
        f"- runtime_selector_changed: `{summary['runtime_selector_changed']}`",
        f"- next_step: `{summary['next_step']}`",
        "",
        "| matrix | candidate | status | solver | preconditioner | iters | rel_res | sol_err | reason |",
        "|---|---|---|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['matrix_id']} | "
            f"{row['candidate_id']} | "
            f"{row['status']} | "
            f"{row['solver']} | "
            f"{row['preconditioner']} | "
            f"{row.get('num_iterations') or ''} | "
            f"{_fmt(row.get('final_relative_residual'))} | "
            f"{_fmt(row.get('solution_relative_error'))} | "
            f"{row.get('failure_reason') or ''} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _counts(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _max_finite(values: Iterable[Any]) -> float | None:
    finite = [float(value) for value in values if _finite(value)]
    return None if not finite else max(finite)


def _min_finite(values: Iterable[Any]) -> float | None:
    finite = [float(value) for value in values if _finite(value)]
    return None if not finite else min(finite)


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _fmt(value: Any) -> str:
    if not _finite(value):
        return ""
    return f"{float(value):.6g}"


if __name__ == "__main__":
    main()
