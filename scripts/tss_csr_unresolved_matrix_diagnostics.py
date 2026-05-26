"""Diagnose unresolved guarded CSR fallback matrices after CPU screens."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transsolvestack.datasets.csr import CsrMatrix, csr_matrix_from_record
from transsolvestack.datasets.csr_diagnostics import (
    build_csr_matrix_diagnostics,
    classify_unresolved_matrix,
)
from transsolvestack.profiling.artifacts import read_jsonl, write_jsonl
from transsolvestack.profiling.provenance import (
    CORE_CSR_UNRESOLVED_MATRIX_DIAGNOSTICS_PROVENANCE_FILES,
    build_artifact_manifest,
    git_commit_or_unknown,
    write_manifest,
)


SCHEMA_VERSION = "phase1_csr_unresolved_matrix_diagnostics_v1"
DIAGNOSTIC_ID = "phase1_csr_unresolved_matrix_diagnostics_m63"
DEFAULT_CSR_RECORD_PATHS = (
    "runs/phase1_suitesparse_csr_import/csr_matrices.jsonl",
    "runs/phase1_csr_micro_campaign/csr_matrices.jsonl",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cpu-screen-results",
        default="runs/phase1_csr_unresolved_fallback_cpu_screen/csr_unresolved_fallback_cpu_screen_results.jsonl",
    )
    parser.add_argument(
        "--cpu-screen-summary",
        default="runs/phase1_csr_unresolved_fallback_cpu_screen/csr_unresolved_fallback_cpu_screen_summary.json",
    )
    parser.add_argument(
        "--csr-records",
        action="append",
        default=list(DEFAULT_CSR_RECORD_PATHS),
        help="CSR matrix JSONL records. May be provided more than once.",
    )
    parser.add_argument("--out", default="runs/phase1_csr_unresolved_matrix_diagnostics")
    args = parser.parse_args()

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    screen_rows = tuple(read_jsonl(args.cpu_screen_results))
    screen_summary = json.loads(Path(args.cpu_screen_summary).read_text(encoding="utf-8"))
    matrix_ids = tuple(sorted({str(row["matrix_id"]) for row in screen_rows}))
    if not matrix_ids:
        raise SystemExit("expected at least one unresolved matrix")
    csr_by_id = _load_csr_records(tuple(args.csr_records))
    missing = sorted(set(matrix_ids) - set(csr_by_id))
    if missing:
        raise SystemExit(f"missing CSR records for diagnostics: {missing}")

    rows = tuple(
        _diagnostic_row(matrix_id, csr_by_id[matrix_id], screen_rows)
        for matrix_id in matrix_ids
    )
    summary = _summary(
        rows,
        screen_rows,
        screen_summary=screen_summary,
        cpu_screen_results_path=args.cpu_screen_results,
        cpu_screen_summary_path=args.cpu_screen_summary,
        csr_record_paths=tuple(args.csr_records),
    )
    paths = {
        "diagnostics": output / "csr_unresolved_matrix_diagnostics.jsonl",
        "summary": output / "csr_unresolved_matrix_diagnostics_summary.json",
        "schema": output / "csr_unresolved_matrix_diagnostics_schema.json",
        "report": output / "csr_unresolved_matrix_diagnostics_report.md",
        "manifest": output / "artifact_manifest.json",
    }
    write_jsonl(rows, paths["diagnostics"])
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
            artifact_kind="csr_unresolved_matrix_diagnostics",
            command="scripts/tss_csr_unresolved_matrix_diagnostics.py",
            tracked_files=CORE_CSR_UNRESOLVED_MATRIX_DIAGNOSTICS_PROVENANCE_FILES,
            metadata={
                "git_commit": git_commit_or_unknown(),
                "status": summary["status"],
                "schema_version": summary["schema_version"],
                "diagnosed_matrices": summary["diagnosed_matrices"],
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


def _diagnostic_row(
    matrix_id: str,
    csr: CsrMatrix,
    screen_rows: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    matrix_screen_rows = tuple(row for row in screen_rows if row["matrix_id"] == matrix_id)
    best_residual = _min_finite(row.get("final_relative_residual") for row in matrix_screen_rows)
    best_solution_error = _min_finite(row.get("solution_relative_error") for row in matrix_screen_rows)
    diagnostics = build_csr_matrix_diagnostics(csr)
    route = classify_unresolved_matrix(
        diagnostics,
        best_solution_error=best_solution_error,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "diagnostic_id": DIAGNOSTIC_ID,
        "matrix_id": matrix_id,
        "m62_attempted_candidates": len(matrix_screen_rows),
        "m62_best_relative_residual": best_residual,
        "m62_best_solution_relative_error": best_solution_error,
        "m62_failure_reasons": tuple(
            sorted({str(row["failure_reason"]) for row in matrix_screen_rows})
        ),
        "recommended_route": route,
        "recommended_next_candidate_family": _candidate_family(matrix_id, route),
        "executes_gpu": False,
        "runtime_selector_changed": False,
        "diagnostics": diagnostics,
    }


def _candidate_family(matrix_id: str, route: str) -> str:
    if matrix_id == "suitesparse:Goodwin/Goodwin_010":
        return "bicgstab_or_gmres_with_ilu0_or_row_column_scaling"
    if matrix_id == "suitesparse:Gset/G17":
        return "graph_laplacian_or_shifted_diagonal_formulation_before_solver_tuning"
    if matrix_id == "suitesparse:HB/bcsstk07":
        return "pcg_with_ic0_or_symmetric_equilibration"
    if matrix_id == "suitesparse:Zitney/extr1b":
        return "gmres_or_bicgstab_with_nonsymmetric_scaling_or_ilu0"
    return route


def _summary(
    rows: tuple[dict[str, Any], ...],
    screen_rows: tuple[dict[str, Any], ...],
    *,
    screen_summary: dict[str, Any],
    cpu_screen_results_path: str,
    cpu_screen_summary_path: str,
    csr_record_paths: tuple[str, ...],
) -> dict[str, Any]:
    routes = _counts(row["recommended_route"] for row in rows)
    status = (
        "passed"
        if screen_summary["status"] == "passed"
        and screen_summary["gpu_probe_ready_candidates"] == 0
        and len(screen_rows) == int(screen_summary["attempted_cpu_screens"])
        and len(rows) == len({row["matrix_id"] for row in screen_rows})
        and all(row["executes_gpu"] is False for row in rows)
        and all(row["runtime_selector_changed"] is False for row in rows)
        else "failed"
    )
    return {
        "status": status,
        "schema_version": SCHEMA_VERSION,
        "diagnostic_id": DIAGNOSTIC_ID,
        "cpu_screen_results_path": cpu_screen_results_path,
        "cpu_screen_summary_path": cpu_screen_summary_path,
        "csr_record_paths": csr_record_paths,
        "source_cpu_screen_status": screen_summary["status"],
        "source_gpu_probe_ready_candidates": screen_summary["gpu_probe_ready_candidates"],
        "diagnosed_matrices": len(rows),
        "source_cpu_screen_rows": len(screen_rows),
        "executes_gpu": False,
        "runtime_selector_changed": False,
        "by_recommended_route": routes,
        "matrix_routes": {
            row["matrix_id"]: row["recommended_route"]
            for row in sorted(rows, key=lambda item: item["matrix_id"])
        },
        "next_step": "implement_and_test_preconditioner_or_formulation_candidates_before_gpu_probe",
    }


def _schema() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "task": "diagnose_unresolved_guarded_fallback_matrices_after_cpu_screen",
        "integration_boundary": {
            "diagnostic_only": True,
            "executes_gpu": False,
            "runtime_selector_changed": False,
        },
        "diagnostic_rules": [
            "zero diagonal plus negative symmetric-spectrum Ritz values points to formulation diagnostics",
            "nonsymmetric unresolved matrices are routed to ILU/scaling families",
            "symmetric positive dense Cholesky probes with solution-error failures are routed to IC0/equilibration families",
        ],
    }


def _write_report(
    rows: tuple[dict[str, Any], ...],
    summary: dict[str, Any],
    path: Path,
) -> None:
    lines = [
        "# CSR Unresolved Matrix Diagnostics",
        "",
        f"- status: `{summary['status']}`",
        f"- diagnosed_matrices: `{summary['diagnosed_matrices']}`",
        f"- source_gpu_probe_ready_candidates: `{summary['source_gpu_probe_ready_candidates']}`",
        f"- executes_gpu: `{summary['executes_gpu']}`",
        f"- runtime_selector_changed: `{summary['runtime_selector_changed']}`",
        f"- next_step: `{summary['next_step']}`",
        "",
        "| matrix | route | zero_diag | asym | cholesky | ritz_min | ritz_max | best_res | best_sol_err |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        diagnostics = row["diagnostics"]
        lines.append(
            "| "
            f"{row['matrix_id']} | "
            f"{row['recommended_route']} | "
            f"{diagnostics['diagonal']['zero_count']} | "
            f"{_fmt(diagnostics['symmetry_check']['relative_frobenius_asymmetry'])} | "
            f"{diagnostics['dense_cholesky_probe']['status']} | "
            f"{_fmt(diagnostics['symmetric_spectrum_probe'].get('ritz_min'))} | "
            f"{_fmt(diagnostics['symmetric_spectrum_probe'].get('ritz_max'))} | "
            f"{_fmt(row['m62_best_relative_residual'])} | "
            f"{_fmt(row['m62_best_solution_relative_error'])} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _counts(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


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
