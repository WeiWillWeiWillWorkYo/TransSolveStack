"""Probe exact profiled-success fallbacks for non-success learned CSR choices."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transsolvestack.datasets.csr import CsrMatrix, csr_matrix_from_record
from transsolvestack.operators.taichi_diffusion2d import ensure_taichi_cuda
from transsolvestack.policies.csr_selector_data import build_csr_selector_export
from transsolvestack.profiling.artifacts import read_jsonl, write_jsonl
from transsolvestack.profiling.csr_micro_campaign import (
    _run_gpu_once,
    _screen_candidate,
    _screened_out_row,
)
from transsolvestack.profiling.provenance import (
    CORE_CSR_NON_SUCCESS_FALLBACK_PROBE_PROVENANCE_FILES,
    build_artifact_manifest,
    git_commit_or_unknown,
    write_manifest,
)
from transsolvestack.runtime.engine import TaichiExecutionEngine


SCHEMA_VERSION = "phase1_csr_non_success_fallback_probe_v1"
TARGET_SCENARIO_KIND = "fixture_non_success_candidate_block"
FALLBACK_CONTEXTS = ("phase1_csr_selector", "phase1_csr_micro_campaign")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--coverage-plan",
        default="runs/phase1_csr_guarded_promotion_coverage_plan/csr_guarded_promotion_coverage_scenarios.jsonl",
    )
    parser.add_argument("--device-memory-gb", type=float, default=0.25)
    parser.add_argument("--max-iter", type=int, default=512)
    parser.add_argument("--tolerance-rel", type=float, default=1.0e-5)
    parser.add_argument("--out", default="runs/phase1_csr_non_success_fallback_probe")
    args = parser.parse_args()

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    scenario_rows = [
        row
        for row in read_jsonl(args.coverage_plan)
        if row["scenario_kind"] == TARGET_SCENARIO_KIND
    ]
    if not scenario_rows:
        raise SystemExit("coverage plan has no non-success candidate scenarios")
    csr_by_matrix, csr_records = _load_required_csr(scenario_rows)
    candidate_rows = [
        candidate
        for scenario in scenario_rows
        for candidate in _candidate_rows_for_scenario(
            scenario,
            max_iter=args.max_iter,
            tolerance_rel=args.tolerance_rel,
        )
    ]
    result_rows, selector_rows = _run_probe(
        scenario_rows,
        candidate_rows,
        csr_by_matrix,
        csr_records,
        device_memory_gb=args.device_memory_gb,
    )
    summary = _summary(
        scenario_rows,
        candidate_rows,
        result_rows,
        selector_rows,
        coverage_plan_path=args.coverage_plan,
        device_memory_gb=args.device_memory_gb,
        max_iter=args.max_iter,
        tolerance_rel=args.tolerance_rel,
    )
    paths = {
        "results": output / "csr_non_success_fallback_probe_results.jsonl",
        "selector_rows": output / "csr_non_success_fallback_selector_rows.jsonl",
        "summary": output / "csr_non_success_fallback_probe_summary.json",
        "schema": output / "csr_non_success_fallback_probe_schema.json",
        "report": output / "csr_non_success_fallback_probe_report.md",
        "manifest": output / "artifact_manifest.json",
    }
    write_jsonl(result_rows, paths["results"])
    write_jsonl(selector_rows, paths["selector_rows"])
    paths["summary"].write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths["schema"].write_text(
        json.dumps(_schema(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_report(result_rows, summary, paths["report"])
    write_manifest(
        build_artifact_manifest(
            artifact_kind="csr_non_success_fallback_probe",
            command="scripts/tss_csr_non_success_fallback_probe.py",
            tracked_files=CORE_CSR_NON_SUCCESS_FALLBACK_PROBE_PROVENANCE_FILES,
            metadata={
                "git_commit": git_commit_or_unknown(),
                "status": summary["status"],
                "schema_version": summary["schema_version"],
                "target_scenarios": summary["target_scenarios"],
                "candidate_attempts": summary["candidate_attempts"],
                "gpu_success_rows": summary["gpu_success_rows"],
                "cpu_screened_out_rows": summary["cpu_screened_out_rows"],
                "resolved_matrix_count": summary["resolved_matrix_count"],
                "unresolved_matrix_count": summary["unresolved_matrix_count"],
                "runtime_selector_changed": summary["runtime_selector_changed"],
                "executes_gpu": summary["executes_gpu"],
            },
        ),
        paths["manifest"],
    )
    for key, value in paths.items():
        print(f"{key}: {value}")
    print(f"status: {summary['status']}")
    if summary["status"] != "passed":
        raise SystemExit(2)


def _run_probe(
    scenario_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    csr_by_matrix: dict[str, CsrMatrix],
    csr_records: dict[str, dict[str, Any]],
    *,
    device_memory_gb: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    screens: list[tuple[dict[str, Any], CsrMatrix, dict[str, Any]]] = []
    for candidate in candidate_rows:
        csr = csr_by_matrix[str(candidate["matrix_id"])]
        screen = _screen_candidate(csr, candidate)
        screens.append((candidate, csr, screen))
    if any(screen["success"] for _, _, screen in screens):
        ensure_taichi_cuda(device_memory_gb=device_memory_gb)
    engine = TaichiExecutionEngine()
    results: list[dict[str, Any]] = []
    for candidate, csr, screen in screens:
        if screen["success"]:
            row = _run_gpu_once(
                csr,
                candidate,
                screen=screen,
                engine=engine,
                context_id=str(candidate["context_id"]),
                repeat_index=0,
            )
        else:
            row = _screened_out_row(csr, candidate, screen)
        results.append(_probe_row(row, candidate, screen))

    selector_rows: list[dict[str, Any]] = []
    for context_id in FALLBACK_CONTEXTS:
        context_results = [row for row in results if row["context_id"] == context_id]
        if not context_results:
            continue
        context_matrix_ids = {row["matrix_id"] for row in context_results}
        context_records = [
            csr_records[matrix_id]
            for matrix_id in sorted(context_matrix_ids)
            if matrix_id in csr_records
        ]
        export = build_csr_selector_export(
            context_records,
            context_results,
            context_id=context_id,
        )
        selector_rows.extend(asdict(row) for row in export.selector_rows)
    return results, selector_rows


def _probe_row(
    row: dict[str, Any],
    candidate: dict[str, Any],
    screen: dict[str, Any],
) -> dict[str, Any]:
    payload = dict(row)
    payload["schema_version"] = SCHEMA_VERSION
    payload["source_scenario_id"] = candidate["source_scenario_id"]
    payload["source_selected_candidate_id"] = candidate["source_selected_candidate_id"]
    payload["context_id"] = candidate["context_id"]
    payload["candidate_role"] = candidate["candidate_role"]
    payload["planned_as_fallback_probe"] = True
    payload["gpu_executed"] = payload["status"] in {"success", "failed"} and payload["backend"] == "taichi_gpu"
    payload["cpu_screen_success"] = bool(screen["success"])
    payload["cpu_screen_relative_residual"] = screen.get("relative_residual")
    payload["cpu_screen_solution_relative_error"] = screen.get("solution_relative_error")
    payload["cpu_screen_breakdown"] = screen.get("breakdown")
    return payload


def _candidate_rows_for_scenario(
    scenario: dict[str, Any],
    *,
    max_iter: int,
    tolerance_rel: float,
) -> list[dict[str, Any]]:
    matrix_id = str(scenario["matrix_id"])
    context_id = str(scenario["context_id"])
    selected_candidate = _candidate_from_id(
        str(scenario["selected_candidate_id"]),
        matrix_id=matrix_id,
        context_id=context_id,
        role="learned_non_success_candidate",
        max_iter=max_iter,
        tolerance_rel=tolerance_rel,
        source_scenario_id=str(scenario["scenario_id"]),
        source_selected_candidate_id=str(scenario["selected_candidate_id"]),
    )
    fallback_candidates = [
        _candidate(
            matrix_id=matrix_id,
            context_id=context_id,
            solver="bicgstab",
            preconditioner="none",
            solver_parameters={},
            role="fallback_probe",
            max_iter=max_iter,
            tolerance_rel=tolerance_rel,
            source_scenario_id=str(scenario["scenario_id"]),
            source_selected_candidate_id=str(scenario["selected_candidate_id"]),
        ),
        _candidate(
            matrix_id=matrix_id,
            context_id=context_id,
            solver="bicgstab",
            preconditioner="jacobi",
            solver_parameters={},
            role="fallback_probe",
            max_iter=max_iter,
            tolerance_rel=tolerance_rel,
            source_scenario_id=str(scenario["scenario_id"]),
            source_selected_candidate_id=str(scenario["selected_candidate_id"]),
        ),
        _candidate(
            matrix_id=matrix_id,
            context_id=context_id,
            solver="gmres",
            preconditioner="jacobi",
            solver_parameters={"restart": 32},
            role="fallback_probe",
            max_iter=max_iter,
            tolerance_rel=tolerance_rel,
            source_scenario_id=str(scenario["scenario_id"]),
            source_selected_candidate_id=str(scenario["selected_candidate_id"]),
        ),
    ]
    unique: dict[str, dict[str, Any]] = {}
    for candidate in [selected_candidate, *fallback_candidates]:
        unique.setdefault(str(candidate["candidate_id"]), candidate)
    return list(unique.values())


def _candidate_from_id(
    candidate_id: str,
    *,
    matrix_id: str,
    context_id: str,
    role: str,
    max_iter: int,
    tolerance_rel: float,
    source_scenario_id: str,
    source_selected_candidate_id: str,
) -> dict[str, Any]:
    if "_gmres_" in candidate_id:
        restart = 16
        for item in (8, 16, 32):
            if f"restart{item}" in candidate_id:
                restart = item
        return _candidate(
            matrix_id=matrix_id,
            context_id=context_id,
            solver="gmres",
            preconditioner="jacobi",
            solver_parameters={"restart": restart},
            role=role,
            max_iter=max_iter,
            tolerance_rel=tolerance_rel,
            source_scenario_id=source_scenario_id,
            source_selected_candidate_id=source_selected_candidate_id,
        )
    if "_richardson_" in candidate_id:
        return _candidate(
            matrix_id=matrix_id,
            context_id=context_id,
            solver="richardson",
            preconditioner="jacobi",
            solver_parameters={},
            role=role,
            max_iter=max_iter,
            tolerance_rel=tolerance_rel,
            source_scenario_id=source_scenario_id,
            source_selected_candidate_id=source_selected_candidate_id,
        )
    if "_pcg_" in candidate_id:
        return _candidate(
            matrix_id=matrix_id,
            context_id=context_id,
            solver="pcg",
            preconditioner="jacobi",
            solver_parameters={},
            role=role,
            max_iter=max_iter,
            tolerance_rel=tolerance_rel,
            source_scenario_id=source_scenario_id,
            source_selected_candidate_id=source_selected_candidate_id,
        )
    if "_cg_" in candidate_id:
        return _candidate(
            matrix_id=matrix_id,
            context_id=context_id,
            solver="cg",
            preconditioner="none",
            solver_parameters={},
            role=role,
            max_iter=max_iter,
            tolerance_rel=tolerance_rel,
            source_scenario_id=source_scenario_id,
            source_selected_candidate_id=source_selected_candidate_id,
        )
    if "_bicgstab_jacobi_" in candidate_id:
        preconditioner = "jacobi"
    else:
        preconditioner = "none"
    return _candidate(
        matrix_id=matrix_id,
        context_id=context_id,
        solver="bicgstab",
        preconditioner=preconditioner,
        solver_parameters={},
        role=role,
        max_iter=max_iter,
        tolerance_rel=tolerance_rel,
        source_scenario_id=source_scenario_id,
        source_selected_candidate_id=source_selected_candidate_id,
    )


def _candidate(
    *,
    matrix_id: str,
    context_id: str,
    solver: str,
    preconditioner: str,
    solver_parameters: dict[str, Any],
    role: str,
    max_iter: int,
    tolerance_rel: float,
    source_scenario_id: str,
    source_selected_candidate_id: str,
) -> dict[str, Any]:
    suffix = ""
    if solver == "gmres":
        suffix = f"_restart{int(solver_parameters.get('restart', 16))}"
    candidate_id = f"taichi_csr_{solver}_{preconditioner}{suffix}_float64"
    return {
        "schema_version": SCHEMA_VERSION,
        "matrix_id": matrix_id,
        "context_id": context_id,
        "candidate_id": candidate_id,
        "solver": solver,
        "preconditioner": preconditioner,
        "precision": "float64",
        "solver_parameters": dict(solver_parameters),
        "measurement_repeats": 1,
        "max_iter": max_iter,
        "tolerance_rel": tolerance_rel,
        "candidate_role": role,
        "source_scenario_id": source_scenario_id,
        "source_selected_candidate_id": source_selected_candidate_id,
    }


def _load_required_csr(
    scenario_rows: list[dict[str, Any]],
) -> tuple[dict[str, CsrMatrix], dict[str, dict[str, Any]]]:
    source_paths = {
        str(row["matrix_source_path"])
        for row in scenario_rows
        if row.get("matrix_source_path")
    }
    required = {str(row["matrix_id"]) for row in scenario_rows}
    csr_by_matrix: dict[str, CsrMatrix] = {}
    records: dict[str, dict[str, Any]] = {}
    for source in source_paths:
        for row in read_jsonl(source):
            matrix_id = str(row["matrix_id"])
            if matrix_id in required and matrix_id not in csr_by_matrix:
                csr_by_matrix[matrix_id] = csr_matrix_from_record(row)
                records[matrix_id] = dict(row)
    missing = sorted(required - set(csr_by_matrix))
    if missing:
        raise SystemExit(f"missing CSR records for fallback probe: {missing}")
    return csr_by_matrix, records


def _summary(
    scenario_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    result_rows: list[dict[str, Any]],
    selector_rows: list[dict[str, Any]],
    *,
    coverage_plan_path: str,
    device_memory_gb: float,
    max_iter: int,
    tolerance_rel: float,
) -> dict[str, Any]:
    success_rows = [row for row in result_rows if row["status"] == "success"]
    screen_rows = [row for row in result_rows if row["status"] == "screened_out"]
    failed_rows = [row for row in result_rows if row["status"] == "failed"]
    resolved_matrices = sorted({row["matrix_id"] for row in success_rows})
    target_matrices = sorted({row["matrix_id"] for row in scenario_rows})
    unresolved_matrices = sorted(set(target_matrices) - set(resolved_matrices))
    max_final = max(
        (float(row.get("final_relative_residual") or 0.0) for row in success_rows),
        default=0.0,
    )
    max_cpu = max(
        (
            float(row.get("cpu_recomputed_relative_residual") or 0.0)
            for row in success_rows
        ),
        default=0.0,
    )
    max_solution = max(
        (float(row.get("solution_relative_error") or 0.0) for row in success_rows),
        default=0.0,
    )
    selector_success_rows = [
        row for row in selector_rows if row.get("target_status") == "success"
    ]
    status = (
        "passed"
        if len(scenario_rows) == 4
        and len(candidate_rows) >= len(scenario_rows)
        and len(result_rows) == len(candidate_rows)
        and len(success_rows) >= 1
        and len(failed_rows) == 0
        and len(selector_success_rows) == len(success_rows)
        else "failed"
    )
    return {
        "status": status,
        "schema_version": SCHEMA_VERSION,
        "coverage_plan_path": coverage_plan_path,
        "device_memory_gb": device_memory_gb,
        "max_iter": max_iter,
        "tolerance_rel": tolerance_rel,
        "executes_gpu": bool(success_rows),
        "runtime_selector_changed": False,
        "target_scenarios": len(scenario_rows),
        "target_matrices": target_matrices,
        "candidate_attempts": len(candidate_rows),
        "gpu_success_rows": len(success_rows),
        "cpu_screened_out_rows": len(screen_rows),
        "gpu_failed_rows": len(failed_rows),
        "selector_rows": len(selector_rows),
        "selector_success_rows": len(selector_success_rows),
        "resolved_matrix_count": len(resolved_matrices),
        "resolved_matrices": resolved_matrices,
        "unresolved_matrix_count": len(unresolved_matrices),
        "unresolved_matrices": unresolved_matrices,
        "max_final_relative_residual": max_final,
        "max_cpu_recomputed_relative_residual": max_cpu,
        "max_solution_relative_error": max_solution,
        "by_status": _counts(row["status"] for row in result_rows),
        "by_solver_status": _counts(
            f"{row['solver']}:{row['status']}" for row in result_rows
        ),
        "next_step": "merge_resolved_fallback_rows_into_guarded_coverage_selector_fixture_and_keep_unresolved_as_documented_gaps",
    }


def _schema() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "task": "cpu_screened_probe_for_exact_profiled_success_fallbacks",
        "source_plan_schema": "phase1_csr_guarded_promotion_coverage_plan_v1",
        "screening_policy": {
            "cpu_screen_required": True,
            "gpu_execution": "only_after_candidate_cpu_screen_success",
            "success_requires": [
                "relative_residual_within_tolerance",
                "solution_error_to_ones_within_5e-3",
                "no_gpu_failure",
            ],
        },
        "runtime_selector_changed": False,
        "expected_outcome": "partial_resolution_allowed_but_unresolved_matrices_must_be_recorded",
    }


def _write_report(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    path: Path,
) -> None:
    lines = [
        "# CSR Non-Success Fallback Probe",
        "",
        f"- status: `{summary['status']}`",
        f"- target_matrices: `{len(summary['target_matrices'])}`",
        f"- candidate_attempts: `{summary['candidate_attempts']}`",
        f"- gpu_success_rows: `{summary['gpu_success_rows']}`",
        f"- cpu_screened_out_rows: `{summary['cpu_screened_out_rows']}`",
        f"- gpu_failed_rows: `{summary['gpu_failed_rows']}`",
        f"- resolved_matrices: `{', '.join(summary['resolved_matrices'])}`",
        f"- unresolved_matrices: `{', '.join(summary['unresolved_matrices'])}`",
        f"- max_final_relative_residual: `{summary['max_final_relative_residual']:.6g}`",
        f"- max_cpu_recomputed_relative_residual: "
        f"`{summary['max_cpu_recomputed_relative_residual']:.6g}`",
        f"- max_solution_relative_error: `{summary['max_solution_relative_error']:.6g}`",
        "",
        "| matrix | candidate | role | status | solver | preconditioner | gpu | rel_res | sol_err | screen_breakdown |",
        "|---|---|---|---|---|---|---|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['matrix_id']} | "
            f"{row['candidate_id']} | "
            f"{row['candidate_role']} | "
            f"{row['status']} | "
            f"{row['solver']} | "
            f"{row['preconditioner']} | "
            f"{row['gpu_executed']} | "
            f"{_fmt(row.get('final_relative_residual'))} | "
            f"{_fmt(row.get('solution_relative_error'))} | "
            f"{row.get('cpu_screen_breakdown') or ''} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _counts(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    return dict(sorted(counts.items()))


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return f"{float(value):.6g}"


if __name__ == "__main__":
    main()
