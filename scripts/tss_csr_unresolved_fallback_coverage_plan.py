"""Plan minimal coverage for unresolved guarded CSR fallback gaps."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transsolvestack.profiling.artifacts import read_jsonl, write_jsonl
from transsolvestack.profiling.provenance import (
    CORE_CSR_UNRESOLVED_FALLBACK_COVERAGE_PLAN_PROVENANCE_FILES,
    build_artifact_manifest,
    git_commit_or_unknown,
    write_manifest,
)


SCHEMA_VERSION = "phase1_csr_unresolved_fallback_coverage_plan_v1"
PLAN_ID = "phase1_csr_unresolved_fallback_coverage_m61"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fallback-probe-results",
        default="runs/phase1_csr_non_success_fallback_probe/csr_non_success_fallback_probe_results.jsonl",
    )
    parser.add_argument(
        "--integration-results",
        default="runs/phase1_csr_guarded_non_success_fallback_integration/csr_guarded_non_success_fallback_integration_results.jsonl",
    )
    parser.add_argument(
        "--integration-summary",
        default="runs/phase1_csr_guarded_non_success_fallback_integration/csr_guarded_non_success_fallback_integration_summary.json",
    )
    parser.add_argument(
        "--out",
        default="runs/phase1_csr_unresolved_fallback_coverage_plan",
    )
    args = parser.parse_args()

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    probe_rows = read_jsonl(args.fallback_probe_results)
    integration_rows = read_jsonl(args.integration_results)
    integration_summary = json.loads(
        Path(args.integration_summary).read_text(encoding="utf-8")
    )
    unresolved_rows = [
        row
        for row in integration_rows
        if row["resolved_exact_fallback_available"] is False
    ]
    if not unresolved_rows:
        raise SystemExit("expected at least one unresolved row")

    rows = _plan_rows(unresolved_rows, probe_rows)
    summary = _summary(
        rows,
        unresolved_rows,
        probe_rows=probe_rows,
        integration_summary=integration_summary,
        fallback_probe_results_path=args.fallback_probe_results,
        integration_results_path=args.integration_results,
        integration_summary_path=args.integration_summary,
    )
    paths = {
        "candidate_plan": output / "csr_unresolved_fallback_candidate_plan.jsonl",
        "matrix_diagnostics": output / "csr_unresolved_fallback_matrix_diagnostics.jsonl",
        "summary": output / "csr_unresolved_fallback_coverage_plan_summary.json",
        "schema": output / "csr_unresolved_fallback_coverage_plan_schema.json",
        "report": output / "csr_unresolved_fallback_coverage_plan_report.md",
        "manifest": output / "artifact_manifest.json",
    }
    write_jsonl(rows, paths["candidate_plan"])
    write_jsonl(
        (_matrix_diagnostic(row, probe_rows) for row in unresolved_rows),
        paths["matrix_diagnostics"],
    )
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
            artifact_kind="csr_unresolved_fallback_coverage_plan",
            command="scripts/tss_csr_unresolved_fallback_coverage_plan.py",
            tracked_files=CORE_CSR_UNRESOLVED_FALLBACK_COVERAGE_PLAN_PROVENANCE_FILES,
            metadata={
                "git_commit": git_commit_or_unknown(),
                "status": summary["status"],
                "schema_version": summary["schema_version"],
                "unresolved_matrix_count": summary["unresolved_matrix_count"],
                "candidate_plan_rows": summary["candidate_plan_rows"],
                "cpu_screen_ready_candidates": summary["cpu_screen_ready_candidates"],
                "future_dependency_candidates": summary["future_dependency_candidates"],
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


def _plan_rows(
    unresolved_rows: list[dict[str, Any]],
    probe_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in sorted(unresolved_rows, key=lambda row: str(row["matrix_id"])):
        matrix_id = str(source["matrix_id"])
        if matrix_id == "suitesparse:Goodwin/Goodwin_010":
            rows.extend(_goodwin_rows(source, probe_rows))
        elif matrix_id == "suitesparse:Gset/G17":
            rows.extend(_gset_rows(source, probe_rows))
        elif matrix_id == "suitesparse:HB/bcsstk07":
            rows.extend(_bcsstk_rows(source, probe_rows))
        elif matrix_id == "suitesparse:Zitney/extr1b":
            rows.extend(_zitney_rows(source, probe_rows))
        else:
            raise SystemExit(f"unexpected unresolved matrix: {matrix_id}")
    return rows


def _goodwin_rows(
    source: dict[str, Any],
    probe_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    evidence = _evidence(source, probe_rows)
    return [
        _row(
            source,
            evidence=evidence,
            local_index=1,
            plan_kind="existing_solver_parameter_sweep",
            candidate_id="taichi_csr_gmres_jacobi_restart64_float64",
            solver="gmres",
            preconditioner="jacobi",
            solver_parameters={"restart": 64, "max_iter": 1024},
            execution_stage="cpu_screen_ready",
            blocker_addressed="restart32_stagnation_and_solution_error",
            expected_value="test_whether_larger_krylov_subspace_reduces_solution_error",
            risk="moderate",
        ),
        _row(
            source,
            evidence=evidence,
            local_index=2,
            plan_kind="existing_solver_parameter_sweep",
            candidate_id="taichi_csr_gmres_none_restart64_float64",
            solver="gmres",
            preconditioner="none",
            solver_parameters={"restart": 64, "max_iter": 1024},
            execution_stage="cpu_screen_ready",
            blocker_addressed="jacobi_preconditioner_may_distort_general_matrix_solution",
            expected_value="separate_preconditioner_effect_from_gmres_restart_effect",
            risk="moderate",
        ),
        _row(
            source,
            evidence=evidence,
            local_index=3,
            plan_kind="future_preconditioner_required",
            candidate_id="taichi_csr_bicgstab_ilu0_float64",
            solver="bicgstab",
            preconditioner="ilu0",
            solver_parameters={"max_iter": 1024},
            execution_stage="blocked_until_preconditioner_exists",
            blocker_addressed="small_residual_but_large_solution_error",
            expected_value="stronger_nonsymmetric_preconditioner_candidate",
            risk="future_work",
        ),
    ]


def _gset_rows(
    source: dict[str, Any],
    probe_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    evidence = _evidence(source, probe_rows)
    return [
        _row(
            source,
            evidence=evidence,
            local_index=1,
            plan_kind="existing_solver_parameter_sweep",
            candidate_id="taichi_csr_gmres_none_restart64_float64",
            solver="gmres",
            preconditioner="none",
            solver_parameters={"restart": 64, "max_iter": 2048},
            execution_stage="cpu_screen_ready",
            blocker_addressed="zero_diagonal_blocks_jacobi_and_restart32_stagnates",
            expected_value="try_non_jacobi_gmres_on_pattern_matrix_before_gpu",
            risk="high",
        ),
        _row(
            source,
            evidence=evidence,
            local_index=2,
            plan_kind="existing_solver_parameter_sweep",
            candidate_id="taichi_csr_bicgstab_none_maxiter2048_float64",
            solver="bicgstab",
            preconditioner="none",
            solver_parameters={"max_iter": 2048},
            execution_stage="cpu_screen_ready",
            blocker_addressed="bicgstab512_near_converges_residual_but_fails_solution_error",
            expected_value="measure_whether_more_iterations_cross_solution_error_gate",
            risk="high",
        ),
        _row(
            source,
            evidence=evidence,
            local_index=3,
            plan_kind="matrix_formulation_diagnostic",
            candidate_id="graph_laplacian_or_shifted_diagonal_formulation_check",
            solver="diagnostic",
            preconditioner="none",
            solver_parameters={},
            execution_stage="diagnostic_only_no_gpu",
            blocker_addressed="pattern_adjacency_has_zero_diagonal_and_is_not_spd_system",
            expected_value="decide_whether_this_matrix_requires_formulation_change_not_solver_tuning",
            risk="future_work",
        ),
    ]


def _bcsstk_rows(
    source: dict[str, Any],
    probe_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    evidence = _evidence(source, probe_rows)
    return [
        _row(
            source,
            evidence=evidence,
            local_index=1,
            plan_kind="existing_solver_parameter_sweep",
            candidate_id="taichi_csr_pcg_jacobi_maxiter2048_float64",
            solver="pcg",
            preconditioner="jacobi",
            solver_parameters={"max_iter": 2048},
            execution_stage="cpu_screen_ready",
            blocker_addressed="pcg_residual_passes_but_solution_error_remains_large",
            expected_value="check_whether_more_iterations_improve_solution_error",
            risk="moderate",
        ),
        _row(
            source,
            evidence=evidence,
            local_index=2,
            plan_kind="existing_solver_parameter_sweep",
            candidate_id="taichi_csr_gmres_jacobi_restart64_float64",
            solver="gmres",
            preconditioner="jacobi",
            solver_parameters={"restart": 64, "max_iter": 1024},
            execution_stage="cpu_screen_ready",
            blocker_addressed="symmetric_matrix_still_fails_solution_error_across_current_krylov_rows",
            expected_value="test_nonsymmetric_krylov_as_robust_fallback",
            risk="moderate",
        ),
        _row(
            source,
            evidence=evidence,
            local_index=3,
            plan_kind="future_preconditioner_required",
            candidate_id="taichi_csr_pcg_ic0_float64",
            solver="pcg",
            preconditioner="ic0",
            solver_parameters={"max_iter": 1024},
            execution_stage="blocked_until_preconditioner_exists",
            blocker_addressed="ill_scaled_spd_like_matrix_needs_stronger_preconditioner",
            expected_value="incomplete_cholesky_family_for_structural_symmetric_cases",
            risk="future_work",
        ),
    ]


def _zitney_rows(
    source: dict[str, Any],
    probe_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    evidence = _evidence(source, probe_rows)
    return [
        _row(
            source,
            evidence=evidence,
            local_index=1,
            plan_kind="existing_solver_parameter_sweep",
            candidate_id="taichi_csr_gmres_none_restart64_float64",
            solver="gmres",
            preconditioner="none",
            solver_parameters={"restart": 64, "max_iter": 2048},
            execution_stage="cpu_screen_ready",
            blocker_addressed="pcg_curvature_breakdown_and_bicgstab_divergence",
            expected_value="test_unpreconditioned_restarted_gmres_before_gpu_probe",
            risk="high",
        ),
        _row(
            source,
            evidence=evidence,
            local_index=2,
            plan_kind="existing_solver_parameter_sweep",
            candidate_id="taichi_csr_gmres_jacobi_restart64_float64",
            solver="gmres",
            preconditioner="jacobi",
            solver_parameters={"restart": 64, "max_iter": 2048},
            execution_stage="cpu_screen_ready",
            blocker_addressed="restart32_gmres_has_large_residual_and_solution_error",
            expected_value="measure_whether_larger_krylov_subspace_changes_failure_class",
            risk="high",
        ),
        _row(
            source,
            evidence=evidence,
            local_index=3,
            plan_kind="future_preconditioner_required",
            candidate_id="taichi_csr_bicgstab_ilu0_or_scaling_float64",
            solver="bicgstab",
            preconditioner="ilu0_or_nonsymmetric_scaling",
            solver_parameters={"max_iter": 2048},
            execution_stage="blocked_until_preconditioner_exists",
            blocker_addressed="general_matrix_existing_krylov_rows_are_unstable_or_inaccurate",
            expected_value="stronger_nonsymmetric_preconditioner_or_scaling_route",
            risk="future_work",
        ),
    ]


def _row(
    source: dict[str, Any],
    *,
    evidence: dict[str, Any],
    local_index: int,
    plan_kind: str,
    candidate_id: str,
    solver: str,
    preconditioner: str,
    solver_parameters: dict[str, Any],
    execution_stage: str,
    blocker_addressed: str,
    expected_value: str,
    risk: str,
) -> dict[str, Any]:
    matrix_id = str(source["matrix_id"])
    row_id = f"m61_{_slug(matrix_id)}_{local_index:02d}"
    cpu_ready = execution_stage == "cpu_screen_ready"
    return {
        "schema_version": SCHEMA_VERSION,
        "plan_id": PLAN_ID,
        "row_id": row_id,
        "matrix_id": matrix_id,
        "context_id": str(source["context_id"]),
        "source_scenario_id": str(source["source_scenario_id"]),
        "source_selected_candidate_id": str(source["source_selected_candidate_id"]),
        "source_selected_target_status": str(source["source_selected_target_status"]),
        "plan_kind": plan_kind,
        "priority": local_index,
        "candidate_id": candidate_id,
        "solver": solver,
        "preconditioner": preconditioner,
        "solver_parameters": solver_parameters,
        "execution_stage": execution_stage,
        "cpu_screen_first": cpu_ready,
        "gpu_allowed_after_cpu_screen": cpu_ready,
        "runtime_selector_changed": False,
        "executes_gpu": False,
        "blocker_class": evidence["blocker_class"],
        "blocker_addressed": blocker_addressed,
        "expected_value": expected_value,
        "risk": risk,
        "source_evidence": evidence,
    }


def _evidence(
    source: dict[str, Any],
    probe_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    matrix_rows = [
        row for row in probe_rows if row["matrix_id"] == source["matrix_id"]
    ]
    finite_residuals = [
        float(row["final_relative_residual"])
        for row in matrix_rows
        if _finite(row.get("final_relative_residual"))
    ]
    finite_solution_errors = [
        float(row["solution_relative_error"])
        for row in matrix_rows
        if _finite(row.get("solution_relative_error"))
    ]
    zero_diagonal = any(
        row.get("cpu_screen_breakdown") == "zero_diagonal" for row in matrix_rows
    )
    best_residual_row = min(
        (row for row in matrix_rows if _finite(row.get("final_relative_residual"))),
        key=lambda row: float(row["final_relative_residual"]),
        default=None,
    )
    best_solution_row = min(
        (row for row in matrix_rows if _finite(row.get("solution_relative_error"))),
        key=lambda row: float(row["solution_relative_error"]),
        default=None,
    )
    if zero_diagonal:
        blocker = "zero_diagonal_and_solution_error_gate"
    elif finite_solution_errors and min(finite_solution_errors) > 5.0e-3:
        blocker = "solution_error_above_gate_despite_small_residual"
    else:
        blocker = "no_exact_profiled_success"
    return {
        "learned_guard_status": source["learned_guard_status"],
        "learned_guard_reasons": source["learned_guard_reasons"],
        "probe_attempt_count": len(matrix_rows),
        "best_relative_residual": None if not finite_residuals else min(finite_residuals),
        "best_residual_candidate_id": (
            None if best_residual_row is None else best_residual_row["candidate_id"]
        ),
        "best_solution_relative_error": (
            None if not finite_solution_errors else min(finite_solution_errors)
        ),
        "best_solution_candidate_id": (
            None if best_solution_row is None else best_solution_row["candidate_id"]
        ),
        "zero_diagonal_observed": zero_diagonal,
        "blocker_class": blocker,
    }


def _matrix_diagnostic(
    source: dict[str, Any],
    probe_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "plan_id": PLAN_ID,
        "matrix_id": source["matrix_id"],
        "context_id": source["context_id"],
        **_evidence(source, probe_rows),
    }


def _summary(
    rows: list[dict[str, Any]],
    unresolved_rows: list[dict[str, Any]],
    *,
    probe_rows: list[dict[str, Any]],
    integration_summary: dict[str, Any],
    fallback_probe_results_path: str,
    integration_results_path: str,
    integration_summary_path: str,
) -> dict[str, Any]:
    by_kind = _counts(row["plan_kind"] for row in rows)
    by_stage = _counts(row["execution_stage"] for row in rows)
    ready_count = sum(1 for row in rows if row["cpu_screen_first"])
    future_count = len(rows) - ready_count
    unresolved_matrices = sorted({row["matrix_id"] for row in unresolved_rows})
    expected_rows = 3 * len(unresolved_matrices)
    expected_ready = 2 * len(unresolved_matrices)
    expected_future = len(unresolved_matrices)
    status = (
        "passed"
        if len(rows) == expected_rows
        and ready_count == expected_ready
        and future_count == expected_future
        and by_kind.get("existing_solver_parameter_sweep") == expected_ready
        and by_stage.get("cpu_screen_ready") == expected_ready
        and all(row["executes_gpu"] is False for row in rows)
        and all(row["runtime_selector_changed"] is False for row in rows)
        else "failed"
    )
    return {
        "status": status,
        "schema_version": SCHEMA_VERSION,
        "plan_id": PLAN_ID,
        "fallback_probe_results_path": fallback_probe_results_path,
        "integration_results_path": integration_results_path,
        "integration_summary_path": integration_summary_path,
        "runtime_selector_changed": False,
        "executes_gpu": False,
        "source_probe_attempts": len(probe_rows),
        "source_integration_status": integration_summary["status"],
        "unresolved_matrix_count": len(unresolved_matrices),
        "unresolved_matrices": unresolved_matrices,
        "candidate_plan_rows": len(rows),
        "cpu_screen_ready_candidates": ready_count,
        "future_dependency_candidates": future_count,
        "by_plan_kind": by_kind,
        "by_execution_stage": by_stage,
        "next_step": "run_cpu_screen_only_for_cpu_screen_ready_candidates_before_any_gpu_execution",
    }


def _schema() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "task": "minimal_plan_for_remaining_unresolved_guarded_fallback_gaps",
        "integration_boundary": {
            "plan_only": True,
            "executes_gpu": False,
            "runtime_selector_changed": False,
        },
        "execution_rules": [
            "cpu_screen_ready candidates must pass CPU residual and solution-error gates before GPU",
            "future_preconditioner_required rows are not executable until the named preconditioner exists",
            "matrix_formulation_diagnostic rows must not be treated as same-system solver candidates",
        ],
    }


def _write_report(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    path: Path,
) -> None:
    lines = [
        "# CSR Unresolved Fallback Coverage Plan",
        "",
        f"- status: `{summary['status']}`",
        f"- unresolved_matrices: `{', '.join(summary['unresolved_matrices'])}`",
        f"- candidate_plan_rows: `{summary['candidate_plan_rows']}`",
        f"- cpu_screen_ready_candidates: `{summary['cpu_screen_ready_candidates']}`",
        f"- future_dependency_candidates: `{summary['future_dependency_candidates']}`",
        f"- executes_gpu: `{summary['executes_gpu']}`",
        f"- runtime_selector_changed: `{summary['runtime_selector_changed']}`",
        "",
        "| row | matrix | kind | candidate | stage | blocker |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['row_id']} | "
            f"{row['matrix_id']} | "
            f"{row['plan_kind']} | "
            f"{row['candidate_id']} | "
            f"{row['execution_stage']} | "
            f"{row['blocker_class']} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _counts(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _finite(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)


def _slug(matrix_id: str) -> str:
    return (
        matrix_id.replace("suitesparse:", "")
        .replace("/", "_")
        .replace("-", "_")
        .lower()
    )


if __name__ == "__main__":
    main()
