"""Integrate resolved non-success fallback rows into guarded CSR coverage."""

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

import transsolvestack as tss
from transsolvestack.core.types import SolveContext
from transsolvestack.datasets.csr import CsrMatrix, csr_matrix_from_record
from transsolvestack.profiling.artifacts import read_jsonl, write_jsonl
from transsolvestack.profiling.provenance import (
    CORE_CSR_GUARDED_NON_SUCCESS_FALLBACK_INTEGRATION_PROVENANCE_FILES,
    build_artifact_manifest,
    git_commit_or_unknown,
    write_manifest,
)
from transsolvestack.profiling.trace import trace_to_record


SCHEMA_VERSION = "phase1_csr_guarded_non_success_fallback_integration_v1"
FIXTURE_MODEL_ID = "m60_fixture_non_success_ranker"
TARGET_SCENARIO_KIND = "fixture_non_success_candidate_block"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--coverage-plan",
        default="runs/phase1_csr_guarded_promotion_coverage_plan/csr_guarded_promotion_coverage_scenarios.jsonl",
    )
    parser.add_argument(
        "--base-selector-rows",
        default="runs/phase1_csr_transformer_ready/combined_csr_selector_rows.jsonl",
    )
    parser.add_argument(
        "--fallback-probe-selector-rows",
        default="runs/phase1_csr_non_success_fallback_probe/csr_non_success_fallback_selector_rows.jsonl",
    )
    parser.add_argument("--device-memory-gb", type=float, default=0.25)
    parser.add_argument(
        "--out",
        default="runs/phase1_csr_guarded_non_success_fallback_integration",
    )
    args = parser.parse_args()

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    plan_rows = [
        row
        for row in read_jsonl(args.coverage_plan)
        if row["scenario_kind"] == TARGET_SCENARIO_KIND
    ]
    if len(plan_rows) != 4:
        raise SystemExit(f"expected 4 non-success scenarios, got {len(plan_rows)}")

    base_selector_rows = read_jsonl(args.base_selector_rows)
    probe_selector_rows = read_jsonl(args.fallback_probe_selector_rows)
    probe_success_rows = [
        row
        for row in probe_selector_rows
        if row["target_status"] == "success"
        and float(row.get("target_success_rate", 0.0)) >= 1.0
    ]
    augmented_selector_rows = _augmented_selector_rows(
        base_selector_rows,
        probe_success_rows,
    )
    paths = {
        "augmented_selector_rows": output / "augmented_csr_selector_rows.jsonl",
        "fixture_predictions": output / "fixture_non_success_predictions.jsonl",
        "fixture_quality_gate": output / "fixture_runtime_eligible_quality_gate_summary.json",
        "results": output / "csr_guarded_non_success_fallback_integration_results.jsonl",
        "summary": output / "csr_guarded_non_success_fallback_integration_summary.json",
        "schema": output / "csr_guarded_non_success_fallback_integration_schema.json",
        "report": output / "csr_guarded_non_success_fallback_integration_report.md",
        "manifest": output / "artifact_manifest.json",
    }
    write_jsonl(augmented_selector_rows, paths["augmented_selector_rows"])
    write_jsonl(
        _fixture_predictions(plan_rows, augmented_selector_rows),
        paths["fixture_predictions"],
    )
    paths["fixture_quality_gate"].write_text(
        json.dumps(_fixture_quality_gate(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    csr_by_matrix = _load_required_csr(plan_rows)
    success_by_key = _exact_success_profiles(augmented_selector_rows)
    records = []
    for row in plan_rows:
        exact_success = success_by_key.get((row["matrix_id"], row["context_id"]), ())
        if exact_success:
            records.append(
                _resolved_fallback_record(
                    row,
                    csr=csr_by_matrix[row["matrix_id"]],
                    selector_rows_path=str(paths["augmented_selector_rows"]),
                    learned_predictions_path=str(paths["fixture_predictions"]),
                    quality_gate_summary_path=str(paths["fixture_quality_gate"]),
                    exact_success_rows=exact_success,
                    device_memory_gb=args.device_memory_gb,
                )
            )
        else:
            records.append(
                _unresolved_guard_only_record(
                    row,
                    csr=csr_by_matrix[row["matrix_id"]],
                    selector_rows_path=str(paths["augmented_selector_rows"]),
                    learned_predictions_path=str(paths["fixture_predictions"]),
                    quality_gate_summary_path=str(paths["fixture_quality_gate"]),
                )
            )

    summary = _summary(
        records,
        base_selector_rows=base_selector_rows,
        probe_selector_rows=probe_selector_rows,
        probe_success_rows=probe_success_rows,
        augmented_selector_rows=augmented_selector_rows,
        coverage_plan_path=args.coverage_plan,
        base_selector_rows_path=args.base_selector_rows,
        fallback_probe_selector_rows_path=args.fallback_probe_selector_rows,
        augmented_selector_rows_path=str(paths["augmented_selector_rows"]),
        fixture_predictions_path=str(paths["fixture_predictions"]),
        fixture_quality_gate_path=str(paths["fixture_quality_gate"]),
        device_memory_gb=args.device_memory_gb,
    )
    write_jsonl(records, paths["results"])
    paths["summary"].write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths["schema"].write_text(
        json.dumps(_schema(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_report(records, summary, paths["report"])
    write_manifest(
        build_artifact_manifest(
            artifact_kind="csr_guarded_non_success_fallback_integration",
            command="scripts/tss_csr_guarded_non_success_fallback_integration.py",
            tracked_files=CORE_CSR_GUARDED_NON_SUCCESS_FALLBACK_INTEGRATION_PROVENANCE_FILES,
            metadata={
                "git_commit": git_commit_or_unknown(),
                "status": summary["status"],
                "schema_version": summary["schema_version"],
                "target_non_success_scenarios": summary["target_non_success_scenarios"],
                "resolved_exact_fallback_scenarios": summary[
                    "resolved_exact_fallback_scenarios"
                ],
                "unresolved_exact_fallback_scenarios": summary[
                    "unresolved_exact_fallback_scenarios"
                ],
                "executed_gpu_scenarios": summary["executed_gpu_scenarios"],
                "guard_only_scenarios": summary["guard_only_scenarios"],
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


def _augmented_selector_rows(
    base_rows: list[dict[str, Any]],
    probe_success_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = list(base_rows)
    existing = {
        (row["matrix_id"], row["context_id"], row["candidate_id"], row["target_status"])
        for row in rows
    }
    for row in probe_success_rows:
        key = (
            row["matrix_id"],
            row["context_id"],
            row["candidate_id"],
            row["target_status"],
        )
        if key not in existing:
            payload = dict(row)
            payload["features"] = dict(row.get("features", {}))
            payload["features"]["source_augmented_from"] = (
                "phase1_csr_non_success_fallback_probe"
            )
            rows.append(payload)
            existing.add(key)
    return rows


def _fixture_predictions(
    plan_rows: list[dict[str, Any]],
    selector_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    success_by_key = _exact_success_profiles(selector_rows)
    rows_by_key = _selector_rows_by_key(selector_rows)
    predictions = []
    for row in plan_rows:
        key = (str(row["matrix_id"]), str(row["context_id"]))
        exact_success = success_by_key.get(key, ())
        selected_row = _fixture_non_success_selection(row, rows_by_key.get(key, ()), exact_success)
        selected = str(selected_row["candidate_id"])
        ranked = [selected]
        if exact_success:
            fallback_id = str(exact_success[0]["candidate_id"])
            if fallback_id != selected:
                ranked.append(fallback_id)
        scores = {
            candidate_id: (9.0 if index == 0 else 1.0)
            for index, candidate_id in enumerate(ranked)
        }
        predictions.append(
            {
                "schema_version": "phase1_csr_transformer_ranker_v1",
                "model_id": FIXTURE_MODEL_ID,
                "request_id": (
                    f"m60:{row['scenario_id']}:{row['matrix_id']}:{row['context_id']}"
                ),
                "split": "eval",
                "matrix_id": row["matrix_id"],
                "context_id": row["context_id"],
                "selected_candidate_id": selected,
                "selected_target_status": selected_row.get("target_status"),
                "selected_label_class": selected_row.get("label_class"),
                "selected_score": scores[selected],
                "evaluation_status": selected_row.get("target_evaluation_status"),
                "oracle_candidate_id": (
                    None if not exact_success else exact_success[0]["candidate_id"]
                ),
                "oracle_rank": None if not exact_success else 2,
                "regret_vs_oracle_ms": None,
                "ranked_candidate_ids": ranked,
                "scores": scores,
            }
        )
    return predictions


def _selector_rows_by_key(
    selector_rows: list[dict[str, Any]],
) -> dict[tuple[str, str], tuple[dict[str, Any], ...]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in selector_rows:
        key = (str(row["matrix_id"]), str(row["context_id"]))
        grouped.setdefault(key, []).append(row)
    return {key: tuple(rows) for key, rows in grouped.items()}


def _fixture_non_success_selection(
    plan_row: dict[str, Any],
    selector_rows: tuple[dict[str, Any], ...],
    exact_success: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    success_ids = {str(row["candidate_id"]) for row in exact_success}
    original_id = str(plan_row["selected_candidate_id"])
    non_success_rows = [
        row
        for row in selector_rows
        if str(row["candidate_id"]) not in success_ids
        and row.get("target_status") != "success"
    ]
    for row in non_success_rows:
        if str(row["candidate_id"]) == original_id:
            return row
    if non_success_rows:
        return sorted(
            non_success_rows,
            key=lambda item: (
                str(item.get("target_status")),
                str(item.get("candidate_id")),
            ),
        )[0]
    return {
        "candidate_id": original_id,
        "target_status": plan_row.get("selected_target_status"),
        "label_class": plan_row.get("selected_target_status"),
        "target_evaluation_status": plan_row.get("selected_evaluation_status"),
    }


def _fixture_quality_gate() -> dict[str, Any]:
    return {
        "status": "passed",
        "schema_version": "phase1_csr_selector_model_eval_v1",
        "evaluation_id": "csr_selector_model_quality_gate_v1",
        "baseline_model_id": "m60_fixture_baseline",
        "challenger_model_id": FIXTURE_MODEL_ID,
        "best_offline_model_id": FIXTURE_MODEL_ID,
        "runtime_selected_model_id": FIXTURE_MODEL_ID,
        "runtime_selector_changed": False,
        "challenger_beats_baseline": True,
        "challenger_runtime_eligible": True,
        "challenger_gate_failures": [],
        "num_eval_predictions": 4,
        "num_eval_oracle_requests": 1,
    }


def _resolved_fallback_record(
    plan_row: dict[str, Any],
    *,
    csr: CsrMatrix,
    selector_rows_path: str,
    learned_predictions_path: str,
    quality_gate_summary_path: str,
    exact_success_rows: tuple[dict[str, Any], ...],
    device_memory_gb: float,
) -> dict[str, Any]:
    context = _context(plan_row)
    rhs = _ones_rhs(csr)
    result = tss.auto_solve_csr_guarded(
        csr,
        rhs,
        selector_path=selector_rows_path,
        learned_predictions_path=learned_predictions_path,
        quality_gate_summary_path=quality_gate_summary_path,
        context=context,
        learned_policy_mode="promote_if_safe",
        device_memory_gb=device_memory_gb,
    )
    trace = trace_to_record(result.trace)
    learned_guard = dict(result.metadata["learned_policy_guard"])
    runtime_guard = dict(result.metadata["runtime_guard"])
    solution = tuple(float(value) for value in result.solution)
    cpu_residual = _relative_residual(csr, solution, rhs)
    solution_error = _relative_error_to_ones(solution)
    exact_ids = tuple(row["candidate_id"] for row in exact_success_rows)
    failure_reasons = _numeric_failure_reasons(
        result_status=str(result.status),
        trace=trace,
        context=context,
        cpu_residual=cpu_residual,
        solution_error=solution_error,
    )
    if learned_guard["guard_status"] != "blocked_non_success_candidate":
        failure_reasons.append("learned_non_success_candidate_not_blocked")
    if learned_guard["runtime_selection_source"] != "artifact":
        failure_reasons.append("unexpected_runtime_selection_source")
    if learned_guard["runtime_candidate_id"] not in exact_ids:
        failure_reasons.append("artifact_runtime_candidate_not_exact_fallback")
    if learned_guard["runtime_selector_changed"] is not False:
        failure_reasons.append("unexpected_runtime_selector_change")
    if runtime_guard["guard_status"] != "success":
        failure_reasons.append("runtime_guard_not_success")
    if runtime_guard["used_fallback"] is not False:
        failure_reasons.append("unexpected_runtime_fallback")
    return _base_record(
        plan_row,
        status="success" if not failure_reasons else "failed",
        failure_reasons=failure_reasons,
        resolved_exact_fallback_available=True,
        exact_success_candidate_ids=exact_ids,
        exact_success_candidate_count=len(exact_success_rows),
        executed_gpu=True,
        final_result_status=str(result.status),
        runtime_selection_source=learned_guard["runtime_selection_source"],
        learned_guard_status=learned_guard["guard_status"],
        learned_guard_reasons=tuple(learned_guard["guard_reasons"]),
        learned_runtime_selector_changed=learned_guard["runtime_selector_changed"],
        candidate_id=learned_guard["runtime_candidate_id"],
        runtime_guard_status=runtime_guard["guard_status"],
        runtime_guard_used_fallback=runtime_guard["used_fallback"],
        runtime_guard_attempt_count=len(runtime_guard["attempts"]),
        runtime_guard_attempts=runtime_guard["attempts"],
        trace=trace,
        cpu_recomputed_relative_residual=cpu_residual,
        solution_relative_error=solution_error,
        solver=trace["metadata"]["solver"],
        preconditioner=trace["metadata"]["preconditioner"],
        precision=trace["metadata"]["precision"],
    )


def _unresolved_guard_only_record(
    plan_row: dict[str, Any],
    *,
    csr: CsrMatrix,
    selector_rows_path: str,
    learned_predictions_path: str,
    quality_gate_summary_path: str,
) -> dict[str, Any]:
    decision = tss.plan_csr_with_learned_guard(
        csr,
        selector_path=selector_rows_path,
        learned_predictions_path=learned_predictions_path,
        quality_gate_summary_path=quality_gate_summary_path,
        context=_context(plan_row),
        mode="promote_if_safe",
        min_confidence=0.0,
    )
    failure_reasons: list[str] = []
    if not _blocks_non_success_candidate(
        decision.guard_status,
        tuple(decision.guard_reasons),
    ):
        failure_reasons.append("unexpected_guard_status")
    if decision.runtime_selector_changed is not False:
        failure_reasons.append("unexpected_runtime_selector_change")
    if decision.learned_prediction is None:
        failure_reasons.append("missing_fixture_prediction")
    return _base_record(
        plan_row,
        status="success" if not failure_reasons else "failed",
        failure_reasons=failure_reasons,
        resolved_exact_fallback_available=False,
        exact_success_candidate_ids=(),
        exact_success_candidate_count=0,
        executed_gpu=False,
        final_result_status="guard_plan_only_unresolved_exact_fallback",
        runtime_selection_source=decision.runtime_selection_source,
        learned_guard_status=decision.guard_status,
        learned_guard_reasons=tuple(decision.guard_reasons),
        learned_runtime_selector_changed=decision.runtime_selector_changed,
        candidate_id=decision.runtime_candidate_id,
        runtime_guard_status="not_executed",
        runtime_guard_used_fallback=False,
        runtime_guard_attempt_count=0,
        runtime_guard_attempts=(),
        trace=None,
        cpu_recomputed_relative_residual=None,
        solution_relative_error=None,
        solver=None,
        preconditioner=None,
        precision=None,
    )


def _base_record(
    plan_row: dict[str, Any],
    *,
    status: str,
    failure_reasons: list[str],
    resolved_exact_fallback_available: bool,
    exact_success_candidate_ids: tuple[str, ...],
    exact_success_candidate_count: int,
    executed_gpu: bool,
    final_result_status: str,
    runtime_selection_source: str,
    learned_guard_status: str,
    learned_guard_reasons: tuple[str, ...],
    learned_runtime_selector_changed: bool,
    candidate_id: str | None,
    runtime_guard_status: str,
    runtime_guard_used_fallback: bool,
    runtime_guard_attempt_count: int,
    runtime_guard_attempts: Any,
    trace: dict[str, Any] | None,
    cpu_recomputed_relative_residual: float | None,
    solution_relative_error: float | None,
    solver: str | None,
    preconditioner: str | None,
    precision: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "source_plan_id": plan_row["plan_id"],
        "source_scenario_id": plan_row["scenario_id"],
        "scenario_kind": plan_row["scenario_kind"],
        "matrix_id": plan_row["matrix_id"],
        "context_id": plan_row["context_id"],
        "source_selected_candidate_id": plan_row["selected_candidate_id"],
        "source_selected_target_status": plan_row["selected_target_status"],
        "resolved_exact_fallback_available": resolved_exact_fallback_available,
        "exact_success_candidate_ids": exact_success_candidate_ids,
        "exact_success_candidate_count": exact_success_candidate_count,
        "executed_gpu": executed_gpu,
        "candidate_id": candidate_id,
        "runtime_selection_source": runtime_selection_source,
        "learned_guard_status": learned_guard_status,
        "learned_guard_reasons": learned_guard_reasons,
        "learned_runtime_selector_changed": learned_runtime_selector_changed,
        "runtime_guard_status": runtime_guard_status,
        "runtime_guard_used_fallback": runtime_guard_used_fallback,
        "runtime_guard_attempt_count": runtime_guard_attempt_count,
        "runtime_guard_attempts": runtime_guard_attempts,
        "solver": solver,
        "preconditioner": preconditioner,
        "precision": precision,
        "final_result_status": final_result_status,
        "status": status,
        "failure_reasons": tuple(failure_reasons),
        "trace": trace,
        "cpu_recomputed_relative_residual": cpu_recomputed_relative_residual,
        "solution_relative_error": solution_relative_error,
    }


def _summary(
    records: list[dict[str, Any]],
    *,
    base_selector_rows: list[dict[str, Any]],
    probe_selector_rows: list[dict[str, Any]],
    probe_success_rows: list[dict[str, Any]],
    augmented_selector_rows: list[dict[str, Any]],
    coverage_plan_path: str,
    base_selector_rows_path: str,
    fallback_probe_selector_rows_path: str,
    augmented_selector_rows_path: str,
    fixture_predictions_path: str,
    fixture_quality_gate_path: str,
    device_memory_gb: float,
) -> dict[str, Any]:
    gpu_rows = [row for row in records if row["executed_gpu"]]
    guard_only_rows = [row for row in records if not row["executed_gpu"]]
    resolved_rows = [row for row in records if row["resolved_exact_fallback_available"]]
    unresolved_rows = [row for row in records if not row["resolved_exact_fallback_available"]]
    residuals = [
        float(row["trace"]["final_residual_norm"])
        for row in gpu_rows
        if row["trace"] is not None
    ]
    cpu_residuals = [
        float(row["cpu_recomputed_relative_residual"])
        for row in gpu_rows
        if row["cpu_recomputed_relative_residual"] is not None
    ]
    solution_errors = [
        float(row["solution_relative_error"])
        for row in gpu_rows
        if row["solution_relative_error"] is not None
    ]
    status = (
        "passed"
        if len(records) == 4
        and len(resolved_rows) >= 1
        and len(unresolved_rows) >= 1
        and len(gpu_rows) == len(resolved_rows)
        and len(guard_only_rows) == len(unresolved_rows)
        and all(row["status"] == "success" for row in records)
        and all(_record_blocks_non_success_candidate(row) for row in records)
        and all(row["learned_runtime_selector_changed"] is False for row in records)
        and all(row["runtime_selection_source"] == "artifact" for row in records)
        and max(residuals, default=math.inf) <= 1.0e-5
        and max(cpu_residuals, default=math.inf) <= 1.0e-4
        and max(solution_errors, default=math.inf) <= 5.0e-3
        else "failed"
    )
    return {
        "status": status,
        "schema_version": SCHEMA_VERSION,
        "coverage_plan_path": coverage_plan_path,
        "base_selector_rows_path": base_selector_rows_path,
        "fallback_probe_selector_rows_path": fallback_probe_selector_rows_path,
        "augmented_selector_rows_path": augmented_selector_rows_path,
        "fixture_predictions_path": fixture_predictions_path,
        "fixture_quality_gate_path": fixture_quality_gate_path,
        "device_memory_gb": device_memory_gb,
        "runtime_selector_changed": False,
        "target_non_success_scenarios": len(records),
        "base_selector_rows": len(base_selector_rows),
        "fallback_probe_selector_rows": len(probe_selector_rows),
        "fallback_probe_success_rows_imported": len(probe_success_rows),
        "augmented_selector_rows": len(augmented_selector_rows),
        "resolved_exact_fallback_scenarios": len(resolved_rows),
        "resolved_matrices": sorted(row["matrix_id"] for row in resolved_rows),
        "unresolved_exact_fallback_scenarios": len(unresolved_rows),
        "unresolved_matrices": sorted(row["matrix_id"] for row in unresolved_rows),
        "executed_gpu_scenarios": len(gpu_rows),
        "guard_only_scenarios": len(guard_only_rows),
        "learned_non_success_blocks": sum(
            1 for row in records if _record_blocks_non_success_candidate(row)
        ),
        "artifact_fallback_solves": sum(
            1
            for row in gpu_rows
            if row["runtime_selection_source"] == "artifact"
        ),
        "runtime_guard_success_count": sum(
            1 for row in gpu_rows if row["runtime_guard_status"] == "success"
        ),
        "selected_solver_set": sorted(
            {str(row["solver"]) for row in gpu_rows if row["solver"] is not None}
        ),
        "max_final_relative_residual": max(residuals, default=0.0),
        "max_cpu_recomputed_relative_residual": max(cpu_residuals, default=0.0),
        "max_solution_relative_error": max(solution_errors, default=0.0),
        "next_step": "plan_remaining_guard_only_solver_or_formulation_coverage",
    }


def _schema() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "task": "integrate_resolved_non_success_fallback_rows_without_promoting_non_success_learned_candidates",
        "source_probe_schema": "phase1_csr_non_success_fallback_probe_v1",
        "safety_policy": {
            "runtime_selector_changed": False,
            "learned_non_success_candidate": "must_remain_blocked",
            "gpu_execution": "only_when_exact_matrix_context_profiled_success_fallback_exists",
            "unresolved_matrices": "guard_only_no_gpu",
        },
    }


def _record_blocks_non_success_candidate(row: dict[str, Any]) -> bool:
    return _blocks_non_success_candidate(
        str(row["learned_guard_status"]),
        tuple(str(item) for item in row["learned_guard_reasons"]),
    )


def _blocks_non_success_candidate(
    guard_status: str,
    guard_reasons: tuple[str, ...],
) -> bool:
    if guard_status == "blocked_non_success_candidate":
        return True
    return guard_status == "blocked_fallback_chain" and any(
        "not a profiled success" in reason for reason in guard_reasons
    )


def _load_required_csr(rows: list[dict[str, Any]]) -> dict[str, CsrMatrix]:
    source_paths = {
        str(row["matrix_source_path"])
        for row in rows
        if row.get("matrix_source_path")
    }
    required = {str(row["matrix_id"]) for row in rows}
    loaded: dict[str, CsrMatrix] = {}
    for source in source_paths:
        for row in read_jsonl(source):
            matrix_id = str(row["matrix_id"])
            if matrix_id in required and matrix_id not in loaded:
                loaded[matrix_id] = csr_matrix_from_record(row)
    missing = sorted(required - set(loaded))
    if missing:
        raise SystemExit(f"missing CSR rows for fallback integration: {missing}")
    return loaded


def _exact_success_profiles(
    selector_rows: list[dict[str, Any]],
) -> dict[tuple[str, str], tuple[dict[str, Any], ...]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in selector_rows:
        if row["target_status"] != "success":
            continue
        if float(row.get("target_success_rate", 0.0)) < 1.0:
            continue
        key = (str(row["matrix_id"]), str(row["context_id"]))
        grouped.setdefault(key, []).append(row)
    return {
        key: tuple(sorted(rows, key=_profile_sort_key))
        for key, rows in grouped.items()
    }


def _profile_sort_key(row: dict[str, Any]) -> tuple[float, float, str]:
    solve_time = row.get("target_median_solve_time_ms")
    if solve_time is None:
        solve_time = row.get("target_solve_time_ms")
    iqr = row.get("target_solve_time_iqr_ms")
    return (
        math.inf if solve_time is None else float(solve_time),
        0.0 if iqr is None else float(iqr),
        str(row["candidate_id"]),
    )


def _context(row: dict[str, Any]) -> SolveContext:
    return SolveContext(
        context_id=str(row["context_id"]),
        tolerance_abs=0.0,
        tolerance_rel=1.0e-5,
        max_iter=512,
        precision="float64",
        required_backend="taichi_gpu",
    )


def _ones_rhs(csr: CsrMatrix) -> tuple[float, ...]:
    return csr.matvec(tuple(1.0 for _ in range(csr.n_cols)))


def _numeric_failure_reasons(
    *,
    result_status: str,
    trace: dict[str, Any],
    context: SolveContext,
    cpu_residual: float,
    solution_error: float,
) -> list[str]:
    reasons: list[str] = []
    if result_status not in {"success", "fallback_success"}:
        reasons.append("solver_failed")
    trace_residual = trace.get("final_residual_norm")
    if trace_residual is None or float(trace_residual) > context.tolerance_rel:
        reasons.append("trace_residual_above_tolerance")
    if cpu_residual > 1.0e-4:
        reasons.append("cpu_recomputed_residual_above_tolerance")
    if solution_error > 5.0e-3:
        reasons.append("solution_error_above_tolerance")
    if trace["backend"] != "taichi_gpu":
        reasons.append("final_result_not_taichi_gpu")
    return reasons


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


def _write_report(
    records: list[dict[str, Any]],
    summary: dict[str, Any],
    path: Path,
) -> None:
    lines = [
        "# CSR Guarded Non-Success Fallback Integration",
        "",
        f"- status: `{summary['status']}`",
        f"- target_non_success_scenarios: `{summary['target_non_success_scenarios']}`",
        f"- resolved_exact_fallback_scenarios: `{summary['resolved_exact_fallback_scenarios']}`",
        f"- unresolved_exact_fallback_scenarios: `{summary['unresolved_exact_fallback_scenarios']}`",
        f"- executed_gpu_scenarios: `{summary['executed_gpu_scenarios']}`",
        f"- guard_only_scenarios: `{summary['guard_only_scenarios']}`",
        f"- learned_non_success_blocks: `{summary['learned_non_success_blocks']}`",
        f"- artifact_fallback_solves: `{summary['artifact_fallback_solves']}`",
        f"- selected_solver_set: `{', '.join(summary['selected_solver_set'])}`",
        f"- max_final_relative_residual: `{summary['max_final_relative_residual']:.6g}`",
        f"- max_cpu_recomputed_relative_residual: "
        f"`{summary['max_cpu_recomputed_relative_residual']:.6g}`",
        f"- max_solution_relative_error: `{summary['max_solution_relative_error']:.6g}`",
        "",
        "| scenario | matrix | selected | exact fallback | executed_gpu | runtime candidate | solver | status | rel_res |",
        "|---|---|---|---:|---:|---|---|---|---:|",
    ]
    for row in records:
        residual = "" if row["trace"] is None else f"{float(row['trace']['final_residual_norm']):.6g}"
        lines.append(
            "| "
            f"{row['source_scenario_id']} | "
            f"{row['matrix_id']} | "
            f"{row['source_selected_candidate_id']} | "
            f"{row['exact_success_candidate_count']} | "
            f"{row['executed_gpu']} | "
            f"{row['candidate_id'] or ''} | "
            f"{row['solver'] or ''} | "
            f"{row['status']} | "
            f"{residual} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
