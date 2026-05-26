"""Execute a light guarded learned-policy CSR promotion coverage subset."""

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
from transsolvestack.core.result import PolicyPlan
from transsolvestack.core.types import SolveContext
from transsolvestack.datasets.csr import CsrMatrix, csr_matrix_from_record
from transsolvestack.policies.csr_artifact_selector import CsrArtifactPolicySelector
from transsolvestack.profiling.artifacts import read_jsonl, write_jsonl
from transsolvestack.profiling.provenance import (
    CORE_CSR_GUARDED_PROMOTION_COVERAGE_EXEC_PROVENANCE_FILES,
    build_artifact_manifest,
    git_commit_or_unknown,
    write_manifest,
)
from transsolvestack.profiling.trace import trace_to_record
from transsolvestack.runtime.guarded_fallback import (
    RuntimeGuardConfig,
    execute_with_fallback_guard,
)


FIXTURE_MODEL_ID = "m58_fixture_promotable_ranker"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--coverage-plan",
        default="runs/phase1_csr_guarded_promotion_coverage_plan/csr_guarded_promotion_coverage_scenarios.jsonl",
    )
    parser.add_argument(
        "--selector-rows",
        default="runs/phase1_csr_transformer_ready/combined_csr_selector_rows.jsonl",
    )
    parser.add_argument(
        "--learned-predictions",
        default="runs/phase1_csr_transformer_ranker/csr_transformer_ranker_predictions.jsonl",
    )
    parser.add_argument(
        "--quality-gate-summary",
        default="runs/phase1_csr_transformer_quality_gate/csr_transformer_quality_gate_summary.json",
    )
    parser.add_argument("--max-current-blocked", type=int, default=4)
    parser.add_argument(
        "--max-changed-promotions",
        dest="max_fixture_promotions",
        type=int,
        default=2,
        help=(
            "Fixture profiled-success promotions to execute. Changed-candidate "
            "promotions are prioritized when available."
        ),
    )
    parser.add_argument("--max-runtime-fallbacks", type=int, default=2)
    parser.add_argument("--device-memory-gb", type=float, default=0.25)
    parser.add_argument("--out", default="runs/phase1_csr_guarded_promotion_coverage_exec")
    args = parser.parse_args()

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    plan_rows = read_jsonl(args.coverage_plan)
    selected = _select_coverage_subset(
        plan_rows,
        max_current_blocked=args.max_current_blocked,
        max_fixture_promotions=args.max_fixture_promotions,
        max_runtime_fallbacks=args.max_runtime_fallbacks,
    )
    fixture_paths = _write_fixture_policy_artifacts(selected, output)
    csr_by_matrix = _load_required_csr(selected)

    records: list[dict[str, Any]] = []
    for row in selected["current_blocked"]:
        records.append(
            _guarded_auto_solve_record(
                row,
                csr=csr_by_matrix[row["matrix_id"]],
                selector_rows_path=args.selector_rows,
                learned_predictions_path=args.learned_predictions,
                quality_gate_summary_path=args.quality_gate_summary,
                expected_guard_status="blocked_quality_gate",
                expected_runtime_source="artifact",
                device_memory_gb=args.device_memory_gb,
            )
        )
    for row in selected["fixture_promotions"]:
        records.append(
            _guarded_auto_solve_record(
                row,
                csr=csr_by_matrix[row["matrix_id"]],
                selector_rows_path=args.selector_rows,
                learned_predictions_path=str(fixture_paths["predictions"]),
                quality_gate_summary_path=str(fixture_paths["quality_gate"]),
                expected_guard_status="promoted",
                expected_runtime_source="learned",
                device_memory_gb=args.device_memory_gb,
            )
        )
    for row in selected["non_success_blocks"]:
        records.append(
            _guard_only_non_success_record(
                row,
                csr=csr_by_matrix[row["matrix_id"]],
                selector_rows_path=args.selector_rows,
                learned_predictions_path=str(fixture_paths["predictions"]),
                quality_gate_summary_path=str(fixture_paths["quality_gate"]),
            )
        )
    for row in selected["runtime_fallbacks"]:
        records.append(
            _runtime_exception_fallback_record(
                row,
                csr=csr_by_matrix[row["matrix_id"]],
                selector_rows_path=args.selector_rows,
                device_memory_gb=args.device_memory_gb,
            )
        )

    summary = _build_summary(
        records,
        coverage_plan_path=args.coverage_plan,
        selector_rows_path=args.selector_rows,
        learned_predictions_path=args.learned_predictions,
        quality_gate_summary_path=args.quality_gate_summary,
        fixture_predictions_path=str(fixture_paths["predictions"]),
        fixture_quality_gate_path=str(fixture_paths["quality_gate"]),
        device_memory_gb=args.device_memory_gb,
    )
    paths = {
        "results": output / "csr_guarded_promotion_coverage_exec_results.jsonl",
        "summary": output / "csr_guarded_promotion_coverage_exec_summary.json",
        "schema": output / "csr_guarded_promotion_coverage_exec_schema.json",
        "report": output / "csr_guarded_promotion_coverage_exec_report.md",
        "manifest": output / "artifact_manifest.json",
    }
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
            artifact_kind="csr_guarded_promotion_coverage_exec",
            command="scripts/tss_csr_guarded_promotion_coverage_exec.py",
            tracked_files=CORE_CSR_GUARDED_PROMOTION_COVERAGE_EXEC_PROVENANCE_FILES,
            metadata={
                "git_commit": git_commit_or_unknown(),
                "status": summary["status"],
                "schema_version": summary["schema_version"],
                "num_scenarios": summary["num_scenarios"],
                "executed_gpu_scenarios": summary["executed_gpu_scenarios"],
                "guard_only_scenarios": summary["guard_only_scenarios"],
                "current_quality_gate_blocks": summary["current_quality_gate_blocks"],
                "fixture_learned_promotions": summary["fixture_learned_promotions"],
                "non_success_blocks": summary["non_success_blocks"],
                "runtime_exception_fallbacks": summary["runtime_exception_fallbacks"],
                "current_runtime_selector_changed": summary[
                    "current_runtime_selector_changed"
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


def _select_coverage_subset(
    rows: list[dict[str, Any]],
    *,
    max_current_blocked: int,
    max_fixture_promotions: int,
    max_runtime_fallbacks: int,
) -> dict[str, list[dict[str, Any]]]:
    current = [
        row
        for row in rows
        if row["scenario_kind"] == "current_ranker_quality_gate_block"
    ][:max_current_blocked]
    promotion_candidates = [
        row
        for row in rows
        if row["scenario_kind"] == "fixture_profiled_success_promotion"
    ]
    changed_promotions = [
        row
        for row in promotion_candidates
        if row["artifact_candidate_id"] != row["selected_candidate_id"]
    ]
    unchanged_promotions = [
        row
        for row in promotion_candidates
        if row["artifact_candidate_id"] == row["selected_candidate_id"]
    ]
    promotions = (changed_promotions + unchanged_promotions)[:max_fixture_promotions]
    non_success = [
        row
        for row in rows
        if row["scenario_kind"] == "fixture_non_success_candidate_block"
    ]
    runtime_fallbacks = [
        row
        for row in rows
        if row["scenario_kind"] == "runtime_exception_fallback"
    ][:max_runtime_fallbacks]
    if len(current) != max_current_blocked:
        raise SystemExit("not enough current blocked coverage scenarios")
    if len(promotions) != max_fixture_promotions:
        raise SystemExit("not enough fixture promotion scenarios")
    if not changed_promotions:
        raise SystemExit("missing changed-candidate promotion scenario")
    if len(runtime_fallbacks) != max_runtime_fallbacks:
        raise SystemExit("not enough runtime fallback scenarios")
    if not non_success:
        raise SystemExit("missing non-success block coverage scenarios")
    return {
        "current_blocked": current,
        "fixture_promotions": promotions,
        "non_success_blocks": non_success,
        "runtime_fallbacks": runtime_fallbacks,
    }


def _guarded_auto_solve_record(
    plan_row: dict[str, Any],
    *,
    csr: CsrMatrix,
    selector_rows_path: str,
    learned_predictions_path: str,
    quality_gate_summary_path: str,
    expected_guard_status: str,
    expected_runtime_source: str,
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
    failure_reasons = _common_failure_reasons(
        result_status=str(result.status),
        trace=trace,
        context=context,
        cpu_residual=cpu_residual,
        solution_error=solution_error,
    )
    if learned_guard["guard_status"] != expected_guard_status:
        failure_reasons.append("unexpected_learned_guard_status")
    if learned_guard["runtime_selection_source"] != expected_runtime_source:
        failure_reasons.append("unexpected_runtime_selection_source")
    if runtime_guard["guard_status"] != "success":
        failure_reasons.append("runtime_guard_not_success")
    if runtime_guard["used_fallback"] is not False:
        failure_reasons.append("unexpected_runtime_fallback")
    if expected_runtime_source == "learned" and learned_guard["runtime_candidate_id"] != plan_row["selected_candidate_id"]:
        failure_reasons.append("unexpected_promoted_candidate")
    return _base_record(
        plan_row,
        status="success" if not failure_reasons else "failed",
        failure_reasons=failure_reasons,
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


def _guard_only_non_success_record(
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
    if decision.guard_status != "blocked_non_success_candidate":
        if not (
            decision.guard_status == "blocked_fallback_chain"
            and any("not a profiled success" in reason for reason in decision.guard_reasons)
        ):
            failure_reasons.append("unexpected_guard_status")
    if decision.runtime_selector_changed is not False:
        failure_reasons.append("unexpected_runtime_selector_change")
    if decision.learned_prediction is None:
        failure_reasons.append("missing_fixture_prediction")
    elif decision.learned_prediction.selected_candidate_id != plan_row["selected_candidate_id"]:
        failure_reasons.append("fixture_prediction_candidate_mismatch")
    return _base_record(
        plan_row,
        status="success" if not failure_reasons else "failed",
        failure_reasons=failure_reasons,
        final_result_status="guard_plan_only",
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


def _runtime_exception_fallback_record(
    plan_row: dict[str, Any],
    *,
    csr: CsrMatrix,
    selector_rows_path: str,
    device_memory_gb: float,
) -> dict[str, Any]:
    context = _context(plan_row)
    rhs = _ones_rhs(csr)
    selection = CsrArtifactPolicySelector.from_selector_rows(selector_rows_path).select(
        csr,
        context_id=context.context_id,
    )
    bad_plan = PolicyPlan(
        plan_id=f"m58_invalid_cg_primary:{csr.matrix_id}:{context.context_id}",
        backend="taichi_gpu",
        solver={"name": "cg", "precision": context.precision},
        preconditioner={"name": "none"},
        audit={
            "candidate_id": "m58_invalid_cg_primary",
            "selection_reason": "runtime_exception_fallback_fixture",
        },
    )

    def execute(active_plan: PolicyPlan):
        return tss.solve_csr(
            csr,
            rhs,
            context=context,
            solver=str(active_plan.solver["name"]),
            preconditioner=active_plan.preconditioner.get("name"),
            precision=active_plan.solver.get("precision", context.precision),
            solver_parameters={
                key: value
                for key, value in active_plan.solver.items()
                if key not in {"name", "precision"}
            },
            policy_plan=active_plan,
            device_memory_gb=device_memory_gb,
        )

    outcome = execute_with_fallback_guard(
        (bad_plan, selection.plan),
        execute,
        config=RuntimeGuardConfig(),
    )
    result = outcome.result
    trace = trace_to_record(result.trace)
    runtime_guard = dict(result.metadata["runtime_guard"])
    solution = tuple(float(value) for value in result.solution)
    cpu_residual = _relative_residual(csr, solution, rhs)
    solution_error = _relative_error_to_ones(solution)
    failure_reasons = _common_failure_reasons(
        result_status=str(result.status),
        trace=trace,
        context=context,
        cpu_residual=cpu_residual,
        solution_error=solution_error,
    )
    if runtime_guard["guard_status"] != "success":
        failure_reasons.append("runtime_guard_not_success")
    if runtime_guard["used_fallback"] is not True:
        failure_reasons.append("runtime_fallback_not_used")
    attempts = runtime_guard["attempts"]
    if len(attempts) != 2:
        failure_reasons.append("runtime_attempt_count_mismatch")
    elif attempts[0]["status"] != "exception":
        failure_reasons.append("first_attempt_not_exception")
    if str(result.status) != "fallback_success":
        failure_reasons.append("final_result_not_fallback_success")
    return _base_record(
        plan_row,
        status="success" if not failure_reasons else "failed",
        failure_reasons=failure_reasons,
        final_result_status=str(result.status),
        runtime_selection_source="runtime_guard_fallback",
        learned_guard_status="not_applicable_runtime_guard_only",
        learned_guard_reasons=(),
        learned_runtime_selector_changed=False,
        candidate_id=selection.candidate_id,
        runtime_guard_status=runtime_guard["guard_status"],
        runtime_guard_used_fallback=runtime_guard["used_fallback"],
        runtime_guard_attempt_count=len(attempts),
        runtime_guard_attempts=attempts,
        trace=trace,
        cpu_recomputed_relative_residual=cpu_residual,
        solution_relative_error=solution_error,
        solver=trace["metadata"]["solver"],
        preconditioner=trace["metadata"]["preconditioner"],
        precision=trace["metadata"]["precision"],
    )


def _base_record(
    plan_row: dict[str, Any],
    *,
    status: str,
    failure_reasons: list[str],
    final_result_status: str,
    runtime_selection_source: str,
    learned_guard_status: str | None,
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
        "schema_version": "phase1_csr_guarded_promotion_coverage_exec_v1",
        "source_plan_id": plan_row["plan_id"],
        "source_scenario_id": plan_row["scenario_id"],
        "scenario_kind": plan_row["scenario_kind"],
        "matrix_id": plan_row["matrix_id"],
        "context_id": plan_row["context_id"],
        "requires_gpu_execution": bool(plan_row["requires_gpu_execution"]),
        "candidate_id": candidate_id,
        "source_selected_candidate_id": plan_row["selected_candidate_id"],
        "source_artifact_candidate_id": plan_row["artifact_candidate_id"],
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


def _build_summary(
    records: list[dict[str, Any]],
    *,
    coverage_plan_path: str,
    selector_rows_path: str,
    learned_predictions_path: str,
    quality_gate_summary_path: str,
    fixture_predictions_path: str,
    fixture_quality_gate_path: str,
    device_memory_gb: float,
) -> dict[str, Any]:
    num_success = sum(1 for row in records if row["status"] == "success")
    num_failed = len(records) - num_success
    gpu_rows = [row for row in records if row["requires_gpu_execution"]]
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
    current_blocks = sum(
        1
        for row in records
        if row["scenario_kind"] == "current_ranker_quality_gate_block"
        and row["learned_guard_status"] == "blocked_quality_gate"
    )
    promotions = sum(
        1
        for row in records
        if row["scenario_kind"] == "fixture_profiled_success_promotion"
        and row["runtime_selection_source"] == "learned"
    )
    non_success = sum(
        1
        for row in records
        if row["scenario_kind"] == "fixture_non_success_candidate_block"
        and (
            row["learned_guard_status"] == "blocked_non_success_candidate"
            or (
                row["learned_guard_status"] == "blocked_fallback_chain"
                and any(
                    "not a profiled success" in reason
                    for reason in row["learned_guard_reasons"]
                )
            )
        )
    )
    fallbacks = sum(
        1
        for row in records
        if row["scenario_kind"] == "runtime_exception_fallback"
        and row["runtime_guard_used_fallback"] is True
    )
    status = (
        "passed"
        if len(records) == 12
        and num_failed == 0
        and len(gpu_rows) == 8
        and current_blocks == 4
        and promotions == 2
        and non_success == 4
        and fallbacks == 2
        and all(row["trace"]["backend"] == "taichi_gpu" for row in gpu_rows)
        and max(residuals, default=math.inf) <= 1.0e-5
        and max(cpu_residuals, default=math.inf) <= 1.0e-4
        and max(solution_errors, default=math.inf) <= 5.0e-3
        else "failed"
    )
    return {
        "status": status,
        "schema_version": "phase1_csr_guarded_promotion_coverage_exec_v1",
        "coverage_plan_path": coverage_plan_path,
        "selector_rows_path": selector_rows_path,
        "learned_predictions_path": learned_predictions_path,
        "quality_gate_summary_path": quality_gate_summary_path,
        "fixture_predictions_path": fixture_predictions_path,
        "fixture_quality_gate_path": fixture_quality_gate_path,
        "device_memory_gb": device_memory_gb,
        "num_scenarios": len(records),
        "num_success": num_success,
        "num_failed": num_failed,
        "executed_gpu_scenarios": len(gpu_rows),
        "guard_only_scenarios": len(records) - len(gpu_rows),
        "current_quality_gate_blocks": current_blocks,
        "fixture_learned_promotions": promotions,
        "fixture_runtime_selector_changed_count": sum(
            1 for row in records if row["learned_runtime_selector_changed"] is True
        ),
        "current_runtime_selector_changed": any(
            row["learned_runtime_selector_changed"] is True
            for row in records
            if row["scenario_kind"] == "current_ranker_quality_gate_block"
        ),
        "non_success_blocks": non_success,
        "runtime_exception_fallbacks": fallbacks,
        "runtime_fallback_used_count": sum(
            1 for row in records if row["runtime_guard_used_fallback"] is True
        ),
        "runtime_guard_success_count": sum(
            1 for row in gpu_rows if row["runtime_guard_status"] == "success"
        ),
        "selected_solver_set": sorted(
            {str(row["solver"]) for row in gpu_rows if row["solver"] is not None}
        ),
        "max_final_relative_residual": max(residuals, default=math.inf),
        "max_cpu_recomputed_relative_residual": max(cpu_residuals, default=math.inf),
        "max_solution_relative_error": max(solution_errors, default=math.inf),
    }


def _schema() -> dict[str, Any]:
    return {
        "schema_version": "phase1_csr_guarded_promotion_coverage_exec_v1",
        "task": "execute_light_guarded_learned_policy_runtime_coverage_subset",
        "source_plan_schema": "phase1_csr_guarded_promotion_coverage_plan_v1",
        "execution_policy": {
            "current_ranker_quality_gate_block": "execute_gpu",
            "fixture_profiled_success_promotion": (
                "execute_gpu; changed-candidate promotions are prioritized"
            ),
            "fixture_non_success_candidate_block": "guard_only_no_gpu",
            "runtime_exception_fallback": "execute_gpu",
        },
        "pass_conditions": [
            "current ranker remains blocked by real quality gate",
            "fixture profiled-success promotions solve on GPU",
            "at least one fixture promotion changes the runtime candidate",
            "non-success candidates are blocked before GPU execution",
            "runtime exception fallbacks end in profiled GPU solves",
            "all final GPU solves satisfy residual and solution checks",
        ],
    }


def _write_fixture_policy_artifacts(
    selected: dict[str, list[dict[str, Any]]],
    output: Path,
) -> dict[str, Path]:
    predictions = output / "fixture_coverage_predictions.jsonl"
    quality_gate = output / "fixture_runtime_eligible_quality_gate_summary.json"
    fixture_rows = [
        *_fixture_predictions_from_plan_rows(selected["fixture_promotions"]),
        *_fixture_predictions_from_plan_rows(selected["non_success_blocks"]),
    ]
    write_jsonl(fixture_rows, predictions)
    quality_gate.write_text(
        json.dumps(_fixture_quality_gate(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"predictions": predictions, "quality_gate": quality_gate}


def _fixture_predictions_from_plan_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    predictions = []
    for row in rows:
        selected = str(row["selected_candidate_id"])
        artifact = row.get("artifact_candidate_id")
        ranked = [selected]
        if artifact and artifact != selected:
            ranked.append(str(artifact))
        scores = {
            candidate_id: (9.0 if index == 0 else 1.0)
            for index, candidate_id in enumerate(ranked)
        }
        predictions.append(
            {
                "schema_version": "phase1_csr_transformer_ranker_v1",
                "model_id": FIXTURE_MODEL_ID,
                "request_id": f"{row['scenario_id']}:{row['matrix_id']}:{row['context_id']}",
                "split": "eval",
                "matrix_id": row["matrix_id"],
                "context_id": row["context_id"],
                "selected_candidate_id": selected,
                "selected_target_status": row.get("selected_target_status"),
                "selected_label_class": row.get("selected_target_status"),
                "selected_score": scores[selected],
                "evaluation_status": row.get("selected_evaluation_status"),
                "oracle_candidate_id": artifact,
                "oracle_rank": 2 if artifact and artifact != selected else 1,
                "regret_vs_oracle_ms": None,
                "ranked_candidate_ids": ranked,
                "scores": scores,
            }
        )
    return predictions


def _fixture_quality_gate() -> dict[str, Any]:
    return {
        "status": "passed",
        "schema_version": "phase1_csr_selector_model_eval_v1",
        "evaluation_id": "csr_selector_model_quality_gate_v1",
        "baseline_model_id": "m58_fixture_baseline",
        "challenger_model_id": FIXTURE_MODEL_ID,
        "best_offline_model_id": FIXTURE_MODEL_ID,
        "runtime_selected_model_id": FIXTURE_MODEL_ID,
        "runtime_selector_changed": False,
        "challenger_beats_baseline": True,
        "challenger_runtime_eligible": True,
        "challenger_gate_failures": [],
        "num_eval_predictions": 6,
        "num_eval_oracle_requests": 6,
    }


def _load_required_csr(selected: dict[str, list[dict[str, Any]]]) -> dict[str, CsrMatrix]:
    source_paths = {
        row["matrix_source_path"]
        for rows in selected.values()
        for row in rows
        if row.get("matrix_source_path")
    }
    matrix_ids = {
        row["matrix_id"]
        for rows in selected.values()
        for row in rows
    }
    loaded: dict[str, CsrMatrix] = {}
    for source in source_paths:
        for row in read_jsonl(source):
            matrix_id = str(row["matrix_id"])
            if matrix_id in matrix_ids and matrix_id not in loaded:
                loaded[matrix_id] = csr_matrix_from_record(row)
    missing = sorted(matrix_ids - set(loaded))
    if missing:
        raise SystemExit(f"missing CSR rows for coverage execution: {missing}")
    return loaded


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


def _common_failure_reasons(
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


def _write_report(records: list[dict[str, Any]], summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# CSR Guarded Promotion Coverage Exec",
        "",
        f"- status: `{summary['status']}`",
        f"- scenarios: `{summary['num_scenarios']}`",
        f"- executed_gpu_scenarios: `{summary['executed_gpu_scenarios']}`",
        f"- guard_only_scenarios: `{summary['guard_only_scenarios']}`",
        f"- current_quality_gate_blocks: `{summary['current_quality_gate_blocks']}`",
        f"- fixture_learned_promotions: `{summary['fixture_learned_promotions']}`",
        f"- non_success_blocks: `{summary['non_success_blocks']}`",
        f"- runtime_exception_fallbacks: `{summary['runtime_exception_fallbacks']}`",
        f"- runtime_fallback_used_count: `{summary['runtime_fallback_used_count']}`",
        f"- current_runtime_selector_changed: `{summary['current_runtime_selector_changed']}`",
        f"- selected_solver_set: `{', '.join(summary['selected_solver_set'])}`",
        f"- max_final_relative_residual: `{summary['max_final_relative_residual']:.6g}`",
        f"- max_cpu_recomputed_relative_residual: "
        f"`{summary['max_cpu_recomputed_relative_residual']:.6g}`",
        f"- max_solution_relative_error: `{summary['max_solution_relative_error']:.6g}`",
        "",
        "| scenario | kind | matrix | source | solver | gpu | fallback | status | rel_res |",
        "|---|---|---|---|---|---|---|---|---:|",
    ]
    for row in records:
        residual = "" if row["trace"] is None else f"{float(row['trace']['final_residual_norm']):.6g}"
        lines.append(
            "| "
            f"{row['source_scenario_id']} | "
            f"{row['scenario_kind']} | "
            f"{row['matrix_id']} | "
            f"{row['runtime_selection_source']} | "
            f"{row['solver'] or ''} | "
            f"{row['requires_gpu_execution']} | "
            f"{row['runtime_guard_used_fallback']} | "
            f"{row['status']} | "
            f"{residual} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
