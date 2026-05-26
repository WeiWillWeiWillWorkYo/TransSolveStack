"""Exercise learned CSR guard promotion and fallback readiness on real GPU solves."""

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
    CORE_CSR_GUARDED_PROMOTION_READINESS_PROVENANCE_FILES,
    build_artifact_manifest,
    git_commit_or_unknown,
    write_manifest,
)
from transsolvestack.profiling.trace import trace_to_record
from transsolvestack.runtime.guarded_fallback import (
    RuntimeGuardConfig,
    execute_with_fallback_guard,
)


BLOCKED_MATRIX_ID = "suitesparse:FIDAP/ex5"
PROMOTION_MATRIX_ID = "suitesparse:FIDAP/ex5"
FALLBACK_MATRIX_ID = "suitesparse:HB/curtis54"
PROMOTED_CANDIDATE_ID = "taichi_csr_pcg_jacobi_float64"


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
    parser.add_argument(
        "--learned-predictions",
        default="runs/phase1_csr_transformer_ranker/csr_transformer_ranker_predictions.jsonl",
    )
    parser.add_argument(
        "--quality-gate-summary",
        default="runs/phase1_csr_transformer_quality_gate/csr_transformer_quality_gate_summary.json",
    )
    parser.add_argument(
        "--out",
        default="runs/phase1_csr_guarded_promotion_readiness",
    )
    parser.add_argument("--device-memory-gb", type=float, default=0.25)
    args = parser.parse_args()

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    fixture_paths = _write_fixture_policy_artifacts(output)
    csr_rows = {str(row["matrix_id"]): row for row in read_jsonl(args.csr)}
    required = {BLOCKED_MATRIX_ID, PROMOTION_MATRIX_ID, FALLBACK_MATRIX_ID}
    missing = sorted(required - set(csr_rows))
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
    blocked_csr = csr_matrix_from_record(csr_rows[BLOCKED_MATRIX_ID])
    promoted_csr = csr_matrix_from_record(csr_rows[PROMOTION_MATRIX_ID])
    fallback_csr = csr_matrix_from_record(csr_rows[FALLBACK_MATRIX_ID])

    records = [
        _guarded_auto_solve_record(
            scenario_id="current_ranker_blocked_by_quality_gate",
            scenario_kind="current_ranker_blocked",
            csr=blocked_csr,
            context=context,
            selector_rows_path=args.selector_rows,
            learned_predictions_path=args.learned_predictions,
            quality_gate_summary_path=args.quality_gate_summary,
            expected_guard_status="blocked_quality_gate",
            expected_runtime_source="artifact",
            device_memory_gb=args.device_memory_gb,
        ),
        _guarded_auto_solve_record(
            scenario_id="fixture_gate_promotes_profiled_success_candidate",
            scenario_kind="fixture_learned_promotion",
            csr=promoted_csr,
            context=context,
            selector_rows_path=args.selector_rows,
            learned_predictions_path=str(fixture_paths["predictions"]),
            quality_gate_summary_path=str(fixture_paths["quality_gate"]),
            expected_guard_status="promoted",
            expected_runtime_source="learned",
            device_memory_gb=args.device_memory_gb,
        ),
        _runtime_exception_fallback_record(
            csr=fallback_csr,
            context=context,
            selector_rows_path=args.selector_rows,
            device_memory_gb=args.device_memory_gb,
        ),
    ]
    summary = _build_summary(
        records,
        selector_rows_path=args.selector_rows,
        learned_predictions_path=args.learned_predictions,
        quality_gate_summary_path=args.quality_gate_summary,
        fixture_predictions_path=str(fixture_paths["predictions"]),
        fixture_quality_gate_path=str(fixture_paths["quality_gate"]),
        source_csr_path=args.csr,
        device_memory_gb=args.device_memory_gb,
    )
    paths = {
        "results": output / "csr_guarded_promotion_readiness_results.jsonl",
        "summary": output / "csr_guarded_promotion_readiness_summary.json",
        "schema": output / "csr_guarded_promotion_readiness_schema.json",
        "report": output / "csr_guarded_promotion_readiness_report.md",
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
            artifact_kind="csr_guarded_promotion_readiness",
            command="scripts/tss_csr_guarded_promotion_readiness.py",
            tracked_files=CORE_CSR_GUARDED_PROMOTION_READINESS_PROVENANCE_FILES,
            metadata={
                "git_commit": git_commit_or_unknown(),
                "status": summary["status"],
                "num_scenarios": summary["num_scenarios"],
                "num_success": summary["num_success"],
                "current_quality_gate_blocks": summary["current_quality_gate_blocks"],
                "fixture_learned_promotions": summary["fixture_learned_promotions"],
                "runtime_exception_fallbacks": summary["runtime_exception_fallbacks"],
                "runtime_fallback_used_count": summary["runtime_fallback_used_count"],
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


def _guarded_auto_solve_record(
    *,
    scenario_id: str,
    scenario_kind: str,
    csr: CsrMatrix,
    context: SolveContext,
    selector_rows_path: str,
    learned_predictions_path: str,
    quality_gate_summary_path: str,
    expected_guard_status: str,
    expected_runtime_source: str,
    device_memory_gb: float,
) -> dict[str, Any]:
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
        failure_reasons.append("unexpected_fallback_retry")
    if expected_runtime_source == "learned" and learned_guard["runtime_candidate_id"] != PROMOTED_CANDIDATE_ID:
        failure_reasons.append("unexpected_promoted_candidate")
    return {
        "schema_version": "phase1_csr_guarded_promotion_readiness_v1",
        "scenario_id": scenario_id,
        "scenario_kind": scenario_kind,
        "matrix_id": csr.matrix_id,
        "candidate_id": learned_guard["runtime_candidate_id"],
        "artifact_candidate_id": learned_guard["artifact_candidate_id"],
        "learned_selected_candidate_id": (
            None
            if learned_guard["learned_prediction"] is None
            else learned_guard["learned_prediction"]["selected_candidate_id"]
        ),
        "runtime_selection_source": learned_guard["runtime_selection_source"],
        "learned_guard_status": learned_guard["guard_status"],
        "learned_guard_reasons": tuple(learned_guard["guard_reasons"]),
        "learned_runtime_selector_changed": learned_guard["runtime_selector_changed"],
        "fallback_chain_enforced": learned_guard["fallback_chain_enforced"],
        "runtime_guard_status": runtime_guard["guard_status"],
        "runtime_guard_used_fallback": runtime_guard["used_fallback"],
        "runtime_guard_attempt_count": len(runtime_guard["attempts"]),
        "runtime_guard_attempts": runtime_guard["attempts"],
        "solver": trace["metadata"]["solver"],
        "preconditioner": trace["metadata"]["preconditioner"],
        "precision": trace["metadata"]["precision"],
        "final_result_status": result.status,
        "status": "success" if not failure_reasons else "failed",
        "failure_reasons": tuple(failure_reasons),
        "trace": trace,
        "cpu_recomputed_relative_residual": cpu_residual,
        "solution_relative_error": solution_error,
    }


def _runtime_exception_fallback_record(
    *,
    csr: CsrMatrix,
    context: SolveContext,
    selector_rows_path: str,
    device_memory_gb: float,
) -> dict[str, Any]:
    rhs = _ones_rhs(csr)
    fallback_selection = CsrArtifactPolicySelector.from_selector_rows(
        selector_rows_path
    ).select(csr, context_id=context.context_id)
    bad_plan = PolicyPlan(
        plan_id=f"m56_invalid_cg_primary:{csr.matrix_id}",
        backend="taichi_gpu",
        solver={"name": "cg", "precision": context.precision},
        preconditioner={"name": "none"},
        audit={
            "candidate_id": "m56_invalid_cg_primary",
            "selection_reason": "runtime_exception_fallback_fixture",
        },
    )

    def execute(plan: PolicyPlan):
        return tss.solve_csr(
            csr,
            rhs,
            context=context,
            solver=str(plan.solver["name"]),
            preconditioner=plan.preconditioner.get("name"),
            precision=plan.solver.get("precision", context.precision),
            solver_parameters={
                key: value
                for key, value in plan.solver.items()
                if key not in {"name", "precision"}
            },
            policy_plan=plan,
            device_memory_gb=device_memory_gb,
        )

    outcome = execute_with_fallback_guard(
        (bad_plan, fallback_selection.plan),
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
    attempts = runtime_guard["attempts"]
    if runtime_guard["guard_status"] != "success":
        failure_reasons.append("runtime_guard_not_success")
    if runtime_guard["used_fallback"] is not True:
        failure_reasons.append("runtime_fallback_not_used")
    if len(attempts) != 2:
        failure_reasons.append("runtime_attempt_count_mismatch")
    elif attempts[0]["status"] != "exception" or attempts[0]["exception_type"] != "ValueError":
        failure_reasons.append("first_attempt_not_value_error_exception")
    if str(result.status) != "fallback_success":
        failure_reasons.append("final_result_not_fallback_success")
    if trace["backend"] != "taichi_gpu":
        failure_reasons.append("fallback_result_not_taichi_gpu")
    return {
        "schema_version": "phase1_csr_guarded_promotion_readiness_v1",
        "scenario_id": "runtime_exception_retries_profiled_gpu_fallback",
        "scenario_kind": "runtime_exception_fallback",
        "matrix_id": csr.matrix_id,
        "candidate_id": fallback_selection.candidate_id,
        "artifact_candidate_id": fallback_selection.candidate_id,
        "learned_selected_candidate_id": None,
        "runtime_selection_source": "runtime_guard_fallback_fixture",
        "learned_guard_status": None,
        "learned_guard_reasons": (),
        "learned_runtime_selector_changed": False,
        "fallback_chain_enforced": True,
        "runtime_guard_status": runtime_guard["guard_status"],
        "runtime_guard_used_fallback": runtime_guard["used_fallback"],
        "runtime_guard_attempt_count": len(attempts),
        "runtime_guard_attempts": attempts,
        "solver": trace["metadata"]["solver"],
        "preconditioner": trace["metadata"]["preconditioner"],
        "precision": trace["metadata"]["precision"],
        "final_result_status": result.status,
        "status": "success" if not failure_reasons else "failed",
        "failure_reasons": tuple(failure_reasons),
        "trace": trace,
        "cpu_recomputed_relative_residual": cpu_residual,
        "solution_relative_error": solution_error,
    }


def _build_summary(
    records: list[dict[str, Any]],
    *,
    selector_rows_path: str,
    learned_predictions_path: str,
    quality_gate_summary_path: str,
    fixture_predictions_path: str,
    fixture_quality_gate_path: str,
    source_csr_path: str,
    device_memory_gb: float,
) -> dict[str, Any]:
    num_success = sum(1 for row in records if row["status"] == "success")
    num_failed = len(records) - num_success
    current_quality_gate_blocks = sum(
        1
        for row in records
        if row["scenario_kind"] == "current_ranker_blocked"
        and row["learned_guard_status"] == "blocked_quality_gate"
    )
    fixture_learned_promotions = sum(
        1
        for row in records
        if row["scenario_kind"] == "fixture_learned_promotion"
        and row["runtime_selection_source"] == "learned"
    )
    runtime_exception_fallbacks = sum(
        1
        for row in records
        if row["scenario_kind"] == "runtime_exception_fallback"
        and row["runtime_guard_used_fallback"] is True
    )
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
        if len(records) == 3
        and num_failed == 0
        and current_quality_gate_blocks == 1
        and fixture_learned_promotions == 1
        and runtime_exception_fallbacks == 1
        and all(row["trace"]["backend"] == "taichi_gpu" for row in records)
        and max_final_residual <= 1.0e-5
        and max_cpu_residual <= 1.0e-4
        and max_solution_error <= 5.0e-3
        else "failed"
    )
    return {
        "status": status,
        "schema_version": "phase1_csr_guarded_promotion_readiness_v1",
        "selector_rows_path": selector_rows_path,
        "learned_predictions_path": learned_predictions_path,
        "quality_gate_summary_path": quality_gate_summary_path,
        "fixture_predictions_path": fixture_predictions_path,
        "fixture_quality_gate_path": fixture_quality_gate_path,
        "source_csr_path": source_csr_path,
        "device_memory_gb": device_memory_gb,
        "num_scenarios": len(records),
        "num_success": num_success,
        "num_failed": num_failed,
        "current_quality_gate_blocks": current_quality_gate_blocks,
        "fixture_learned_promotions": fixture_learned_promotions,
        "fixture_runtime_selector_changed_count": sum(
            1 for row in records if row["learned_runtime_selector_changed"] is True
        ),
        "current_runtime_selector_changed": any(
            row["learned_runtime_selector_changed"] is True
            for row in records
            if row["scenario_kind"] == "current_ranker_blocked"
        ),
        "runtime_exception_fallbacks": runtime_exception_fallbacks,
        "runtime_guard_success_count": sum(
            1 for row in records if row["runtime_guard_status"] == "success"
        ),
        "runtime_fallback_used_count": sum(
            1 for row in records if row["runtime_guard_used_fallback"] is True
        ),
        "real_gpu_final_result_count": sum(
            1 for row in records if row["trace"]["backend"] == "taichi_gpu"
        ),
        "selected_solver_set": sorted({str(row["solver"]) for row in records}),
        "max_final_relative_residual": max_final_residual,
        "max_cpu_recomputed_relative_residual": max_cpu_residual,
        "max_solution_relative_error": max_solution_error,
    }


def _schema() -> dict[str, Any]:
    return {
        "schema_version": "phase1_csr_guarded_promotion_readiness_v1",
        "task": "learned_policy_runtime_promotion_readiness_matrix",
        "scenarios": [
            {
                "scenario_kind": "current_ranker_blocked",
                "expectation": "current Transformer ranker remains blocked by quality gate",
            },
            {
                "scenario_kind": "fixture_learned_promotion",
                "expectation": "runtime-eligible high-confidence profiled-success learned candidate promotes and solves on GPU",
            },
            {
                "scenario_kind": "runtime_exception_fallback",
                "expectation": "runtime exception triggers fallback retry and the fallback solves on GPU",
            },
        ],
        "runtime": {
            "backend": "taichi_gpu",
            "learned_policy_mode": "promote_if_safe",
            "production_runtime_selector_changed": False,
            "fixture_promotion_allowed": True,
        },
        "pass_conditions": [
            "current learned ranker is blocked by the real quality gate",
            "fixture runtime-eligible learned prediction promotes only when it is a profiled success",
            "runtime exception fallback reaches a profiled artifact GPU candidate",
            "all final solve results satisfy numeric residual and solution checks",
        ],
    }


def _write_report(records: list[dict[str, Any]], summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# CSR Guarded Promotion Readiness",
        "",
        f"- status: `{summary['status']}`",
        f"- scenarios: `{summary['num_scenarios']}`",
        f"- successes: `{summary['num_success']}`",
        f"- current_quality_gate_blocks: `{summary['current_quality_gate_blocks']}`",
        f"- fixture_learned_promotions: `{summary['fixture_learned_promotions']}`",
        f"- runtime_exception_fallbacks: `{summary['runtime_exception_fallbacks']}`",
        f"- runtime_fallback_used_count: `{summary['runtime_fallback_used_count']}`",
        f"- current_runtime_selector_changed: `{summary['current_runtime_selector_changed']}`",
        f"- selected_solver_set: `{', '.join(summary['selected_solver_set'])}`",
        f"- max_final_relative_residual: `{summary['max_final_relative_residual']:.6g}`",
        f"- max_cpu_recomputed_relative_residual: "
        f"`{summary['max_cpu_recomputed_relative_residual']:.6g}`",
        f"- max_solution_relative_error: `{summary['max_solution_relative_error']:.6g}`",
        "",
        "| scenario | matrix | source | guard | solver | fallback | status | rel_res |",
        "|---|---|---|---|---|---|---|---:|",
    ]
    for row in records:
        lines.append(
            "| "
            f"{row['scenario_id']} | "
            f"{row['matrix_id']} | "
            f"{row['runtime_selection_source']} | "
            f"{row['learned_guard_status'] or ''} | "
            f"{row['solver']} | "
            f"{row['runtime_guard_used_fallback']} | "
            f"{row['status']} | "
            f"{float(row['trace']['final_residual_norm']):.6g} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_fixture_policy_artifacts(output: Path) -> dict[str, Path]:
    predictions = output / "fixture_promotable_predictions.jsonl"
    quality_gate = output / "fixture_runtime_eligible_quality_gate_summary.json"
    write_jsonl((_fixture_prediction(),), predictions)
    quality_gate.write_text(
        json.dumps(_fixture_quality_gate(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"predictions": predictions, "quality_gate": quality_gate}


def _fixture_prediction() -> dict[str, Any]:
    return {
        "schema_version": "phase1_csr_transformer_ranker_v1",
        "model_id": "m56_fixture_promotable_ranker",
        "request_id": f"{PROMOTION_MATRIX_ID}:phase1_csr_selector:m56_fixture",
        "split": "eval",
        "matrix_id": PROMOTION_MATRIX_ID,
        "context_id": "phase1_csr_selector",
        "selected_candidate_id": PROMOTED_CANDIDATE_ID,
        "selected_target_status": "success",
        "selected_label_class": "success_non_oracle",
        "selected_score": 9.0,
        "evaluation_status": "profiled_success_non_oracle",
        "oracle_candidate_id": "taichi_csr_cg_none_float64",
        "oracle_rank": 2,
        "regret_vs_oracle_ms": 8.442404679954052,
        "ranked_candidate_ids": [
            PROMOTED_CANDIDATE_ID,
            "taichi_csr_cg_none_float64",
        ],
        "scores": {
            PROMOTED_CANDIDATE_ID: 9.0,
            "taichi_csr_cg_none_float64": 2.0,
        },
    }


def _fixture_quality_gate() -> dict[str, Any]:
    return {
        "status": "passed",
        "schema_version": "phase1_csr_selector_model_eval_v1",
        "evaluation_id": "csr_selector_model_quality_gate_v1",
        "baseline_model_id": "m56_fixture_baseline",
        "challenger_model_id": "m56_fixture_promotable_ranker",
        "best_offline_model_id": "m56_fixture_promotable_ranker",
        "runtime_selected_model_id": "m56_fixture_promotable_ranker",
        "runtime_selector_changed": False,
        "challenger_beats_baseline": True,
        "challenger_runtime_eligible": True,
        "challenger_gate_failures": [],
        "num_eval_predictions": 4,
        "num_eval_oracle_requests": 4,
    }


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
    if float(trace["final_residual_norm"] or math.inf) > context.tolerance_rel:
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


if __name__ == "__main__":
    main()
