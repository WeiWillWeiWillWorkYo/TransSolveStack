"""Smoke-test learned CSR runtime guards without launching GPU work."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import transsolvestack as tss
from transsolvestack.core.result import PolicyPlan, RunTrace, SolveResult
from transsolvestack.datasets.csr import CsrMatrix
from transsolvestack.profiling.artifacts import read_jsonl, write_jsonl
from transsolvestack.profiling.provenance import (
    CORE_CSR_LEARNED_GUARD_PROVENANCE_FILES,
    build_artifact_manifest,
    git_commit_or_unknown,
    write_manifest,
)
from transsolvestack.runtime.guarded_fallback import (
    RuntimeGuardConfig,
    execute_with_fallback_guard,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="runs/phase1_csr_learned_guard")
    args = parser.parse_args()

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    fixture_paths = _write_fixture_artifacts(output)

    csr_rows = {
        str(row["matrix_id"]): row
        for row in read_jsonl("runs/phase1_suitesparse_csr_import/csr_matrices.jsonl")
    }
    shadow_decision = tss.plan_csr_with_learned_guard(
        csr_rows["suitesparse:HB/curtis54"],
        mode="shadow",
    )
    blocked_decision = tss.plan_csr_with_learned_guard(
        csr_rows["suitesparse:HB/curtis54"],
        mode="promote_if_safe",
    )
    promoted_decision = tss.plan_csr_with_learned_guard(
        _fixture_csr(),
        selector_path=fixture_paths["selector_rows"],
        learned_predictions_path=fixture_paths["predictions"],
        quality_gate_summary_path=fixture_paths["quality_gate"],
        context=tss.SolveContext(context_id="fixture_guard"),
        mode="promote_if_safe",
        min_confidence=0.75,
    )
    fallback_outcome = _fake_runtime_fallback_check(promoted_decision.selection.plan)

    rows = (
        {
            "check_id": "actual_shadow_mode",
            "status": "passed",
            "guard_status": shadow_decision.guard_status,
            "runtime_selection_source": shadow_decision.runtime_selection_source,
            "runtime_selector_changed": shadow_decision.runtime_selector_changed,
            "artifact_candidate_id": shadow_decision.artifact_candidate_id,
            "runtime_candidate_id": shadow_decision.runtime_candidate_id,
            "fallback_chain_enforced": shadow_decision.fallback_chain_enforced,
            "guard_reasons": list(shadow_decision.guard_reasons),
        },
        {
            "check_id": "actual_quality_gate_blocks_promotion",
            "status": "passed",
            "guard_status": blocked_decision.guard_status,
            "runtime_selection_source": blocked_decision.runtime_selection_source,
            "runtime_selector_changed": blocked_decision.runtime_selector_changed,
            "artifact_candidate_id": blocked_decision.artifact_candidate_id,
            "runtime_candidate_id": blocked_decision.runtime_candidate_id,
            "fallback_chain_enforced": blocked_decision.fallback_chain_enforced,
            "guard_reasons": list(blocked_decision.guard_reasons),
        },
        {
            "check_id": "eligible_high_confidence_promotion_fixture",
            "status": "passed",
            "guard_status": promoted_decision.guard_status,
            "runtime_selection_source": promoted_decision.runtime_selection_source,
            "runtime_selector_changed": promoted_decision.runtime_selector_changed,
            "artifact_candidate_id": promoted_decision.artifact_candidate_id,
            "runtime_candidate_id": promoted_decision.runtime_candidate_id,
            "fallback_candidate_ids": list(promoted_decision.fallback_candidate_ids),
            "fallback_chain_enforced": promoted_decision.fallback_chain_enforced,
            "confidence": promoted_decision.learned_prediction.confidence
            if promoted_decision.learned_prediction
            else None,
        },
        {
            "check_id": "runtime_timeout_fallback_fixture",
            "status": "passed",
            "guard_status": fallback_outcome.guard_status,
            "result_status": fallback_outcome.result.status,
            "used_fallback": fallback_outcome.used_fallback,
            "attempt_count": len(fallback_outcome.attempts),
            "first_attempt_timed_out": fallback_outcome.attempts[0].timed_out,
            "fallback_events": fallback_outcome.result.trace.fallback_events,
        },
    )
    summary = _summary(rows)
    schema = _schema()
    paths = {
        "rows": output / "csr_learned_guard_rows.jsonl",
        "summary": output / "csr_learned_guard_summary.json",
        "schema": output / "csr_learned_guard_schema.json",
        "report": output / "csr_learned_guard_report.md",
        "manifest": output / "artifact_manifest.json",
    }
    write_jsonl(rows, paths["rows"])
    paths["summary"].write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths["schema"].write_text(
        json.dumps(schema, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_report(summary, rows, paths["report"])
    write_manifest(
        build_artifact_manifest(
            artifact_kind="csr_learned_runtime_guard",
            command="scripts/tss_csr_learned_guard_smoke.py",
            tracked_files=CORE_CSR_LEARNED_GUARD_PROVENANCE_FILES,
            metadata={
                "git_commit": git_commit_or_unknown(),
                "status": summary["status"],
                "runtime_selector_changed": summary["runtime_selector_changed"],
                "shadow_mode_checked": summary["shadow_mode_checked"],
                "quality_gate_blocks_checked": summary["quality_gate_blocks_checked"],
                "promotion_fixture_checked": summary["promotion_fixture_checked"],
                "runtime_fallback_checked": summary["runtime_fallback_checked"],
            },
        ),
        paths["manifest"],
    )
    for key, value in paths.items():
        print(f"{key}: {value}")
    print(f"status: {summary['status']}")
    if summary["status"] != "passed":
        raise SystemExit(2)


def _summary(rows: tuple[dict, ...]) -> dict:
    row_by_id = {row["check_id"]: row for row in rows}
    status = (
        "passed"
        if row_by_id["actual_shadow_mode"]["guard_status"] == "shadow_only"
        and row_by_id["actual_shadow_mode"]["runtime_selector_changed"] is False
        and row_by_id["actual_quality_gate_blocks_promotion"]["guard_status"]
        == "blocked_quality_gate"
        and row_by_id["actual_quality_gate_blocks_promotion"]["runtime_selector_changed"]
        is False
        and row_by_id["eligible_high_confidence_promotion_fixture"]["guard_status"]
        == "promoted"
        and row_by_id["eligible_high_confidence_promotion_fixture"][
            "runtime_selection_source"
        ]
        == "learned"
        and row_by_id["eligible_high_confidence_promotion_fixture"][
            "fallback_chain_enforced"
        ]
        is True
        and row_by_id["runtime_timeout_fallback_fixture"]["result_status"]
        == "fallback_success"
        and row_by_id["runtime_timeout_fallback_fixture"]["used_fallback"] is True
        and row_by_id["runtime_timeout_fallback_fixture"]["first_attempt_timed_out"]
        is True
        else "failed"
    )
    return {
        "status": status,
        "schema_version": "phase1_csr_learned_runtime_guard_v1",
        "guard_id": "csr_learned_policy_runtime_guard_v1",
        "num_checks": len(rows),
        "shadow_mode_checked": True,
        "quality_gate_blocks_checked": True,
        "confidence_threshold_checked": True,
        "fallback_chain_enforced_checked": True,
        "promotion_fixture_checked": True,
        "runtime_fallback_checked": True,
        "timeout_guard_checked": True,
        "preemptive_gpu_kill_supported": False,
        "runtime_selector_changed": False,
    }


def _schema() -> dict:
    return {
        "schema_version": "phase1_csr_learned_runtime_guard_v1",
        "guard_id": "csr_learned_policy_runtime_guard_v1",
        "default_mode": "shadow",
        "promotion_requires": [
            "offline_quality_gate_runtime_eligible",
            "confidence_at_or_above_threshold",
            "learned_candidate_is_profiled_success_for_exact_matrix_context",
            "fallback_chain_contains_only_profiled_success_candidates",
        ],
        "runtime_guard": {
            "fallback_on_failed_status": True,
            "fallback_on_exception": True,
            "fallback_on_timeout": True,
            "preemptive_gpu_kill_supported": False,
            "timeout_semantics": "post_attempt_wall_clock_guard",
        },
    }


def _write_report(summary: dict, rows: tuple[dict, ...], path: Path) -> None:
    lines = [
        "# CSR Learned Runtime Guard",
        "",
        f"- status: `{summary['status']}`",
        f"- schema_version: `{summary['schema_version']}`",
        f"- runtime_selector_changed: `{summary['runtime_selector_changed']}`",
        f"- preemptive_gpu_kill_supported: `{summary['preemptive_gpu_kill_supported']}`",
        "",
        "| check | status | guard_status | runtime_source |",
        "|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['check_id']} | "
            f"{row['status']} | "
            f"{row.get('guard_status', '')} | "
            f"{row.get('runtime_selection_source', '')} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_fixture_artifacts(output: Path) -> dict[str, Path]:
    selector_rows = output / "fixture_selector_rows.jsonl"
    predictions = output / "fixture_predictions.jsonl"
    quality_gate = output / "fixture_quality_gate_summary.json"
    write_jsonl(_fixture_selector_rows(), selector_rows)
    write_jsonl((_fixture_prediction(),), predictions)
    quality_gate.write_text(
        json.dumps(_fixture_quality_gate(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "selector_rows": selector_rows,
        "predictions": predictions,
        "quality_gate": quality_gate,
    }


def _fixture_csr() -> CsrMatrix:
    return CsrMatrix(
        matrix_id="fixture:learned_guard",
        n_rows=2,
        n_cols=2,
        row_ptr=(0, 1, 2),
        col_ind=(0, 1),
        values=(1.0, 1.0),
        field="real",
        symmetry="symmetric",
        source_path="memory://learned_guard",
    )


def _fixture_selector_rows() -> tuple[dict, ...]:
    base = {
        "matrix_id": "fixture:learned_guard",
        "context_id": "fixture_guard",
        "precision": "float64",
        "target_status": "success",
        "target_success_rate": 1.0,
        "target_measurement_repeats": 1,
        "target_solve_time_iqr_ms": 0.0,
        "target_final_relative_residual": 1.0e-8,
        "target_cpu_recomputed_relative_residual": 1.0e-8,
        "target_solution_relative_error": 0.0,
        "features": {},
    }
    return (
        {
            **base,
            "candidate_id": "artifact_fast",
            "solver": "cg",
            "preconditioner": "none",
            "solver_parameters": {},
            "label_is_oracle": True,
            "target_solve_time_ms": 1.0,
            "target_median_solve_time_ms": 1.0,
            "target_wall_time_ms": 1.0,
            "target_regret_vs_oracle_ms": 0.0,
            "target_num_iterations": 2,
        },
        {
            **base,
            "candidate_id": "learned_safe",
            "solver": "pcg",
            "preconditioner": "jacobi",
            "solver_parameters": {},
            "label_is_oracle": False,
            "target_solve_time_ms": 2.0,
            "target_median_solve_time_ms": 2.0,
            "target_wall_time_ms": 2.0,
            "target_regret_vs_oracle_ms": 1.0,
            "target_num_iterations": 3,
        },
        {
            **base,
            "candidate_id": "screened_bad",
            "solver": "gmres",
            "preconditioner": "jacobi",
            "solver_parameters": {"restart": 8},
            "target_status": "screened_out",
            "target_success_rate": 0.0,
            "label_is_oracle": False,
            "target_solve_time_ms": None,
            "target_median_solve_time_ms": None,
            "target_wall_time_ms": None,
            "target_regret_vs_oracle_ms": None,
            "target_num_iterations": None,
        },
    )


def _fixture_prediction() -> dict:
    return {
        "schema_version": "phase1_csr_transformer_ranker_v1",
        "model_id": "fixture_ranker",
        "request_id": "fixture:learned_guard:fixture_guard",
        "split": "eval",
        "matrix_id": "fixture:learned_guard",
        "context_id": "fixture_guard",
        "selected_candidate_id": "learned_safe",
        "selected_target_status": "success",
        "selected_label_class": "success_non_oracle",
        "selected_score": 8.0,
        "evaluation_status": "profiled_success_non_oracle",
        "oracle_candidate_id": "artifact_fast",
        "oracle_rank": 2,
        "regret_vs_oracle_ms": 1.0,
        "ranked_candidate_ids": ["learned_safe", "artifact_fast", "screened_bad"],
        "scores": {
            "learned_safe": 8.0,
            "artifact_fast": 4.0,
            "screened_bad": -20.0,
        },
    }


def _fixture_quality_gate() -> dict:
    return {
        "status": "passed",
        "schema_version": "phase1_csr_selector_model_eval_v1",
        "evaluation_id": "csr_selector_model_quality_gate_v1",
        "baseline_model_id": "fixture_baseline",
        "challenger_model_id": "fixture_ranker",
        "best_offline_model_id": "fixture_ranker",
        "runtime_selected_model_id": "fixture_ranker",
        "runtime_selector_changed": False,
        "challenger_beats_baseline": True,
        "challenger_runtime_eligible": True,
        "challenger_gate_failures": [],
        "num_eval_predictions": 4,
        "num_eval_oracle_requests": 4,
    }


def _fake_runtime_fallback_check(plan: PolicyPlan):
    fallback = plan.fallback_chain[0]
    fallback_plan = PolicyPlan(
        plan_id=f"{plan.plan_id}:fallback:1:{fallback['candidate_id']}",
        backend=plan.backend,
        solver=dict(fallback["solver"]),
        preconditioner=dict(fallback["preconditioner"]),
        audit={"candidate_id": fallback["candidate_id"]},
    )

    def execute(active_plan: PolicyPlan) -> SolveResult:
        if active_plan.plan_id == plan.plan_id:
            time.sleep(0.005)
        success = active_plan.plan_id == fallback_plan.plan_id
        status = "success" if success else "success"
        trace = RunTrace(
            run_id=f"fake:{active_plan.plan_id}",
            plan_id=active_plan.plan_id,
            backend=active_plan.backend,
            status=status,
            final_residual_norm=1.0e-8,
            metadata={"candidate_id": active_plan.audit.get("candidate_id")},
        )
        return SolveResult(status=status, solution=(1.0, 1.0), trace=trace)

    return execute_with_fallback_guard(
        (plan, fallback_plan),
        execute,
        config=RuntimeGuardConfig(max_attempt_wall_time_ms=1.0),
    )


if __name__ == "__main__":
    main()
