"""Integrate ILU0 success evidence into guarded CSR selector fixtures."""

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
    CORE_CSR_ILU0_GUARDED_INTEGRATION_PROVENANCE_FILES,
    build_artifact_manifest,
    git_commit_or_unknown,
    write_manifest,
)
from transsolvestack.profiling.trace import trace_to_record


SCHEMA_VERSION = "phase1_csr_ilu0_guarded_integration_v1"
SELECTOR_SCHEMA_VERSION = "phase1_csr_selector_features_v8"
FIXTURE_MODEL_ID = "m69_fixture_ilu0_ranker"
CONTEXT_ID = "phase1_taichi_csr_ilu0"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ilu0-results",
        default="runs/phase1_taichi_csr_ilu0/csr_ilu0_results.jsonl",
    )
    parser.add_argument(
        "--base-selector-rows",
        default="runs/phase1_csr_transformer_ready/combined_csr_selector_rows.jsonl",
    )
    parser.add_argument(
        "--csr-records",
        action="append",
        default=[
            "runs/phase1_suitesparse_csr_import/csr_matrices.jsonl",
            "runs/phase1_csr_micro_campaign/csr_matrices.jsonl",
        ],
    )
    parser.add_argument("--device-memory-gb", type=float, default=0.25)
    parser.add_argument(
        "--out",
        default="runs/phase1_csr_ilu0_guarded_integration",
    )
    args = parser.parse_args()

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    ilu0_rows = list(read_jsonl(args.ilu0_results))
    if len(ilu0_rows) < 2:
        raise SystemExit("expected at least two ILU0 rows")
    base_selector_rows = list(read_jsonl(args.base_selector_rows))
    csr_by_id = _load_required_csr(tuple(args.csr_records), ilu0_rows)
    selector_rows = _ilu0_selector_rows(ilu0_rows, base_selector_rows)
    fixture_predictions = _fixture_predictions(selector_rows)
    fixture_quality_gate = _fixture_quality_gate()

    paths = {
        "selector_rows": output / "csr_ilu0_guarded_selector_rows.jsonl",
        "predictions": output / "fixture_ilu0_predictions.jsonl",
        "quality_gate": output / "fixture_runtime_eligible_quality_gate_summary.json",
        "results": output / "csr_ilu0_guarded_integration_results.jsonl",
        "summary": output / "csr_ilu0_guarded_integration_summary.json",
        "schema": output / "csr_ilu0_guarded_integration_schema.json",
        "report": output / "csr_ilu0_guarded_integration_report.md",
        "manifest": output / "artifact_manifest.json",
    }
    write_jsonl(selector_rows, paths["selector_rows"])
    write_jsonl(fixture_predictions, paths["predictions"])
    paths["quality_gate"].write_text(
        json.dumps(fixture_quality_gate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    records = []
    for row in selector_rows:
        csr = csr_by_id[str(row["matrix_id"])]
        if row["target_status"] == "success":
            records.append(
                _promoted_gpu_record(
                    selector_row=row,
                    csr=csr,
                    selector_rows_path=str(paths["selector_rows"]),
                    learned_predictions_path=str(paths["predictions"]),
                    quality_gate_summary_path=str(paths["quality_gate"]),
                    device_memory_gb=float(args.device_memory_gb),
                )
            )
        else:
            records.append(
                _blocked_failed_numeric_gate_record(
                    selector_row=row,
                    csr=csr,
                    selector_rows_path=str(paths["selector_rows"]),
                    learned_predictions_path=str(paths["predictions"]),
                    quality_gate_summary_path=str(paths["quality_gate"]),
                )
            )

    summary = _summary(
        records,
        selector_rows=selector_rows,
        ilu0_rows=ilu0_rows,
        ilu0_results_path=args.ilu0_results,
        base_selector_rows_path=args.base_selector_rows,
        selector_rows_path=str(paths["selector_rows"]),
        learned_predictions_path=str(paths["predictions"]),
        quality_gate_summary_path=str(paths["quality_gate"]),
        csr_record_paths=tuple(args.csr_records),
        device_memory_gb=float(args.device_memory_gb),
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
            artifact_kind="csr_ilu0_guarded_integration",
            command="scripts/tss_csr_ilu0_guarded_integration.py",
            tracked_files=CORE_CSR_ILU0_GUARDED_INTEGRATION_PROVENANCE_FILES,
            metadata={
                "git_commit": git_commit_or_unknown(),
                "status": summary["status"],
                "schema_version": summary["schema_version"],
                "ilu0_selector_rows": summary["ilu0_selector_rows"],
                "ilu0_success_rows_imported": summary["ilu0_success_rows_imported"],
                "failed_numeric_gate_rows_blocked": summary[
                    "failed_numeric_gate_rows_blocked"
                ],
                "executed_gpu_scenarios": summary["executed_gpu_scenarios"],
                "production_runtime_selector_changed": summary[
                    "production_runtime_selector_changed"
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


def _ilu0_selector_rows(
    ilu0_rows: list[dict[str, Any]],
    base_selector_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    base_features_by_matrix = {
        str(row["matrix_id"]): dict(row.get("features", {}))
        for row in base_selector_rows
    }
    rows = []
    success_rows = [row for row in ilu0_rows if row["numeric_status"] == "success"]
    for row in sorted(ilu0_rows, key=lambda item: str(item["matrix_id"])):
        success = row["numeric_status"] == "success" and row["candidate_promoted"] is True
        features = dict(base_features_by_matrix.get(str(row["matrix_id"]), {}))
        features.update(
            {
                "candidate_id": row["candidate_id"],
                "solver": row["solver"],
                "preconditioner": row["preconditioner"],
                "precision": row["precision"],
                "applicability_status": "applicable",
                "applicability_reason": None,
                "skip_reason": None,
                "screened_out_source": None,
                "success_rate": 1.0 if success else 0.0,
                "measurement_repeats": 1,
                "median_solve_time_ms": row["solve_time_ms"] if success else None,
                "solve_time_iqr_ms": 0.0,
                "failure_reason": None if success else _failure_reason(row),
                "solver_parameters": {},
                "source_augmented_from": "phase1_taichi_csr_ilu0",
                "ilu0_numeric_status": row["numeric_status"],
                "ilu0_min_abs_pivot": row.get("ilu0_min_abs_pivot"),
                "ilu0_pivot_tolerance": row.get("pivot_tolerance"),
                "ilu0_candidate_promoted": bool(row["candidate_promoted"]),
            }
        )
        rows.append(
            {
                "schema_version": SELECTOR_SCHEMA_VERSION,
                "matrix_id": row["matrix_id"],
                "context_id": CONTEXT_ID,
                "candidate_id": row["candidate_id"],
                "solver": row["solver"],
                "preconditioner": row["preconditioner"],
                "precision": row["precision"],
                "solver_parameters": {},
                "label_is_oracle": success and len(success_rows) == 1,
                "target_status": "success" if success else "failed_numeric_gate",
                "target_success_rate": 1.0 if success else 0.0,
                "target_measurement_repeats": 1,
                "target_solve_time_ms": row["solve_time_ms"] if success else None,
                "target_median_solve_time_ms": row["solve_time_ms"] if success else None,
                "target_solve_time_iqr_ms": 0.0,
                "target_wall_time_ms": row.get("wall_time_ms") if success else None,
                "target_regret_vs_oracle_ms": 0.0 if success else None,
                "target_num_iterations": row.get("num_iterations"),
                "target_final_relative_residual": row.get("final_relative_residual"),
                "target_cpu_recomputed_relative_residual": row.get(
                    "cpu_recomputed_relative_residual"
                ),
                "target_solution_relative_error": row.get("solution_relative_error"),
                "target_failure_reason": None if success else _failure_reason(row),
                "target_applicability_status": "applicable",
                "target_applicability_reason": None,
                "features": features,
            }
        )
    return rows


def _fixture_predictions(selector_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    predictions = []
    for row in selector_rows:
        selected = str(row["candidate_id"])
        scores = {selected: 8.0, "artifact_reference": -8.0}
        predictions.append(
            {
                "schema_version": "phase1_csr_transformer_ranker_v1",
                "model_id": FIXTURE_MODEL_ID,
                "request_id": f"m69:{row['matrix_id']}:{row['context_id']}",
                "split": "eval",
                "matrix_id": row["matrix_id"],
                "context_id": row["context_id"],
                "selected_candidate_id": selected,
                "selected_target_status": row["target_status"],
                "selected_label_class": row["target_status"],
                "selected_score": scores[selected],
                "evaluation_status": (
                    "oracle_match"
                    if row["label_is_oracle"]
                    else "failed_numeric_gate_selected"
                ),
                "oracle_candidate_id": selected if row["label_is_oracle"] else None,
                "oracle_rank": 1 if row["label_is_oracle"] else None,
                "regret_vs_oracle_ms": 0.0 if row["label_is_oracle"] else None,
                "ranked_candidate_ids": [selected, "artifact_reference"],
                "scores": scores,
            }
        )
    return predictions


def _fixture_quality_gate() -> dict[str, Any]:
    return {
        "status": "passed",
        "schema_version": "phase1_csr_selector_model_eval_v1",
        "evaluation_id": "csr_selector_model_quality_gate_v1",
        "baseline_model_id": "m69_fixture_baseline",
        "challenger_model_id": FIXTURE_MODEL_ID,
        "best_offline_model_id": FIXTURE_MODEL_ID,
        "runtime_selected_model_id": FIXTURE_MODEL_ID,
        "runtime_selector_changed": False,
        "challenger_beats_baseline": True,
        "challenger_runtime_eligible": True,
        "challenger_gate_failures": [],
        "num_eval_predictions": 2,
        "num_eval_oracle_requests": 1,
    }


def _promoted_gpu_record(
    *,
    selector_row: dict[str, Any],
    csr: CsrMatrix,
    selector_rows_path: str,
    learned_predictions_path: str,
    quality_gate_summary_path: str,
    device_memory_gb: float,
) -> dict[str, Any]:
    context = _context()
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
    failure_reasons = _numeric_failure_reasons(
        result_status=str(result.status),
        trace=trace,
        cpu_residual=cpu_residual,
        solution_error=solution_error,
        context=context,
    )
    if learned_guard["guard_status"] != "promoted":
        failure_reasons.append("ilu0_success_not_promoted_by_fixture_guard")
    if learned_guard["runtime_selection_source"] != "learned":
        failure_reasons.append("ilu0_fixture_not_selected_from_learned_source")
    if learned_guard["runtime_candidate_id"] != selector_row["candidate_id"]:
        failure_reasons.append("ilu0_promoted_candidate_mismatch")
    if runtime_guard["guard_status"] != "success":
        failure_reasons.append("runtime_guard_not_success")
    if runtime_guard["used_fallback"] is not False:
        failure_reasons.append("unexpected_runtime_fallback")
    return _record(
        selector_row=selector_row,
        status="success" if not failure_reasons else "failed",
        failure_reasons=failure_reasons,
        executed_gpu=True,
        guard_status=learned_guard["guard_status"],
        guard_reasons=tuple(learned_guard["guard_reasons"]),
        runtime_selection_source=learned_guard["runtime_selection_source"],
        runtime_selector_changed=learned_guard["runtime_selector_changed"],
        learned_selected_candidate_id=(
            None
            if learned_guard["learned_prediction"] is None
            else learned_guard["learned_prediction"]["selected_candidate_id"]
        ),
        learned_selected_target_status=(
            None
            if learned_guard["learned_prediction"] is None
            else learned_guard["learned_prediction"]["selected_target_status"]
        ),
        runtime_candidate_id=learned_guard["runtime_candidate_id"],
        artifact_candidate_id=learned_guard["artifact_candidate_id"],
        fallback_candidate_ids=tuple(learned_guard["fallback_candidate_ids"]),
        runtime_guard_status=runtime_guard["guard_status"],
        runtime_guard_used_fallback=runtime_guard["used_fallback"],
        runtime_guard_attempt_count=len(runtime_guard["attempts"]),
        runtime_guard_attempts=runtime_guard["attempts"],
        trace=trace,
        cpu_recomputed_relative_residual=cpu_residual,
        solution_relative_error=solution_error,
        final_result_status=str(result.status),
        solver=trace["metadata"]["solver"],
        preconditioner=trace["metadata"]["preconditioner"],
        precision=trace["metadata"]["precision"],
    )


def _blocked_failed_numeric_gate_record(
    *,
    selector_row: dict[str, Any],
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
        context=_context(),
        mode="promote_if_safe",
        min_confidence=0.75,
    )
    failure_reasons: list[str] = []
    if decision.guard_status != "blocked_non_success_candidate":
        failure_reasons.append("failed_numeric_gate_candidate_not_blocked")
    if not any("not a profiled success" in reason for reason in decision.guard_reasons):
        failure_reasons.append("missing_profiled_success_guard_reason")
    if decision.runtime_selector_changed is not False:
        failure_reasons.append("failed_numeric_gate_changed_runtime_selector")
    if decision.learned_prediction is None:
        failure_reasons.append("missing_fixture_prediction")
    elif decision.learned_prediction.selected_target_status != "failed_numeric_gate":
        failure_reasons.append("fixture_prediction_target_status_mismatch")
    return _record(
        selector_row=selector_row,
        status="success" if not failure_reasons else "failed",
        failure_reasons=failure_reasons,
        executed_gpu=False,
        guard_status=decision.guard_status,
        guard_reasons=tuple(decision.guard_reasons),
        runtime_selection_source=decision.runtime_selection_source,
        runtime_selector_changed=decision.runtime_selector_changed,
        learned_selected_candidate_id=(
            None
            if decision.learned_prediction is None
            else decision.learned_prediction.selected_candidate_id
        ),
        learned_selected_target_status=(
            None
            if decision.learned_prediction is None
            else decision.learned_prediction.selected_target_status
        ),
        runtime_candidate_id=decision.runtime_candidate_id,
        artifact_candidate_id=decision.artifact_candidate_id,
        fallback_candidate_ids=tuple(decision.fallback_candidate_ids),
        runtime_guard_status="not_executed",
        runtime_guard_used_fallback=False,
        runtime_guard_attempt_count=0,
        runtime_guard_attempts=(),
        trace=None,
        cpu_recomputed_relative_residual=None,
        solution_relative_error=None,
        final_result_status="guard_plan_only_failed_numeric_gate",
        solver=None,
        preconditioner=None,
        precision=None,
    )


def _record(
    *,
    selector_row: dict[str, Any],
    status: str,
    failure_reasons: list[str],
    executed_gpu: bool,
    guard_status: str,
    guard_reasons: tuple[str, ...],
    runtime_selection_source: str,
    runtime_selector_changed: bool,
    learned_selected_candidate_id: str | None,
    learned_selected_target_status: str | None,
    runtime_candidate_id: str,
    artifact_candidate_id: str,
    fallback_candidate_ids: tuple[str, ...],
    runtime_guard_status: str,
    runtime_guard_used_fallback: bool,
    runtime_guard_attempt_count: int,
    runtime_guard_attempts: Any,
    trace: dict[str, Any] | None,
    cpu_recomputed_relative_residual: float | None,
    solution_relative_error: float | None,
    final_result_status: str,
    solver: str | None,
    preconditioner: str | None,
    precision: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "matrix_id": selector_row["matrix_id"],
        "context_id": selector_row["context_id"],
        "candidate_id": selector_row["candidate_id"],
        "selector_target_status": selector_row["target_status"],
        "selector_label_is_oracle": selector_row["label_is_oracle"],
        "learned_selected_candidate_id": learned_selected_candidate_id,
        "learned_selected_target_status": learned_selected_target_status,
        "executed_gpu": executed_gpu,
        "runtime_candidate_id": runtime_candidate_id,
        "artifact_candidate_id": artifact_candidate_id,
        "fallback_candidate_ids": fallback_candidate_ids,
        "runtime_selection_source": runtime_selection_source,
        "learned_guard_status": guard_status,
        "learned_guard_reasons": guard_reasons,
        "learned_runtime_selector_changed": runtime_selector_changed,
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
    selector_rows: list[dict[str, Any]],
    ilu0_rows: list[dict[str, Any]],
    ilu0_results_path: str,
    base_selector_rows_path: str,
    selector_rows_path: str,
    learned_predictions_path: str,
    quality_gate_summary_path: str,
    csr_record_paths: tuple[str, ...],
    device_memory_gb: float,
) -> dict[str, Any]:
    gpu_rows = [row for row in records if row["executed_gpu"]]
    blocked_rows = [
        row
        for row in records
        if row["learned_guard_status"] == "blocked_non_success_candidate"
    ]
    promoted_rows = [row for row in records if row["learned_guard_status"] == "promoted"]
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
        if len(records) == 2
        and len(selector_rows) == 2
        and len(promoted_rows) == 1
        and len(blocked_rows) == 1
        and len(gpu_rows) == 1
        and all(row["status"] == "success" for row in records)
        and all(row["preconditioner"] == "ilu0" for row in gpu_rows)
        and all(row["solver"] == "bicgstab" for row in gpu_rows)
        and max(residuals, default=math.inf) <= 1.0e-5
        and max(cpu_residuals, default=math.inf) <= 1.0e-4
        and max(solution_errors, default=math.inf) <= 5.0e-3
        else "failed"
    )
    failed_selector_rows = [
        row for row in selector_rows if row["target_status"] == "failed_numeric_gate"
    ]
    return {
        "status": status,
        "schema_version": SCHEMA_VERSION,
        "ilu0_results_path": ilu0_results_path,
        "base_selector_rows_path": base_selector_rows_path,
        "selector_rows_path": selector_rows_path,
        "learned_predictions_path": learned_predictions_path,
        "quality_gate_summary_path": quality_gate_summary_path,
        "csr_record_paths": csr_record_paths,
        "device_memory_gb": device_memory_gb,
        "production_runtime_selector_changed": False,
        "fixture_runtime_selector_changed_count": sum(
            1 for row in records if row["learned_runtime_selector_changed"] is True
        ),
        "ilu0_source_rows": len(ilu0_rows),
        "ilu0_selector_rows": len(selector_rows),
        "ilu0_success_rows_imported": sum(
            1 for row in selector_rows if row["target_status"] == "success"
        ),
        "ilu0_failed_numeric_gate_rows_imported": len(failed_selector_rows),
        "promoted_success_rows": len(promoted_rows),
        "failed_numeric_gate_rows_blocked": len(blocked_rows),
        "executed_gpu_scenarios": len(gpu_rows),
        "guard_only_scenarios": len(records) - len(gpu_rows),
        "by_matrix_guard_status": {
            row["matrix_id"]: row["learned_guard_status"] for row in records
        },
        "by_matrix_runtime_execution": {
            row["matrix_id"]: row["executed_gpu"] for row in records
        },
        "selected_solver_set": sorted(
            {str(row["solver"]) for row in gpu_rows if row["solver"] is not None}
        ),
        "selected_preconditioner_set": sorted(
            {
                str(row["preconditioner"])
                for row in gpu_rows
                if row["preconditioner"] is not None
            }
        ),
        "max_final_relative_residual": max(residuals, default=0.0),
        "max_cpu_recomputed_relative_residual": max(cpu_residuals, default=0.0),
        "max_solution_relative_error": max(solution_errors, default=0.0),
        "next_step": (
            "merge ILU0 selector rows into broader transformer training only "
            "after more exact matrix/context successes are collected"
        ),
    }


def _schema() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "task": "prove_ilu0_success_rows_are_guarded_selector_compatible",
        "source_schema": "phase1_taichi_csr_ilu0_v1",
        "selector_schema": SELECTOR_SCHEMA_VERSION,
        "safety_policy": {
            "production_runtime_selector_changed": False,
            "failed_numeric_gate_rows": "must remain guard-only",
            "gpu_execution": "only exact matrix/context target_status=success rows",
            "fixture_quality_gate": "used only to exercise guard promotion branch",
        },
    }


def _write_report(
    records: list[dict[str, Any]],
    summary: dict[str, Any],
    path: Path,
) -> None:
    lines = [
        "# CSR ILU0 Guarded Integration",
        "",
        f"- status: `{summary['status']}`",
        f"- ilu0_selector_rows: `{summary['ilu0_selector_rows']}`",
        f"- ilu0_success_rows_imported: `{summary['ilu0_success_rows_imported']}`",
        f"- ilu0_failed_numeric_gate_rows_imported: `{summary['ilu0_failed_numeric_gate_rows_imported']}`",
        f"- promoted_success_rows: `{summary['promoted_success_rows']}`",
        f"- failed_numeric_gate_rows_blocked: `{summary['failed_numeric_gate_rows_blocked']}`",
        f"- executed_gpu_scenarios: `{summary['executed_gpu_scenarios']}`",
        f"- production_runtime_selector_changed: `{summary['production_runtime_selector_changed']}`",
        "",
        "| matrix | target_status | guard | gpu | solver | preconditioner | rel_res | sol_err | failures |",
        "|---|---|---|---:|---|---|---:|---:|---|",
    ]
    for row in records:
        trace = row["trace"] or {}
        lines.append(
            "| "
            f"{row['matrix_id']} | "
            f"{row['selector_target_status']} | "
            f"{row['learned_guard_status']} | "
            f"{row['executed_gpu']} | "
            f"{row['solver']} | "
            f"{row['preconditioner']} | "
            f"{_fmt(trace.get('final_residual_norm'))} | "
            f"{_fmt(row['solution_relative_error'])} | "
            f"{','.join(row['failure_reasons']) or 'none'} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _load_required_csr(
    paths: tuple[str, ...],
    ilu0_rows: list[dict[str, Any]],
) -> dict[str, CsrMatrix]:
    required = {str(row["matrix_id"]) for row in ilu0_rows}
    loaded: dict[str, CsrMatrix] = {}
    for path in paths:
        for row in read_jsonl(path):
            matrix_id = str(row["matrix_id"])
            if matrix_id in required and matrix_id not in loaded:
                loaded[matrix_id] = csr_matrix_from_record(row)
    missing = sorted(required - set(loaded))
    if missing:
        raise SystemExit(f"missing CSR records for ILU0 guard integration: {missing}")
    return loaded


def _context() -> SolveContext:
    return SolveContext(
        context_id=CONTEXT_ID,
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
    cpu_residual: float,
    solution_error: float,
    context: SolveContext,
) -> list[str]:
    reasons: list[str] = []
    if result_status not in {"success", "fallback_success"}:
        reasons.append("runtime_status_not_success")
    if str(trace.get("backend")) != "taichi_gpu":
        reasons.append("backend_not_taichi_gpu")
    if trace.get("metadata", {}).get("preconditioner") != "ilu0":
        reasons.append("preconditioner_not_ilu0")
    if trace.get("metadata", {}).get("solver") != "bicgstab":
        reasons.append("solver_not_bicgstab")
    if float(trace.get("final_residual_norm", math.inf)) > context.tolerance_rel:
        reasons.append("trace_residual_above_tolerance")
    if cpu_residual > 1.0e-4:
        reasons.append("cpu_recomputed_residual_above_tolerance")
    if solution_error > 5.0e-3:
        reasons.append("solution_error_above_tolerance")
    return reasons


def _relative_residual(
    csr: CsrMatrix,
    solution: tuple[float, ...],
    rhs: tuple[float, ...],
) -> float:
    ax = csr.matvec(solution)
    residual_sq = sum((float(b) - float(a)) ** 2 for a, b in zip(ax, rhs))
    rhs_sq = sum(float(value) ** 2 for value in rhs)
    return math.sqrt(residual_sq) / max(math.sqrt(rhs_sq), 1.0e-30)


def _relative_error_to_ones(solution: tuple[float, ...]) -> float:
    error_sq = sum((float(value) - 1.0) ** 2 for value in solution)
    truth_sq = float(len(solution))
    return math.sqrt(error_sq) / max(math.sqrt(truth_sq), 1.0e-30)


def _failure_reason(row: dict[str, Any]) -> str:
    reasons = tuple(str(item) for item in row.get("failure_reasons", ()))
    return reasons[0] if reasons else str(row.get("numeric_status", "failed_numeric_gate"))


def _fmt(value: Any) -> str:
    if value is None:
        return "na"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "nan"
    if not math.isfinite(numeric):
        return str(numeric)
    return f"{numeric:.6g}"


if __name__ == "__main__":
    main()
