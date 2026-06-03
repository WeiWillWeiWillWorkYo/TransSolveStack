"""Exercise guarded public auto_solve_csr on real CSR selector fixtures."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import transsolvestack as tss
from transsolvestack.core.types import SolveContext
from transsolvestack.datasets.csr import CsrMatrix, csr_matrix_from_record
from transsolvestack.profiling.artifacts import read_jsonl, write_jsonl
from transsolvestack.profiling.provenance import (
    CORE_CSR_GUARDED_AUTO_SOLVE_PROVENANCE_FILES,
    build_artifact_manifest,
    git_commit_or_unknown,
    write_manifest,
)
from transsolvestack.profiling.trace import trace_to_record


SELECTED_MATRIX_IDS = (
    "suitesparse:FIDAP/ex5",
    "suitesparse:HB/curtis54",
)


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
        "--learned-model-artifact",
        default="runs/phase1_csr_policy_model_artifact/csr_policy_model_artifact.json",
    )
    parser.add_argument(
        "--learned-model",
        default="runs/phase1_csr_transformer_ranker/csr_transformer_ranker_model.json",
    )
    parser.add_argument(
        "--learned-tensors",
        default="runs/phase1_csr_transformer_ready/csr_transformer_training_tensors.json",
    )
    parser.add_argument(
        "--learned-request-index",
        default="runs/phase1_csr_transformer_ready/csr_transformer_request_index.jsonl",
    )
    parser.add_argument(
        "--quality-gate-summary",
        default="runs/phase1_csr_transformer_quality_gate/csr_transformer_quality_gate_summary.json",
    )
    parser.add_argument("--out", default="runs/phase1_csr_guarded_auto_solve")
    parser.add_argument("--device-memory-gb", type=float, default=0.25)
    args = parser.parse_args()

    rows = {str(row["matrix_id"]): row for row in read_jsonl(args.csr)}
    missing = [matrix_id for matrix_id in SELECTED_MATRIX_IDS if matrix_id not in rows]
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
    result_records = []
    for matrix_id in SELECTED_MATRIX_IDS:
        csr = csr_matrix_from_record(rows[matrix_id])
        rhs = csr.matvec(tuple(1.0 for _ in range(csr.n_cols)))
        result = tss.auto_solve_csr_guarded(
            csr,
            rhs,
            selector_path=args.selector_rows,
            learned_predictions_path=args.learned_predictions,
            learned_model_artifact_path=args.learned_model_artifact,
            quality_gate_summary_path=args.quality_gate_summary,
            context=context,
            learned_policy_mode="promote_if_safe",
            device_memory_gb=args.device_memory_gb,
        )
        result_records.append(
            _result_record(
                csr,
                result,
                rhs,
                context=context,
            )
        )

    summary = _build_summary(
        result_records,
        selector_rows_path=args.selector_rows,
        learned_predictions_path=args.learned_predictions,
        learned_model_artifact_path=args.learned_model_artifact,
        learned_model_path=args.learned_model,
        learned_tensor_path=args.learned_tensors,
        learned_request_index_path=args.learned_request_index,
        quality_gate_summary_path=args.quality_gate_summary,
        source_csr_path=args.csr,
        device_memory_gb=args.device_memory_gb,
    )
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "results": output / "csr_guarded_auto_solve_results.jsonl",
        "summary": output / "csr_guarded_auto_solve_summary.json",
        "schema": output / "csr_guarded_auto_solve_schema.json",
        "report": output / "csr_guarded_auto_solve_report.md",
        "manifest": output / "artifact_manifest.json",
    }
    write_jsonl(result_records, paths["results"])
    paths["summary"].write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths["schema"].write_text(
        json.dumps(_schema(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_report(result_records, summary, paths["report"])
    write_manifest(
        build_artifact_manifest(
            artifact_kind="csr_guarded_auto_solve_smoke",
            command="scripts/tss_csr_guarded_auto_solve_smoke.py",
            tracked_files=CORE_CSR_GUARDED_AUTO_SOLVE_PROVENANCE_FILES,
            metadata={
                "git_commit": git_commit_or_unknown(),
                "status": summary["status"],
                "num_solves": summary["num_solves"],
                "num_success": summary["num_success"],
                "num_failed": summary["num_failed"],
                "learned_prediction_source": summary["learned_prediction_source"],
                "saved_model_loaded_count": summary["saved_model_loaded_count"],
                "learned_runtime_promotions": summary["learned_runtime_promotions"],
                "quality_gate_blocks": summary["quality_gate_blocks"],
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


def _result_record(
    csr: CsrMatrix,
    result,
    rhs: tuple[float, ...],
    *,
    context: SolveContext,
) -> dict:
    trace = trace_to_record(result.trace)
    learned_guard = dict(result.metadata["learned_policy_guard"])
    learned_policy_source = dict(learned_guard["learned_policy_source"])
    runtime_guard = dict(result.metadata["runtime_guard"])
    solution = tuple(float(value) for value in result.solution)
    cpu_residual = _relative_residual(csr, solution, rhs)
    solution_error = _relative_error_to_ones(solution)
    failure_reasons: list[str] = []
    if result.status != "success":
        failure_reasons.append(result.trace.failure_class or "solver_failed")
    if float(result.trace.final_residual_norm or math.inf) > context.tolerance_rel:
        failure_reasons.append("trace_residual_above_tolerance")
    if cpu_residual > 1.0e-4:
        failure_reasons.append("cpu_recomputed_residual_above_tolerance")
    if solution_error > 5.0e-3:
        failure_reasons.append("solution_error_above_tolerance")
    if learned_guard["runtime_selection_source"] != "artifact":
        failure_reasons.append("unexpected_learned_runtime_promotion")
    if learned_guard["guard_status"] != "blocked_quality_gate":
        failure_reasons.append("learned_guard_not_blocked_by_quality_gate")
    if runtime_guard["guard_status"] != "success":
        failure_reasons.append("runtime_guard_not_success")
    if runtime_guard["used_fallback"] is not False:
        failure_reasons.append("unexpected_runtime_fallback")
    return {
        "matrix_id": csr.matrix_id,
        "candidate_id": learned_guard["runtime_candidate_id"],
        "artifact_candidate_id": learned_guard["artifact_candidate_id"],
        "runtime_selection_source": learned_guard["runtime_selection_source"],
        "learned_policy_source_kind": learned_policy_source["source_kind"],
        "learned_model_loaded": learned_policy_source["model_loaded"],
        "learned_policy_source": learned_policy_source,
        "learned_guard_status": learned_guard["guard_status"],
        "learned_guard_reasons": tuple(learned_guard["guard_reasons"]),
        "learned_selected_candidate_id": (
            None
            if learned_guard["learned_prediction"] is None
            else learned_guard["learned_prediction"]["selected_candidate_id"]
        ),
        "runtime_guard_status": runtime_guard["guard_status"],
        "runtime_guard_used_fallback": runtime_guard["used_fallback"],
        "runtime_guard_attempt_count": len(runtime_guard["attempts"]),
        "fallback_chain_enforced": learned_guard["fallback_chain_enforced"],
        "solver": trace["metadata"]["solver"],
        "preconditioner": trace["metadata"]["preconditioner"],
        "precision": trace["metadata"]["precision"],
        "status": "success" if not failure_reasons else "failed",
        "failure_reasons": tuple(failure_reasons),
        "trace": trace,
        "cpu_recomputed_relative_residual": cpu_residual,
        "solution_relative_error": solution_error,
    }


def _build_summary(
    records: list[dict],
    *,
    selector_rows_path: str,
    learned_predictions_path: str,
    learned_model_artifact_path: str,
    learned_model_path: str,
    learned_tensor_path: str,
    learned_request_index_path: str,
    quality_gate_summary_path: str,
    source_csr_path: str,
    device_memory_gb: float,
) -> dict:
    num_success = sum(1 for row in records if row["status"] == "success")
    num_failed = len(records) - num_success
    selected_solver_set = sorted({row["solver"] for row in records})
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
    quality_gate_blocks = sum(
        1 for row in records if row["learned_guard_status"] == "blocked_quality_gate"
    )
    learned_runtime_promotions = sum(
        1 for row in records if row["runtime_selection_source"] == "learned"
    )
    saved_model_loaded_count = sum(
        1
        for row in records
        if row["learned_policy_source_kind"] == "model_artifact"
        and row["learned_model_loaded"] is True
    )
    status = (
        "passed"
        if len(records) == 2
        and num_failed == 0
        and saved_model_loaded_count == 2
        and quality_gate_blocks == 2
        and learned_runtime_promotions == 0
        and all(row["runtime_selection_source"] == "artifact" for row in records)
        and all(row["fallback_chain_enforced"] is True for row in records)
        and all(row["runtime_guard_status"] == "success" for row in records)
        and all(row["runtime_guard_used_fallback"] is False for row in records)
        and all(row["trace"]["backend"] == "taichi_gpu" for row in records)
        and max_final_residual <= 1.0e-5
        and max_cpu_residual <= 1.0e-4
        and max_solution_error <= 5.0e-3
        else "failed"
    )
    return {
        "status": status,
        "schema_version": "phase1_csr_guarded_auto_solve_v1",
        "selector_rows_path": selector_rows_path,
        "learned_predictions_path": learned_predictions_path,
        "learned_model_artifact_path": learned_model_artifact_path,
        "learned_model_path": learned_model_path,
        "learned_tensor_path": learned_tensor_path,
        "learned_request_index_path": learned_request_index_path,
        "learned_prediction_source": "model_artifact",
        "saved_model_loaded_count": saved_model_loaded_count,
        "quality_gate_summary_path": quality_gate_summary_path,
        "source_csr_path": source_csr_path,
        "device_memory_gb": device_memory_gb,
        "num_solves": len(records),
        "num_success": num_success,
        "num_failed": num_failed,
        "selected_solver_set": selected_solver_set,
        "quality_gate_blocks": quality_gate_blocks,
        "learned_runtime_promotions": learned_runtime_promotions,
        "runtime_selector_changed": learned_runtime_promotions > 0,
        "fallback_chain_enforced_count": sum(
            1 for row in records if row["fallback_chain_enforced"] is True
        ),
        "runtime_guard_success_count": sum(
            1 for row in records if row["runtime_guard_status"] == "success"
        ),
        "runtime_fallback_used_count": sum(
            1 for row in records if row["runtime_guard_used_fallback"] is True
        ),
        "max_final_relative_residual": max_final_residual,
        "max_cpu_recomputed_relative_residual": max_cpu_residual,
        "max_solution_relative_error": max_solution_error,
    }


def _schema() -> dict:
    return {
        "schema_version": "phase1_csr_guarded_auto_solve_v1",
        "task": "real_gpu_csr_auto_solve_through_learned_policy_runtime_guard",
        "runtime": {
            "backend": "taichi_gpu",
            "learned_policy_mode": "promote_if_safe",
            "learned_prediction_source": "model_artifact",
            "expected_current_guard_status": "blocked_quality_gate",
            "expected_runtime_selection_source": "artifact",
        },
        "pass_conditions": [
            "all solves succeed numerically",
            "learned guard blocks current ranker by quality gate",
            "no learned runtime promotion occurs",
            "fallback chain is enforced",
            "runtime guard records success",
        ],
    }


def _write_report(records: list[dict], summary: dict, path: Path) -> Path:
    lines = [
        "# CSR Guarded Auto Solve Smoke",
        "",
        f"- status: `{summary['status']}`",
        f"- solves: `{summary['num_solves']}`",
        f"- successes: `{summary['num_success']}`",
        f"- quality_gate_blocks: `{summary['quality_gate_blocks']}`",
        f"- learned_prediction_source: `{summary['learned_prediction_source']}`",
        f"- saved_model_loaded_count: `{summary['saved_model_loaded_count']}`",
        f"- learned_runtime_promotions: `{summary['learned_runtime_promotions']}`",
        f"- runtime_selector_changed: `{summary['runtime_selector_changed']}`",
        f"- selected_solver_set: `{', '.join(summary['selected_solver_set'])}`",
        f"- max_final_relative_residual: `{summary['max_final_relative_residual']:.6g}`",
        f"- max_cpu_recomputed_relative_residual: "
        f"`{summary['max_cpu_recomputed_relative_residual']:.6g}`",
        f"- max_solution_relative_error: `{summary['max_solution_relative_error']:.6g}`",
        "",
        "| matrix | runtime candidate | learned candidate | guard | solver | status | rel_res |",
        "|---|---|---|---|---|---|---:|",
    ]
    for row in records:
        lines.append(
            "| "
            f"{row['matrix_id']} | "
            f"{row['candidate_id']} | "
            f"{row['learned_selected_candidate_id'] or ''} | "
            f"{row['learned_guard_status']} | "
            f"{row['solver']} | "
            f"{row['status']} | "
            f"{float(row['trace']['final_residual_norm']):.6g} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


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
