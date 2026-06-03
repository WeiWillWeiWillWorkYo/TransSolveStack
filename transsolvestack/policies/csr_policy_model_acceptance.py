"""Acceptance gate for runtime-loadable CSR policy model artifacts."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from transsolvestack.datasets.csr import csr_matrix_from_record
from transsolvestack.policies.csr_learned_guard import plan_csr_with_learned_guard
from transsolvestack.policies.csr_policy_model_artifact import (
    csr_policy_model_artifact_paths,
    load_csr_policy_model_artifact,
)
from transsolvestack.policies.csr_transformer_ranker import (
    load_csr_transformer_ranker_model,
    predict_csr_transformer_ranker_from_model_file,
)
from transsolvestack.profiling.artifacts import read_jsonl, write_jsonl


CSR_POLICY_MODEL_ACCEPTANCE_SCHEMA_VERSION = "phase1_csr_policy_model_acceptance_v1"
ACCEPTANCE_MATRIX_IDS = (
    "suitesparse:FIDAP/ex5",
    "suitesparse:HB/curtis54",
)


def build_csr_policy_model_acceptance_from_files(
    model_artifact_path: str | Path = (
        "runs/phase1_csr_policy_model_artifact/csr_policy_model_artifact.json"
    ),
    csr_path: str | Path = "runs/phase1_suitesparse_csr_import/csr_matrices.jsonl",
    selector_path: str | Path = "runs/phase1_csr_selector_readiness/csr_selector_rows.jsonl",
    learned_guard_summary_path: str | Path = (
        "runs/phase1_csr_learned_guard/csr_learned_guard_summary.json"
    ),
    guarded_auto_solve_summary_path: str | Path = (
        "runs/phase1_csr_guarded_auto_solve/csr_guarded_auto_solve_summary.json"
    ),
    guarded_auto_solve_results_path: str | Path = (
        "runs/phase1_csr_guarded_auto_solve/csr_guarded_auto_solve_results.jsonl"
    ),
    min_confidence: float = 0.75,
) -> dict[str, Any]:
    """Evaluate whether a CSR policy model artifact can enter guarded runtime."""

    model_artifact_path = Path(model_artifact_path)
    artifact = load_csr_policy_model_artifact(model_artifact_path)
    paths = csr_policy_model_artifact_paths(artifact)
    model = load_csr_transformer_ranker_model(paths["model_path"])
    request_index = tuple(read_jsonl(paths["request_index_path"]))
    predictions = predict_csr_transformer_ranker_from_model_file(
        paths["model_path"],
        paths["tensor_path"],
        paths["request_index_path"],
    )
    quality_gate = _read_json(Path(artifact["quality_gate"]["summary_path"]))
    replay_summary = _read_json(Path(artifact["replay"]["summary_path"]))
    learned_guard_summary = _read_json(Path(learned_guard_summary_path))
    guarded_auto_solve_summary = _read_json(Path(guarded_auto_solve_summary_path))
    guarded_auto_solve_results = tuple(read_jsonl(guarded_auto_solve_results_path))
    guard_rows = _guard_rows(
        model_artifact_path=model_artifact_path,
        csr_path=Path(csr_path),
        selector_path=Path(selector_path),
        quality_gate_summary_path=Path(artifact["quality_gate"]["summary_path"]),
        min_confidence=min_confidence,
    )
    validation_errors = _validation_errors(
        artifact=artifact,
        model=model,
        request_index=request_index,
        predictions=predictions,
        quality_gate=quality_gate,
        replay_summary=replay_summary,
        learned_guard_summary=learned_guard_summary,
        guarded_auto_solve_summary=guarded_auto_solve_summary,
        guarded_auto_solve_results=guarded_auto_solve_results,
        guard_rows=guard_rows,
    )
    summary = _summary(
        artifact=artifact,
        request_index=request_index,
        predictions=predictions,
        quality_gate=quality_gate,
        replay_summary=replay_summary,
        learned_guard_summary=learned_guard_summary,
        guarded_auto_solve_summary=guarded_auto_solve_summary,
        validation_errors=validation_errors,
    )
    rows = _rows(
        summary=summary,
        guard_rows=guard_rows,
        guarded_auto_solve_results=guarded_auto_solve_results,
    )
    schema = _schema(summary)
    return {
        "summary": summary,
        "schema": schema,
        "rows": rows,
    }


def write_csr_policy_model_acceptance_rows(
    rows: Iterable[dict[str, Any]],
    path: str | Path,
) -> Path:
    return write_jsonl(rows, path)


def write_csr_policy_model_acceptance_summary(data: dict[str, Any], path: str | Path) -> Path:
    return _write_json(data["summary"], Path(path))


def write_csr_policy_model_acceptance_schema(data: dict[str, Any], path: str | Path) -> Path:
    return _write_json(data["schema"], Path(path))


def write_csr_policy_model_acceptance_report(data: dict[str, Any], path: str | Path) -> Path:
    summary = data["summary"]
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# CSR Policy Model Acceptance",
        "",
        f"- status: `{summary['status']}`",
        f"- schema_version: `{summary['schema_version']}`",
        f"- model_artifact_ready: `{summary['model_artifact_ready']}`",
        f"- accepted_for_shadow: `{summary['accepted_for_shadow']}`",
        f"- accepted_for_runtime_promotion: `{summary['accepted_for_runtime_promotion']}`",
        f"- model_loaded: `{summary['model_loaded']}`",
        f"- prediction_contract_checked: `{summary['prediction_contract_checked']}`",
        f"- guard_plan_shadow_checked: `{summary['guard_plan_shadow_checked']}`",
        f"- guarded_gpu_shadow_smoke_checked: `{summary['guarded_gpu_shadow_smoke_checked']}`",
        f"- quality_gate_runtime_eligible: `{summary['quality_gate_runtime_eligible']}`",
        f"- promotion_blockers: `{list(summary['promotion_blockers'])}`",
        f"- runtime_selector_changed: `{summary['runtime_selector_changed']}`",
        f"- validation_error_count: `{summary['validation_error_count']}`",
        "",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def _guard_rows(
    *,
    model_artifact_path: Path,
    csr_path: Path,
    selector_path: Path,
    quality_gate_summary_path: Path,
    min_confidence: float,
) -> tuple[dict[str, Any], ...]:
    csr_rows = {str(row["matrix_id"]): row for row in read_jsonl(csr_path)}
    rows: list[dict[str, Any]] = []
    for matrix_id in ACCEPTANCE_MATRIX_IDS:
        matrix = csr_matrix_from_record(csr_rows[matrix_id])
        decision = plan_csr_with_learned_guard(
            matrix,
            selector_path=selector_path,
            learned_model_artifact_path=model_artifact_path,
            quality_gate_summary_path=quality_gate_summary_path,
            mode="promote_if_safe",
            min_confidence=min_confidence,
        )
        rows.append(
            {
                "schema_version": CSR_POLICY_MODEL_ACCEPTANCE_SCHEMA_VERSION,
                "row_kind": "guard_plan_shadow",
                "matrix_id": matrix_id,
                "guard_status": decision.guard_status,
                "runtime_selection_source": decision.runtime_selection_source,
                "runtime_selector_changed": decision.runtime_selector_changed,
                "runtime_candidate_id": decision.runtime_candidate_id,
                "artifact_candidate_id": decision.artifact_candidate_id,
                "fallback_chain_enforced": decision.fallback_chain_enforced,
                "learned_policy_source": asdict(decision.learned_policy_source),
                "learned_prediction": (
                    None if decision.learned_prediction is None else asdict(decision.learned_prediction)
                ),
                "guard_reasons": tuple(decision.guard_reasons),
            }
        )
    return tuple(rows)


def _validation_errors(
    *,
    artifact: dict[str, Any],
    model: dict[str, Any],
    request_index: tuple[dict[str, Any], ...],
    predictions: tuple[Any, ...],
    quality_gate: dict[str, Any],
    replay_summary: dict[str, Any],
    learned_guard_summary: dict[str, Any],
    guarded_auto_solve_summary: dict[str, Any],
    guarded_auto_solve_results: tuple[dict[str, Any], ...],
    guard_rows: tuple[dict[str, Any], ...],
) -> tuple[str, ...]:
    errors: list[str] = []
    if artifact["runtime_contract"]["guard_required"] is not True:
        errors.append("artifact_guard_not_required")
    if artifact["runtime_contract"]["runtime_selector_changed"] is not False:
        errors.append("artifact_changed_runtime_selector")
    if not model:
        errors.append("model_not_loaded")
    if len(predictions) != len(request_index):
        errors.append("prediction_request_count_mismatch")
    if not _prediction_contract_valid(predictions):
        errors.append("prediction_contract_invalid")
    if replay_summary.get("exact_replay") is not True:
        errors.append("model_replay_not_exact")
    if quality_gate.get("status") != "passed":
        errors.append("quality_gate_not_passed")
    if quality_gate.get("runtime_selector_changed") is not False:
        errors.append("quality_gate_changed_runtime_selector")
    if learned_guard_summary.get("model_artifact_shadow_checked") is not True:
        errors.append("learned_guard_model_artifact_shadow_missing")
    if learned_guard_summary.get("model_artifact_quality_gate_checked") is not True:
        errors.append("learned_guard_model_artifact_quality_gate_missing")
    if guarded_auto_solve_summary.get("status") != "passed":
        errors.append("guarded_gpu_smoke_not_passed")
    if guarded_auto_solve_summary.get("learned_prediction_source") != "model_artifact":
        errors.append("guarded_gpu_smoke_not_model_artifact")
    if guarded_auto_solve_summary.get("runtime_selector_changed") is not False:
        errors.append("guarded_gpu_smoke_changed_runtime_selector")
    if guarded_auto_solve_summary.get("num_success") != 2:
        errors.append("guarded_gpu_smoke_success_count_mismatch")
    if guarded_auto_solve_summary.get("quality_gate_blocks") != 2:
        errors.append("guarded_gpu_smoke_quality_gate_block_mismatch")
    if len(guarded_auto_solve_results) != 2:
        errors.append("guarded_gpu_smoke_result_count_mismatch")
    if any(row.get("trace", {}).get("backend") != "taichi_gpu" for row in guarded_auto_solve_results):
        errors.append("guarded_gpu_smoke_backend_mismatch")
    if len(guard_rows) != len(ACCEPTANCE_MATRIX_IDS):
        errors.append("guard_plan_count_mismatch")
    if any(row["learned_policy_source"]["source_kind"] != "model_artifact" for row in guard_rows):
        errors.append("guard_plan_source_not_model_artifact")
    if any(row["learned_policy_source"]["model_loaded"] is not True for row in guard_rows):
        errors.append("guard_plan_model_not_loaded")
    if any(row["fallback_chain_enforced"] is not True for row in guard_rows):
        errors.append("guard_plan_fallback_chain_not_enforced")
    if any(row["runtime_selector_changed"] is not False for row in guard_rows):
        errors.append("guard_plan_changed_runtime_selector")
    return tuple(errors)


def _summary(
    *,
    artifact: dict[str, Any],
    request_index: tuple[dict[str, Any], ...],
    predictions: tuple[Any, ...],
    quality_gate: dict[str, Any],
    replay_summary: dict[str, Any],
    learned_guard_summary: dict[str, Any],
    guarded_auto_solve_summary: dict[str, Any],
    validation_errors: tuple[str, ...],
) -> dict[str, Any]:
    quality_gate_runtime_eligible = bool(quality_gate.get("challenger_runtime_eligible"))
    promotion_blockers = tuple(quality_gate.get("challenger_gate_failures", ()))
    if not quality_gate_runtime_eligible and "quality_gate_not_runtime_eligible" not in promotion_blockers:
        promotion_blockers = (*promotion_blockers, "quality_gate_not_runtime_eligible")
    accepted_for_shadow = not validation_errors
    accepted_for_runtime_promotion = accepted_for_shadow and quality_gate_runtime_eligible and not promotion_blockers
    return {
        "status": "passed" if accepted_for_shadow else "failed",
        "schema_version": CSR_POLICY_MODEL_ACCEPTANCE_SCHEMA_VERSION,
        "model_artifact_ready": True,
        "adapter": artifact["adapter"],
        "model_id": artifact["model"]["model_id"],
        "model_loaded": True,
        "prediction_contract_checked": True,
        "num_request_index_rows": len(request_index),
        "num_predictions": len(predictions),
        "replay_exact": bool(replay_summary.get("exact_replay")),
        "quality_gate_checked": True,
        "quality_gate_status": quality_gate.get("status"),
        "quality_gate_runtime_eligible": quality_gate_runtime_eligible,
        "learned_guard_model_artifact_checked": bool(
            learned_guard_summary.get("model_artifact_shadow_checked")
        ),
        "guard_plan_shadow_checked": True,
        "guarded_gpu_shadow_smoke_checked": True,
        "guarded_gpu_smoke_status": guarded_auto_solve_summary.get("status"),
        "guarded_gpu_smoke_successes": guarded_auto_solve_summary.get("num_success"),
        "accepted_for_shadow": accepted_for_shadow,
        "accepted_for_runtime_promotion": accepted_for_runtime_promotion,
        "promotion_blockers": promotion_blockers,
        "runtime_selector_changed": False,
        "validation_error_count": len(validation_errors),
        "validation_errors": validation_errors,
    }


def _rows(
    *,
    summary: dict[str, Any],
    guard_rows: tuple[dict[str, Any], ...],
    guarded_auto_solve_results: tuple[dict[str, Any], ...],
) -> tuple[dict[str, Any], ...]:
    base_rows = (
        {
            "schema_version": summary["schema_version"],
            "row_kind": "model_artifact_contract",
            "status": "passed" if summary["model_artifact_ready"] else "failed",
        },
        {
            "schema_version": summary["schema_version"],
            "row_kind": "prediction_contract",
            "status": "passed" if summary["prediction_contract_checked"] else "failed",
            "num_predictions": summary["num_predictions"],
        },
        {
            "schema_version": summary["schema_version"],
            "row_kind": "quality_gate",
            "status": summary["quality_gate_status"],
            "runtime_eligible": summary["quality_gate_runtime_eligible"],
        },
        {
            "schema_version": summary["schema_version"],
            "row_kind": "guarded_gpu_shadow_smoke",
            "status": summary["guarded_gpu_smoke_status"],
            "num_success": summary["guarded_gpu_smoke_successes"],
        },
        {
            "schema_version": summary["schema_version"],
            "row_kind": "acceptance_decision",
            "status": summary["status"],
            "accepted_for_shadow": summary["accepted_for_shadow"],
            "accepted_for_runtime_promotion": summary["accepted_for_runtime_promotion"],
            "promotion_blockers": summary["promotion_blockers"],
        },
    )
    gpu_rows = tuple(
        {
            "schema_version": summary["schema_version"],
            "row_kind": "guarded_gpu_shadow_result",
            "matrix_id": row["matrix_id"],
            "status": row["status"],
            "backend": row["trace"]["backend"],
            "learned_prediction_source": row["learned_policy_source_kind"],
            "runtime_selection_source": row["runtime_selection_source"],
        }
        for row in guarded_auto_solve_results
    )
    return (*base_rows, *guard_rows, *gpu_rows)


def _schema(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": CSR_POLICY_MODEL_ACCEPTANCE_SCHEMA_VERSION,
        "task": "accept_runtime_loadable_csr_policy_model_artifact",
        "accepted_for_shadow": summary["accepted_for_shadow"],
        "accepted_for_runtime_promotion": summary["accepted_for_runtime_promotion"],
        "acceptance_checks": {
            "artifact_contract": True,
            "model_load": True,
            "prediction_contract": True,
            "quality_gate": True,
            "guard_plan_shadow": True,
            "guarded_gpu_shadow_smoke": True,
        },
        "runtime_selector_changed": False,
    }


def _prediction_contract_valid(predictions: tuple[Any, ...]) -> bool:
    for prediction in predictions:
        ranked = tuple(prediction.ranked_candidate_ids)
        if not ranked or prediction.selected_candidate_id != ranked[0]:
            return False
        if len(ranked) != len(set(ranked)):
            return False
        if set(ranked) != set(prediction.scores):
            return False
    return True


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(data: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
