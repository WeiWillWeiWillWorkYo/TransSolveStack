"""Formal CSR Transformer training entrypoint contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from transsolvestack.profiling.artifacts import read_jsonl, write_jsonl


CSR_TRANSFORMER_TRAINING_ENTRYPOINT_SCHEMA_VERSION = (
    "phase1_csr_transformer_training_entrypoint_v1"
)


def build_csr_transformer_training_entrypoint_from_files(
    ready_dir: str | Path = "runs/phase1_csr_transformer_ready",
    quality_gate_summary_path: str | Path = (
        "runs/phase1_csr_transformer_quality_gate/"
        "csr_transformer_quality_gate_summary.json"
    ),
    output_dir: str | Path = "runs/phase1_csr_transformer_training_entrypoint",
    *,
    model_family: str = "csr_transformer_policy_v1",
) -> dict[str, Any]:
    """Write the formal training contract for a future CSR Transformer policy.

    This boundary is intentionally not a training run. It validates that the
    Tensor/contract/gate artifacts are present and records the exact files a
    future large-scale Transformer trainer must consume and produce before it
    can be considered for runtime promotion.
    """

    ready = Path(ready_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    paths = _input_paths(ready, Path(quality_gate_summary_path))
    ready_summary = _read_json(paths["ready_summary"])
    ready_schema = _read_json(paths["ready_schema"])
    tensor_summary = _read_json(paths["tensor_summary"])
    contract_summary = _read_json(paths["contract_summary"])
    quality_gate = _read_json(paths["quality_gate_summary"])
    request_index = read_jsonl(paths["request_index"])
    requests = read_jsonl(paths["requests"])
    targets = read_jsonl(paths["targets"])

    validation_errors = _validation_errors(
        paths=paths,
        ready_summary=ready_summary,
        ready_schema=ready_schema,
        tensor_summary=tensor_summary,
        contract_summary=contract_summary,
        quality_gate=quality_gate,
        request_index=request_index,
        requests=requests,
        targets=targets,
    )
    training_entrypoint_ready = not validation_errors
    summary = {
        "status": "passed" if training_entrypoint_ready else "failed",
        "schema_version": CSR_TRANSFORMER_TRAINING_ENTRYPOINT_SCHEMA_VERSION,
        "training_entrypoint_ready": training_entrypoint_ready,
        "model_family": model_family,
        "model_training_required": True,
        "model_trained": False,
        "runtime_selector_changed": False,
        "transformer_connectable": bool(ready_summary.get("transformer_connectable")),
        "quality_gate_enforced": True,
        "current_quality_gate_runtime_eligible": bool(
            quality_gate.get("challenger_runtime_eligible")
        ),
        "current_quality_gate_failures": tuple(
            quality_gate.get("challenger_gate_failures", ())
        ),
        "num_selector_rows": int(ready_summary["num_selector_rows"]),
        "num_matrices": int(ready_summary["num_matrices"]),
        "num_success_rows": int(ready_summary["num_success_rows"]),
        "num_screened_out_rows": int(ready_summary["num_screened_out_rows"]),
        "num_oracle_rows": int(ready_summary["num_oracle_rows"]),
        "num_model_requests": int(ready_summary["num_model_requests"]),
        "num_tensor_requests": int(tensor_summary["num_requests"]),
        "num_global_candidates": int(tensor_summary["num_global_candidates"]),
        "num_active_candidate_slots": int(tensor_summary["num_active_candidate_slots"]),
        "matrix_feature_dim": int(tensor_summary["matrix_feature_dim"]),
        "candidate_feature_dim": int(tensor_summary["candidate_feature_dim"]),
        "validation_error_count": len(validation_errors),
        "validation_errors": tuple(validation_errors),
        "next_step": "run_large_scale_transformer_training_then_quality_gate",
        "public_release_boundary": (
            "platform_closed_for_training_contract_runtime_stays_artifact_backed"
        ),
    }
    rows = _rows(paths=paths, summary=summary)
    job_spec = _job_spec(paths=paths, output_dir=output, model_family=model_family)
    quality_contract = _quality_contract(quality_gate=quality_gate)
    schema = _schema(summary=summary)

    output_paths = {
        "rows": output / "csr_transformer_training_entrypoint_rows.jsonl",
        "summary": output / "csr_transformer_training_entrypoint_summary.json",
        "schema": output / "csr_transformer_training_entrypoint_schema.json",
        "job_spec": output / "csr_transformer_training_job_spec.json",
        "quality_contract": output / "csr_transformer_training_quality_contract.json",
        "report": output / "csr_transformer_training_entrypoint_report.md",
    }
    write_jsonl(rows, output_paths["rows"])
    _write_json(summary, output_paths["summary"])
    _write_json(schema, output_paths["schema"])
    _write_json(job_spec, output_paths["job_spec"])
    _write_json(quality_contract, output_paths["quality_contract"])
    _write_report(summary, output_paths["report"])

    return {
        "rows": rows,
        "summary": summary,
        "schema": schema,
        "job_spec": job_spec,
        "quality_contract": quality_contract,
        "paths": output_paths,
    }


def _input_paths(ready: Path, quality_gate_summary: Path) -> dict[str, Path]:
    return {
        "ready_summary": ready / "csr_transformer_ready_summary.json",
        "ready_schema": ready / "csr_transformer_ready_schema.json",
        "tensor_summary": ready / "csr_transformer_tensor_summary.json",
        "contract_summary": ready / "csr_transformer_model_contract_summary.json",
        "requests": ready / "csr_transformer_model_requests.jsonl",
        "targets": ready / "csr_transformer_model_targets.jsonl",
        "tensors": ready / "csr_transformer_training_tensors.json",
        "request_index": ready / "csr_transformer_request_index.jsonl",
        "quality_gate_summary": quality_gate_summary,
    }


def _validation_errors(
    *,
    paths: dict[str, Path],
    ready_summary: dict[str, Any],
    ready_schema: dict[str, Any],
    tensor_summary: dict[str, Any],
    contract_summary: dict[str, Any],
    quality_gate: dict[str, Any],
    request_index: list[dict[str, Any]],
    requests: list[dict[str, Any]],
    targets: list[dict[str, Any]],
) -> list[str]:
    errors: list[str] = []
    for name, path in paths.items():
        if not path.exists():
            errors.append(f"missing_input:{name}:{path}")
    if ready_summary.get("status") != "passed":
        errors.append("ready_summary_not_passed")
    if ready_summary.get("transformer_connectable") is not True:
        errors.append("transformer_bundle_not_connectable")
    if ready_summary.get("runtime_selector_changed") is not False:
        errors.append("ready_bundle_changed_runtime_selector")
    if ready_schema.get("transformer_connectable") is not True:
        errors.append("ready_schema_not_connectable")
    if tensor_summary.get("status") != "passed":
        errors.append("tensor_summary_not_passed")
    if tensor_summary.get("model_required") is not False:
        errors.append("tensor_export_requires_model")
    if contract_summary.get("validation_error_count") != 0:
        errors.append("model_contract_has_validation_errors")
    if quality_gate.get("runtime_selector_changed") is not False:
        errors.append("quality_gate_changed_runtime_selector")
    if len(request_index) != int(tensor_summary.get("num_requests", -1)):
        errors.append("request_index_count_mismatch")
    if len(requests) != int(ready_summary.get("num_model_requests", -1)):
        errors.append("model_request_count_mismatch")
    if len(targets) != int(ready_summary.get("num_model_targets", -1)):
        errors.append("model_target_count_mismatch")
    return errors


def _rows(*, paths: dict[str, Path], summary: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    return (
        {
            "schema_version": summary["schema_version"],
            "row_kind": "input_tensor",
            "path": str(paths["tensors"]),
            "required": True,
            "description": "numeric CSR request/candidate tensor arrays",
        },
        {
            "schema_version": summary["schema_version"],
            "row_kind": "input_request_index",
            "path": str(paths["request_index"]),
            "required": True,
            "description": "request split, matrix, and context metadata",
        },
        {
            "schema_version": summary["schema_version"],
            "row_kind": "input_model_contract",
            "path": str(paths["requests"]),
            "required": True,
            "description": "label-free prediction requests",
        },
        {
            "schema_version": summary["schema_version"],
            "row_kind": "offline_targets",
            "path": str(paths["targets"]),
            "required": True,
            "description": "offline-only labels and timing targets",
        },
        {
            "schema_version": summary["schema_version"],
            "row_kind": "quality_gate",
            "path": str(paths["quality_gate_summary"]),
            "required": True,
            "description": "runtime promotion gate that must pass before learned use",
        },
        {
            "schema_version": summary["schema_version"],
            "row_kind": "runtime_guard",
            "path": "transsolvestack/policies/csr_learned_guard.py",
            "required": True,
            "description": "shadow/promotion/fallback enforcement boundary",
        },
    )


def _job_spec(
    *,
    paths: dict[str, Path],
    output_dir: Path,
    model_family: str,
) -> dict[str, Any]:
    return {
        "schema_version": CSR_TRANSFORMER_TRAINING_ENTRYPOINT_SCHEMA_VERSION,
        "job_kind": "csr_transformer_policy_training",
        "model_family": model_family,
        "runtime_selector_changed": False,
        "inputs": {
            "tensor_arrays": str(paths["tensors"]),
            "request_index": str(paths["request_index"]),
            "model_requests": str(paths["requests"]),
            "offline_targets": str(paths["targets"]),
        },
        "expected_outputs": {
            "model": str(output_dir / "future_csr_transformer_policy_model.json"),
            "predictions": str(output_dir / "future_csr_transformer_policy_predictions.jsonl"),
            "summary": str(output_dir / "future_csr_transformer_policy_summary.json"),
            "schema": str(output_dir / "future_csr_transformer_policy_schema.json"),
        },
        "required_post_training_checks": (
            "validate_predictions_against_phase1_csr_model_contract_v1",
            "run_csr_selector_model_quality_gate",
            "require_no_non_success_runtime_selections",
            "require_profiled_success_fallback_chain_for_promotions",
            "keep_runtime_selector_artifact_backed_until_gate_passes",
        ),
        "non_goals": (
            "do_not_train_inside_this_entrypoint_artifact",
            "do_not_change_runtime_selector",
            "do_not_execute_gpu_benchmarks",
        ),
    }


def _quality_contract(*, quality_gate: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": CSR_TRANSFORMER_TRAINING_ENTRYPOINT_SCHEMA_VERSION,
        "quality_gate_schema_version": "phase1_csr_selector_model_eval_v1",
        "runtime_selector_changed": False,
        "current_gate_status": {
            "challenger_runtime_eligible": bool(
                quality_gate.get("challenger_runtime_eligible")
            ),
            "challenger_gate_failures": tuple(
                quality_gate.get("challenger_gate_failures", ())
            ),
            "best_offline_model_id": quality_gate.get("best_offline_model_id"),
            "runtime_selected_model_id": quality_gate.get("runtime_selected_model_id"),
        },
        "promotion_requirements": {
            "min_required_eval_oracle_requests": quality_gate.get(
                "min_required_eval_oracle_requests"
            ),
            "min_runtime_oracle_top1_accuracy": quality_gate.get(
                "min_runtime_oracle_top1_accuracy"
            ),
            "min_runtime_profiled_success_rate": quality_gate.get(
                "min_runtime_profiled_success_rate"
            ),
            "non_success_eval_selection_count": 0,
            "guard_mode": "promote_if_safe",
        },
    }


def _schema(*, summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": CSR_TRANSFORMER_TRAINING_ENTRYPOINT_SCHEMA_VERSION,
        "task": "formalize_csr_transformer_training_boundary",
        "training_entrypoint_ready": summary["training_entrypoint_ready"],
        "model_training_required": True,
        "model_trained": False,
        "runtime_selector_changed": False,
        "input_contract": {
            "tensor_schema": "phase1_csr_training_tensors_v1",
            "model_contract_schema": "phase1_csr_model_contract_v1",
            "ready_bundle_schema": "phase1_csr_transformer_ready_v1",
        },
        "runtime_integration": {
            "selector_before_training": "artifact_backed",
            "quality_gate_required": True,
            "learned_guard_required": True,
            "automatic_runtime_promotion": False,
        },
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(data: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_report(summary: dict[str, Any], path: Path) -> Path:
    lines = [
        "# CSR Transformer Training Entrypoint",
        "",
        f"- status: `{summary['status']}`",
        f"- schema_version: `{summary['schema_version']}`",
        f"- training_entrypoint_ready: `{summary['training_entrypoint_ready']}`",
        f"- model_training_required: `{summary['model_training_required']}`",
        f"- model_trained: `{summary['model_trained']}`",
        f"- runtime_selector_changed: `{summary['runtime_selector_changed']}`",
        f"- transformer_connectable: `{summary['transformer_connectable']}`",
        f"- current_quality_gate_runtime_eligible: `{summary['current_quality_gate_runtime_eligible']}`",
        f"- current_quality_gate_failures: `{list(summary['current_quality_gate_failures'])}`",
        f"- selector_rows: `{summary['num_selector_rows']}`",
        f"- matrices: `{summary['num_matrices']}`",
        f"- model_requests: `{summary['num_model_requests']}`",
        f"- tensor_requests: `{summary['num_tensor_requests']}`",
        f"- active_candidate_slots: `{summary['num_active_candidate_slots']}`",
        f"- matrix_feature_dim: `{summary['matrix_feature_dim']}`",
        f"- candidate_feature_dim: `{summary['candidate_feature_dim']}`",
        f"- validation_error_count: `{summary['validation_error_count']}`",
        f"- next_step: `{summary['next_step']}`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
