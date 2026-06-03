"""Runtime-loadable CSR policy model artifact contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from transsolvestack.policies.csr_transformer_ranker import (
    CSR_TRANSFORMER_RANKER_SCHEMA_VERSION,
    MODEL_FAMILY as CSR_TRANSFORMER_RANKER_MODEL_FAMILY,
    load_csr_transformer_ranker_model,
    predict_csr_transformer_ranker_from_model_file,
)
from transsolvestack.profiling.artifacts import read_jsonl, write_jsonl


CSR_POLICY_MODEL_ARTIFACT_SCHEMA_VERSION = "phase1_csr_policy_model_artifact_v1"
CSR_POLICY_MODEL_ARTIFACT_KIND = "csr_policy_model_artifact"
CSR_TRANSFORMER_RANKER_ADAPTER = "csr_transformer_ranker_saved_model_v1"


def build_csr_policy_model_artifact_from_files(
    model_path: str | Path = "runs/phase1_csr_transformer_ranker/csr_transformer_ranker_model.json",
    tensor_path: str | Path = (
        "runs/phase1_csr_transformer_ready/csr_transformer_training_tensors.json"
    ),
    request_index_path: str | Path = (
        "runs/phase1_csr_transformer_ready/csr_transformer_request_index.jsonl"
    ),
    quality_gate_summary_path: str | Path = (
        "runs/phase1_csr_transformer_quality_gate/"
        "csr_transformer_quality_gate_summary.json"
    ),
    replay_summary_path: str | Path = (
        "runs/phase1_csr_transformer_model_replay/"
        "csr_transformer_model_replay_summary.json"
    ),
) -> dict[str, Any]:
    """Build a runtime loader contract around a saved CSR policy model.

    This is the adapter boundary that future large Transformer weights must
    satisfy before the guarded runtime can shadow or promote them.
    """

    model_path = Path(model_path)
    tensor_path = Path(tensor_path)
    request_index_path = Path(request_index_path)
    quality_gate_summary_path = Path(quality_gate_summary_path)
    replay_summary_path = Path(replay_summary_path)

    model = load_csr_transformer_ranker_model(model_path)
    request_index = tuple(read_jsonl(request_index_path))
    predictions = predict_csr_transformer_ranker_from_model_file(
        model_path,
        tensor_path,
        request_index_path,
    )
    quality_gate = _read_json(quality_gate_summary_path)
    replay_summary = _read_json(replay_summary_path)
    validation_errors = _validation_errors(
        model=model,
        request_index=request_index,
        predictions=predictions,
        quality_gate=quality_gate,
        replay_summary=replay_summary,
    )
    summary = _summary(
        model=model,
        request_index=request_index,
        predictions=predictions,
        quality_gate=quality_gate,
        replay_summary=replay_summary,
        validation_errors=validation_errors,
    )
    artifact = _artifact(
        model=model,
        model_path=model_path,
        tensor_path=tensor_path,
        request_index_path=request_index_path,
        quality_gate_summary_path=quality_gate_summary_path,
        replay_summary_path=replay_summary_path,
        summary=summary,
    )
    schema = _schema(summary)
    rows = _rows(artifact)
    return {
        "artifact": artifact,
        "summary": summary,
        "schema": schema,
        "rows": rows,
    }


def load_csr_policy_model_artifact(path: str | Path) -> dict[str, Any]:
    artifact = _read_json(Path(path))
    validate_csr_policy_model_artifact(artifact)
    return artifact


def validate_csr_policy_model_artifact(artifact: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "artifact_kind",
        "adapter",
        "model",
        "inputs",
        "quality_gate",
        "runtime_contract",
    }
    missing = sorted(required - set(artifact))
    if missing:
        raise ValueError(f"CSR policy model artifact missing required keys: {missing}")
    if artifact["schema_version"] != CSR_POLICY_MODEL_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("CSR policy model artifact schema_version mismatch")
    if artifact["artifact_kind"] != CSR_POLICY_MODEL_ARTIFACT_KIND:
        raise ValueError("CSR policy model artifact kind mismatch")
    if artifact["adapter"] != CSR_TRANSFORMER_RANKER_ADAPTER:
        raise ValueError("unsupported CSR policy model adapter")
    model = artifact["model"]
    if model["source_model_schema_version"] != CSR_TRANSFORMER_RANKER_SCHEMA_VERSION:
        raise ValueError("CSR policy model source schema mismatch")
    if model["model_family"] != CSR_TRANSFORMER_RANKER_MODEL_FAMILY:
        raise ValueError("CSR policy model family mismatch")
    inputs = artifact["inputs"]
    for key in ("model_path", "tensor_path", "request_index_path"):
        if not inputs.get(key):
            raise ValueError(f"CSR policy model artifact missing input path: {key}")
    runtime = artifact["runtime_contract"]
    if runtime.get("guard_required") is not True:
        raise ValueError("CSR policy model artifact must require guard")
    if runtime.get("runtime_selector_changed") is not False:
        raise ValueError("CSR policy model artifact must not change runtime selector")


def csr_policy_model_artifact_paths(artifact: dict[str, Any]) -> dict[str, str]:
    validate_csr_policy_model_artifact(artifact)
    inputs = artifact["inputs"]
    return {
        "model_path": str(inputs["model_path"]),
        "tensor_path": str(inputs["tensor_path"]),
        "request_index_path": str(inputs["request_index_path"]),
    }


def write_csr_policy_model_artifact(data: dict[str, Any], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data["artifact"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def write_csr_policy_model_artifact_rows(
    rows: Iterable[dict[str, Any]],
    path: str | Path,
) -> Path:
    return write_jsonl(rows, path)


def write_csr_policy_model_artifact_summary(data: dict[str, Any], path: str | Path) -> Path:
    return _write_json(data["summary"], Path(path))


def write_csr_policy_model_artifact_schema(data: dict[str, Any], path: str | Path) -> Path:
    return _write_json(data["schema"], Path(path))


def write_csr_policy_model_artifact_report(data: dict[str, Any], path: str | Path) -> Path:
    summary = data["summary"]
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# CSR Policy Model Artifact",
        "",
        f"- status: `{summary['status']}`",
        f"- schema_version: `{summary['schema_version']}`",
        f"- artifact_ready: `{summary['artifact_ready']}`",
        f"- adapter: `{summary['adapter']}`",
        f"- model_id: `{summary['model_id']}`",
        f"- model_family: `{summary['model_family']}`",
        f"- model_loaded: `{summary['model_loaded']}`",
        f"- prediction_contract_checked: `{summary['prediction_contract_checked']}`",
        f"- replay_exact: `{summary['replay_exact']}`",
        f"- guard_required: `{summary['guard_required']}`",
        f"- runtime_selector_changed: `{summary['runtime_selector_changed']}`",
        f"- current_quality_gate_runtime_eligible: `{summary['current_quality_gate_runtime_eligible']}`",
        f"- validation_error_count: `{summary['validation_error_count']}`",
        "",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def _validation_errors(
    *,
    model: dict[str, Any],
    request_index: tuple[dict[str, Any], ...],
    predictions: tuple[Any, ...],
    quality_gate: dict[str, Any],
    replay_summary: dict[str, Any],
) -> tuple[str, ...]:
    errors: list[str] = []
    if model["schema_version"] != CSR_TRANSFORMER_RANKER_SCHEMA_VERSION:
        errors.append("source_model_schema_mismatch")
    if model["model_family"] != CSR_TRANSFORMER_RANKER_MODEL_FAMILY:
        errors.append("source_model_family_mismatch")
    if not request_index:
        errors.append("empty_request_index")
    if len(predictions) != len(request_index):
        errors.append("prediction_request_count_mismatch")
    if not _prediction_contract_valid(predictions):
        errors.append("prediction_contract_invalid")
    if quality_gate.get("runtime_selector_changed") is not False:
        errors.append("quality_gate_changed_runtime_selector")
    if replay_summary.get("exact_replay") is not True:
        errors.append("saved_model_replay_not_exact")
    if replay_summary.get("model_loaded") is not True:
        errors.append("saved_model_replay_not_loaded")
    return tuple(errors)


def _summary(
    *,
    model: dict[str, Any],
    request_index: tuple[dict[str, Any], ...],
    predictions: tuple[Any, ...],
    quality_gate: dict[str, Any],
    replay_summary: dict[str, Any],
    validation_errors: tuple[str, ...],
) -> dict[str, Any]:
    artifact_ready = not validation_errors
    return {
        "status": "passed" if artifact_ready else "failed",
        "schema_version": CSR_POLICY_MODEL_ARTIFACT_SCHEMA_VERSION,
        "artifact_kind": CSR_POLICY_MODEL_ARTIFACT_KIND,
        "artifact_ready": artifact_ready,
        "adapter": CSR_TRANSFORMER_RANKER_ADAPTER,
        "model_id": str(model["model_id"]),
        "model_family": str(model["model_family"]),
        "source_model_schema_version": str(model["schema_version"]),
        "model_loaded": True,
        "prediction_contract_checked": True,
        "num_request_index_rows": len(request_index),
        "num_predictions": len(predictions),
        "replay_exact": bool(replay_summary.get("exact_replay")),
        "guard_required": True,
        "quality_gate_status": quality_gate.get("status"),
        "current_quality_gate_runtime_eligible": bool(
            quality_gate.get("challenger_runtime_eligible")
        ),
        "current_quality_gate_failures": tuple(
            quality_gate.get("challenger_gate_failures", ())
        ),
        "runtime_selector_changed": False,
        "validation_error_count": len(validation_errors),
        "validation_errors": validation_errors,
    }


def _artifact(
    *,
    model: dict[str, Any],
    model_path: Path,
    tensor_path: Path,
    request_index_path: Path,
    quality_gate_summary_path: Path,
    replay_summary_path: Path,
    summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": CSR_POLICY_MODEL_ARTIFACT_SCHEMA_VERSION,
        "artifact_kind": CSR_POLICY_MODEL_ARTIFACT_KIND,
        "adapter": CSR_TRANSFORMER_RANKER_ADAPTER,
        "model": {
            "model_id": summary["model_id"],
            "model_family": summary["model_family"],
            "source_model_schema_version": summary["source_model_schema_version"],
        },
        "inputs": {
            "model_path": str(model_path),
            "tensor_path": str(tensor_path),
            "request_index_path": str(request_index_path),
            "token_feature_names": tuple(model["token_feature_names"]),
            "scorer_feature_names": tuple(model["scorer_feature_names"]),
        },
        "quality_gate": {
            "summary_path": str(quality_gate_summary_path),
            "runtime_eligible": summary["current_quality_gate_runtime_eligible"],
            "guard_required_before_promotion": True,
        },
        "replay": {
            "summary_path": str(replay_summary_path),
            "exact_replay_required": True,
            "exact_replay": summary["replay_exact"],
        },
        "runtime_contract": {
            "entrypoint": "transsolvestack.api.plan_csr_with_learned_guard",
            "guard_required": True,
            "default_mode": "shadow",
            "promotion_mode": "promote_if_safe",
            "confidence_threshold_required": True,
            "fallback_chain_enforced": True,
            "runtime_selector_changed": False,
        },
    }


def _rows(artifact: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    schema_version = artifact["schema_version"]
    return (
        {
            "schema_version": schema_version,
            "row_kind": "model",
            "path": artifact["inputs"]["model_path"],
            "required": True,
        },
        {
            "schema_version": schema_version,
            "row_kind": "tensor_contract",
            "path": artifact["inputs"]["tensor_path"],
            "required": True,
        },
        {
            "schema_version": schema_version,
            "row_kind": "request_index",
            "path": artifact["inputs"]["request_index_path"],
            "required": True,
        },
        {
            "schema_version": schema_version,
            "row_kind": "quality_gate",
            "path": artifact["quality_gate"]["summary_path"],
            "required": True,
        },
        {
            "schema_version": schema_version,
            "row_kind": "runtime_contract",
            "path": artifact["runtime_contract"]["entrypoint"],
            "required": True,
        },
    )


def _schema(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": CSR_POLICY_MODEL_ARTIFACT_SCHEMA_VERSION,
        "artifact_kind": CSR_POLICY_MODEL_ARTIFACT_KIND,
        "artifact_ready": summary["artifact_ready"],
        "adapter": CSR_TRANSFORMER_RANKER_ADAPTER,
        "supported_source_model_schema": CSR_TRANSFORMER_RANKER_SCHEMA_VERSION,
        "runtime_contract": {
            "guard_required": True,
            "shadow_mode_default": True,
            "promotion_requires_quality_gate": True,
            "promotion_requires_confidence": True,
            "promotion_requires_profiled_success_candidate": True,
            "runtime_selector_changed": False,
        },
    }


def _prediction_contract_valid(predictions: tuple[Any, ...]) -> bool:
    for prediction in predictions:
        ranked = tuple(prediction.ranked_candidate_ids)
        if not ranked or ranked[0] != prediction.selected_candidate_id:
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
