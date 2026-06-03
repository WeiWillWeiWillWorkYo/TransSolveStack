"""Adapter boundary for externally trained CSR Transformer policy checkpoints."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from transsolvestack.policies.csr_transformer_ranker import (
    CSR_TRANSFORMER_RANKER_SCHEMA_VERSION,
    MODEL_FAMILY as CSR_TRANSFORMER_RANKER_MODEL_FAMILY,
    CsrTransformerRankerPrediction,
    load_csr_transformer_ranker_model,
    predict_csr_transformer_ranker_from_model,
)
from transsolvestack.profiling.artifacts import read_jsonl, write_jsonl


CSR_EXTERNAL_RANKER_CHECKPOINT_SCHEMA_VERSION = "phase1_csr_external_ranker_checkpoint_v1"
CSR_EXTERNAL_MODEL_ADAPTER_SCHEMA_VERSION = "phase1_csr_external_model_adapter_v1"
CSR_EXTERNAL_RANKER_CHECKPOINT_KIND = "csr_external_ranker_checkpoint"
CSR_EXTERNAL_RANKER_ADAPTER = "csr_external_json_ranker_checkpoint_v1"


def build_reference_csr_external_ranker_checkpoint_from_files(
    model_path: str | Path = "runs/phase1_csr_transformer_ranker/csr_transformer_ranker_model.json",
    tensor_path: str | Path = (
        "runs/phase1_csr_transformer_ready/csr_transformer_training_tensors.json"
    ),
    request_index_path: str | Path = (
        "runs/phase1_csr_transformer_ready/csr_transformer_request_index.jsonl"
    ),
    *,
    contributor_id: str = "wei_cui_reference",
    contributor_name: str = "Wei CUI",
    training_statement: str = (
        "Reference checkpoint exported from the Phase 1 CSR Transformer ranker."
    ),
) -> dict[str, Any]:
    """Wrap a saved TSS ranker model in the external-checkpoint schema.

    This produces a local reference fixture for the adapter. Future PyTorch or
    other training code should export the same checkpoint schema, then use the
    adapter path below without touching runtime selector code.
    """

    model_path = Path(model_path)
    tensor_path = Path(tensor_path)
    request_index_path = Path(request_index_path)
    model = load_csr_transformer_ranker_model(model_path)
    arrays = _read_json(tensor_path)
    request_index = tuple(read_jsonl(request_index_path))
    return {
        "schema_version": CSR_EXTERNAL_RANKER_CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_kind": CSR_EXTERNAL_RANKER_CHECKPOINT_KIND,
        "adapter": CSR_EXTERNAL_RANKER_ADAPTER,
        "framework": "tss_json_reference_export",
        "contributor": {
            "id": contributor_id,
            "name": contributor_name,
        },
        "source": {
            "model_path": str(model_path),
            "tensor_path": str(tensor_path),
            "request_index_path": str(request_index_path),
            "training_statement": training_statement,
        },
        "model": model,
        "tensor_contract": _tensor_contract(arrays, request_index, model=model),
        "runtime_contract": {
            "guard_required": True,
            "default_mode": "shadow",
            "promotion_requires_quality_gate": True,
            "runtime_selector_changed": False,
        },
    }


def adapt_csr_external_ranker_checkpoint_from_files(
    checkpoint_path: str | Path,
    tensor_path: str | Path = (
        "runs/phase1_csr_transformer_ready/csr_transformer_training_tensors.json"
    ),
    request_index_path: str | Path = (
        "runs/phase1_csr_transformer_ready/csr_transformer_request_index.jsonl"
    ),
) -> dict[str, Any]:
    """Convert an external checkpoint into a TSS saved ranker model artifact."""

    checkpoint_path = Path(checkpoint_path)
    tensor_path = Path(tensor_path)
    request_index_path = Path(request_index_path)
    checkpoint = _read_json(checkpoint_path)
    _validate_external_checkpoint(checkpoint)
    arrays = _read_json(tensor_path)
    request_index = tuple(read_jsonl(request_index_path))
    adapted_model = dict(checkpoint["model"])
    predictions = predict_csr_transformer_ranker_from_model(
        adapted_model,
        arrays,
        request_index,
    )
    validation_errors = _validation_errors(
        checkpoint=checkpoint,
        arrays=arrays,
        request_index=request_index,
        predictions=predictions,
    )
    adapted_ranker_summary = _adapted_ranker_summary(
        model=adapted_model,
        arrays=arrays,
        predictions=predictions,
        validation_errors=validation_errors,
    )
    summary = _adapter_summary(
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path,
        tensor_path=tensor_path,
        request_index_path=request_index_path,
        predictions=predictions,
        adapted_ranker_summary=adapted_ranker_summary,
        validation_errors=validation_errors,
    )
    rows = _rows(summary, checkpoint)
    schema = _schema(summary)
    report = _report_text(summary)
    return {
        "checkpoint": checkpoint,
        "adapted_model": adapted_model,
        "predictions": predictions,
        "adapted_ranker_summary": adapted_ranker_summary,
        "summary": summary,
        "schema": schema,
        "rows": rows,
        "report": report,
    }


def write_csr_external_ranker_checkpoint(
    checkpoint: dict[str, Any],
    path: str | Path,
) -> Path:
    return _write_json(checkpoint, Path(path))


def write_csr_external_model_adapter_model(
    data: dict[str, Any],
    path: str | Path,
) -> Path:
    return _write_json(data["adapted_model"], Path(path))


def write_csr_external_model_adapter_predictions(
    predictions: Iterable[CsrTransformerRankerPrediction],
    path: str | Path,
) -> Path:
    return write_jsonl((asdict(row) for row in predictions), path)


def write_csr_external_model_adapter_ranker_summary(
    data: dict[str, Any],
    path: str | Path,
) -> Path:
    return _write_json(data["adapted_ranker_summary"], Path(path))


def write_csr_external_model_adapter_rows(
    rows: Iterable[dict[str, Any]],
    path: str | Path,
) -> Path:
    return write_jsonl(rows, path)


def write_csr_external_model_adapter_summary(
    data: dict[str, Any],
    path: str | Path,
) -> Path:
    return _write_json(data["summary"], Path(path))


def write_csr_external_model_adapter_schema(
    data: dict[str, Any],
    path: str | Path,
) -> Path:
    return _write_json(data["schema"], Path(path))


def write_csr_external_model_adapter_report(
    data: dict[str, Any],
    path: str | Path,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(str(data["report"]), encoding="utf-8")
    return output


def _validate_external_checkpoint(checkpoint: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "checkpoint_kind",
        "adapter",
        "framework",
        "model",
        "tensor_contract",
        "runtime_contract",
    }
    missing = sorted(required - set(checkpoint))
    if missing:
        raise ValueError(f"external checkpoint missing required keys: {missing}")
    if checkpoint["schema_version"] != CSR_EXTERNAL_RANKER_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("external checkpoint schema_version mismatch")
    if checkpoint["checkpoint_kind"] != CSR_EXTERNAL_RANKER_CHECKPOINT_KIND:
        raise ValueError("external checkpoint kind mismatch")
    if checkpoint["adapter"] != CSR_EXTERNAL_RANKER_ADAPTER:
        raise ValueError("unsupported external checkpoint adapter")
    model = checkpoint["model"]
    if model.get("schema_version") != CSR_TRANSFORMER_RANKER_SCHEMA_VERSION:
        raise ValueError("external checkpoint model schema mismatch")
    if model.get("model_family") != CSR_TRANSFORMER_RANKER_MODEL_FAMILY:
        raise ValueError("external checkpoint model family mismatch")
    runtime = checkpoint["runtime_contract"]
    if runtime.get("guard_required") is not True:
        raise ValueError("external checkpoint must require runtime guard")
    if runtime.get("runtime_selector_changed") is not False:
        raise ValueError("external checkpoint must not change runtime selector")


def _validation_errors(
    *,
    checkpoint: dict[str, Any],
    arrays: dict[str, Any],
    request_index: tuple[dict[str, Any], ...],
    predictions: tuple[CsrTransformerRankerPrediction, ...],
) -> tuple[str, ...]:
    errors: list[str] = []
    contract = checkpoint["tensor_contract"]
    if tuple(contract.get("token_feature_names", ())) != tuple(_token_feature_names(arrays)):
        errors.append("token_feature_contract_mismatch")
    d_model = int(checkpoint["model"]["encoder"]["d_model"])
    if tuple(contract.get("scorer_feature_names", ())) != _scorer_feature_names(
        arrays,
        d_model=d_model,
    ):
        errors.append("scorer_feature_contract_mismatch")
    if int(contract.get("num_requests", -1)) != len(request_index):
        errors.append("request_count_contract_mismatch")
    if len(predictions) != len(request_index):
        errors.append("prediction_request_count_mismatch")
    if not _prediction_contract_valid(predictions):
        errors.append("prediction_contract_invalid")
    if checkpoint["model"].get("runtime_integration", {}).get(
        "runtime_selector_changed"
    ) is not False:
        errors.append("model_runtime_selector_changed")
    return tuple(errors)


def _adapter_summary(
    *,
    checkpoint: dict[str, Any],
    checkpoint_path: Path,
    tensor_path: Path,
    request_index_path: Path,
    predictions: tuple[CsrTransformerRankerPrediction, ...],
    adapted_ranker_summary: dict[str, Any],
    validation_errors: tuple[str, ...],
) -> dict[str, Any]:
    adapter_ready = not validation_errors
    return {
        "status": "passed" if adapter_ready else "failed",
        "schema_version": CSR_EXTERNAL_MODEL_ADAPTER_SCHEMA_VERSION,
        "checkpoint_schema_version": checkpoint["schema_version"],
        "checkpoint_kind": checkpoint["checkpoint_kind"],
        "adapter": checkpoint["adapter"],
        "framework": checkpoint["framework"],
        "checkpoint_loaded": True,
        "adapter_ready": adapter_ready,
        "adapted_model_schema_version": adapted_ranker_summary["schema_version"],
        "adapted_model_id": adapted_ranker_summary["model_id"],
        "adapted_model_family": adapted_ranker_summary["model_family"],
        "adapted_model_trained": adapted_ranker_summary["model_trained"],
        "quality_gate_input_ready": adapter_ready,
        "policy_model_artifact_input_ready": adapter_ready,
        "prediction_contract_checked": True,
        "prediction_contract_valid": _prediction_contract_valid(predictions),
        "num_predictions": len(predictions),
        "num_eval_predictions": adapted_ranker_summary["num_eval_requests"],
        "eval_oracle_top1_accuracy": adapted_ranker_summary[
            "eval_oracle_top1_accuracy"
        ],
        "eval_profiled_success_selection_rate": adapted_ranker_summary[
            "eval_profiled_success_selection_rate"
        ],
        "eval_non_success_selection_count": adapted_ranker_summary[
            "eval_non_success_selection_count"
        ],
        "checkpoint_path": str(checkpoint_path),
        "tensor_path": str(tensor_path),
        "request_index_path": str(request_index_path),
        "runtime_selector_changed": False,
        "validation_error_count": len(validation_errors),
        "validation_errors": validation_errors,
    }


def _adapted_ranker_summary(
    *,
    model: dict[str, Any],
    arrays: dict[str, Any],
    predictions: tuple[CsrTransformerRankerPrediction, ...],
    validation_errors: tuple[str, ...],
) -> dict[str, Any]:
    train_predictions = tuple(row for row in predictions if row.split == "train")
    eval_predictions = tuple(row for row in predictions if row.split == "eval")
    train_oracle = tuple(row for row in train_predictions if row.oracle_candidate_id)
    eval_oracle = tuple(row for row in eval_predictions if row.oracle_candidate_id)
    eval_success = tuple(row for row in eval_predictions if row.selected_target_status == "success")
    eval_regrets = [
        float(row.regret_vs_oracle_ms)
        for row in eval_predictions
        if row.regret_vs_oracle_ms is not None
    ]
    encoder = model["encoder"]
    training = model.get("training", {})
    return {
        "status": "passed" if predictions and not validation_errors else "failed",
        "schema_version": CSR_TRANSFORMER_RANKER_SCHEMA_VERSION,
        "model_family": model["model_family"],
        "model_id": model["model_id"],
        "model_trained": True,
        "runtime_selector_changed": False,
        "encoder_training": model.get("encoder_training", "external_checkpoint_adapter"),
        "num_requests": len(predictions),
        "num_predictions": len(predictions),
        "num_train_requests": len(train_predictions),
        "num_eval_requests": len(eval_predictions),
        "num_train_oracle_requests": len(train_oracle),
        "num_eval_oracle_requests": len(eval_oracle),
        "num_global_candidates": len(arrays["global_candidate_ids"]),
        "token_feature_dim": len(_token_feature_names(arrays)),
        "d_model": int(encoder["d_model"]),
        "num_attention_heads": int(encoder["num_attention_heads"]),
        "feedforward_dim": int(encoder["feedforward_dim"]),
        "scorer_feature_dim": len(model["scorer_feature_names"]),
        "num_epochs": int(training.get("epochs", 0)),
        "learning_rate": float(training.get("learning_rate", 0.0)),
        "l2_regularization": float(training.get("l2_regularization", 0.0)),
        "num_pairwise_constraints": int(training.get("num_pairwise_constraints", 0)),
        "num_pairwise_updates": int(training.get("num_pairwise_updates", 0)),
        "final_train_pairwise_loss": float(
            training.get("final_train_pairwise_loss", 0.0)
        ),
        "train_oracle_top1_accuracy": _oracle_accuracy(train_oracle),
        "eval_oracle_top1_accuracy": _oracle_accuracy(eval_oracle),
        "eval_oracle_top1_accuracy_all_requests": _oracle_accuracy(eval_predictions),
        "eval_profiled_success_selection_rate": (
            len(eval_success) / len(eval_predictions) if eval_predictions else 0.0
        ),
        "eval_non_success_selection_count": len(eval_predictions) - len(eval_success),
        "eval_mean_regret_ms": (
            sum(eval_regrets) / len(eval_regrets) if eval_regrets else None
        ),
        "eval_max_regret_ms": max(eval_regrets) if eval_regrets else None,
    }


def _rows(
    summary: dict[str, Any],
    checkpoint: dict[str, Any],
) -> tuple[dict[str, Any], ...]:
    schema = summary["schema_version"]
    return (
        {
            "schema_version": schema,
            "row_kind": "checkpoint_contract",
            "checkpoint_schema_version": summary["checkpoint_schema_version"],
            "adapter": summary["adapter"],
            "framework": summary["framework"],
        },
        {
            "schema_version": schema,
            "row_kind": "tensor_contract",
            "num_requests": checkpoint["tensor_contract"]["num_requests"],
            "token_feature_dim": len(checkpoint["tensor_contract"]["token_feature_names"]),
            "scorer_feature_dim": len(
                checkpoint["tensor_contract"]["scorer_feature_names"]
            ),
        },
        {
            "schema_version": schema,
            "row_kind": "adapted_model",
            "model_id": summary["adapted_model_id"],
            "model_family": summary["adapted_model_family"],
            "quality_gate_input_ready": summary["quality_gate_input_ready"],
        },
        {
            "schema_version": schema,
            "row_kind": "prediction_contract",
            "prediction_contract_valid": summary["prediction_contract_valid"],
            "num_predictions": summary["num_predictions"],
        },
        {
            "schema_version": schema,
            "row_kind": "runtime_boundary",
            "guard_required": True,
            "default_mode": "shadow",
            "runtime_selector_changed": False,
        },
        {
            "schema_version": schema,
            "row_kind": "adapter_decision",
            "status": summary["status"],
            "adapter_ready": summary["adapter_ready"],
            "validation_errors": summary["validation_errors"],
        },
    )


def _schema(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": CSR_EXTERNAL_MODEL_ADAPTER_SCHEMA_VERSION,
        "checkpoint_schema_version": CSR_EXTERNAL_RANKER_CHECKPOINT_SCHEMA_VERSION,
        "adapter": CSR_EXTERNAL_RANKER_ADAPTER,
        "outputs": {
            "adapted_model_schema_version": CSR_TRANSFORMER_RANKER_SCHEMA_VERSION,
            "quality_gate_input_ready": summary["quality_gate_input_ready"],
            "policy_model_artifact_input_ready": summary[
                "policy_model_artifact_input_ready"
            ],
        },
        "runtime_boundary": {
            "guard_required": True,
            "default_mode": "shadow",
            "runtime_selector_changed": False,
        },
    }


def _report_text(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# CSR External Model Adapter",
            "",
            f"- status: `{summary['status']}`",
            f"- schema_version: `{summary['schema_version']}`",
            f"- checkpoint_schema_version: `{summary['checkpoint_schema_version']}`",
            f"- adapter: `{summary['adapter']}`",
            f"- framework: `{summary['framework']}`",
            f"- adapted_model_id: `{summary['adapted_model_id']}`",
            f"- quality_gate_input_ready: `{summary['quality_gate_input_ready']}`",
            f"- policy_model_artifact_input_ready: `{summary['policy_model_artifact_input_ready']}`",
            f"- prediction_contract_valid: `{summary['prediction_contract_valid']}`",
            f"- num_predictions: `{summary['num_predictions']}`",
            f"- eval_oracle_top1_accuracy: `{summary['eval_oracle_top1_accuracy']:.6g}`",
            f"- eval_profiled_success_selection_rate: `{summary['eval_profiled_success_selection_rate']:.6g}`",
            f"- eval_non_success_selection_count: `{summary['eval_non_success_selection_count']}`",
            f"- runtime_selector_changed: `{summary['runtime_selector_changed']}`",
            f"- validation_error_count: `{summary['validation_error_count']}`",
            "",
        ]
    )


def _tensor_contract(
    arrays: dict[str, Any],
    request_index: tuple[dict[str, Any], ...],
    *,
    model: dict[str, Any],
) -> dict[str, Any]:
    return {
        "tensor_schema": "csr_transformer_training_tensors_json_v1",
        "num_requests": len(request_index),
        "num_global_candidates": len(arrays["global_candidate_ids"]),
        "matrix_feature_dim": len(arrays["matrix_feature_names"]),
        "candidate_feature_dim": len(arrays["candidate_feature_names"]),
        "token_feature_names": tuple(model["token_feature_names"]),
        "scorer_feature_names": tuple(model["scorer_feature_names"]),
    }


def _prediction_contract_valid(
    predictions: tuple[CsrTransformerRankerPrediction, ...],
) -> bool:
    for prediction in predictions:
        ranked = tuple(prediction.ranked_candidate_ids)
        if not ranked or prediction.selected_candidate_id != ranked[0]:
            return False
        if len(ranked) != len(set(ranked)):
            return False
        if set(ranked) != set(prediction.scores):
            return False
        if any(not math.isfinite(float(value)) for value in prediction.scores.values()):
            return False
    return True


def _token_feature_names(arrays: dict[str, Any]) -> tuple[str, ...]:
    return (
        *(f"matrix:{name}" for name in arrays["matrix_feature_names"]),
        *(f"candidate:{name}" for name in arrays["candidate_feature_names"]),
    )


def _scorer_feature_names(arrays: dict[str, Any], *, d_model: int) -> tuple[str, ...]:
    return (
        "bias",
        *(f"encoded:{index}" for index in range(d_model)),
        *_token_feature_names(arrays),
    )


def _oracle_accuracy(predictions: tuple[CsrTransformerRankerPrediction, ...]) -> float:
    oracle_predictions = tuple(row for row in predictions if row.oracle_candidate_id)
    if not oracle_predictions:
        return 0.0
    hits = sum(row.selected_candidate_id == row.oracle_candidate_id for row in oracle_predictions)
    return hits / len(oracle_predictions)


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(data: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
