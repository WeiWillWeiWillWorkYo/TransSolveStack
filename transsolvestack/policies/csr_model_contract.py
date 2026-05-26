"""CSR learned-selector request, target, and prediction contracts."""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from transsolvestack.profiling.artifacts import read_jsonl, write_jsonl


CSR_MODEL_CONTRACT_SCHEMA_VERSION = "phase1_csr_model_contract_v1"

_MATRIX_FEATURE_KEYS = (
    "actual_symmetric",
    "cg_candidate",
    "csr_nnz",
    "declared_symmetry",
    "diagonal_present_count",
    "field",
    "has_nonpositive_diagonal",
    "is_square",
    "max_abs_diagonal",
    "min_abs_diagonal",
    "n_cols",
    "n_rows",
    "recommended_precision",
    "symmetry_max_abs_error",
    "symmetry_missing_pairs",
    "symmetry_relative_error",
    "zero_diagonal_count",
)


@dataclass(frozen=True)
class CsrSelectorModelRequest:
    schema_version: str
    request_id: str
    split: str
    matrix_id: str
    context_id: str
    candidate_ids: tuple[str, ...]
    matrix_features: dict[str, Any] = field(default_factory=dict)
    candidate_features: dict[str, dict[str, Any]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CsrSelectorModelTarget:
    schema_version: str
    request_id: str
    split: str
    matrix_id: str
    context_id: str
    oracle_candidate_id: str | None
    profiled_success_candidate_ids: tuple[str, ...]
    candidate_label_classes: dict[str, str]
    candidate_target_statuses: dict[str, str]
    candidate_median_solve_time_ms: dict[str, float | None]
    candidate_regret_vs_oracle_ms: dict[str, float | None]


@dataclass(frozen=True)
class CsrSelectorModelPrediction:
    schema_version: str
    request_id: str
    split: str
    matrix_id: str
    context_id: str
    model_id: str
    ranked_candidate_ids: tuple[str, ...]
    scores: dict[str, float]
    selected_candidate_id: str | None
    evaluation_status: str
    oracle_candidate_id: str | None
    regret_vs_oracle_ms: float | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CsrSelectorModelContractSummary:
    status: str
    schema_version: str
    prediction_source: str
    model_required: bool
    runtime_selector_changed: bool
    num_requests: int
    num_targets: int
    num_predictions: int
    num_train_requests: int
    num_eval_requests: int
    num_request_candidates: int
    min_candidates_per_request: int
    max_candidates_per_request: int
    target_label_class_counts: dict[str, int]
    num_predictions_with_selected_candidate: int
    num_predictions_with_profiled_success_selection: int
    eval_oracle_top1_accuracy: float
    eval_profiled_selection_rate: float
    eval_mean_regret_ms: float | None
    eval_max_regret_ms: float | None
    validation_error_count: int


@dataclass(frozen=True)
class CsrSelectorModelContractExport:
    requests: tuple[CsrSelectorModelRequest, ...]
    targets: tuple[CsrSelectorModelTarget, ...]
    predictions: tuple[CsrSelectorModelPrediction, ...]
    summary: CsrSelectorModelContractSummary


def build_csr_model_contract_export(
    learning_rows_path: str | Path,
    baseline_predictions_path: str | Path,
    *,
    prediction_source: str = "csr_learning_baseline_adapter_v1",
) -> CsrSelectorModelContractExport:
    learning_rows = tuple(read_jsonl(learning_rows_path))
    baseline_predictions = tuple(read_jsonl(baseline_predictions_path))
    grouped = _group_learning_rows(learning_rows)
    requests = tuple(
        _request_from_rows(request_id, rows)
        for request_id, rows in sorted(grouped.items())
    )
    targets = tuple(
        _target_from_rows(request_id, rows)
        for request_id, rows in sorted(grouped.items())
    )
    request_by_id = {request.request_id: request for request in requests}
    target_by_id = {target.request_id: target for target in targets}
    predictions = tuple(
        _prediction_from_baseline(
            row,
            request_by_id=_request_by_matrix_context_split(requests),
            target_by_id=target_by_id,
            prediction_source=prediction_source,
        )
        for row in baseline_predictions
    )
    validation_errors = _validation_errors(request_by_id, predictions)
    summary = _summary(
        requests,
        targets,
        predictions,
        prediction_source=prediction_source,
        validation_error_count=len(validation_errors),
    )
    return CsrSelectorModelContractExport(
        requests=requests,
        targets=targets,
        predictions=predictions,
        summary=summary,
    )


def validate_csr_model_prediction(
    request: CsrSelectorModelRequest,
    prediction: CsrSelectorModelPrediction,
) -> None:
    if prediction.schema_version != request.schema_version:
        raise ValueError("CSR model prediction schema version mismatch")
    if prediction.request_id != request.request_id:
        raise ValueError("CSR model prediction request_id mismatch")
    if prediction.matrix_id != request.matrix_id:
        raise ValueError("CSR model prediction matrix_id mismatch")
    if prediction.context_id != request.context_id:
        raise ValueError("CSR model prediction context_id mismatch")
    if prediction.split != request.split:
        raise ValueError("CSR model prediction split mismatch")
    requested = set(request.candidate_ids)
    ranked = tuple(prediction.ranked_candidate_ids)
    if set(ranked) != requested:
        raise ValueError("CSR model prediction must rank every requested candidate")
    if len(ranked) != len(request.candidate_ids):
        raise ValueError("CSR model prediction contains duplicate candidates")
    missing_scores = requested - set(prediction.scores)
    if missing_scores:
        raise ValueError(f"CSR model prediction missing scores: {sorted(missing_scores)}")
    nonfinite_scores = [
        candidate_id
        for candidate_id, score in prediction.scores.items()
        if candidate_id in requested and not math.isfinite(float(score))
    ]
    if nonfinite_scores:
        raise ValueError(f"CSR model prediction has nonfinite scores: {nonfinite_scores}")
    if prediction.selected_candidate_id is not None:
        if prediction.selected_candidate_id not in requested:
            raise ValueError("CSR model prediction selected unknown candidate")
        if not ranked or ranked[0] != prediction.selected_candidate_id:
            raise ValueError("CSR model prediction selected candidate must be ranked first")


def write_csr_model_requests(
    rows: Iterable[CsrSelectorModelRequest],
    path: str | Path,
) -> Path:
    return write_jsonl((asdict(row) for row in rows), path)


def write_csr_model_targets(
    rows: Iterable[CsrSelectorModelTarget],
    path: str | Path,
) -> Path:
    return write_jsonl((asdict(row) for row in rows), path)


def write_csr_model_predictions(
    rows: Iterable[CsrSelectorModelPrediction],
    path: str | Path,
) -> Path:
    return write_jsonl((asdict(row) for row in rows), path)


def write_csr_model_contract_summary(
    summary: CsrSelectorModelContractSummary,
    path: str | Path,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def write_csr_model_contract_schema(path: str | Path) -> Path:
    schema = {
        "schema_version": CSR_MODEL_CONTRACT_SCHEMA_VERSION,
        "task": "rank_real_csr_solver_candidates",
        "model_required": False,
        "request": {
            "unit": "matrix_id_context_id_split",
            "candidate_ids": "all candidate rows for the matrix/context",
            "matrix_features": list(_MATRIX_FEATURE_KEYS),
            "candidate_features": [
                "candidate_id",
                "solver",
                "preconditioner",
                "precision",
                "solver_parameters",
            ],
            "label_free": True,
        },
        "target": {
            "offline_only": True,
            "fields": [
                "oracle_candidate_id",
                "profiled_success_candidate_ids",
                "candidate_label_classes",
                "candidate_target_statuses",
                "candidate_median_solve_time_ms",
                "candidate_regret_vs_oracle_ms",
            ],
        },
        "prediction": {
            "required_fields": [
                "ranked_candidate_ids",
                "scores",
                "selected_candidate_id",
                "model_id",
            ],
            "validation": [
                "schema_version matches request",
                "request/matrix/context/split match",
                "ranked_candidate_ids contain each requested candidate exactly once",
                "scores are finite for every requested candidate",
                "selected_candidate_id is null or ranked first",
            ],
        },
        "runtime_integration": {
            "status": "contract_only_not_runtime_selector",
            "runtime_selector_changed": False,
            "current_runtime_selector": "csr_artifact_success_rows",
            "next_step": "train_or_connect_ranker_then_gate_against_artifact_selector",
        },
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def write_csr_model_contract_report(
    export: CsrSelectorModelContractExport,
    path: str | Path,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = export.summary
    lines = [
        "# CSR Model Contract",
        "",
        f"- status: `{summary.status}`",
        f"- schema_version: `{summary.schema_version}`",
        f"- prediction_source: `{summary.prediction_source}`",
        f"- model_required: `{summary.model_required}`",
        f"- runtime_selector_changed: `{summary.runtime_selector_changed}`",
        f"- requests: `{summary.num_requests}`",
        f"- targets: `{summary.num_targets}`",
        f"- predictions: `{summary.num_predictions}`",
        f"- train_requests: `{summary.num_train_requests}`",
        f"- eval_requests: `{summary.num_eval_requests}`",
        f"- request_candidates: `{summary.num_request_candidates}`",
        f"- candidates_per_request: `{summary.min_candidates_per_request}` to `{summary.max_candidates_per_request}`",
        f"- target_label_class_counts: `{summary.target_label_class_counts}`",
        f"- eval_oracle_top1_accuracy: `{summary.eval_oracle_top1_accuracy:.6g}`",
        f"- eval_profiled_selection_rate: `{summary.eval_profiled_selection_rate:.6g}`",
        f"- eval_mean_regret_ms: `{_fmt(summary.eval_mean_regret_ms)}`",
        f"- eval_max_regret_ms: `{_fmt(summary.eval_max_regret_ms)}`",
        f"- validation_error_count: `{summary.validation_error_count}`",
        "",
        "| matrix | selected | oracle | status | regret_ms | model_id |",
        "|---|---|---|---|---:|---|",
    ]
    for row in export.predictions:
        lines.append(
            "| "
            f"{row.matrix_id} | "
            f"{row.selected_candidate_id or ''} | "
            f"{row.oracle_candidate_id or ''} | "
            f"{row.evaluation_status} | "
            f"{_fmt(row.regret_vs_oracle_ms)} | "
            f"{row.model_id} |"
        )
    lines.append("")
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def _group_learning_rows(rows: Iterable[dict[str, Any]]) -> dict[str, tuple[dict[str, Any], ...]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        request_id = _request_id(
            split=str(row["split"]),
            matrix_id=str(row["matrix_id"]),
            context_id=str(row["context_id"]),
        )
        grouped.setdefault(request_id, []).append(row)
    return {key: tuple(value) for key, value in grouped.items()}


def _request_from_rows(
    request_id: str,
    rows: tuple[dict[str, Any], ...],
) -> CsrSelectorModelRequest:
    first = rows[0]
    candidate_ids = tuple(sorted(str(row["candidate_id"]) for row in rows))
    feature_source = dict(first.get("features", {}))
    return CsrSelectorModelRequest(
        schema_version=CSR_MODEL_CONTRACT_SCHEMA_VERSION,
        request_id=request_id,
        split=str(first["split"]),
        matrix_id=str(first["matrix_id"]),
        context_id=str(first["context_id"]),
        candidate_ids=candidate_ids,
        matrix_features={
            key: feature_source.get(key)
            for key in _MATRIX_FEATURE_KEYS
            if key in feature_source
        },
        candidate_features={
            str(row["candidate_id"]): _candidate_features(row)
            for row in sorted(rows, key=lambda item: str(item["candidate_id"]))
        },
        metadata={
            "source_schema_version": str(first["schema_version"]),
            "feature_policy": "label_free_static_matrix_and_candidate_features",
        },
    )


def _target_from_rows(
    request_id: str,
    rows: tuple[dict[str, Any], ...],
) -> CsrSelectorModelTarget:
    first = rows[0]
    sorted_rows = tuple(sorted(rows, key=lambda item: str(item["candidate_id"])))
    oracle = next((row for row in sorted_rows if row["label_is_oracle"]), None)
    return CsrSelectorModelTarget(
        schema_version=CSR_MODEL_CONTRACT_SCHEMA_VERSION,
        request_id=request_id,
        split=str(first["split"]),
        matrix_id=str(first["matrix_id"]),
        context_id=str(first["context_id"]),
        oracle_candidate_id=str(oracle["candidate_id"]) if oracle else None,
        profiled_success_candidate_ids=tuple(
            str(row["candidate_id"])
            for row in sorted_rows
            if row["target_status"] == "success" and float(row["target_success_rate"]) >= 1.0
        ),
        candidate_label_classes={
            str(row["candidate_id"]): str(row["label_class"]) for row in sorted_rows
        },
        candidate_target_statuses={
            str(row["candidate_id"]): str(row["target_status"]) for row in sorted_rows
        },
        candidate_median_solve_time_ms={
            str(row["candidate_id"]): _optional_float(row.get("target_median_solve_time_ms"))
            for row in sorted_rows
        },
        candidate_regret_vs_oracle_ms={
            str(row["candidate_id"]): _optional_float(row.get("target_regret_vs_oracle_ms"))
            for row in sorted_rows
        },
    )


def _prediction_from_baseline(
    row: dict[str, Any],
    *,
    request_by_id: dict[tuple[str, str, str], CsrSelectorModelRequest],
    target_by_id: dict[str, CsrSelectorModelTarget],
    prediction_source: str,
) -> CsrSelectorModelPrediction:
    request = _request_for_baseline_prediction(row, request_by_id)
    target = target_by_id[request.request_id]
    selected_candidate_id = _optional_str(row.get("predicted_candidate_id"))
    ranked = _ranked_candidates(request.candidate_ids, selected_candidate_id)
    scores = {
        candidate_id: float(len(ranked) - index)
        for index, candidate_id in enumerate(ranked)
    }
    return CsrSelectorModelPrediction(
        schema_version=CSR_MODEL_CONTRACT_SCHEMA_VERSION,
        request_id=request.request_id,
        split=request.split,
        matrix_id=request.matrix_id,
        context_id=request.context_id,
        model_id=prediction_source,
        ranked_candidate_ids=ranked,
        scores=scores,
        selected_candidate_id=selected_candidate_id,
        evaluation_status=str(row["status"]),
        oracle_candidate_id=target.oracle_candidate_id,
        regret_vs_oracle_ms=_optional_float(row.get("regret_vs_oracle_ms")),
        metadata={
            "source_prediction_schema_version": str(row["schema_version"]),
            "source_reason": str(row["reason"]),
            "adapter_note": "baseline prediction adapted to model contract; no model inference ran",
        },
    )


def _request_for_baseline_prediction(
    row: dict[str, Any],
    request_by_id: dict[tuple[str, str, str], CsrSelectorModelRequest],
) -> CsrSelectorModelRequest:
    split = str(row["split"])
    matrix_id = str(row["matrix_id"])
    context_id = row.get("context_id")
    if context_id is not None:
        key = (split, matrix_id, str(context_id))
        if key in request_by_id:
            return request_by_id[key]
    matches = tuple(
        request
        for (request_split, request_matrix_id, _request_context), request in request_by_id.items()
        if request_split == split and request_matrix_id == matrix_id
    )
    if len(matches) != 1:
        raise KeyError(
            f"expected exactly one request for baseline prediction "
            f"split={split!r} matrix_id={matrix_id!r}, got {len(matches)}"
        )
    return matches[0]


def _request_by_matrix_context_split(
    requests: Iterable[CsrSelectorModelRequest],
) -> dict[tuple[str, str, str], CsrSelectorModelRequest]:
    return {
        (request.split, request.matrix_id, request.context_id): request
        for request in requests
    }


def _validation_errors(
    request_by_id: dict[str, CsrSelectorModelRequest],
    predictions: Iterable[CsrSelectorModelPrediction],
) -> tuple[str, ...]:
    errors: list[str] = []
    for prediction in predictions:
        request = request_by_id.get(prediction.request_id)
        if request is None:
            errors.append(f"unknown request_id: {prediction.request_id}")
            continue
        try:
            validate_csr_model_prediction(request, prediction)
        except ValueError as exc:
            errors.append(f"{prediction.request_id}: {exc}")
    return tuple(errors)


def _summary(
    requests: tuple[CsrSelectorModelRequest, ...],
    targets: tuple[CsrSelectorModelTarget, ...],
    predictions: tuple[CsrSelectorModelPrediction, ...],
    *,
    prediction_source: str,
    validation_error_count: int,
) -> CsrSelectorModelContractSummary:
    target_by_id = {target.request_id: target for target in targets}
    selected_predictions = tuple(
        row for row in predictions if row.selected_candidate_id is not None
    )
    profiled_success_predictions = tuple(
        row
        for row in selected_predictions
        if row.selected_candidate_id
        in set(target_by_id[row.request_id].profiled_success_candidate_ids)
    )
    regrets = [
        float(row.regret_vs_oracle_ms)
        for row in profiled_success_predictions
        if row.regret_vs_oracle_ms is not None
    ]
    candidate_counts = [len(request.candidate_ids) for request in requests]
    accuracy = (
        sum(
            1
            for row in predictions
            if row.selected_candidate_id is not None
            and row.selected_candidate_id == target_by_id[row.request_id].oracle_candidate_id
        )
        / len(predictions)
        if predictions
        else 0.0
    )
    profiled_rate = (
        len(profiled_success_predictions) / len(predictions) if predictions else 0.0
    )
    label_counts = Counter()
    for target in targets:
        label_counts.update(target.candidate_label_classes.values())
    status = (
        "passed"
        if requests
        and targets
        and predictions
        and len(requests) == len(targets)
        and validation_error_count == 0
        and all(request.candidate_ids for request in requests)
        else "failed"
    )
    return CsrSelectorModelContractSummary(
        status=status,
        schema_version=CSR_MODEL_CONTRACT_SCHEMA_VERSION,
        prediction_source=prediction_source,
        model_required=False,
        runtime_selector_changed=False,
        num_requests=len(requests),
        num_targets=len(targets),
        num_predictions=len(predictions),
        num_train_requests=sum(1 for request in requests if request.split == "train"),
        num_eval_requests=sum(1 for request in requests if request.split == "eval"),
        num_request_candidates=sum(candidate_counts),
        min_candidates_per_request=min(candidate_counts) if candidate_counts else 0,
        max_candidates_per_request=max(candidate_counts) if candidate_counts else 0,
        target_label_class_counts=dict(sorted(label_counts.items())),
        num_predictions_with_selected_candidate=len(selected_predictions),
        num_predictions_with_profiled_success_selection=len(profiled_success_predictions),
        eval_oracle_top1_accuracy=accuracy,
        eval_profiled_selection_rate=profiled_rate,
        eval_mean_regret_ms=(sum(regrets) / len(regrets) if regrets else None),
        eval_max_regret_ms=(max(regrets) if regrets else None),
        validation_error_count=validation_error_count,
    )


def _candidate_features(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": str(row["candidate_id"]),
        "solver": str(row["solver"]),
        "preconditioner": str(row["preconditioner"]),
        "precision": str(row["precision"]),
        "solver_parameters": dict(row.get("solver_parameters", {})),
    }


def _ranked_candidates(
    candidate_ids: tuple[str, ...],
    selected_candidate_id: str | None,
) -> tuple[str, ...]:
    if selected_candidate_id is None:
        return tuple(sorted(candidate_ids))
    remaining = tuple(sorted(candidate_id for candidate_id in candidate_ids if candidate_id != selected_candidate_id))
    return (selected_candidate_id, *remaining)


def _request_id(*, split: str, matrix_id: str, context_id: str) -> str:
    return f"{split}:{matrix_id}:{context_id}"


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _fmt(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6g}"
