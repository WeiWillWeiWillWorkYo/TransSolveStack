"""Replay saved CSR Transformer ranker models without retraining."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from transsolvestack.policies.csr_transformer_ranker import (
    CsrTransformerRankerPrediction,
    load_csr_transformer_ranker_model,
    predict_csr_transformer_ranker_from_model,
)
from transsolvestack.profiling.artifacts import read_jsonl, write_jsonl


CSR_TRANSFORMER_MODEL_REPLAY_SCHEMA_VERSION = "phase1_csr_transformer_model_replay_v1"


@dataclass(frozen=True)
class CsrTransformerModelReplaySummary:
    status: str
    schema_version: str
    source_model_schema_version: str
    model_family: str
    model_id: str
    model_loaded: bool
    model_trained: bool
    runtime_selector_changed: bool
    num_requests: int
    num_predictions: int
    num_reference_predictions: int
    exact_replay: bool
    selected_candidate_mismatch_count: int
    ranked_order_mismatch_count: int
    evaluation_status_mismatch_count: int
    missing_reference_count: int
    max_abs_score_delta: float
    prediction_contract_valid: bool


@dataclass(frozen=True)
class CsrTransformerModelReplayExport:
    predictions: tuple[CsrTransformerRankerPrediction, ...]
    summary: CsrTransformerModelReplaySummary
    schema: dict[str, Any]
    comparison_rows: tuple[dict[str, Any], ...]


def build_csr_transformer_model_replay_from_files(
    model_path: str | Path = "runs/phase1_csr_transformer_ranker/csr_transformer_ranker_model.json",
    tensor_path: str | Path = (
        "runs/phase1_csr_transformer_ready/csr_transformer_training_tensors.json"
    ),
    request_index_path: str | Path = (
        "runs/phase1_csr_transformer_ready/csr_transformer_request_index.jsonl"
    ),
    reference_predictions_path: str | Path = (
        "runs/phase1_csr_transformer_ranker/csr_transformer_ranker_predictions.jsonl"
    ),
) -> CsrTransformerModelReplayExport:
    """Replay a saved model and compare it with the training-time predictions."""

    model = load_csr_transformer_ranker_model(model_path)
    arrays = json.loads(Path(tensor_path).read_text(encoding="utf-8"))
    request_index = tuple(read_jsonl(request_index_path))
    predictions = predict_csr_transformer_ranker_from_model(model, arrays, request_index)
    references = tuple(read_jsonl(reference_predictions_path))
    comparison_rows = _comparison_rows(predictions, references)
    summary = _summary(model=model, predictions=predictions, references=references, rows=comparison_rows)
    return CsrTransformerModelReplayExport(
        predictions=predictions,
        summary=summary,
        schema=_schema(),
        comparison_rows=comparison_rows,
    )


def write_csr_transformer_model_replay_predictions(
    predictions: Iterable[CsrTransformerRankerPrediction],
    path: str | Path,
) -> Path:
    return write_jsonl((asdict(row) for row in predictions), path)


def write_csr_transformer_model_replay_comparison_rows(
    rows: Iterable[dict[str, Any]],
    path: str | Path,
) -> Path:
    return write_jsonl(rows, path)


def write_csr_transformer_model_replay_summary(
    summary: CsrTransformerModelReplaySummary,
    path: str | Path,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def write_csr_transformer_model_replay_schema(schema: dict[str, Any], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def write_csr_transformer_model_replay_report(
    export: CsrTransformerModelReplayExport,
    path: str | Path,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = export.summary
    lines = [
        "# CSR Transformer Model Replay",
        "",
        f"- status: `{summary.status}`",
        f"- schema_version: `{summary.schema_version}`",
        f"- model_family: `{summary.model_family}`",
        f"- model_id: `{summary.model_id}`",
        f"- model_loaded: `{summary.model_loaded}`",
        f"- model_trained: `{summary.model_trained}`",
        f"- runtime_selector_changed: `{summary.runtime_selector_changed}`",
        f"- predictions: `{summary.num_predictions}`",
        f"- reference_predictions: `{summary.num_reference_predictions}`",
        f"- exact_replay: `{summary.exact_replay}`",
        f"- selected_candidate_mismatch_count: `{summary.selected_candidate_mismatch_count}`",
        f"- ranked_order_mismatch_count: `{summary.ranked_order_mismatch_count}`",
        f"- evaluation_status_mismatch_count: `{summary.evaluation_status_mismatch_count}`",
        f"- missing_reference_count: `{summary.missing_reference_count}`",
        f"- max_abs_score_delta: `{summary.max_abs_score_delta:.6g}`",
        f"- prediction_contract_valid: `{summary.prediction_contract_valid}`",
        "",
        "| request | selected_match | rank_match | status_match | max_score_delta |",
        "|---|---|---|---|---:|",
    ]
    for row in export.comparison_rows:
        lines.append(
            "| "
            f"{row['request_id']} | "
            f"{row['selected_candidate_match']} | "
            f"{row['ranked_order_match']} | "
            f"{row['evaluation_status_match']} | "
            f"{row['max_abs_score_delta']:.6g} |"
        )
    lines.append("")
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def _comparison_rows(
    predictions: tuple[CsrTransformerRankerPrediction, ...],
    references: tuple[dict[str, Any], ...],
) -> tuple[dict[str, Any], ...]:
    reference_by_id = {str(row["request_id"]): row for row in references}
    rows: list[dict[str, Any]] = []
    for prediction in predictions:
        reference = reference_by_id.get(prediction.request_id)
        if reference is None:
            rows.append(
                {
                    "schema_version": CSR_TRANSFORMER_MODEL_REPLAY_SCHEMA_VERSION,
                    "request_id": prediction.request_id,
                    "selected_candidate_match": False,
                    "ranked_order_match": False,
                    "evaluation_status_match": False,
                    "missing_reference": True,
                    "max_abs_score_delta": math.inf,
                }
            )
            continue
        ranked_match = tuple(reference["ranked_candidate_ids"]) == prediction.ranked_candidate_ids
        score_delta = _max_score_delta(prediction.scores, reference["scores"])
        rows.append(
            {
                "schema_version": CSR_TRANSFORMER_MODEL_REPLAY_SCHEMA_VERSION,
                "request_id": prediction.request_id,
                "selected_candidate_match": (
                    str(reference["selected_candidate_id"]) == prediction.selected_candidate_id
                ),
                "ranked_order_match": ranked_match,
                "evaluation_status_match": (
                    str(reference["evaluation_status"]) == prediction.evaluation_status
                ),
                "missing_reference": False,
                "max_abs_score_delta": score_delta,
            }
        )
    return tuple(rows)


def _max_score_delta(left: dict[str, float], right: dict[str, Any]) -> float:
    keys = set(left) | {str(key) for key in right}
    if not keys:
        return 0.0
    deltas = []
    for key in keys:
        if key not in left or key not in right:
            return math.inf
        deltas.append(abs(float(left[key]) - float(right[key])))
    return max(deltas)


def _summary(
    *,
    model: dict[str, Any],
    predictions: tuple[CsrTransformerRankerPrediction, ...],
    references: tuple[dict[str, Any], ...],
    rows: tuple[dict[str, Any], ...],
) -> CsrTransformerModelReplaySummary:
    selected_mismatches = sum(not row["selected_candidate_match"] for row in rows)
    rank_mismatches = sum(not row["ranked_order_match"] for row in rows)
    status_mismatches = sum(not row["evaluation_status_match"] for row in rows)
    missing_references = sum(row["missing_reference"] for row in rows)
    max_score_delta = max((float(row["max_abs_score_delta"]) for row in rows), default=0.0)
    contract_valid = _prediction_contract_valid(predictions)
    exact_replay = (
        len(predictions) == len(references)
        and selected_mismatches == 0
        and rank_mismatches == 0
        and status_mismatches == 0
        and missing_references == 0
        and max_score_delta <= 1.0e-9
    )
    status = "passed" if predictions and exact_replay and contract_valid else "failed"
    return CsrTransformerModelReplaySummary(
        status=status,
        schema_version=CSR_TRANSFORMER_MODEL_REPLAY_SCHEMA_VERSION,
        source_model_schema_version=str(model["schema_version"]),
        model_family=str(model["model_family"]),
        model_id=str(model["model_id"]),
        model_loaded=True,
        model_trained=True,
        runtime_selector_changed=False,
        num_requests=len(predictions),
        num_predictions=len(predictions),
        num_reference_predictions=len(references),
        exact_replay=exact_replay,
        selected_candidate_mismatch_count=selected_mismatches,
        ranked_order_mismatch_count=rank_mismatches,
        evaluation_status_mismatch_count=status_mismatches,
        missing_reference_count=missing_references,
        max_abs_score_delta=max_score_delta,
        prediction_contract_valid=contract_valid,
    )


def _prediction_contract_valid(predictions: tuple[CsrTransformerRankerPrediction, ...]) -> bool:
    for prediction in predictions:
        ranked = prediction.ranked_candidate_ids
        if not ranked or prediction.selected_candidate_id != ranked[0]:
            return False
        if len(ranked) != len(set(ranked)):
            return False
        if set(ranked) != set(prediction.scores):
            return False
        if not all(math.isfinite(float(value)) for value in prediction.scores.values()):
            return False
    return True


def _schema() -> dict[str, Any]:
    return {
        "schema_version": CSR_TRANSFORMER_MODEL_REPLAY_SCHEMA_VERSION,
        "task": "saved_csr_transformer_ranker_model_replay",
        "runtime_selector_changed": False,
        "checks": {
            "model_json_load": True,
            "tensor_schema_match": True,
            "prediction_contract": True,
            "reference_prediction_replay": True,
        },
        "runtime_integration": {
            "status": "offline_model_loading_boundary",
            "guard_required_before_execution": True,
        },
    }
