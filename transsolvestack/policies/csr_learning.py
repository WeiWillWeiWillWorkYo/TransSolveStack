"""CSR selector learning-readiness dataset and baseline evaluator."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from transsolvestack.profiling.artifacts import read_jsonl, write_jsonl


CSR_LEARNING_SCHEMA_VERSION = "phase1_csr_learning_features_v1"


@dataclass(frozen=True)
class CsrLearningRow:
    schema_version: str
    split: str
    matrix_id: str
    context_id: str
    candidate_id: str
    solver: str
    preconditioner: str
    precision: str
    solver_parameters: dict[str, Any]
    label_class: str
    label_is_oracle: bool
    target_status: str
    target_success_rate: float
    target_median_solve_time_ms: float | None
    target_regret_vs_oracle_ms: float | None
    target_failure_reason: str | None
    target_applicability_status: str
    target_applicability_reason: str | None
    features: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CsrBaselinePrediction:
    schema_version: str
    split: str
    matrix_id: str
    predicted_candidate_id: str | None
    oracle_candidate_id: str | None
    status: str
    predicted_label_class: str | None
    oracle_median_solve_time_ms: float | None
    predicted_median_solve_time_ms: float | None
    regret_vs_oracle_ms: float | None
    reason: str


@dataclass(frozen=True)
class CsrLearningSummary:
    status: str
    schema_version: str
    baseline_id: str
    num_rows: int
    num_train_rows: int
    num_eval_rows: int
    num_matrices: int
    num_train_matrices: int
    num_eval_matrices: int
    label_class_counts: dict[str, int]
    num_eval_matrices_with_success: int
    num_eval_predictions: int
    num_eval_profiled_success_predictions: int
    eval_oracle_top1_accuracy: float
    eval_profiled_success_rate: float
    eval_mean_regret_ms: float | None
    eval_max_regret_ms: float | None
    baseline_candidate_order: tuple[str, ...]


@dataclass(frozen=True)
class CsrLearningExport:
    rows: tuple[CsrLearningRow, ...]
    predictions: tuple[CsrBaselinePrediction, ...]
    summary: CsrLearningSummary


def build_csr_learning_export_from_selector_rows(
    selector_rows_path: str | Path,
    *,
    eval_fraction: float = 0.25,
    min_eval_matrices: int = 2,
    baseline_id: str = "candidate_prior_success_median_v1",
) -> CsrLearningExport:
    selector_rows = tuple(read_jsonl(selector_rows_path))
    split_by_matrix = _matrix_split(
        selector_rows,
        eval_fraction=eval_fraction,
        min_eval_matrices=min_eval_matrices,
    )
    rows = tuple(_learning_row(row, split_by_matrix[str(row["matrix_id"])]) for row in selector_rows)
    train_rows = tuple(row for row in rows if row.split == "train")
    eval_rows = tuple(row for row in rows if row.split == "eval")
    baseline_order = _baseline_candidate_order(train_rows)
    predictions = _baseline_predictions(eval_rows, baseline_order)
    summary = _summary(
        rows,
        predictions,
        baseline_id=baseline_id,
        baseline_order=baseline_order,
    )
    return CsrLearningExport(rows=rows, predictions=predictions, summary=summary)


def write_csr_learning_rows(rows: Iterable[CsrLearningRow], path: str | Path) -> Path:
    return write_jsonl((asdict(row) for row in rows), path)


def write_csr_baseline_predictions(
    rows: Iterable[CsrBaselinePrediction],
    path: str | Path,
) -> Path:
    return write_jsonl((asdict(row) for row in rows), path)


def write_csr_learning_summary(summary: CsrLearningSummary, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def write_csr_learning_schema(path: str | Path) -> Path:
    schema = {
        "schema_version": CSR_LEARNING_SCHEMA_VERSION,
        "task": "learn_rank_and_filter_real_csr_solver_candidates",
        "model_required": False,
        "baseline": "candidate_prior_success_median_v1",
        "splitting": {
            "unit": "matrix_id",
            "deterministic": True,
            "eval_fraction_default": 0.25,
            "min_eval_matrices_default": 2,
        },
        "labels": [
            "success_oracle",
            "success_non_oracle",
            "screened_out",
            "not_applicable",
            "not_profiled",
        ],
        "targets": [
            "label_class",
            "label_is_oracle",
            "target_status",
            "target_success_rate",
            "target_median_solve_time_ms",
            "target_regret_vs_oracle_ms",
            "target_failure_reason",
            "target_applicability_status",
            "target_applicability_reason",
        ],
        "integration_boundary": {
            "status": "ready_for_baseline_model_comparison",
            "next_step": "scale_csr_rows_then_train_transformer_ranker",
        },
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(schema, indent=2, sort_keys=True), encoding="utf-8")
    return output


def write_csr_learning_report(export: CsrLearningExport, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = export.summary
    lines = [
        "# CSR Learning Readiness",
        "",
        f"- status: `{summary.status}`",
        f"- schema_version: `{summary.schema_version}`",
        f"- baseline_id: `{summary.baseline_id}`",
        f"- rows: `{summary.num_rows}`",
        f"- train_rows: `{summary.num_train_rows}`",
        f"- eval_rows: `{summary.num_eval_rows}`",
        f"- train_matrices: `{summary.num_train_matrices}`",
        f"- eval_matrices: `{summary.num_eval_matrices}`",
        f"- label_class_counts: `{summary.label_class_counts}`",
        f"- eval_oracle_top1_accuracy: `{summary.eval_oracle_top1_accuracy:.6g}`",
        f"- eval_profiled_success_rate: `{summary.eval_profiled_success_rate:.6g}`",
        f"- eval_mean_regret_ms: `{_fmt(summary.eval_mean_regret_ms)}`",
        f"- eval_max_regret_ms: `{_fmt(summary.eval_max_regret_ms)}`",
        "",
        "| matrix | predicted | oracle | status | regret_ms | reason |",
        "|---|---|---|---|---:|---|",
    ]
    for row in export.predictions:
        lines.append(
            "| "
            f"{row.matrix_id} | "
            f"{row.predicted_candidate_id or ''} | "
            f"{row.oracle_candidate_id or ''} | "
            f"{row.status} | "
            f"{_fmt(row.regret_vs_oracle_ms)} | "
            f"{row.reason} |"
        )
    lines.append("")
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def _matrix_split(
    rows: Iterable[dict[str, Any]],
    *,
    eval_fraction: float,
    min_eval_matrices: int,
) -> dict[str, str]:
    rows_tuple = tuple(rows)
    matrix_ids = tuple(sorted({str(row["matrix_id"]) for row in rows_tuple}))
    if not matrix_ids:
        return {}
    eval_count = max(min_eval_matrices, int(round(len(matrix_ids) * eval_fraction)))
    eval_count = min(max(eval_count, 1), max(len(matrix_ids) - 1, 1))
    success_ids = tuple(
        sorted(
            {
                str(row["matrix_id"])
                for row in rows_tuple
                if row.get("target_status") == "success"
            }
        )
    )
    eval_success_count = min(
        max(1, min(min_eval_matrices, eval_count)),
        max(len(success_ids) - 1, 1),
    )
    eval_ids = set(_pick_evenly(success_ids, eval_success_count))
    remaining = eval_count - len(eval_ids)
    if remaining > 0:
        non_success_only = tuple(matrix_id for matrix_id in matrix_ids if matrix_id not in success_ids)
        eval_ids.update(non_success_only[-remaining:])
    return {matrix_id: "eval" if matrix_id in eval_ids else "train" for matrix_id in matrix_ids}


def _pick_evenly(items: tuple[str, ...], count: int) -> tuple[str, ...]:
    if count <= 0 or not items:
        return ()
    if count >= len(items):
        return items
    if count == 1:
        return (items[0],)
    return tuple(items[int(index * len(items) / count)] for index in range(count))


def _learning_row(row: dict[str, Any], split: str) -> CsrLearningRow:
    return CsrLearningRow(
        schema_version=CSR_LEARNING_SCHEMA_VERSION,
        split=split,
        matrix_id=str(row["matrix_id"]),
        context_id=str(row["context_id"]),
        candidate_id=str(row["candidate_id"]),
        solver=str(row["solver"]),
        preconditioner=str(row["preconditioner"]),
        precision=str(row["precision"]),
        solver_parameters=dict(row.get("solver_parameters", {})),
        label_class=_label_class(row),
        label_is_oracle=bool(row["label_is_oracle"]),
        target_status=str(row["target_status"]),
        target_success_rate=float(row.get("target_success_rate", 0.0)),
        target_median_solve_time_ms=_optional_float(row.get("target_median_solve_time_ms")),
        target_regret_vs_oracle_ms=_optional_float(row.get("target_regret_vs_oracle_ms")),
        target_failure_reason=_optional_str(row.get("target_failure_reason")),
        target_applicability_status=str(row.get("target_applicability_status", "unknown")),
        target_applicability_reason=_optional_str(row.get("target_applicability_reason")),
        features=dict(row.get("features", {})),
    )


def _label_class(row: dict[str, Any]) -> str:
    status = str(row["target_status"])
    if status == "success":
        return "success_oracle" if row["label_is_oracle"] else "success_non_oracle"
    return status


def _baseline_candidate_order(rows: tuple[CsrLearningRow, ...]) -> tuple[str, ...]:
    successful = [row for row in rows if row.target_status == "success"]
    grouped: dict[str, list[CsrLearningRow]] = {}
    for row in successful:
        grouped.setdefault(row.candidate_id, []).append(row)
    ranked = []
    for candidate_id, candidate_rows in grouped.items():
        median_times = [
            float(row.target_median_solve_time_ms)
            for row in candidate_rows
            if row.target_median_solve_time_ms is not None
        ]
        oracle_count = sum(1 for row in candidate_rows if row.label_is_oracle)
        ranked.append(
            (
                -len(candidate_rows),
                -oracle_count,
                sum(median_times) / len(median_times) if median_times else float("inf"),
                candidate_id,
            )
        )
    return tuple(item[3] for item in sorted(ranked))


def _baseline_predictions(
    eval_rows: tuple[CsrLearningRow, ...],
    baseline_order: tuple[str, ...],
) -> tuple[CsrBaselinePrediction, ...]:
    by_matrix: dict[str, list[CsrLearningRow]] = {}
    for row in eval_rows:
        by_matrix.setdefault(row.matrix_id, []).append(row)
    predictions: list[CsrBaselinePrediction] = []
    for matrix_id in sorted(by_matrix):
        rows = tuple(by_matrix[matrix_id])
        oracle = _oracle_row(rows)
        successful_by_candidate = {
            row.candidate_id: row
            for row in rows
            if row.target_status == "success" and row.target_success_rate >= 1.0
        }
        selected = None
        for candidate_id in baseline_order:
            if candidate_id in successful_by_candidate:
                selected = successful_by_candidate[candidate_id]
                break
        if selected is None:
            predictions.append(
                CsrBaselinePrediction(
                    schema_version=CSR_LEARNING_SCHEMA_VERSION,
                    split="eval",
                    matrix_id=matrix_id,
                    predicted_candidate_id=None,
                    oracle_candidate_id=oracle.candidate_id if oracle else None,
                    status="no_profiled_success_candidate",
                    predicted_label_class=None,
                    oracle_median_solve_time_ms=(
                        oracle.target_median_solve_time_ms if oracle else None
                    ),
                    predicted_median_solve_time_ms=None,
                    regret_vs_oracle_ms=None,
                    reason="no_training_prior_candidate_available_in_eval_success_rows",
                )
            )
            continue
        predictions.append(
            CsrBaselinePrediction(
                schema_version=CSR_LEARNING_SCHEMA_VERSION,
                split="eval",
                matrix_id=matrix_id,
                predicted_candidate_id=selected.candidate_id,
                oracle_candidate_id=oracle.candidate_id if oracle else None,
                status=(
                    "oracle_match"
                    if oracle is not None and selected.candidate_id == oracle.candidate_id
                    else "profiled_success_non_oracle"
                ),
                predicted_label_class=selected.label_class,
                oracle_median_solve_time_ms=(
                    oracle.target_median_solve_time_ms if oracle else None
                ),
                predicted_median_solve_time_ms=selected.target_median_solve_time_ms,
                regret_vs_oracle_ms=(
                    None
                    if oracle is None
                    or selected.target_median_solve_time_ms is None
                    or oracle.target_median_solve_time_ms is None
                    else selected.target_median_solve_time_ms - oracle.target_median_solve_time_ms
                ),
                reason="first_training_candidate_prior_with_eval_profiled_success",
            )
        )
    return tuple(predictions)


def _oracle_row(rows: tuple[CsrLearningRow, ...]) -> CsrLearningRow | None:
    for row in rows:
        if row.label_is_oracle:
            return row
    return None


def _summary(
    rows: tuple[CsrLearningRow, ...],
    predictions: tuple[CsrBaselinePrediction, ...],
    *,
    baseline_id: str,
    baseline_order: tuple[str, ...],
) -> CsrLearningSummary:
    train_rows = tuple(row for row in rows if row.split == "train")
    eval_rows = tuple(row for row in rows if row.split == "eval")
    eval_with_success = {
        row.matrix_id for row in eval_rows if row.target_status == "success"
    }
    successful_predictions = tuple(
        row
        for row in predictions
        if row.status in {"oracle_match", "profiled_success_non_oracle"}
    )
    regrets = [
        float(row.regret_vs_oracle_ms)
        for row in successful_predictions
        if row.regret_vs_oracle_ms is not None
    ]
    accuracy = (
        sum(1 for row in predictions if row.status == "oracle_match") / len(predictions)
        if predictions
        else 0.0
    )
    profiled_success_rate = (
        len(successful_predictions) / len(predictions) if predictions else 0.0
    )
    status = (
        "passed"
        if rows
        and train_rows
        and eval_rows
        and predictions
        and eval_with_success
        and baseline_order
        and all(row.split == "eval" for row in predictions)
        else "failed"
    )
    return CsrLearningSummary(
        status=status,
        schema_version=CSR_LEARNING_SCHEMA_VERSION,
        baseline_id=baseline_id,
        num_rows=len(rows),
        num_train_rows=len(train_rows),
        num_eval_rows=len(eval_rows),
        num_matrices=len({row.matrix_id for row in rows}),
        num_train_matrices=len({row.matrix_id for row in train_rows}),
        num_eval_matrices=len({row.matrix_id for row in eval_rows}),
        label_class_counts=dict(sorted(Counter(row.label_class for row in rows).items())),
        num_eval_matrices_with_success=len(eval_with_success),
        num_eval_predictions=len(predictions),
        num_eval_profiled_success_predictions=len(successful_predictions),
        eval_oracle_top1_accuracy=accuracy,
        eval_profiled_success_rate=profiled_success_rate,
        eval_mean_regret_ms=(sum(regrets) / len(regrets) if regrets else None),
        eval_max_regret_ms=(max(regrets) if regrets else None),
        baseline_candidate_order=baseline_order,
    )


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _fmt(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6g}"
