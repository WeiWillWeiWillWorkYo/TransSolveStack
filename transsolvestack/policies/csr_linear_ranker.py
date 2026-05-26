"""Trainable linear ranker baseline for CSR selector tensors."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from transsolvestack.profiling.artifacts import read_jsonl, write_jsonl


CSR_LINEAR_RANKER_SCHEMA_VERSION = "phase1_csr_linear_ranker_v1"
MODEL_FAMILY = "pairwise_linear_ranker_v1"


@dataclass(frozen=True)
class CsrLinearRankerPrediction:
    schema_version: str
    model_id: str
    request_id: str
    split: str
    matrix_id: str
    context_id: str
    selected_candidate_id: str
    oracle_candidate_id: str | None
    oracle_rank: int | None
    selected_score: float
    selected_label_class: str
    selected_target_status: str
    evaluation_status: str
    regret_vs_oracle_ms: float | None
    ranked_candidate_ids: tuple[str, ...]
    scores: dict[str, float]


@dataclass(frozen=True)
class CsrLinearRankerSummary:
    status: str
    schema_version: str
    model_family: str
    model_id: str
    model_trained: bool
    runtime_selector_changed: bool
    num_requests: int
    num_predictions: int
    num_train_requests: int
    num_eval_requests: int
    num_train_oracle_requests: int
    num_eval_oracle_requests: int
    num_global_candidates: int
    num_features: int
    num_epochs: int
    learning_rate: float
    l2_regularization: float
    num_pairwise_constraints: int
    num_pairwise_updates: int
    final_train_pairwise_loss: float
    train_oracle_top1_accuracy: float
    eval_oracle_top1_accuracy: float
    eval_oracle_top1_accuracy_all_requests: float
    eval_profiled_success_selection_rate: float
    eval_non_success_selection_count: int
    eval_mean_regret_ms: float | None
    eval_max_regret_ms: float | None


@dataclass(frozen=True)
class CsrLinearRankerExport:
    model: dict[str, Any]
    predictions: tuple[CsrLinearRankerPrediction, ...]
    summary: CsrLinearRankerSummary
    schema: dict[str, Any]


def train_csr_linear_ranker_from_tensor_file(
    tensor_path: str | Path,
    request_index_path: str | Path,
    *,
    model_id: str = "csr_pairwise_linear_ranker_v1",
    epochs: int = 80,
    learning_rate: float = 0.05,
    l2_regularization: float = 1.0e-4,
) -> CsrLinearRankerExport:
    arrays = json.loads(Path(tensor_path).read_text(encoding="utf-8"))
    request_index = tuple(read_jsonl(request_index_path))
    return train_csr_linear_ranker(
        arrays,
        request_index,
        model_id=model_id,
        epochs=epochs,
        learning_rate=learning_rate,
        l2_regularization=l2_regularization,
    )


def train_csr_linear_ranker(
    arrays: dict[str, Any],
    request_index_rows: Iterable[dict[str, Any]],
    *,
    model_id: str = "csr_pairwise_linear_ranker_v1",
    epochs: int = 80,
    learning_rate: float = 0.05,
    l2_regularization: float = 1.0e-4,
) -> CsrLinearRankerExport:
    request_index = tuple(sorted(request_index_rows, key=lambda row: int(row["row_index"])))
    feature_names = _feature_names(arrays)
    raw_features = _raw_feature_grid(arrays)
    train_slots = _active_train_slots(arrays)
    means, scales = _normalization(raw_features, train_slots)
    features = _normalized_feature_grid(raw_features, means, scales)
    weights, constraint_count, update_count, final_loss = _fit_pairwise(
        features,
        arrays,
        epochs=epochs,
        learning_rate=learning_rate,
        l2_regularization=l2_regularization,
    )
    predictions = _predict(
        arrays,
        request_index,
        features,
        weights,
        model_id=model_id,
    )
    summary = _summary(
        arrays,
        predictions,
        weights=weights,
        model_id=model_id,
        epochs=epochs,
        learning_rate=learning_rate,
        l2_regularization=l2_regularization,
        constraint_count=constraint_count,
        update_count=update_count,
        final_loss=final_loss,
    )
    model = {
        "schema_version": CSR_LINEAR_RANKER_SCHEMA_VERSION,
        "model_family": MODEL_FAMILY,
        "model_id": model_id,
        "feature_names": list(feature_names),
        "weights": {
            name: weight for name, weight in zip(feature_names, weights)
        },
        "normalization": {
            "mean": {name: mean for name, mean in zip(feature_names, means)},
            "scale": {name: scale for name, scale in zip(feature_names, scales)},
        },
        "training": {
            "objective": "label_utility_pairwise_hinge_with_oracle_priority",
            "epochs": epochs,
            "learning_rate": learning_rate,
            "l2_regularization": l2_regularization,
            "num_pairwise_constraints": constraint_count,
            "num_pairwise_updates": update_count,
            "final_train_pairwise_loss": final_loss,
        },
        "runtime_integration": {
            "runtime_selector_changed": False,
            "current_runtime_selector": "csr_artifact_success_rows",
        },
    }
    return CsrLinearRankerExport(
        model=model,
        predictions=predictions,
        summary=summary,
        schema=_schema(feature_names),
    )


def write_csr_linear_ranker_model(model: dict[str, Any], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(model, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def write_csr_linear_ranker_predictions(
    predictions: Iterable[CsrLinearRankerPrediction],
    path: str | Path,
) -> Path:
    return write_jsonl((asdict(row) for row in predictions), path)


def write_csr_linear_ranker_summary(
    summary: CsrLinearRankerSummary,
    path: str | Path,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def write_csr_linear_ranker_schema(schema: dict[str, Any], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def write_csr_linear_ranker_report(export: CsrLinearRankerExport, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = export.summary
    lines = [
        "# CSR Linear Ranker",
        "",
        f"- status: `{summary.status}`",
        f"- schema_version: `{summary.schema_version}`",
        f"- model_family: `{summary.model_family}`",
        f"- model_id: `{summary.model_id}`",
        f"- model_trained: `{summary.model_trained}`",
        f"- runtime_selector_changed: `{summary.runtime_selector_changed}`",
        f"- requests: `{summary.num_requests}`",
        f"- predictions: `{summary.num_predictions}`",
        f"- train/eval requests: `{summary.num_train_requests}` / `{summary.num_eval_requests}`",
        f"- train/eval oracle requests: `{summary.num_train_oracle_requests}` / `{summary.num_eval_oracle_requests}`",
        f"- global_candidates: `{summary.num_global_candidates}`",
        f"- features: `{summary.num_features}`",
        f"- epochs: `{summary.num_epochs}`",
        f"- pairwise_constraints: `{summary.num_pairwise_constraints}`",
        f"- pairwise_updates: `{summary.num_pairwise_updates}`",
        f"- final_train_pairwise_loss: `{summary.final_train_pairwise_loss:.6g}`",
        f"- train_oracle_top1_accuracy: `{summary.train_oracle_top1_accuracy:.6g}`",
        f"- eval_oracle_top1_accuracy: `{summary.eval_oracle_top1_accuracy:.6g}`",
        f"- eval_oracle_top1_accuracy_all_requests: `{summary.eval_oracle_top1_accuracy_all_requests:.6g}`",
        f"- eval_profiled_success_selection_rate: `{summary.eval_profiled_success_selection_rate:.6g}`",
        f"- eval_non_success_selection_count: `{summary.eval_non_success_selection_count}`",
        f"- eval_mean_regret_ms: `{_fmt(summary.eval_mean_regret_ms)}`",
        f"- eval_max_regret_ms: `{_fmt(summary.eval_max_regret_ms)}`",
        "",
        "| split | matrix | selected | oracle | status | oracle_rank | regret_ms |",
        "|---|---|---|---|---|---:|---:|",
    ]
    for row in export.predictions:
        lines.append(
            "| "
            f"{row.split} | "
            f"{row.matrix_id} | "
            f"{row.selected_candidate_id} | "
            f"{row.oracle_candidate_id or ''} | "
            f"{row.evaluation_status} | "
            f"{'' if row.oracle_rank is None else row.oracle_rank} | "
            f"{_fmt(row.regret_vs_oracle_ms)} |"
        )
    lines.append("")
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def _feature_names(arrays: dict[str, Any]) -> tuple[str, ...]:
    matrix_names = tuple(f"matrix:{name}" for name in arrays["matrix_feature_names"])
    candidate_names = tuple(f"candidate:{name}" for name in arrays["candidate_feature_names"])
    interaction_names = tuple(
        f"interaction:{matrix_name}*{candidate_name}"
        for matrix_name in arrays["matrix_feature_names"]
        for candidate_name in arrays["candidate_feature_names"]
    )
    return ("bias", *matrix_names, *candidate_names, *interaction_names)


def _raw_feature_grid(arrays: dict[str, Any]) -> list[list[list[float]]]:
    matrix_rows = arrays["matrix_features"]
    candidate_rows = arrays["candidate_features"]
    result: list[list[list[float]]] = []
    for request_index, matrix_features in enumerate(matrix_rows):
        request_vectors = []
        for candidate_features in candidate_rows[request_index]:
            interactions = [
                float(matrix_value) * float(candidate_value)
                for matrix_value in matrix_features
                for candidate_value in candidate_features
            ]
            request_vectors.append(
                [1.0, *map(float, matrix_features), *map(float, candidate_features), *interactions]
            )
        result.append(request_vectors)
    return result


def _active_train_slots(arrays: dict[str, Any]) -> tuple[tuple[int, int], ...]:
    slots = []
    for request_index, split_id in enumerate(arrays["split_ids"]):
        if int(split_id) != 0:
            continue
        for candidate_index, active in enumerate(arrays["candidate_mask"][request_index]):
            if int(active):
                slots.append((request_index, candidate_index))
    return tuple(slots)


def _normalization(
    features: list[list[list[float]]],
    slots: tuple[tuple[int, int], ...],
) -> tuple[list[float], list[float]]:
    dim = len(features[0][0]) if features and features[0] else 0
    means = [0.0] * dim
    scales = [1.0] * dim
    if not slots:
        return means, scales
    for index in range(1, dim):
        values = [features[request][candidate][index] for request, candidate in slots]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        means[index] = mean
        scales[index] = math.sqrt(variance) if variance > 1.0e-24 else 1.0
    return means, scales


def _normalized_feature_grid(
    features: list[list[list[float]]],
    means: list[float],
    scales: list[float],
) -> list[list[list[float]]]:
    return [
        [
            [
                1.0 if index == 0 else (value - means[index]) / scales[index]
                for index, value in enumerate(candidate_features)
            ]
            for candidate_features in request_features
        ]
        for request_features in features
    ]


def _fit_pairwise(
    features: list[list[list[float]]],
    arrays: dict[str, Any],
    *,
    epochs: int,
    learning_rate: float,
    l2_regularization: float,
) -> tuple[list[float], int, int, float]:
    dim = len(features[0][0]) if features and features[0] else 0
    weights = [0.0] * dim
    update_count = 0
    final_loss = 0.0
    training_pairs = _training_pairs(arrays)
    for _epoch in range(epochs):
        epoch_loss = 0.0
        for request_index, better_index, worse_index, target_margin in training_pairs:
            better_features = features[request_index][better_index]
            worse_features = features[request_index][worse_index]
            margin = target_margin - (
                _dot(weights, better_features) - _dot(weights, worse_features)
            )
            shrink = 1.0 - learning_rate * l2_regularization
            weights = [weight * shrink for weight in weights]
            if margin > 0.0:
                diff = [
                    better_value - worse_value
                    for better_value, worse_value in zip(better_features, worse_features)
                ]
                weights = [
                    weight + learning_rate * value
                    for weight, value in zip(weights, diff)
                ]
                update_count += 1
                epoch_loss += margin
        final_loss = epoch_loss
    return weights, len(training_pairs), update_count, final_loss


def _training_pairs(arrays: dict[str, Any]) -> tuple[tuple[int, int, int, float], ...]:
    label_by_id = {int(value): key for key, value in arrays["label_class_to_id"].items()}
    pairs: list[tuple[int, int, int, float]] = []
    for request_index, split_id in enumerate(arrays["split_ids"]):
        if int(split_id) != 0:
            continue
        active = [
            index
            for index, value in enumerate(arrays["candidate_mask"][request_index])
            if int(value)
        ]
        utilities = {
            index: _candidate_training_utility(
                label_by_id[int(arrays["label_class_ids"][request_index][index])],
                arrays["target_median_solve_time_ms"][request_index][index],
            )
            for index in active
        }
        for left in active:
            for right in active:
                if left == right:
                    continue
                diff = utilities[left] - utilities[right]
                if diff <= 1.0e-9:
                    continue
                margin = 1.0 if diff >= 1.0 else max(0.1, diff)
                pairs.append((request_index, left, right, margin))
    return tuple(pairs)


def _candidate_training_utility(label_class: str, median_solve_time_ms: Any) -> float:
    base = {
        "success_oracle": 4.0,
        "success_non_oracle": 3.0,
        "screened_out": 1.0,
        "not_applicable": 0.0,
        "not_profiled": 0.0,
    }[label_class]
    if not label_class.startswith("success"):
        return base
    median = float(median_solve_time_ms)
    if median <= 0.0 or not math.isfinite(median):
        return base
    return base + min(0.5, 1.0 / (1.0 + math.log1p(median)))


def _predict(
    arrays: dict[str, Any],
    request_index: tuple[dict[str, Any], ...],
    features: list[list[list[float]]],
    weights: list[float],
    *,
    model_id: str,
) -> tuple[CsrLinearRankerPrediction, ...]:
    candidate_ids = tuple(str(item) for item in arrays["global_candidate_ids"])
    label_by_id = {int(value): key for key, value in arrays["label_class_to_id"].items()}
    status_by_id = {int(value): key for key, value in arrays["target_status_to_id"].items()}
    predictions = []
    for request_row in request_index:
        request_i = int(request_row["row_index"])
        active = [
            candidate_i
            for candidate_i, value in enumerate(arrays["candidate_mask"][request_i])
            if int(value)
        ]
        scores = {
            candidate_ids[candidate_i]: _dot(weights, features[request_i][candidate_i])
            for candidate_i in active
        }
        ranked = tuple(
            sorted(
                scores,
                key=lambda candidate_id: (-scores[candidate_id], candidate_id),
            )
        )
        selected = ranked[0]
        selected_index = candidate_ids.index(selected)
        oracle_index = int(arrays["oracle_index"][request_i])
        oracle_candidate_id = candidate_ids[oracle_index] if oracle_index >= 0 else None
        oracle_rank = ranked.index(oracle_candidate_id) + 1 if oracle_candidate_id in ranked else None
        selected_status = status_by_id[int(arrays["target_status_ids"][request_i][selected_index])]
        selected_label = label_by_id[int(arrays["label_class_ids"][request_i][selected_index])]
        regret = arrays["target_regret_vs_oracle_ms"][request_i][selected_index]
        predictions.append(
            CsrLinearRankerPrediction(
                schema_version=CSR_LINEAR_RANKER_SCHEMA_VERSION,
                model_id=model_id,
                request_id=str(request_row["request_id"]),
                split=str(request_row["split"]),
                matrix_id=str(request_row["matrix_id"]),
                context_id=str(request_row["context_id"]),
                selected_candidate_id=selected,
                oracle_candidate_id=oracle_candidate_id,
                oracle_rank=oracle_rank,
                selected_score=float(scores[selected]),
                selected_label_class=selected_label,
                selected_target_status=selected_status,
                evaluation_status=_evaluation_status(
                    selected=selected,
                    oracle=oracle_candidate_id,
                    selected_status=selected_status,
                ),
                regret_vs_oracle_ms=(None if float(regret) < 0.0 else float(regret)),
                ranked_candidate_ids=ranked,
                scores={candidate_id: float(scores[candidate_id]) for candidate_id in ranked},
            )
        )
    return tuple(predictions)


def _summary(
    arrays: dict[str, Any],
    predictions: tuple[CsrLinearRankerPrediction, ...],
    *,
    weights: list[float],
    model_id: str,
    epochs: int,
    learning_rate: float,
    l2_regularization: float,
    constraint_count: int,
    update_count: int,
    final_loss: float,
) -> CsrLinearRankerSummary:
    train_predictions = tuple(row for row in predictions if row.split == "train")
    eval_predictions = tuple(row for row in predictions if row.split == "eval")
    train_oracle = tuple(row for row in train_predictions if row.oracle_candidate_id is not None)
    eval_oracle = tuple(row for row in eval_predictions if row.oracle_candidate_id is not None)
    eval_regrets = [
        float(row.regret_vs_oracle_ms)
        for row in eval_predictions
        if row.regret_vs_oracle_ms is not None
    ]
    eval_success = tuple(
        row for row in eval_predictions if row.selected_target_status == "success"
    )
    status = (
        "passed"
        if predictions
        and weights
        and constraint_count > 0
        and update_count > 0
        and train_oracle
        and eval_predictions
        and all(math.isfinite(value) for value in weights)
        else "failed"
    )
    return CsrLinearRankerSummary(
        status=status,
        schema_version=CSR_LINEAR_RANKER_SCHEMA_VERSION,
        model_family=MODEL_FAMILY,
        model_id=model_id,
        model_trained=True,
        runtime_selector_changed=False,
        num_requests=len(predictions),
        num_predictions=len(predictions),
        num_train_requests=len(train_predictions),
        num_eval_requests=len(eval_predictions),
        num_train_oracle_requests=len(train_oracle),
        num_eval_oracle_requests=len(eval_oracle),
        num_global_candidates=len(arrays["global_candidate_ids"]),
        num_features=len(weights),
        num_epochs=epochs,
        learning_rate=learning_rate,
        l2_regularization=l2_regularization,
        num_pairwise_constraints=constraint_count,
        num_pairwise_updates=update_count,
        final_train_pairwise_loss=float(final_loss),
        train_oracle_top1_accuracy=_oracle_accuracy(train_oracle),
        eval_oracle_top1_accuracy=_oracle_accuracy(eval_oracle),
        eval_oracle_top1_accuracy_all_requests=_oracle_accuracy(eval_predictions),
        eval_profiled_success_selection_rate=(
            len(eval_success) / len(eval_predictions) if eval_predictions else 0.0
        ),
        eval_non_success_selection_count=(
            len(eval_predictions) - len(eval_success)
        ),
        eval_mean_regret_ms=(
            sum(eval_regrets) / len(eval_regrets) if eval_regrets else None
        ),
        eval_max_regret_ms=(max(eval_regrets) if eval_regrets else None),
    )


def _schema(feature_names: tuple[str, ...]) -> dict[str, Any]:
    return {
        "schema_version": CSR_LINEAR_RANKER_SCHEMA_VERSION,
        "model_family": MODEL_FAMILY,
        "training_objective": "label_utility_pairwise_hinge_with_oracle_priority",
        "model_required": True,
        "runtime_selector_changed": False,
        "feature_space": {
            "num_features": len(feature_names),
            "features": list(feature_names),
            "normalization": "train_active_candidate_standardization",
        },
        "prediction": {
            "ranked_candidate_ids": "all active candidates sorted by descending linear score",
            "evaluation_status": [
                "oracle_match",
                "profiled_success_non_oracle",
                "non_success_selected",
                "no_oracle_target",
            ],
        },
        "runtime_integration": {
            "status": "offline_model_artifact_only",
            "current_runtime_selector": "csr_artifact_success_rows",
        },
    }


def _evaluation_status(
    *,
    selected: str,
    oracle: str | None,
    selected_status: str,
) -> str:
    if oracle is None:
        return "no_oracle_target"
    if selected == oracle:
        return "oracle_match"
    if selected_status == "success":
        return "profiled_success_non_oracle"
    return "non_success_selected"


def _oracle_accuracy(rows: tuple[CsrLinearRankerPrediction, ...]) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if row.evaluation_status == "oracle_match") / len(rows)


def _dot(weights: list[float], features: list[float]) -> float:
    return sum(weight * value for weight, value in zip(weights, features))


def _fmt(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6g}"
