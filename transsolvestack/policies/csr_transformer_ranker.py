"""Lightweight masked self-attention ranker for CSR selector tensors."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from transsolvestack.profiling.artifacts import read_jsonl, write_jsonl


CSR_TRANSFORMER_RANKER_SCHEMA_VERSION = "phase1_csr_transformer_ranker_v1"
MODEL_FAMILY = "masked_self_attention_ranker_v1"


@dataclass(frozen=True)
class CsrTransformerRankerPrediction:
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
class CsrTransformerRankerSummary:
    status: str
    schema_version: str
    model_family: str
    model_id: str
    model_trained: bool
    runtime_selector_changed: bool
    encoder_training: str
    num_requests: int
    num_predictions: int
    num_train_requests: int
    num_eval_requests: int
    num_train_oracle_requests: int
    num_eval_oracle_requests: int
    num_global_candidates: int
    token_feature_dim: int
    d_model: int
    num_attention_heads: int
    feedforward_dim: int
    scorer_feature_dim: int
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
class CsrTransformerRankerExport:
    model: dict[str, Any]
    predictions: tuple[CsrTransformerRankerPrediction, ...]
    summary: CsrTransformerRankerSummary
    schema: dict[str, Any]


def train_csr_transformer_ranker_from_tensor_file(
    tensor_path: str | Path,
    request_index_path: str | Path,
    *,
    model_id: str = "csr_masked_self_attention_ranker_v1",
    d_model: int = 24,
    num_attention_heads: int = 4,
    feedforward_dim: int = 48,
    epochs: int = 120,
    learning_rate: float = 0.03,
    l2_regularization: float = 1.0e-4,
    seed: int = 17,
) -> CsrTransformerRankerExport:
    arrays = json.loads(Path(tensor_path).read_text(encoding="utf-8"))
    request_index = tuple(read_jsonl(request_index_path))
    return train_csr_transformer_ranker(
        arrays,
        request_index,
        model_id=model_id,
        d_model=d_model,
        num_attention_heads=num_attention_heads,
        feedforward_dim=feedforward_dim,
        epochs=epochs,
        learning_rate=learning_rate,
        l2_regularization=l2_regularization,
        seed=seed,
    )


def load_csr_transformer_ranker_model(path: str | Path) -> dict[str, Any]:
    """Load and validate a saved CSR Transformer ranker model artifact."""

    model = json.loads(Path(path).read_text(encoding="utf-8"))
    _validate_saved_model(model)
    return model


def predict_csr_transformer_ranker_from_model_file(
    model_path: str | Path,
    tensor_path: str | Path,
    request_index_path: str | Path,
) -> tuple[CsrTransformerRankerPrediction, ...]:
    """Run inference from a saved model without retraining."""

    model = load_csr_transformer_ranker_model(model_path)
    arrays = json.loads(Path(tensor_path).read_text(encoding="utf-8"))
    request_index = tuple(read_jsonl(request_index_path))
    return predict_csr_transformer_ranker_from_model(model, arrays, request_index)


def predict_csr_transformer_ranker_from_model(
    model: dict[str, Any],
    arrays: dict[str, Any],
    request_index_rows: Iterable[dict[str, Any]],
) -> tuple[CsrTransformerRankerPrediction, ...]:
    """Apply a saved CSR Transformer ranker model to tensor arrays."""

    _validate_saved_model(model)
    encoder = _numpy_encoder(model["encoder"])
    d_model = int(encoder["d_model"])
    expected_feature_names = _scorer_feature_names(arrays, d_model=d_model)
    model_feature_names = tuple(str(name) for name in model["scorer_feature_names"])
    if model_feature_names != expected_feature_names:
        raise ValueError("saved model scorer features do not match tensor schema")
    token_feature_names = list(_token_feature_names(arrays))
    if list(model["token_feature_names"]) != token_feature_names:
        raise ValueError("saved model token features do not match tensor schema")

    raw_tokens = _raw_token_grid(arrays)
    means = np.asarray(model["normalization"]["mean"], dtype=np.float64)
    scales = np.asarray(model["normalization"]["scale"], dtype=np.float64)
    token_dim = len(raw_tokens[0][0]) if raw_tokens and raw_tokens[0] else 0
    if means.shape != (token_dim,) or scales.shape != (token_dim,):
        raise ValueError("saved model normalization shape does not match token shape")
    if int(encoder["token_dim"]) != token_dim:
        raise ValueError("saved model encoder token_dim does not match tensor shape")

    normalized_tokens = _normalized_token_grid(raw_tokens, means, scales)
    transformer_features = _transformer_feature_grid(
        normalized_tokens,
        arrays["candidate_mask"],
        encoder,
    )
    weights = [float(model["scorer_head"][name]) for name in model_feature_names]
    request_index = tuple(sorted(request_index_rows, key=lambda row: int(row["row_index"])))
    return _predict(
        arrays,
        request_index,
        transformer_features,
        weights,
        model_id=str(model["model_id"]),
    )


def _validate_saved_model(model: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "model_family",
        "model_id",
        "token_feature_names",
        "scorer_feature_names",
        "normalization",
        "encoder",
        "scorer_head",
    }
    missing = sorted(required - set(model))
    if missing:
        raise ValueError(f"saved model missing required keys: {missing}")
    if model["schema_version"] != CSR_TRANSFORMER_RANKER_SCHEMA_VERSION:
        raise ValueError("saved model schema_version mismatch")
    if model["model_family"] != MODEL_FAMILY:
        raise ValueError("saved model model_family mismatch")
    normalization = model["normalization"]
    if set(normalization) != {"mean", "scale"}:
        raise ValueError("saved model normalization must contain mean and scale")
    if len(normalization["mean"]) != len(normalization["scale"]):
        raise ValueError("saved model normalization mean/scale length mismatch")
    scorer_names = tuple(str(name) for name in model["scorer_feature_names"])
    scorer_head = model["scorer_head"]
    if set(scorer_names) != set(str(name) for name in scorer_head):
        raise ValueError("saved model scorer_head keys do not match scorer_feature_names")


def _numpy_encoder(encoder: dict[str, Any]) -> dict[str, Any]:
    int_keys = {"token_dim", "d_model", "num_attention_heads", "head_dim", "feedforward_dim"}
    result: dict[str, Any] = {}
    for key, value in encoder.items():
        if key in int_keys:
            result[key] = int(value)
        else:
            result[key] = np.asarray(value, dtype=np.float64)
    return result


def train_csr_transformer_ranker(
    arrays: dict[str, Any],
    request_index_rows: Iterable[dict[str, Any]],
    *,
    model_id: str = "csr_masked_self_attention_ranker_v1",
    d_model: int = 24,
    num_attention_heads: int = 4,
    feedforward_dim: int = 48,
    epochs: int = 120,
    learning_rate: float = 0.03,
    l2_regularization: float = 1.0e-4,
    seed: int = 17,
) -> CsrTransformerRankerExport:
    if d_model % num_attention_heads != 0:
        raise ValueError("d_model must be divisible by num_attention_heads")
    request_index = tuple(sorted(request_index_rows, key=lambda row: int(row["row_index"])))
    raw_tokens = _raw_token_grid(arrays)
    train_slots = _active_train_slots(arrays)
    means, scales = _normalization(raw_tokens, train_slots)
    normalized_tokens = _normalized_token_grid(raw_tokens, means, scales)
    encoder = _init_encoder(
        token_dim=len(normalized_tokens[0][0]),
        d_model=d_model,
        num_attention_heads=num_attention_heads,
        feedforward_dim=feedforward_dim,
        seed=seed,
    )
    transformer_features = _transformer_feature_grid(
        normalized_tokens,
        arrays["candidate_mask"],
        encoder,
    )
    feature_names = _scorer_feature_names(arrays, d_model=d_model)
    weights, constraint_count, update_count, final_loss = _fit_pairwise_head(
        transformer_features,
        arrays,
        epochs=epochs,
        learning_rate=learning_rate,
        l2_regularization=l2_regularization,
    )
    predictions = _predict(
        arrays,
        request_index,
        transformer_features,
        weights,
        model_id=model_id,
    )
    summary = _summary(
        arrays,
        predictions,
        weights=weights,
        model_id=model_id,
        token_dim=len(normalized_tokens[0][0]),
        d_model=d_model,
        num_attention_heads=num_attention_heads,
        feedforward_dim=feedforward_dim,
        epochs=epochs,
        learning_rate=learning_rate,
        l2_regularization=l2_regularization,
        constraint_count=constraint_count,
        update_count=update_count,
        final_loss=final_loss,
    )
    model = {
        "schema_version": CSR_TRANSFORMER_RANKER_SCHEMA_VERSION,
        "model_family": MODEL_FAMILY,
        "model_id": model_id,
        "encoder_training": "deterministic_masked_self_attention_encoder_with_trained_pairwise_head",
        "token_feature_names": _token_feature_names(arrays),
        "scorer_feature_names": list(feature_names),
        "normalization": {
            "mean": means.tolist(),
            "scale": scales.tolist(),
        },
        "encoder": _jsonable_encoder(encoder),
        "scorer_head": {
            name: float(weight) for name, weight in zip(feature_names, weights)
        },
        "training": {
            "objective": "label_utility_pairwise_hinge_with_oracle_priority",
            "epochs": epochs,
            "learning_rate": learning_rate,
            "l2_regularization": l2_regularization,
            "num_pairwise_constraints": constraint_count,
            "num_pairwise_updates": update_count,
            "final_train_pairwise_loss": final_loss,
            "seed": seed,
        },
        "runtime_integration": {
            "runtime_selector_changed": False,
            "current_runtime_selector": "csr_artifact_success_rows",
        },
    }
    return CsrTransformerRankerExport(
        model=model,
        predictions=predictions,
        summary=summary,
        schema=_schema(feature_names),
    )


def write_csr_transformer_ranker_model(model: dict[str, Any], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(model, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def write_csr_transformer_ranker_predictions(
    predictions: Iterable[CsrTransformerRankerPrediction],
    path: str | Path,
) -> Path:
    return write_jsonl((asdict(row) for row in predictions), path)


def write_csr_transformer_ranker_summary(
    summary: CsrTransformerRankerSummary,
    path: str | Path,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def write_csr_transformer_ranker_schema(schema: dict[str, Any], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def write_csr_transformer_ranker_report(
    export: CsrTransformerRankerExport,
    path: str | Path,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = export.summary
    lines = [
        "# CSR Transformer Ranker",
        "",
        f"- status: `{summary.status}`",
        f"- schema_version: `{summary.schema_version}`",
        f"- model_family: `{summary.model_family}`",
        f"- model_id: `{summary.model_id}`",
        f"- model_trained: `{summary.model_trained}`",
        f"- runtime_selector_changed: `{summary.runtime_selector_changed}`",
        f"- encoder_training: `{summary.encoder_training}`",
        f"- requests: `{summary.num_requests}`",
        f"- train/eval requests: `{summary.num_train_requests}` / `{summary.num_eval_requests}`",
        f"- train/eval oracle requests: `{summary.num_train_oracle_requests}` / `{summary.num_eval_oracle_requests}`",
        f"- global_candidates: `{summary.num_global_candidates}`",
        f"- token_feature_dim: `{summary.token_feature_dim}`",
        f"- d_model: `{summary.d_model}`",
        f"- attention_heads: `{summary.num_attention_heads}`",
        f"- feedforward_dim: `{summary.feedforward_dim}`",
        f"- scorer_feature_dim: `{summary.scorer_feature_dim}`",
        f"- epochs: `{summary.num_epochs}`",
        f"- pairwise_constraints: `{summary.num_pairwise_constraints}`",
        f"- pairwise_updates: `{summary.num_pairwise_updates}`",
        f"- final_train_pairwise_loss: `{summary.final_train_pairwise_loss:.6g}`",
        f"- train_oracle_top1_accuracy: `{summary.train_oracle_top1_accuracy:.6g}`",
        f"- eval_oracle_top1_accuracy: `{summary.eval_oracle_top1_accuracy:.6g}`",
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


def _raw_token_grid(arrays: dict[str, Any]) -> list[list[list[float]]]:
    result = []
    for request_i, matrix_features in enumerate(arrays["matrix_features"]):
        request_tokens = []
        for candidate_features in arrays["candidate_features"][request_i]:
            request_tokens.append(
                [*map(float, matrix_features), *map(float, candidate_features)]
            )
        result.append(request_tokens)
    return result


def _token_feature_names(arrays: dict[str, Any]) -> list[str]:
    return [
        *(f"matrix:{name}" for name in arrays["matrix_feature_names"]),
        *(f"candidate:{name}" for name in arrays["candidate_feature_names"]),
    ]


def _active_train_slots(arrays: dict[str, Any]) -> tuple[tuple[int, int], ...]:
    return tuple(
        (request_i, candidate_i)
        for request_i, split_id in enumerate(arrays["split_ids"])
        if int(split_id) == 0
        for candidate_i, active in enumerate(arrays["candidate_mask"][request_i])
        if int(active)
    )


def _normalization(
    tokens: list[list[list[float]]],
    slots: tuple[tuple[int, int], ...],
) -> tuple[np.ndarray, np.ndarray]:
    dim = len(tokens[0][0]) if tokens and tokens[0] else 0
    means = np.zeros(dim, dtype=np.float64)
    scales = np.ones(dim, dtype=np.float64)
    if not slots:
        return means, scales
    values = np.asarray([tokens[i][j] for i, j in slots], dtype=np.float64)
    means = np.mean(values, axis=0)
    scales = np.std(values, axis=0)
    scales = np.where(scales > 1.0e-12, scales, 1.0)
    return means, scales


def _normalized_token_grid(
    tokens: list[list[list[float]]],
    means: np.ndarray,
    scales: np.ndarray,
) -> list[np.ndarray]:
    return [
        (np.asarray(request_tokens, dtype=np.float64) - means) / scales
        for request_tokens in tokens
    ]


def _init_encoder(
    *,
    token_dim: int,
    d_model: int,
    num_attention_heads: int,
    feedforward_dim: int,
    seed: int,
) -> dict[str, np.ndarray | int]:
    rng = np.random.default_rng(seed)
    scale = 1.0 / math.sqrt(max(token_dim, 1))
    head_dim = d_model // num_attention_heads
    return {
        "token_dim": token_dim,
        "d_model": d_model,
        "num_attention_heads": num_attention_heads,
        "head_dim": head_dim,
        "feedforward_dim": feedforward_dim,
        "w_in": rng.normal(0.0, scale, size=(token_dim, d_model)),
        "b_in": np.zeros(d_model, dtype=np.float64),
        "w_q": rng.normal(0.0, 1.0 / math.sqrt(d_model), size=(num_attention_heads, d_model, head_dim)),
        "w_k": rng.normal(0.0, 1.0 / math.sqrt(d_model), size=(num_attention_heads, d_model, head_dim)),
        "w_v": rng.normal(0.0, 1.0 / math.sqrt(d_model), size=(num_attention_heads, d_model, head_dim)),
        "w_o": rng.normal(0.0, 1.0 / math.sqrt(d_model), size=(d_model, d_model)),
        "b_o": np.zeros(d_model, dtype=np.float64),
        "w_ff1": rng.normal(0.0, 1.0 / math.sqrt(d_model), size=(d_model, feedforward_dim)),
        "b_ff1": np.zeros(feedforward_dim, dtype=np.float64),
        "w_ff2": rng.normal(0.0, 1.0 / math.sqrt(feedforward_dim), size=(feedforward_dim, d_model)),
        "b_ff2": np.zeros(d_model, dtype=np.float64),
    }


def _transformer_feature_grid(
    tokens: list[np.ndarray],
    candidate_mask: list[list[int]],
    encoder: dict[str, Any],
) -> list[list[list[float]]]:
    features = []
    for request_i, request_tokens in enumerate(tokens):
        encoded = _encode_request(
            request_tokens,
            np.asarray(candidate_mask[request_i], dtype=bool),
            encoder,
        )
        request_features = []
        for candidate_i in range(request_tokens.shape[0]):
            request_features.append(
                [
                    1.0,
                    *encoded[candidate_i].tolist(),
                    *request_tokens[candidate_i].tolist(),
                ]
            )
        features.append(request_features)
    return features


def _encode_request(
    tokens: np.ndarray,
    mask: np.ndarray,
    encoder: dict[str, Any],
) -> np.ndarray:
    x = tokens @ encoder["w_in"] + encoder["b_in"]
    x = _layer_norm(x)
    head_outputs = []
    for head in range(int(encoder["num_attention_heads"])):
        q = x @ encoder["w_q"][head]
        k = x @ encoder["w_k"][head]
        v = x @ encoder["w_v"][head]
        scores = q @ k.T / math.sqrt(float(encoder["head_dim"]))
        scores[:, ~mask] = -1.0e9
        attention = _softmax(scores)
        head_outputs.append(attention @ v)
    attended = np.concatenate(head_outputs, axis=1)
    x = _layer_norm(x + attended @ encoder["w_o"] + encoder["b_o"])
    ff = np.maximum(x @ encoder["w_ff1"] + encoder["b_ff1"], 0.0)
    x = _layer_norm(x + ff @ encoder["w_ff2"] + encoder["b_ff2"])
    return x


def _layer_norm(values: np.ndarray) -> np.ndarray:
    mean = np.mean(values, axis=1, keepdims=True)
    variance = np.mean((values - mean) ** 2, axis=1, keepdims=True)
    return (values - mean) / np.sqrt(variance + 1.0e-6)


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _scorer_feature_names(arrays: dict[str, Any], *, d_model: int) -> tuple[str, ...]:
    return (
        "bias",
        *(f"encoded:{index}" for index in range(d_model)),
        * _token_feature_names(arrays),
    )


def _fit_pairwise_head(
    features: list[list[list[float]]],
    arrays: dict[str, Any],
    *,
    epochs: int,
    learning_rate: float,
    l2_regularization: float,
) -> tuple[list[float], int, int, float]:
    dim = len(features[0][0]) if features and features[0] else 0
    weights = [0.0] * dim
    pairs = _training_pairs(arrays)
    update_count = 0
    final_loss = 0.0
    for _epoch in range(epochs):
        epoch_loss = 0.0
        for request_i, better_i, worse_i, margin_target in pairs:
            better = features[request_i][better_i]
            worse = features[request_i][worse_i]
            margin = margin_target - (_dot(weights, better) - _dot(weights, worse))
            shrink = 1.0 - learning_rate * l2_regularization
            weights = [weight * shrink for weight in weights]
            if margin > 0.0:
                diff = [left - right for left, right in zip(better, worse)]
                weights = [weight + learning_rate * value for weight, value in zip(weights, diff)]
                update_count += 1
                epoch_loss += margin
        final_loss = epoch_loss
    return weights, len(pairs), update_count, final_loss


def _training_pairs(arrays: dict[str, Any]) -> tuple[tuple[int, int, int, float], ...]:
    label_by_id = {int(value): key for key, value in arrays["label_class_to_id"].items()}
    pairs = []
    for request_i, split_id in enumerate(arrays["split_ids"]):
        if int(split_id) != 0:
            continue
        active = [
            candidate_i
            for candidate_i, active_value in enumerate(arrays["candidate_mask"][request_i])
            if int(active_value)
        ]
        utilities = {
            candidate_i: _candidate_training_utility(
                label_by_id[int(arrays["label_class_ids"][request_i][candidate_i])],
                arrays["target_median_solve_time_ms"][request_i][candidate_i],
            )
            for candidate_i in active
        }
        for left in active:
            for right in active:
                if left == right:
                    continue
                diff = utilities[left] - utilities[right]
                if diff <= 1.0e-9:
                    continue
                pairs.append((request_i, left, right, 1.0 if diff >= 1.0 else max(0.1, diff)))
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
) -> tuple[CsrTransformerRankerPrediction, ...]:
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
        ranked = tuple(sorted(scores, key=lambda candidate_id: (-scores[candidate_id], candidate_id)))
        selected = ranked[0]
        selected_i = candidate_ids.index(selected)
        oracle_i = int(arrays["oracle_index"][request_i])
        oracle_candidate_id = candidate_ids[oracle_i] if oracle_i >= 0 else None
        oracle_rank = ranked.index(oracle_candidate_id) + 1 if oracle_candidate_id in ranked else None
        selected_status = status_by_id[int(arrays["target_status_ids"][request_i][selected_i])]
        selected_label = label_by_id[int(arrays["label_class_ids"][request_i][selected_i])]
        regret = arrays["target_regret_vs_oracle_ms"][request_i][selected_i]
        predictions.append(
            CsrTransformerRankerPrediction(
                schema_version=CSR_TRANSFORMER_RANKER_SCHEMA_VERSION,
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
                regret_vs_oracle_ms=None if float(regret) < 0.0 else float(regret),
                ranked_candidate_ids=ranked,
                scores={candidate_id: float(scores[candidate_id]) for candidate_id in ranked},
            )
        )
    return tuple(predictions)


def _summary(
    arrays: dict[str, Any],
    predictions: tuple[CsrTransformerRankerPrediction, ...],
    *,
    weights: list[float],
    model_id: str,
    token_dim: int,
    d_model: int,
    num_attention_heads: int,
    feedforward_dim: int,
    epochs: int,
    learning_rate: float,
    l2_regularization: float,
    constraint_count: int,
    update_count: int,
    final_loss: float,
) -> CsrTransformerRankerSummary:
    train_predictions = tuple(row for row in predictions if row.split == "train")
    eval_predictions = tuple(row for row in predictions if row.split == "eval")
    train_oracle = tuple(row for row in train_predictions if row.oracle_candidate_id is not None)
    eval_oracle = tuple(row for row in eval_predictions if row.oracle_candidate_id is not None)
    eval_success = tuple(row for row in eval_predictions if row.selected_target_status == "success")
    eval_regrets = [
        float(row.regret_vs_oracle_ms)
        for row in eval_predictions
        if row.regret_vs_oracle_ms is not None
    ]
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
    return CsrTransformerRankerSummary(
        status=status,
        schema_version=CSR_TRANSFORMER_RANKER_SCHEMA_VERSION,
        model_family=MODEL_FAMILY,
        model_id=model_id,
        model_trained=True,
        runtime_selector_changed=False,
        encoder_training="deterministic_masked_self_attention_encoder_with_trained_pairwise_head",
        num_requests=len(predictions),
        num_predictions=len(predictions),
        num_train_requests=len(train_predictions),
        num_eval_requests=len(eval_predictions),
        num_train_oracle_requests=len(train_oracle),
        num_eval_oracle_requests=len(eval_oracle),
        num_global_candidates=len(arrays["global_candidate_ids"]),
        token_feature_dim=token_dim,
        d_model=d_model,
        num_attention_heads=num_attention_heads,
        feedforward_dim=feedforward_dim,
        scorer_feature_dim=len(weights),
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
        eval_non_success_selection_count=len(eval_predictions) - len(eval_success),
        eval_mean_regret_ms=(sum(eval_regrets) / len(eval_regrets) if eval_regrets else None),
        eval_max_regret_ms=(max(eval_regrets) if eval_regrets else None),
    )


def _schema(feature_names: tuple[str, ...]) -> dict[str, Any]:
    return {
        "schema_version": CSR_TRANSFORMER_RANKER_SCHEMA_VERSION,
        "model_family": MODEL_FAMILY,
        "training_objective": "label_utility_pairwise_hinge_with_oracle_priority",
        "model_required": True,
        "runtime_selector_changed": False,
        "architecture": {
            "token": "matrix features concatenated with candidate features",
            "encoder": "masked multi-head scaled-dot-product self-attention plus feed-forward block",
            "candidate_mask": "inactive global candidate slots are excluded from attention and scoring",
            "trained_parameters": "pairwise scorer head over frozen deterministic attention features",
        },
        "scorer_features": list(feature_names),
        "prediction": {
            "ranked_candidate_ids": "all active candidates sorted by descending score",
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


def _jsonable_encoder(encoder: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in encoder.items():
        if isinstance(value, np.ndarray):
            result[key] = value.tolist()
        else:
            result[key] = value
    return result


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


def _oracle_accuracy(rows: tuple[CsrTransformerRankerPrediction, ...]) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if row.evaluation_status == "oracle_match") / len(rows)


def _dot(weights: list[float], features: list[float]) -> float:
    return sum(weight * value for weight, value in zip(weights, features))


def _fmt(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6g}"
