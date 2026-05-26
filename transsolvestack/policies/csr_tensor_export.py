"""CSR model-contract tensor/array export."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from transsolvestack.profiling.artifacts import read_jsonl, write_jsonl


CSR_TENSOR_SCHEMA_VERSION = "phase1_csr_training_tensors_v1"

LABEL_CLASS_TO_ID = {
    "not_applicable": 0,
    "not_profiled": 1,
    "screened_out": 2,
    "success_non_oracle": 3,
    "success_oracle": 4,
}

TARGET_STATUS_TO_ID = {
    "not_applicable": 0,
    "not_profiled": 1,
    "screened_out": 2,
    "success": 3,
}

SPLIT_TO_ID = {
    "train": 0,
    "eval": 1,
}

_BASE_MATRIX_FEATURE_NAMES = (
    "actual_symmetric",
    "cg_candidate",
    "has_nonpositive_diagonal",
    "is_square",
    "n_rows",
    "n_cols",
    "csr_nnz",
    "diagonal_present_count",
    "zero_diagonal_count",
    "min_abs_diagonal_log10p",
    "max_abs_diagonal_log10p",
    "symmetry_missing_pairs",
    "symmetry_max_abs_error_log10p",
    "symmetry_relative_error_log10p",
)

_BASE_CANDIDATE_FEATURE_NAMES = (
    "has_restart",
    "restart",
    "has_omega",
    "omega",
    "has_lambda_bounds",
    "lambda_min_log10",
    "lambda_max_log10",
)


@dataclass(frozen=True)
class CsrTensorSummary:
    status: str
    schema_version: str
    source_contract_schema_version: str
    storage_format: str
    num_requests: int
    num_train_requests: int
    num_eval_requests: int
    num_global_candidates: int
    num_active_candidate_slots: int
    matrix_feature_dim: int
    candidate_feature_dim: int
    num_oracle_targets: int
    num_requests_without_oracle: int
    label_class_counts: dict[str, int]
    target_status_counts: dict[str, int]
    has_missing_candidate_slots: bool
    model_required: bool
    runtime_selector_changed: bool


@dataclass(frozen=True)
class CsrTensorExport:
    arrays: dict[str, Any]
    request_index_rows: tuple[dict[str, Any], ...]
    summary: CsrTensorSummary
    schema: dict[str, Any]


def build_csr_tensor_export(
    requests_path: str | Path,
    targets_path: str | Path,
) -> CsrTensorExport:
    requests = tuple(read_jsonl(requests_path))
    targets = tuple(read_jsonl(targets_path))
    target_by_id = {str(row["request_id"]): row for row in targets}
    sorted_requests = tuple(sorted(requests, key=lambda row: str(row["request_id"])))
    if set(target_by_id) != {str(row["request_id"]) for row in sorted_requests}:
        raise ValueError("CSR tensor export requires matching request and target ids")

    global_candidate_ids = tuple(
        sorted(
            {
                str(candidate_id)
                for request in sorted_requests
                for candidate_id in request["candidate_ids"]
            }
        )
    )
    matrix_feature_names = _matrix_feature_names(sorted_requests)
    candidate_feature_names = _candidate_feature_names(sorted_requests)
    arrays = _build_arrays(
        sorted_requests,
        target_by_id=target_by_id,
        global_candidate_ids=global_candidate_ids,
        matrix_feature_names=matrix_feature_names,
        candidate_feature_names=candidate_feature_names,
    )
    request_index_rows = _request_index_rows(
        sorted_requests,
        target_by_id=target_by_id,
        global_candidate_ids=global_candidate_ids,
    )
    schema = _schema(
        matrix_feature_names=matrix_feature_names,
        candidate_feature_names=candidate_feature_names,
        global_candidate_ids=global_candidate_ids,
    )
    summary = _summary(
        arrays,
        targets=tuple(target_by_id[str(request["request_id"])] for request in sorted_requests),
        matrix_feature_names=matrix_feature_names,
        candidate_feature_names=candidate_feature_names,
    )
    return CsrTensorExport(
        arrays=arrays,
        request_index_rows=request_index_rows,
        summary=summary,
        schema=schema,
    )


def write_csr_tensor_arrays(export: CsrTensorExport, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(export.arrays, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def write_csr_tensor_request_index(export: CsrTensorExport, path: str | Path) -> Path:
    return write_jsonl(export.request_index_rows, path)


def write_csr_tensor_summary(summary: CsrTensorSummary, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def write_csr_tensor_schema(schema: dict[str, Any], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def write_csr_tensor_report(export: CsrTensorExport, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = export.summary
    lines = [
        "# CSR Training Tensor Export",
        "",
        f"- status: `{summary.status}`",
        f"- schema_version: `{summary.schema_version}`",
        f"- source_contract_schema_version: `{summary.source_contract_schema_version}`",
        f"- storage_format: `{summary.storage_format}`",
        f"- requests: `{summary.num_requests}`",
        f"- train_requests: `{summary.num_train_requests}`",
        f"- eval_requests: `{summary.num_eval_requests}`",
        f"- global_candidates: `{summary.num_global_candidates}`",
        f"- active_candidate_slots: `{summary.num_active_candidate_slots}`",
        f"- matrix_feature_dim: `{summary.matrix_feature_dim}`",
        f"- candidate_feature_dim: `{summary.candidate_feature_dim}`",
        f"- oracle_targets: `{summary.num_oracle_targets}`",
        f"- requests_without_oracle: `{summary.num_requests_without_oracle}`",
        f"- label_class_counts: `{summary.label_class_counts}`",
        f"- target_status_counts: `{summary.target_status_counts}`",
        f"- has_missing_candidate_slots: `{summary.has_missing_candidate_slots}`",
        f"- model_required: `{summary.model_required}`",
        f"- runtime_selector_changed: `{summary.runtime_selector_changed}`",
        "",
        "| row | split | matrix | oracle_index | active_candidates |",
        "|---:|---|---|---:|---:|",
    ]
    for row in export.request_index_rows:
        lines.append(
            "| "
            f"{row['row_index']} | "
            f"{row['split']} | "
            f"{row['matrix_id']} | "
            f"{row['oracle_index']} | "
            f"{row['num_active_candidates']} |"
        )
    lines.append("")
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def _build_arrays(
    requests: tuple[dict[str, Any], ...],
    *,
    target_by_id: dict[str, dict[str, Any]],
    global_candidate_ids: tuple[str, ...],
    matrix_feature_names: tuple[str, ...],
    candidate_feature_names: tuple[str, ...],
) -> dict[str, Any]:
    candidate_slot_by_id = {
        candidate_id: index for index, candidate_id in enumerate(global_candidate_ids)
    }
    matrix_features: list[list[float]] = []
    candidate_features: list[list[list[float]]] = []
    candidate_mask: list[list[int]] = []
    label_class_ids: list[list[int]] = []
    target_status_ids: list[list[int]] = []
    profiled_success_mask: list[list[int]] = []
    target_median_solve_time_ms: list[list[float]] = []
    target_regret_vs_oracle_ms: list[list[float]] = []
    oracle_index: list[int] = []
    split_ids: list[int] = []
    request_ids: list[str] = []
    matrix_ids: list[str] = []
    context_ids: list[str] = []

    for request in requests:
        request_id = str(request["request_id"])
        target = target_by_id[request_id]
        active_candidates = {str(candidate_id) for candidate_id in request["candidate_ids"]}
        request_ids.append(request_id)
        matrix_ids.append(str(request["matrix_id"]))
        context_ids.append(str(request["context_id"]))
        split_ids.append(SPLIT_TO_ID[str(request["split"])])
        matrix_features.append(_matrix_feature_vector(request, matrix_feature_names))
        candidate_features.append(
            [
                _candidate_feature_vector(
                    request["candidate_features"].get(candidate_id),
                    candidate_feature_names,
                )
                if candidate_id in active_candidates
                else [0.0] * len(candidate_feature_names)
                for candidate_id in global_candidate_ids
            ]
        )
        candidate_mask.append(
            [1 if candidate_id in active_candidates else 0 for candidate_id in global_candidate_ids]
        )
        label_class_ids.append(
            [
                LABEL_CLASS_TO_ID.get(
                    str(target["candidate_label_classes"].get(candidate_id, "")),
                    -1,
                )
                if candidate_id in active_candidates
                else -1
                for candidate_id in global_candidate_ids
            ]
        )
        target_status_ids.append(
            [
                TARGET_STATUS_TO_ID.get(
                    str(target["candidate_target_statuses"].get(candidate_id, "")),
                    -1,
                )
                if candidate_id in active_candidates
                else -1
                for candidate_id in global_candidate_ids
            ]
        )
        success_ids = set(target["profiled_success_candidate_ids"])
        profiled_success_mask.append(
            [1 if candidate_id in success_ids else 0 for candidate_id in global_candidate_ids]
        )
        target_median_solve_time_ms.append(
            [
                _target_float(target["candidate_median_solve_time_ms"].get(candidate_id))
                if candidate_id in active_candidates
                else -1.0
                for candidate_id in global_candidate_ids
            ]
        )
        target_regret_vs_oracle_ms.append(
            [
                _target_float(target["candidate_regret_vs_oracle_ms"].get(candidate_id))
                if candidate_id in active_candidates
                else -1.0
                for candidate_id in global_candidate_ids
            ]
        )
        oracle_candidate_id = target["oracle_candidate_id"]
        oracle_index.append(
            candidate_slot_by_id[str(oracle_candidate_id)]
            if oracle_candidate_id is not None
            else -1
        )

    return {
        "schema_version": CSR_TENSOR_SCHEMA_VERSION,
        "storage_format": "json_numeric_arrays_v1",
        "request_ids": request_ids,
        "matrix_ids": matrix_ids,
        "context_ids": context_ids,
        "global_candidate_ids": list(global_candidate_ids),
        "matrix_feature_names": list(matrix_feature_names),
        "candidate_feature_names": list(candidate_feature_names),
        "split_ids": split_ids,
        "split_to_id": dict(SPLIT_TO_ID),
        "label_class_to_id": dict(LABEL_CLASS_TO_ID),
        "target_status_to_id": dict(TARGET_STATUS_TO_ID),
        "matrix_features": matrix_features,
        "candidate_features": candidate_features,
        "candidate_mask": candidate_mask,
        "label_class_ids": label_class_ids,
        "target_status_ids": target_status_ids,
        "profiled_success_mask": profiled_success_mask,
        "target_median_solve_time_ms": target_median_solve_time_ms,
        "target_regret_vs_oracle_ms": target_regret_vs_oracle_ms,
        "oracle_index": oracle_index,
    }


def _schema(
    *,
    matrix_feature_names: tuple[str, ...],
    candidate_feature_names: tuple[str, ...],
    global_candidate_ids: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "schema_version": CSR_TENSOR_SCHEMA_VERSION,
        "source_contract_schema_version": "phase1_csr_model_contract_v1",
        "storage_format": "json_numeric_arrays_v1",
        "model_required": False,
        "runtime_selector_changed": False,
        "axes": {
            "request": "matrix/context/split request row",
            "candidate": "global candidate slot",
            "matrix_feature": list(matrix_feature_names),
            "candidate_feature": list(candidate_feature_names),
        },
        "global_candidate_ids": list(global_candidate_ids),
        "arrays": {
            "matrix_features": "[num_requests, matrix_feature_dim] float",
            "candidate_features": "[num_requests, num_global_candidates, candidate_feature_dim] float",
            "candidate_mask": "[num_requests, num_global_candidates] int 0/1",
            "label_class_ids": "[num_requests, num_global_candidates] int",
            "target_status_ids": "[num_requests, num_global_candidates] int",
            "profiled_success_mask": "[num_requests, num_global_candidates] int 0/1",
            "target_median_solve_time_ms": "[num_requests, num_global_candidates] float, -1 for missing",
            "target_regret_vs_oracle_ms": "[num_requests, num_global_candidates] float, -1 for missing",
            "oracle_index": "[num_requests] int, -1 when no oracle exists",
            "split_ids": "[num_requests] int",
        },
        "label_class_to_id": dict(LABEL_CLASS_TO_ID),
        "target_status_to_id": dict(TARGET_STATUS_TO_ID),
        "split_to_id": dict(SPLIT_TO_ID),
        "next_step": "train_small_ranker_or_transformer_against_stable_arrays",
    }


def _summary(
    arrays: dict[str, Any],
    *,
    targets: tuple[dict[str, Any], ...],
    matrix_feature_names: tuple[str, ...],
    candidate_feature_names: tuple[str, ...],
) -> CsrTensorSummary:
    split_ids = arrays["split_ids"]
    candidate_mask = arrays["candidate_mask"]
    label_class_ids = arrays["label_class_ids"]
    target_status_ids = arrays["target_status_ids"]
    oracle_index = arrays["oracle_index"]
    label_counts = _id_counts(label_class_ids, LABEL_CLASS_TO_ID)
    status_counts = _id_counts(target_status_ids, TARGET_STATUS_TO_ID)
    num_active = sum(sum(row) for row in candidate_mask)
    num_requests = len(arrays["request_ids"])
    num_candidates = len(arrays["global_candidate_ids"])
    status = (
        "passed"
        if num_requests > 0
        and num_candidates > 0
        and len(arrays["matrix_features"]) == num_requests
        and all(len(row) == len(matrix_feature_names) for row in arrays["matrix_features"])
        and all(len(row) == num_candidates for row in candidate_mask)
        and all(len(row) == num_candidates for row in label_class_ids)
        and all(len(row) == num_candidates for row in target_status_ids)
        and all(
            len(candidate_row) == len(candidate_feature_names)
            for request_row in arrays["candidate_features"]
            for candidate_row in request_row
        )
        else "failed"
    )
    return CsrTensorSummary(
        status=status,
        schema_version=CSR_TENSOR_SCHEMA_VERSION,
        source_contract_schema_version=(
            str(targets[0]["schema_version"]) if targets else "unknown"
        ),
        storage_format="json_numeric_arrays_v1",
        num_requests=num_requests,
        num_train_requests=sum(1 for item in split_ids if int(item) == SPLIT_TO_ID["train"]),
        num_eval_requests=sum(1 for item in split_ids if int(item) == SPLIT_TO_ID["eval"]),
        num_global_candidates=num_candidates,
        num_active_candidate_slots=num_active,
        matrix_feature_dim=len(matrix_feature_names),
        candidate_feature_dim=len(candidate_feature_names),
        num_oracle_targets=sum(1 for item in oracle_index if int(item) >= 0),
        num_requests_without_oracle=sum(1 for item in oracle_index if int(item) < 0),
        label_class_counts=label_counts,
        target_status_counts=status_counts,
        has_missing_candidate_slots=(num_active != num_requests * num_candidates),
        model_required=False,
        runtime_selector_changed=False,
    )


def _request_index_rows(
    requests: tuple[dict[str, Any], ...],
    *,
    target_by_id: dict[str, dict[str, Any]],
    global_candidate_ids: tuple[str, ...],
) -> tuple[dict[str, Any], ...]:
    candidate_slot_by_id = {
        candidate_id: index for index, candidate_id in enumerate(global_candidate_ids)
    }
    rows = []
    for row_index, request in enumerate(requests):
        target = target_by_id[str(request["request_id"])]
        oracle = target["oracle_candidate_id"]
        rows.append(
            {
                "schema_version": CSR_TENSOR_SCHEMA_VERSION,
                "row_index": row_index,
                "request_id": str(request["request_id"]),
                "split": str(request["split"]),
                "split_id": SPLIT_TO_ID[str(request["split"])],
                "matrix_id": str(request["matrix_id"]),
                "context_id": str(request["context_id"]),
                "num_active_candidates": len(request["candidate_ids"]),
                "oracle_candidate_id": oracle,
                "oracle_index": (
                    candidate_slot_by_id[str(oracle)] if oracle is not None else -1
                ),
            }
        )
    return tuple(rows)


def _matrix_feature_names(requests: tuple[dict[str, Any], ...]) -> tuple[str, ...]:
    fields = _categories(requests, "matrix_features", "field")
    symmetries = _categories(requests, "matrix_features", "declared_symmetry")
    precisions = _categories(requests, "matrix_features", "recommended_precision")
    return (
        *_BASE_MATRIX_FEATURE_NAMES,
        *(f"field={value}" for value in fields),
        *(f"declared_symmetry={value}" for value in symmetries),
        *(f"recommended_precision={value}" for value in precisions),
    )


def _candidate_feature_names(requests: tuple[dict[str, Any], ...]) -> tuple[str, ...]:
    candidate_rows = [
        row
        for request in requests
        for row in request["candidate_features"].values()
    ]
    solvers = tuple(sorted({str(row["solver"]) for row in candidate_rows}))
    preconditioners = tuple(sorted({str(row["preconditioner"]) for row in candidate_rows}))
    precisions = tuple(sorted({str(row["precision"]) for row in candidate_rows}))
    return (
        *_BASE_CANDIDATE_FEATURE_NAMES,
        *(f"solver={value}" for value in solvers),
        *(f"preconditioner={value}" for value in preconditioners),
        *(f"precision={value}" for value in precisions),
    )


def _matrix_feature_vector(
    request: dict[str, Any],
    feature_names: tuple[str, ...],
) -> list[float]:
    features = dict(request["matrix_features"])
    values = []
    for name in feature_names:
        if name.endswith("_log10p"):
            source = name.removesuffix("_log10p")
            values.append(_log10p(features.get(source)))
        elif name.startswith("field="):
            values.append(1.0 if str(features.get("field")) == name.split("=", 1)[1] else 0.0)
        elif name.startswith("declared_symmetry="):
            values.append(
                1.0
                if str(features.get("declared_symmetry")) == name.split("=", 1)[1]
                else 0.0
            )
        elif name.startswith("recommended_precision="):
            values.append(
                1.0
                if str(features.get("recommended_precision")) == name.split("=", 1)[1]
                else 0.0
            )
        else:
            values.append(_numeric(features.get(name)))
    return values


def _candidate_feature_vector(
    features: dict[str, Any] | None,
    feature_names: tuple[str, ...],
) -> list[float]:
    if features is None:
        return [0.0] * len(feature_names)
    params = dict(features.get("solver_parameters", {}))
    has_restart = "restart" in params
    has_omega = "omega" in params
    has_lambda = "lambda_min" in params and "lambda_max" in params
    values = []
    for name in feature_names:
        if name == "has_restart":
            values.append(1.0 if has_restart else 0.0)
        elif name == "restart":
            values.append(_numeric(params.get("restart")))
        elif name == "has_omega":
            values.append(1.0 if has_omega else 0.0)
        elif name == "omega":
            values.append(_numeric(params.get("omega")))
        elif name == "has_lambda_bounds":
            values.append(1.0 if has_lambda else 0.0)
        elif name == "lambda_min_log10":
            values.append(_log10_positive(params.get("lambda_min")) if has_lambda else 0.0)
        elif name == "lambda_max_log10":
            values.append(_log10_positive(params.get("lambda_max")) if has_lambda else 0.0)
        elif name.startswith("solver="):
            values.append(1.0 if str(features.get("solver")) == name.split("=", 1)[1] else 0.0)
        elif name.startswith("preconditioner="):
            values.append(
                1.0
                if str(features.get("preconditioner")) == name.split("=", 1)[1]
                else 0.0
            )
        elif name.startswith("precision="):
            values.append(1.0 if str(features.get("precision")) == name.split("=", 1)[1] else 0.0)
        else:
            raise ValueError(f"unknown candidate feature: {name}")
    return values


def _categories(
    requests: tuple[dict[str, Any], ...],
    source_key: str,
    field: str,
) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                str(request[source_key].get(field))
                for request in requests
                if request[source_key].get(field) is not None
            }
        )
    )


def _id_counts(rows: list[list[int]], mapping: dict[str, int]) -> dict[str, int]:
    inverse = {value: key for key, value in mapping.items()}
    counts = {key: 0 for key in mapping}
    for row in rows:
        for value in row:
            if int(value) in inverse:
                counts[inverse[int(value)]] += 1
    return {key: value for key, value in counts.items() if value > 0}


def _numeric(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    return float(value)


def _target_float(value: Any) -> float:
    return -1.0 if value is None else float(value)


def _log10p(value: Any) -> float:
    return math.log10(1.0 + abs(_numeric(value)))


def _log10_positive(value: Any) -> float:
    numeric = _numeric(value)
    if numeric <= 0.0:
        return 0.0
    return math.log10(numeric)
