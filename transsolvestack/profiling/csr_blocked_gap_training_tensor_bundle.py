"""Context-safe training tensor export for D19 blocked-gap integration."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from transsolvestack.policies.csr_learning import (
    build_csr_learning_export_from_selector_rows,
    write_csr_baseline_predictions,
    write_csr_learning_report,
    write_csr_learning_rows,
    write_csr_learning_schema,
    write_csr_learning_summary,
)
from transsolvestack.policies.csr_model_contract import (
    _group_learning_rows,
    _request_from_rows,
    _target_from_rows,
    write_csr_model_contract_schema,
    write_csr_model_requests,
    write_csr_model_targets,
)
from transsolvestack.policies.csr_tensor_export import (
    build_csr_tensor_export,
    write_csr_tensor_arrays,
    write_csr_tensor_report,
    write_csr_tensor_request_index,
    write_csr_tensor_schema,
    write_csr_tensor_summary,
)
from transsolvestack.profiling.artifacts import read_jsonl, write_jsonl


def build_training_tensor_bundle(
    *,
    selector_paths: tuple[Path, ...],
    output: Path,
    eval_fraction: float,
    min_eval_matrices: int,
) -> dict[str, Any]:
    """Build request/target/tensor artifacts without baseline prediction binding.

    The legacy baseline adapter predicts by matrix id. D19 intentionally adds
    another context for already-seen matrices, so this shard exports the full
    request/target tensor contract and keeps baseline predictions informational.
    """

    selector_rows = _combined_selector_rows(selector_paths)
    paths = _paths(output)
    write_jsonl(selector_rows, paths["combined_selector_rows"])
    learning = build_csr_learning_export_from_selector_rows(
        paths["combined_selector_rows"],
        eval_fraction=eval_fraction,
        min_eval_matrices=min_eval_matrices,
    )
    write_csr_learning_rows(learning.rows, paths["learning_rows"])
    write_csr_baseline_predictions(learning.predictions, paths["baseline_predictions"])
    write_csr_learning_summary(learning.summary, paths["learning_summary"])
    write_csr_learning_schema(paths["learning_schema"])
    write_csr_learning_report(learning, paths["learning_report"])

    learning_row_dicts = read_jsonl(paths["learning_rows"])
    grouped = _group_learning_rows(learning_row_dicts)
    requests = tuple(
        _request_from_rows(request_id, rows)
        for request_id, rows in sorted(grouped.items())
    )
    targets = tuple(
        _target_from_rows(request_id, rows)
        for request_id, rows in sorted(grouped.items())
    )
    write_csr_model_requests(requests, paths["model_requests"])
    write_csr_model_targets(targets, paths["model_targets"])
    write_csr_model_contract_schema(paths["model_contract_schema"])
    _write_json(
        _model_contract_summary(requests=requests, targets=targets),
        paths["model_contract_summary"],
    )

    tensors = build_csr_tensor_export(paths["model_requests"], paths["model_targets"])
    write_csr_tensor_arrays(tensors, paths["tensor_arrays"])
    write_csr_tensor_request_index(tensors, paths["request_index"])
    write_csr_tensor_summary(tensors.summary, paths["tensor_summary"])
    write_csr_tensor_schema(tensors.schema, paths["tensor_schema"])
    write_csr_tensor_report(tensors, paths["tensor_report"])
    summary = _summary(
        selector_rows=selector_rows,
        selector_paths=selector_paths,
        learning_summary=asdict(learning.summary),
        tensor_summary=asdict(tensors.summary),
        requests=requests,
        targets=targets,
    )
    return {
        "paths": paths,
        "summary": summary,
        "selector_rows": selector_rows,
        "learning": learning,
        "requests": requests,
        "targets": targets,
        "tensors": tensors,
    }


def _combined_selector_rows(paths: tuple[Path, ...]) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for path in paths:
        for row in read_jsonl(path):
            key = (
                str(row["matrix_id"]),
                str(row["context_id"]),
                str(row["candidate_id"]),
            )
            if key in seen:
                raise ValueError(f"duplicate CSR selector row: {key}")
            seen.add(key)
            rows.append(dict(row))
    return tuple(
        sorted(
            rows,
            key=lambda row: (
                str(row["matrix_id"]),
                str(row["context_id"]),
                str(row["candidate_id"]),
            ),
        )
    )


def _paths(output: Path) -> dict[str, Path]:
    return {
        "combined_selector_rows": output / "combined_csr_selector_rows.jsonl",
        "learning_rows": output / "combined_csr_learning_rows.jsonl",
        "baseline_predictions": output / "combined_csr_baseline_predictions.jsonl",
        "model_requests": output / "csr_blocked_gap_training_model_requests.jsonl",
        "model_targets": output / "csr_blocked_gap_training_model_targets.jsonl",
        "model_contract_schema": (
            output / "csr_blocked_gap_training_model_contract_schema.json"
        ),
        "model_contract_summary": (
            output / "csr_blocked_gap_training_model_contract_summary.json"
        ),
        "tensor_arrays": output / "csr_blocked_gap_training_tensors.json",
        "request_index": output / "csr_blocked_gap_training_request_index.jsonl",
        "tensor_summary": output / "csr_blocked_gap_training_tensor_summary.json",
        "tensor_schema": output / "csr_blocked_gap_training_tensor_schema.json",
        "tensor_report": output / "csr_blocked_gap_training_tensor_report.md",
        "learning_summary": output / "combined_csr_learning_summary.json",
        "learning_schema": output / "combined_csr_learning_schema.json",
        "learning_report": output / "combined_csr_learning_report.md",
    }


def _summary(
    *,
    selector_rows: tuple[dict[str, Any], ...],
    selector_paths: tuple[Path, ...],
    learning_summary: dict[str, Any],
    tensor_summary: dict[str, Any],
    requests: tuple[Any, ...],
    targets: tuple[Any, ...],
) -> dict[str, Any]:
    success_rows = tuple(row for row in selector_rows if row["target_status"] == "success")
    screened_rows = tuple(
        row for row in selector_rows if row["target_status"] == "screened_out"
    )
    oracle_rows = tuple(row for row in selector_rows if bool(row["label_is_oracle"]))
    status = (
        "passed"
        if learning_summary["status"] == "passed"
        and tensor_summary["status"] == "passed"
        and len(requests) == len(targets)
        else "failed"
    )
    return {
        "status": status,
        "schema_version": "phase1_csr_blocked_gap_training_tensor_bundle_v1",
        "source_selector_paths": tuple(str(path) for path in selector_paths),
        "transformer_connectable": status == "passed",
        "runtime_selector_changed": False,
        "model_required": False,
        "baseline_prediction_required": False,
        "num_selector_rows": len(selector_rows),
        "num_matrices": len({str(row["matrix_id"]) for row in selector_rows}),
        "num_matrix_contexts": len(
            {
                (str(row["matrix_id"]), str(row["context_id"]))
                for row in selector_rows
            }
        ),
        "num_success_rows": len(success_rows),
        "num_screened_out_rows": len(screened_rows),
        "num_oracle_rows": len(oracle_rows),
        "num_learning_rows": learning_summary["num_rows"],
        "num_learning_train_rows": learning_summary["num_train_rows"],
        "num_learning_eval_rows": learning_summary["num_eval_rows"],
        "num_model_requests": len(requests),
        "num_model_targets": len(targets),
        "num_model_predictions": 0,
        "num_tensor_requests": tensor_summary["num_requests"],
        "num_global_candidates": tensor_summary["num_global_candidates"],
        "num_active_candidate_slots": tensor_summary["num_active_candidate_slots"],
        "matrix_feature_dim": tensor_summary["matrix_feature_dim"],
        "candidate_feature_dim": tensor_summary["candidate_feature_dim"],
        "num_oracle_targets": tensor_summary["num_oracle_targets"],
        "num_requests_without_oracle": tensor_summary["num_requests_without_oracle"],
        "label_class_counts": tensor_summary["label_class_counts"],
        "target_status_counts": tensor_summary["target_status_counts"],
        "validation_error_count": 0,
    }


def _model_contract_summary(
    *,
    requests: tuple[Any, ...],
    targets: tuple[Any, ...],
) -> dict[str, Any]:
    return {
        "status": "passed" if len(requests) == len(targets) and requests else "failed",
        "schema_version": "phase1_csr_model_contract_v1",
        "model_required": False,
        "runtime_selector_changed": False,
        "prediction_source": "training_only_no_baseline_predictions",
        "num_requests": len(requests),
        "num_targets": len(targets),
        "num_predictions": 0,
        "validation_error_count": 0,
        "contract_note": (
            "D19 exports requests and targets for training tensors; baseline "
            "predictions are not required for this integration shard."
        ),
    }


def _write_json(data: dict[str, Any], path: Path) -> Path:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
