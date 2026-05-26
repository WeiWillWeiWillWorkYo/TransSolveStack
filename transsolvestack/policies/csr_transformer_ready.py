"""Build the CSR Transformer-ready selector training bundle."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from transsolvestack.policies.csr_learning import (
    build_csr_learning_export_from_selector_rows,
    write_csr_baseline_predictions,
    write_csr_learning_report,
    write_csr_learning_rows,
    write_csr_learning_schema,
    write_csr_learning_summary,
)
from transsolvestack.policies.csr_model_contract import (
    build_csr_model_contract_export,
    write_csr_model_contract_report,
    write_csr_model_contract_schema,
    write_csr_model_contract_summary,
    write_csr_model_predictions,
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


CSR_TRANSFORMER_READY_SCHEMA_VERSION = "phase1_csr_transformer_ready_v1"


def build_csr_transformer_ready_bundle_from_files(
    selector_paths: Iterable[str | Path],
    output_dir: str | Path,
    *,
    eval_fraction: float = 0.25,
    min_eval_matrices: int = 4,
) -> dict[str, Any]:
    paths = tuple(Path(path) for path in selector_paths)
    selector_rows = _combined_selector_rows(paths)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    combined_selector_path = output / "combined_csr_selector_rows.jsonl"
    write_jsonl(selector_rows, combined_selector_path)

    learning = build_csr_learning_export_from_selector_rows(
        combined_selector_path,
        eval_fraction=eval_fraction,
        min_eval_matrices=min_eval_matrices,
    )
    learning_rows_path = output / "combined_csr_learning_rows.jsonl"
    baseline_predictions_path = output / "combined_csr_baseline_predictions.jsonl"
    write_csr_learning_rows(learning.rows, learning_rows_path)
    write_csr_baseline_predictions(learning.predictions, baseline_predictions_path)

    contract = build_csr_model_contract_export(
        learning_rows_path,
        baseline_predictions_path,
        prediction_source="csr_transformer_ready_baseline_adapter_v1",
    )
    requests_path = output / "csr_transformer_model_requests.jsonl"
    targets_path = output / "csr_transformer_model_targets.jsonl"
    predictions_path = output / "csr_transformer_baseline_predictions.jsonl"
    write_csr_model_requests(contract.requests, requests_path)
    write_csr_model_targets(contract.targets, targets_path)
    write_csr_model_predictions(contract.predictions, predictions_path)

    tensors = build_csr_tensor_export(requests_path, targets_path)
    arrays_path = output / "csr_transformer_training_tensors.json"
    request_index_path = output / "csr_transformer_request_index.jsonl"
    write_csr_tensor_arrays(tensors, arrays_path)
    write_csr_tensor_request_index(tensors, request_index_path)

    summary = _summary(
        selector_rows,
        selector_paths=paths,
        learning_summary=asdict(learning.summary),
        contract_summary=asdict(contract.summary),
        tensor_summary=asdict(tensors.summary),
    )
    schema = _schema(tensors.schema, summary=summary)
    _write_supporting_artifacts(
        output,
        learning=learning,
        contract=contract,
        tensors=tensors,
        summary=summary,
        schema=schema,
    )
    return {
        "selector_rows": selector_rows,
        "learning": learning,
        "contract": contract,
        "tensors": tensors,
        "summary": summary,
        "schema": schema,
        "paths": {
            "combined_selector_rows": combined_selector_path,
            "learning_rows": learning_rows_path,
            "baseline_predictions": baseline_predictions_path,
            "model_requests": requests_path,
            "model_targets": targets_path,
            "model_predictions": predictions_path,
            "tensor_arrays": arrays_path,
            "request_index": request_index_path,
            "summary": output / "csr_transformer_ready_summary.json",
            "schema": output / "csr_transformer_ready_schema.json",
            "report": output / "csr_transformer_ready_report.md",
        },
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


def _summary(
    selector_rows: tuple[dict[str, Any], ...],
    *,
    selector_paths: tuple[Path, ...],
    learning_summary: dict[str, Any],
    contract_summary: dict[str, Any],
    tensor_summary: dict[str, Any],
) -> dict[str, Any]:
    matrix_ids = {str(row["matrix_id"]) for row in selector_rows}
    success_rows = tuple(row for row in selector_rows if row["target_status"] == "success")
    screened_rows = tuple(
        row for row in selector_rows if row["target_status"] == "screened_out"
    )
    oracle_rows = tuple(row for row in selector_rows if row["label_is_oracle"])
    transformer_connectable = (
        bool(selector_rows)
        and learning_summary["status"] == "passed"
        and contract_summary["status"] == "passed"
        and tensor_summary["status"] == "passed"
        and contract_summary["validation_error_count"] == 0
        and tensor_summary["model_required"] is False
        and tensor_summary["runtime_selector_changed"] is False
    )
    return {
        "status": "passed" if transformer_connectable else "failed",
        "schema_version": CSR_TRANSFORMER_READY_SCHEMA_VERSION,
        "source_selector_paths": tuple(str(path) for path in selector_paths),
        "transformer_connectable": transformer_connectable,
        "runtime_selector_changed": False,
        "model_required": False,
        "num_selector_rows": len(selector_rows),
        "num_matrices": len(matrix_ids),
        "num_success_rows": len(success_rows),
        "num_screened_out_rows": len(screened_rows),
        "num_oracle_rows": len(oracle_rows),
        "num_learning_rows": learning_summary["num_rows"],
        "num_learning_train_rows": learning_summary["num_train_rows"],
        "num_learning_eval_rows": learning_summary["num_eval_rows"],
        "num_model_requests": contract_summary["num_requests"],
        "num_model_targets": contract_summary["num_targets"],
        "num_model_predictions": contract_summary["num_predictions"],
        "num_tensor_requests": tensor_summary["num_requests"],
        "num_global_candidates": tensor_summary["num_global_candidates"],
        "num_active_candidate_slots": tensor_summary["num_active_candidate_slots"],
        "matrix_feature_dim": tensor_summary["matrix_feature_dim"],
        "candidate_feature_dim": tensor_summary["candidate_feature_dim"],
        "num_oracle_targets": tensor_summary["num_oracle_targets"],
        "num_requests_without_oracle": tensor_summary["num_requests_without_oracle"],
        "label_class_counts": tensor_summary["label_class_counts"],
        "target_status_counts": tensor_summary["target_status_counts"],
        "eval_oracle_top1_accuracy": contract_summary["eval_oracle_top1_accuracy"],
        "eval_profiled_selection_rate": contract_summary["eval_profiled_selection_rate"],
        "validation_error_count": contract_summary["validation_error_count"],
    }


def _schema(tensor_schema: dict[str, Any], *, summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": CSR_TRANSFORMER_READY_SCHEMA_VERSION,
        "task": "connect_transformer_ranker_to_real_csr_selector_contract",
        "runtime_selector_changed": False,
        "model_required": False,
        "transformer_connectable": summary["transformer_connectable"],
        "input_contract": {
            "request_file": "csr_transformer_model_requests.jsonl",
            "target_file": "csr_transformer_model_targets.jsonl",
            "tensor_file": "csr_transformer_training_tensors.json",
            "request_index_file": "csr_transformer_request_index.jsonl",
            "prediction_validation": "phase1_csr_model_contract_v1",
        },
        "tensor_axes": tensor_schema["axes"],
        "global_candidate_ids": tensor_schema["global_candidate_ids"],
        "expected_model_output": {
            "ranked_candidate_ids": "per request, rank every candidate with candidate_mask=1",
            "scores": "finite score per requested candidate",
            "selected_candidate_id": "top ranked candidate or null",
        },
        "integration_gate": {
            "required_before_runtime": [
                "validate predictions against csr_model_contract",
                "compare against artifact-backed baseline",
                "require no non-success runtime selections",
                "keep runtime selector artifact-backed until quality gate passes",
            ],
        },
    }


def _write_supporting_artifacts(
    output: Path,
    *,
    learning: Any,
    contract: Any,
    tensors: Any,
    summary: dict[str, Any],
    schema: dict[str, Any],
) -> None:
    write_csr_learning_summary(learning.summary, output / "combined_csr_learning_summary.json")
    write_csr_learning_schema(output / "combined_csr_learning_schema.json")
    write_csr_learning_report(learning, output / "combined_csr_learning_report.md")
    write_csr_model_contract_summary(
        contract.summary,
        output / "csr_transformer_model_contract_summary.json",
    )
    write_csr_model_contract_schema(output / "csr_transformer_model_contract_schema.json")
    write_csr_model_contract_report(contract, output / "csr_transformer_model_contract_report.md")
    write_csr_tensor_summary(tensors.summary, output / "csr_transformer_tensor_summary.json")
    write_csr_tensor_schema(tensors.schema, output / "csr_transformer_tensor_schema.json")
    write_csr_tensor_report(tensors, output / "csr_transformer_tensor_report.md")
    (output / "csr_transformer_ready_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "csr_transformer_ready_schema.json").write_text(
        json.dumps(schema, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_report(summary, output / "csr_transformer_ready_report.md")


def _write_report(summary: dict[str, Any], path: Path) -> Path:
    lines = [
        "# CSR Transformer-Ready Bundle",
        "",
        f"- status: `{summary['status']}`",
        f"- transformer_connectable: `{summary['transformer_connectable']}`",
        f"- runtime_selector_changed: `{summary['runtime_selector_changed']}`",
        f"- selector_rows: `{summary['num_selector_rows']}`",
        f"- matrices: `{summary['num_matrices']}`",
        f"- success_rows: `{summary['num_success_rows']}`",
        f"- screened_out_rows: `{summary['num_screened_out_rows']}`",
        f"- oracle_rows: `{summary['num_oracle_rows']}`",
        f"- model_requests: `{summary['num_model_requests']}`",
        f"- global_candidates: `{summary['num_global_candidates']}`",
        f"- matrix_feature_dim: `{summary['matrix_feature_dim']}`",
        f"- candidate_feature_dim: `{summary['candidate_feature_dim']}`",
        f"- validation_error_count: `{summary['validation_error_count']}`",
        f"- eval_oracle_top1_accuracy: `{summary['eval_oracle_top1_accuracy']:.6g}`",
        f"- eval_profiled_selection_rate: `{summary['eval_profiled_selection_rate']:.6g}`",
        "",
        "## Label Counts",
        "",
        f"- label_class_counts: `{summary['label_class_counts']}`",
        f"- target_status_counts: `{summary['target_status_counts']}`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
