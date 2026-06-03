"""Integrate blocked-gap positive rows into training-only CSR evidence."""

from __future__ import annotations

import json
from collections import Counter
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
from transsolvestack.profiling.csr_blocked_gap_training_tensor_bundle import (
    build_training_tensor_bundle as _build_training_tensor_bundle,
)


CSR_BLOCKED_GAP_TRAINING_INTEGRATION_SCHEMA_VERSION = (
    "phase1_csr_blocked_gap_training_integration_v1"
)


def build_csr_blocked_gap_training_integration_from_files(
    *,
    base_selector_rows_path: str | Path = (
        "runs/phase1_csr_queue_training_pool/"
        "csr_queue_training_pool_selector_rows.jsonl"
    ),
    base_pool_summary_path: str | Path = (
        "runs/phase1_csr_queue_training_pool/csr_queue_training_pool_summary.json"
    ),
    positive_selector_rows_path: str | Path = (
        "runs/phase1_csr_blocked_gap_positive_search/"
        "csr_blocked_gap_positive_selector_rows.jsonl"
    ),
    positive_summary_path: str | Path = (
        "runs/phase1_csr_blocked_gap_positive_search/"
        "csr_blocked_gap_positive_search_summary.json"
    ),
    output_dir: str | Path = "runs/phase1_csr_blocked_gap_training_integration",
    eval_fraction: float = 0.25,
    min_eval_matrices: int = 4,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    base_path = Path(base_selector_rows_path)
    positive_path = Path(positive_selector_rows_path)
    base_rows = tuple(read_jsonl(base_path))
    positive_rows = tuple(read_jsonl(positive_path))
    positive_summary = _read_json(Path(positive_summary_path))
    base_summary = (
        _read_json(Path(base_pool_summary_path))
        if Path(base_pool_summary_path).exists()
        else {}
    )
    duplicate_keys = _duplicate_keys(base_rows, positive_rows)

    bundle = _build_training_tensor_bundle(
        selector_paths=(base_path, positive_path),
        output=output,
        eval_fraction=eval_fraction,
        min_eval_matrices=min_eval_matrices,
    )
    transformer_summary = dict(bundle["summary"])
    source_rows = _source_rows(
        base_rows=base_rows,
        positive_rows=positive_rows,
        base_path=base_path,
        positive_path=positive_path,
        base_summary=base_summary,
        positive_summary=positive_summary,
    )
    positive_membership = _positive_membership(positive_rows)
    summary = _summary(
        base_rows=base_rows,
        positive_rows=positive_rows,
        positive_summary=positive_summary,
        transformer_summary=transformer_summary,
        duplicate_keys=duplicate_keys,
        base_path=base_path,
        positive_path=positive_path,
        base_pool_summary_path=Path(base_pool_summary_path),
        positive_summary_path=Path(positive_summary_path),
    )
    schema = _schema(summary)
    paths = {
        **bundle["paths"],
        "integration_sources": output / "csr_blocked_gap_training_sources.jsonl",
        "positive_membership": (
            output / "csr_blocked_gap_training_positive_membership.jsonl"
        ),
        "integration_summary": (
            output / "csr_blocked_gap_training_integration_summary.json"
        ),
        "integration_schema": (
            output / "csr_blocked_gap_training_integration_schema.json"
        ),
        "integration_report": (
            output / "csr_blocked_gap_training_integration_report.md"
        ),
    }
    write_jsonl(source_rows, paths["integration_sources"])
    write_jsonl(positive_membership, paths["positive_membership"])
    _write_json(summary, paths["integration_summary"])
    _write_json(schema, paths["integration_schema"])
    paths["integration_report"].write_text(_report(summary), encoding="utf-8")
    return {
        "paths": {key: str(value) for key, value in paths.items()},
        "summary": summary,
        "schema": schema,
        "source_rows": source_rows,
        "positive_membership": positive_membership,
        "transformer_ready": bundle,
    }


def _source_rows(
    *,
    base_rows: tuple[dict[str, Any], ...],
    positive_rows: tuple[dict[str, Any], ...],
    base_path: Path,
    positive_path: Path,
    base_summary: dict[str, Any],
    positive_summary: dict[str, Any],
) -> tuple[dict[str, Any], ...]:
    return (
        {
            "schema_version": CSR_BLOCKED_GAP_TRAINING_INTEGRATION_SCHEMA_VERSION,
            "source_id": "queue_training_pool",
            "source_kind": "base_training_pool",
            "selector_rows_path": str(base_path),
            "selector_rows": len(base_rows),
            "matrices": len({str(row["matrix_id"]) for row in base_rows}),
            "matrix_contexts": len(
                {(str(row["matrix_id"]), str(row["context_id"])) for row in base_rows}
            ),
            "success_rows": _count_status(base_rows, "success"),
            "screened_out_rows": _count_status(base_rows, "screened_out"),
            "oracle_rows": _count_oracle(base_rows),
            "training_pool_ready": bool(base_summary.get("training_pool_ready", True)),
            "runtime_selector_changed": bool(
                base_summary.get("runtime_selector_changed", False)
            ),
            "generic_queue_merge": True,
        },
        {
            "schema_version": CSR_BLOCKED_GAP_TRAINING_INTEGRATION_SCHEMA_VERSION,
            "source_id": "blocked_gap_positive_search",
            "source_kind": "bounded_positive_gpu_evidence",
            "selector_rows_path": str(positive_path),
            "selector_rows": len(positive_rows),
            "matrices": len({str(row["matrix_id"]) for row in positive_rows}),
            "matrix_contexts": len(
                {
                    (str(row["matrix_id"]), str(row["context_id"]))
                    for row in positive_rows
                }
            ),
            "success_rows": _count_status(positive_rows, "success"),
            "screened_out_rows": _count_status(positive_rows, "screened_out"),
            "oracle_rows": _count_oracle(positive_rows),
            "positive_evidence_found": bool(
                positive_summary.get("positive_evidence_found", False)
            ),
            "positive_gpu_gap_ids": tuple(
                positive_summary.get("positive_gpu_gap_ids", ())
            ),
            "queue_merge_ready": bool(positive_summary.get("queue_merge_ready", True)),
            "runtime_selector_changed": bool(
                positive_summary.get("runtime_selector_changed", True)
            ),
            "generic_queue_merge": False,
        },
    )


def _positive_membership(
    positive_rows: tuple[dict[str, Any], ...],
) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "schema_version": CSR_BLOCKED_GAP_TRAINING_INTEGRATION_SCHEMA_VERSION,
            "source_id": "blocked_gap_positive_search",
            "matrix_id": str(row["matrix_id"]),
            "context_id": str(row["context_id"]),
            "candidate_id": str(row["candidate_id"]),
            "solver": str(row["solver"]),
            "preconditioner": str(row["preconditioner"]),
            "target_status": str(row["target_status"]),
            "label_is_oracle": bool(row["label_is_oracle"]),
            "source_augmented_from": "phase1_csr_blocked_gap_positive_search",
        }
        for row in positive_rows
    )


def _summary(
    *,
    base_rows: tuple[dict[str, Any], ...],
    positive_rows: tuple[dict[str, Any], ...],
    positive_summary: dict[str, Any],
    transformer_summary: dict[str, Any],
    duplicate_keys: tuple[tuple[str, str, str], ...],
    base_path: Path,
    positive_path: Path,
    base_pool_summary_path: Path,
    positive_summary_path: Path,
) -> dict[str, Any]:
    base_candidate_ids = {str(row["candidate_id"]) for row in base_rows}
    positive_candidate_ids = {str(row["candidate_id"]) for row in positive_rows}
    positive_contexts = tuple(sorted({str(row["context_id"]) for row in positive_rows}))
    positive_gap_row_counts = dict(sorted(Counter(_gap_id(row) for row in positive_rows).items()))
    positive_gap_ids = tuple(sorted(positive_gap_row_counts))
    positive_status_counts = _status_counts(positive_rows)
    transformer_ready = bool(transformer_summary.get("transformer_connectable"))
    positive_ready = (
        positive_summary.get("status") == "passed"
        and positive_summary.get("positive_evidence_found") is True
        and positive_summary.get("queue_merge_ready") is False
        and positive_summary.get("runtime_selector_changed") is False
        and len(positive_rows) == int(positive_summary.get("selector_rows", -1))
    )
    status = (
        "passed"
        if transformer_summary.get("status") == "passed"
        and transformer_ready
        and positive_ready
        and not duplicate_keys
        and positive_status_counts == {"success": len(positive_rows)}
        and _count_status(positive_rows, "success") == 8
        else "failed"
    )
    return {
        "status": status,
        "schema_version": CSR_BLOCKED_GAP_TRAINING_INTEGRATION_SCHEMA_VERSION,
        "runtime_selector_changed": False,
        "executes_gpu": False,
        "imports_matrices": False,
        "model_required": False,
        "training_only_integration": True,
        "generic_queue_merge": False,
        "source_selector_paths": (str(base_path), str(positive_path)),
        "base_pool_summary_path": str(base_pool_summary_path),
        "positive_summary_path": str(positive_summary_path),
        "base_selector_rows": len(base_rows),
        "positive_selector_rows": len(positive_rows),
        "integrated_selector_rows": int(transformer_summary["num_selector_rows"]),
        "base_matrix_contexts": len(
            {(str(row["matrix_id"]), str(row["context_id"])) for row in base_rows}
        ),
        "positive_matrix_contexts": len(
            {
                (str(row["matrix_id"]), str(row["context_id"]))
                for row in positive_rows
            }
        ),
        "integrated_model_requests": int(transformer_summary["num_model_requests"]),
        "model_request_delta": int(transformer_summary["num_model_requests"])
        - len({(str(row["matrix_id"]), str(row["context_id"])) for row in base_rows}),
        "base_success_rows": _count_status(base_rows, "success"),
        "positive_success_rows": _count_status(positive_rows, "success"),
        "integrated_success_rows": int(transformer_summary["num_success_rows"]),
        "base_oracle_rows": _count_oracle(base_rows),
        "positive_oracle_rows": _count_oracle(positive_rows),
        "integrated_oracle_rows": int(transformer_summary["num_oracle_rows"]),
        "positive_target_status_counts": positive_status_counts,
        "positive_contexts": positive_contexts,
        "positive_gap_ids": positive_gap_ids,
        "positive_gap_row_counts": positive_gap_row_counts,
        "positive_candidate_ids": tuple(sorted(positive_candidate_ids)),
        "new_global_candidate_ids": tuple(
            sorted(positive_candidate_ids - base_candidate_ids)
        ),
        "existing_global_candidate_ids": tuple(
            sorted(positive_candidate_ids & base_candidate_ids)
        ),
        "integrated_global_candidates": int(transformer_summary["num_global_candidates"]),
        "integrated_active_candidate_slots": int(
            transformer_summary["num_active_candidate_slots"]
        ),
        "integrated_tensor_requests": int(transformer_summary["num_tensor_requests"]),
        "integrated_transformer_connectable": transformer_ready,
        "validation_error_count": int(transformer_summary["validation_error_count"]),
        "duplicate_key_count": len(duplicate_keys),
        "duplicate_keys": duplicate_keys,
        "positive_source_queue_merge_ready": bool(
            positive_summary.get("queue_merge_ready", True)
        ),
        "positive_source_runtime_selector_changed": bool(
            positive_summary.get("runtime_selector_changed", True)
        ),
        "next_step": (
            "train or replay selector models on the augmented tensor bundle while "
            "keeping runtime promotion behind the learned guard"
        ),
    }


def _schema(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": CSR_BLOCKED_GAP_TRAINING_INTEGRATION_SCHEMA_VERSION,
        "task": "integrate_blocked_gap_positive_rows_into_training_only_bundle",
        "runtime_boundary": {
            "runtime_selector_changed": False,
            "executes_gpu": False,
            "imports_matrices": False,
            "generic_queue_merge": False,
            "model_required": False,
        },
        "input_sources": {
            "base_training_pool": summary["source_selector_paths"][0],
            "blocked_gap_positive_search": summary["source_selector_paths"][1],
        },
        "output_contract": {
            "combined_selector_rows": "combined_csr_selector_rows.jsonl",
            "model_requests": "csr_transformer_model_requests.jsonl",
            "training_tensors": "csr_transformer_training_tensors.json",
            "request_index": "csr_transformer_request_index.jsonl",
        },
        "acceptance_gate": {
            "requires_transformer_connectable": True,
            "requires_no_duplicate_selector_keys": True,
            "requires_positive_rows_success_only": True,
            "requires_positive_source_no_queue_merge": True,
            "requires_positive_source_no_runtime_change": True,
        },
    }


def _report(summary: dict[str, Any]) -> str:
    lines = [
        "# CSR Blocked Gap Training Integration",
        "",
        f"- status: `{summary['status']}`",
        f"- training_only_integration: `{summary['training_only_integration']}`",
        f"- runtime_selector_changed: `{summary['runtime_selector_changed']}`",
        f"- generic_queue_merge: `{summary['generic_queue_merge']}`",
        f"- base_selector_rows: `{summary['base_selector_rows']}`",
        f"- positive_selector_rows: `{summary['positive_selector_rows']}`",
        f"- integrated_selector_rows: `{summary['integrated_selector_rows']}`",
        f"- integrated_model_requests: `{summary['integrated_model_requests']}`",
        f"- integrated_global_candidates: `{summary['integrated_global_candidates']}`",
        f"- positive_success_rows: `{summary['positive_success_rows']}`",
        f"- positive_oracle_rows: `{summary['positive_oracle_rows']}`",
        f"- new_global_candidate_ids: `{summary['new_global_candidate_ids']}`",
        f"- next_step: `{summary['next_step']}`",
        "",
    ]
    return "\n".join(lines)


def _duplicate_keys(
    base_rows: tuple[dict[str, Any], ...],
    positive_rows: tuple[dict[str, Any], ...],
) -> tuple[tuple[str, str, str], ...]:
    base_keys = {
        (str(row["matrix_id"]), str(row["context_id"]), str(row["candidate_id"]))
        for row in base_rows
    }
    return tuple(
        sorted(
            {
                (
                    str(row["matrix_id"]),
                    str(row["context_id"]),
                    str(row["candidate_id"]),
                )
                for row in positive_rows
                if (
                    str(row["matrix_id"]),
                    str(row["context_id"]),
                    str(row["candidate_id"]),
                )
                in base_keys
            }
        )
    )


def _gap_id(row: dict[str, Any]) -> str:
    solver = str(row["solver"])
    preconditioner = str(row["preconditioner"])
    if solver == "bicgstab" and preconditioner == "ilu0":
        return "general_bicgstab_ilu0"
    if solver == "bicgstab" and preconditioner == "row_column_equilibration":
        return "general_bicgstab_row_column_equilibration"
    if solver == "chebyshev" and preconditioner == "jacobi":
        return "symmetric_chebyshev_jacobi"
    if solver == "pcg" and preconditioner == "symmetric_equilibration":
        return "symmetric_pcg_symmetric_equilibration"
    return f"{solver}:{preconditioner}"


def _count_status(rows: tuple[dict[str, Any], ...], status: str) -> int:
    return sum(1 for row in rows if str(row["target_status"]) == status)


def _count_oracle(rows: tuple[dict[str, Any], ...]) -> int:
    return sum(1 for row in rows if bool(row["label_is_oracle"]))


def _status_counts(rows: tuple[dict[str, Any], ...]) -> dict[str, int]:
    return dict(sorted(Counter(str(row["target_status"]) for row in rows).items()))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(data: dict[str, Any], path: Path) -> Path:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
