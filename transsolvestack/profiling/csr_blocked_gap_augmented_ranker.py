"""Train and inspect a shadow ranker on blocked-gap augmented CSR tensors."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from transsolvestack.policies.csr_transformer_ranker import (
    CsrTransformerRankerExport,
    CsrTransformerRankerPrediction,
    train_csr_transformer_ranker_from_tensor_file,
    write_csr_transformer_ranker_model,
    write_csr_transformer_ranker_predictions,
    write_csr_transformer_ranker_report,
    write_csr_transformer_ranker_schema,
    write_csr_transformer_ranker_summary,
)
from transsolvestack.profiling.artifacts import read_jsonl, write_jsonl


CSR_BLOCKED_GAP_AUGMENTED_RANKER_SCHEMA_VERSION = (
    "phase1_csr_blocked_gap_augmented_ranker_v1"
)
POSITIVE_CONTEXT_ID = "phase1_csr_blocked_gap_positive_search"


def build_csr_blocked_gap_augmented_ranker_from_files(
    *,
    tensors_path: str | Path = (
        "runs/phase1_csr_blocked_gap_training_integration/"
        "csr_blocked_gap_training_tensors.json"
    ),
    request_index_path: str | Path = (
        "runs/phase1_csr_blocked_gap_training_integration/"
        "csr_blocked_gap_training_request_index.jsonl"
    ),
    integration_summary_path: str | Path = (
        "runs/phase1_csr_blocked_gap_training_integration/"
        "csr_blocked_gap_training_integration_summary.json"
    ),
    positive_membership_path: str | Path = (
        "runs/phase1_csr_blocked_gap_training_integration/"
        "csr_blocked_gap_training_positive_membership.jsonl"
    ),
    output_dir: str | Path = "runs/phase1_csr_blocked_gap_augmented_ranker",
    epochs: int = 160,
    learning_rate: float = 0.03,
    l2_regularization: float = 1.0e-4,
    d_model: int = 24,
    num_attention_heads: int = 4,
    feedforward_dim: int = 48,
    seed: int = 20,
) -> dict[str, Any]:
    """Train a shadow ranker and summarize blocked-gap positive behavior."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    export = train_csr_transformer_ranker_from_tensor_file(
        tensors_path,
        request_index_path,
        model_id="csr_blocked_gap_augmented_shadow_ranker_v1",
        epochs=epochs,
        learning_rate=learning_rate,
        l2_regularization=l2_regularization,
        d_model=d_model,
        num_attention_heads=num_attention_heads,
        feedforward_dim=feedforward_dim,
        seed=seed,
    )
    integration_summary = _read_json(Path(integration_summary_path))
    positive_membership = tuple(read_jsonl(positive_membership_path))
    request_index = tuple(read_jsonl(request_index_path))
    positive_rows = _positive_prediction_rows(
        export.predictions,
        positive_membership=positive_membership,
        integration_summary=integration_summary,
    )
    summary = _summary(
        export=export,
        integration_summary=integration_summary,
        positive_membership=positive_membership,
        request_index=request_index,
        positive_rows=positive_rows,
        tensors_path=Path(tensors_path),
        request_index_path=Path(request_index_path),
        integration_summary_path=Path(integration_summary_path),
        positive_membership_path=Path(positive_membership_path),
    )
    schema = _schema(summary)
    paths = {
        "model": output / "csr_blocked_gap_augmented_ranker_model.json",
        "predictions": output / "csr_blocked_gap_augmented_ranker_predictions.jsonl",
        "summary": output / "csr_blocked_gap_augmented_ranker_summary.json",
        "schema": output / "csr_blocked_gap_augmented_ranker_schema.json",
        "report": output / "csr_blocked_gap_augmented_ranker_report.md",
        "positive_predictions": (
            output / "csr_blocked_gap_augmented_ranker_positive_predictions.jsonl"
        ),
    }
    write_csr_transformer_ranker_model(export.model, paths["model"])
    write_csr_transformer_ranker_predictions(export.predictions, paths["predictions"])
    write_csr_transformer_ranker_summary(export.summary, paths["summary"])
    write_csr_transformer_ranker_schema(export.schema, paths["schema"])
    write_csr_transformer_ranker_report(export, paths["report"])
    write_jsonl(positive_rows, paths["positive_predictions"])
    _write_json(summary, paths["summary"])
    _write_json(schema, paths["schema"])
    paths["report"].write_text(_report(summary), encoding="utf-8")
    return {
        "export": export,
        "summary": summary,
        "schema": schema,
        "positive_predictions": positive_rows,
        "paths": {key: str(value) for key, value in paths.items()},
    }


def _positive_prediction_rows(
    predictions: tuple[CsrTransformerRankerPrediction, ...],
    *,
    positive_membership: tuple[dict[str, Any], ...],
    integration_summary: dict[str, Any],
) -> tuple[dict[str, Any], ...]:
    positive_candidate_ids = set(map(str, integration_summary["positive_candidate_ids"]))
    new_candidate_ids = set(map(str, integration_summary["new_global_candidate_ids"]))
    positive_keys = {
        (str(row["matrix_id"]), str(row["context_id"]))
        for row in positive_membership
    }
    rows: list[dict[str, Any]] = []
    for prediction in predictions:
        if (prediction.matrix_id, prediction.context_id) not in positive_keys:
            continue
        ranked_positive = tuple(
            candidate_id
            for candidate_id in prediction.ranked_candidate_ids
            if candidate_id in positive_candidate_ids
        )
        ranked_new = tuple(
            candidate_id
            for candidate_id in prediction.ranked_candidate_ids
            if candidate_id in new_candidate_ids
        )
        rows.append(
            {
                **asdict(prediction),
                "schema_version": CSR_BLOCKED_GAP_AUGMENTED_RANKER_SCHEMA_VERSION,
                "source_context_id": POSITIVE_CONTEXT_ID,
                "selected_is_positive_candidate": (
                    prediction.selected_candidate_id in positive_candidate_ids
                ),
                "selected_is_new_candidate": (
                    prediction.selected_candidate_id in new_candidate_ids
                ),
                "selected_is_oracle": (
                    prediction.selected_candidate_id == prediction.oracle_candidate_id
                ),
                "selected_is_profiled_success": (
                    prediction.selected_target_status == "success"
                ),
                "ranked_positive_candidate_ids": ranked_positive,
                "ranked_new_candidate_ids": ranked_new,
                "positive_candidate_count": len(ranked_positive),
                "new_candidate_count": len(ranked_new),
            }
        )
    return tuple(sorted(rows, key=lambda row: str(row["request_id"])))


def _summary(
    *,
    export: CsrTransformerRankerExport,
    integration_summary: dict[str, Any],
    positive_membership: tuple[dict[str, Any], ...],
    request_index: tuple[dict[str, Any], ...],
    positive_rows: tuple[dict[str, Any], ...],
    tensors_path: Path,
    request_index_path: Path,
    integration_summary_path: Path,
    positive_membership_path: Path,
) -> dict[str, Any]:
    ranker = export.summary
    positive_candidate_ids = tuple(map(str, integration_summary["positive_candidate_ids"]))
    new_candidate_ids = tuple(map(str, integration_summary["new_global_candidate_ids"]))
    expected_matrix_contexts = {
        (str(row["matrix_id"]), str(row["context_id"]))
        for row in positive_membership
    }
    positive_index_rows = tuple(
        row
        for row in request_index
        if (str(row["matrix_id"]), str(row["context_id"])) in expected_matrix_contexts
    )
    selected_status_counts = _counts(row["selected_target_status"] for row in positive_rows)
    evaluation_status_counts = _counts(row["evaluation_status"] for row in positive_rows)
    split_counts = _counts(row["split"] for row in positive_rows)
    selected_candidate_ids = tuple(
        sorted({str(row["selected_candidate_id"]) for row in positive_rows})
    )
    ranked_positive_candidate_ids = tuple(
        sorted(
            {
                str(candidate_id)
                for row in positive_rows
                for candidate_id in row["ranked_positive_candidate_ids"]
            }
        )
    )
    ranked_new_candidate_ids = tuple(
        sorted(
            {
                str(candidate_id)
                for row in positive_rows
                for candidate_id in row["ranked_new_candidate_ids"]
            }
        )
    )
    positive_success_predictions = sum(
        1 for row in positive_rows if row["selected_target_status"] == "success"
    )
    positive_oracle_matches = sum(
        1 for row in positive_rows if row["evaluation_status"] == "oracle_match"
    )
    positive_non_success = len(positive_rows) - positive_success_predictions
    positive_eval_rows = tuple(row for row in positive_rows if row["split"] == "eval")
    positive_eval_success = sum(
        1 for row in positive_eval_rows if row["selected_target_status"] == "success"
    )
    coverage_complete = (
        set(ranked_positive_candidate_ids) == set(positive_candidate_ids)
        and set(ranked_new_candidate_ids) == set(new_candidate_ids)
    )
    status = (
        "passed"
        if ranker.status == "passed"
        and integration_summary.get("status") == "passed"
        and integration_summary.get("runtime_selector_changed") is False
        and len(positive_rows) == len(expected_matrix_contexts) == 5
        and len(positive_index_rows) == len(positive_rows)
        and positive_success_predictions == len(positive_rows)
        and positive_non_success == 0
        and positive_oracle_matches >= 4
        and coverage_complete
        else "failed"
    )
    return {
        "status": status,
        "schema_version": CSR_BLOCKED_GAP_AUGMENTED_RANKER_SCHEMA_VERSION,
        "runtime_selector_changed": False,
        "shadow_only": True,
        "model_trained": ranker.model_trained,
        "model_id": ranker.model_id,
        "model_family": ranker.model_family,
        "source_tensors_path": str(tensors_path),
        "source_request_index_path": str(request_index_path),
        "source_integration_summary_path": str(integration_summary_path),
        "source_positive_membership_path": str(positive_membership_path),
        "source_integration_status": integration_summary.get("status"),
        "source_positive_selector_rows": int(integration_summary["positive_selector_rows"]),
        "source_positive_success_rows": int(integration_summary["positive_success_rows"]),
        "source_positive_oracle_rows": int(integration_summary["positive_oracle_rows"]),
        "source_positive_matrix_contexts": int(
            integration_summary["positive_matrix_contexts"]
        ),
        "source_positive_candidate_ids": positive_candidate_ids,
        "source_new_global_candidate_ids": new_candidate_ids,
        "source_existing_global_candidate_ids": tuple(
            map(str, integration_summary["existing_global_candidate_ids"])
        ),
        "ranker_num_requests": ranker.num_requests,
        "ranker_num_predictions": ranker.num_predictions,
        "ranker_num_train_requests": ranker.num_train_requests,
        "ranker_num_eval_requests": ranker.num_eval_requests,
        "ranker_num_train_oracle_requests": ranker.num_train_oracle_requests,
        "ranker_num_eval_oracle_requests": ranker.num_eval_oracle_requests,
        "ranker_num_global_candidates": ranker.num_global_candidates,
        "ranker_token_feature_dim": ranker.token_feature_dim,
        "ranker_scorer_feature_dim": ranker.scorer_feature_dim,
        "ranker_num_epochs": ranker.num_epochs,
        "ranker_num_pairwise_constraints": ranker.num_pairwise_constraints,
        "ranker_num_pairwise_updates": ranker.num_pairwise_updates,
        "ranker_final_train_pairwise_loss": ranker.final_train_pairwise_loss,
        "ranker_train_oracle_top1_accuracy": ranker.train_oracle_top1_accuracy,
        "ranker_eval_oracle_top1_accuracy": ranker.eval_oracle_top1_accuracy,
        "ranker_eval_profiled_success_selection_rate": (
            ranker.eval_profiled_success_selection_rate
        ),
        "ranker_eval_non_success_selection_count": (
            ranker.eval_non_success_selection_count
        ),
        "blocked_gap_prediction_rows": len(positive_rows),
        "blocked_gap_train_predictions": split_counts.get("train", 0),
        "blocked_gap_eval_predictions": split_counts.get("eval", 0),
        "blocked_gap_success_predictions": positive_success_predictions,
        "blocked_gap_non_success_predictions": positive_non_success,
        "blocked_gap_oracle_matches": positive_oracle_matches,
        "blocked_gap_profiled_success_non_oracle": evaluation_status_counts.get(
            "profiled_success_non_oracle",
            0,
        ),
        "blocked_gap_selected_status_counts": selected_status_counts,
        "blocked_gap_evaluation_status_counts": evaluation_status_counts,
        "blocked_gap_split_counts": split_counts,
        "blocked_gap_selected_candidate_ids": selected_candidate_ids,
        "blocked_gap_ranked_positive_candidate_ids": ranked_positive_candidate_ids,
        "blocked_gap_ranked_new_candidate_ids": ranked_new_candidate_ids,
        "blocked_gap_positive_candidate_coverage_complete": coverage_complete,
        "blocked_gap_eval_success_selection_rate": (
            positive_eval_success / len(positive_eval_rows)
            if positive_eval_rows
            else 0.0
        ),
        "overall_eval_is_not_acceptance_gate": True,
        "acceptance_note": (
            "This artifact gates only blocked-gap training-shard behavior; "
            "runtime promotion remains behind learned guard acceptance."
        ),
        "next_step": (
            "compare augmented ranker replay against guarded fallback policy and "
            "prepare a transformer training handoff that treats this model as a "
            "shadow baseline, not a production policy"
        ),
    }


def _schema(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": CSR_BLOCKED_GAP_AUGMENTED_RANKER_SCHEMA_VERSION,
        "task": "train_shadow_ranker_on_blocked_gap_augmented_tensor_bundle",
        "runtime_boundary": {
            "runtime_selector_changed": False,
            "shadow_only": True,
            "model_required_for_runtime": False,
        },
        "input_contract": {
            "tensors": summary["source_tensors_path"],
            "request_index": summary["source_request_index_path"],
            "positive_membership": summary["source_positive_membership_path"],
        },
        "output_contract": {
            "model": "csr_blocked_gap_augmented_ranker_model.json",
            "predictions": "csr_blocked_gap_augmented_ranker_predictions.jsonl",
            "summary": "csr_blocked_gap_augmented_ranker_summary.json",
            "positive_predictions": (
                "csr_blocked_gap_augmented_ranker_positive_predictions.jsonl"
            ),
        },
        "acceptance_gate": {
            "requires_ranker_status_passed": True,
            "requires_positive_context_prediction_count": 5,
            "requires_positive_selected_success_only": True,
            "requires_positive_candidate_coverage_complete": True,
            "requires_no_runtime_selector_change": True,
        },
    }


def _report(summary: dict[str, Any]) -> str:
    lines = [
        "# CSR Blocked Gap Augmented Ranker",
        "",
        f"- status: `{summary['status']}`",
        f"- shadow_only: `{summary['shadow_only']}`",
        f"- runtime_selector_changed: `{summary['runtime_selector_changed']}`",
        f"- model_id: `{summary['model_id']}`",
        f"- ranker_requests: `{summary['ranker_num_requests']}`",
        f"- ranker_global_candidates: `{summary['ranker_num_global_candidates']}`",
        f"- blocked_gap_prediction_rows: `{summary['blocked_gap_prediction_rows']}`",
        f"- blocked_gap_success_predictions: `{summary['blocked_gap_success_predictions']}`",
        f"- blocked_gap_oracle_matches: `{summary['blocked_gap_oracle_matches']}`",
        f"- blocked_gap_eval_success_selection_rate: `{summary['blocked_gap_eval_success_selection_rate']:.6g}`",
        f"- blocked_gap_positive_candidate_coverage_complete: `{summary['blocked_gap_positive_candidate_coverage_complete']}`",
        f"- ranker_eval_profiled_success_selection_rate: `{summary['ranker_eval_profiled_success_selection_rate']:.6g}`",
        f"- ranker_eval_non_success_selection_count: `{summary['ranker_eval_non_success_selection_count']}`",
        f"- acceptance_note: `{summary['acceptance_note']}`",
        f"- next_step: `{summary['next_step']}`",
        "",
    ]
    return "\n".join(lines)


def _counts(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(data: dict[str, Any], path: Path) -> Path:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
