"""Transformer training handoff bundle for CSR selector policy work."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from transsolvestack.profiling.artifacts import read_jsonl, write_jsonl


CSR_TRANSFORMER_HANDOFF_SCHEMA_VERSION = "phase1_csr_transformer_handoff_bundle_v1"


def build_csr_transformer_handoff_bundle_from_files(
    *,
    training_dir: str | Path = "runs/phase1_csr_blocked_gap_training_integration",
    augmented_ranker_dir: str | Path = "runs/phase1_csr_blocked_gap_augmented_ranker",
    guarded_replay_dir: str | Path = "runs/phase1_csr_blocked_gap_guarded_replay",
    submission_contract_dir: str | Path = "runs/phase1_csr_policy_model_submission",
    output_dir: str | Path = "runs/phase1_csr_transformer_handoff_bundle",
    model_family: str = "csr_transformer_policy_v2",
) -> dict[str, Any]:
    """Build a Transformer handoff bundle from augmented CSR training evidence."""

    training = Path(training_dir)
    augmented = Path(augmented_ranker_dir)
    guarded = Path(guarded_replay_dir)
    submission = Path(submission_contract_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    source_paths = _source_paths(
        training=training,
        augmented=augmented,
        guarded=guarded,
        submission=submission,
    )
    training_summary = _read_json(source_paths["training_integration_summary"])
    tensor_summary = _read_json(source_paths["training_tensor_summary"])
    ranker_summary = _read_json(source_paths["augmented_ranker_summary"])
    guarded_summary = _read_json(source_paths["guarded_replay_summary"])
    submission_summary = _read_json(source_paths["model_submission_summary"])
    request_index = read_jsonl(source_paths["request_index"])
    model_requests = read_jsonl(source_paths["model_requests"])
    model_targets = read_jsonl(source_paths["model_targets"])
    guarded_rows = read_jsonl(source_paths["guarded_replay_rows"])
    file_rows = tuple(
        _file_row(label=label, path=path, kind=_source_kind(label))
        for label, path in source_paths.items()
    )
    validation_errors = _validation_errors(
        file_rows=file_rows,
        training_summary=training_summary,
        tensor_summary=tensor_summary,
        ranker_summary=ranker_summary,
        guarded_summary=guarded_summary,
        submission_summary=submission_summary,
        request_index=request_index,
        model_requests=model_requests,
        model_targets=model_targets,
        guarded_rows=guarded_rows,
    )
    summary = _summary(
        validation_errors=validation_errors,
        training_summary=training_summary,
        tensor_summary=tensor_summary,
        ranker_summary=ranker_summary,
        guarded_summary=guarded_summary,
        submission_summary=submission_summary,
        file_rows=file_rows,
        request_index=request_index,
        model_family=model_family,
    )
    rows = _rows(file_rows=file_rows, summary=summary)
    training_spec = _training_spec(summary=summary, source_paths=source_paths)
    guard_contract = _guard_contract(
        summary=summary,
        ranker_summary=ranker_summary,
        guarded_summary=guarded_summary,
    )
    schema = _schema(summary)
    output_paths = {
        "rows": output / "csr_transformer_handoff_bundle_rows.jsonl",
        "summary": output / "csr_transformer_handoff_bundle_summary.json",
        "schema": output / "csr_transformer_handoff_bundle_schema.json",
        "training_spec": output / "csr_transformer_handoff_training_spec.json",
        "guard_contract": output / "csr_transformer_handoff_guard_contract.json",
        "report": output / "csr_transformer_handoff_bundle_report.md",
    }
    write_jsonl(rows, output_paths["rows"])
    _write_json(summary, output_paths["summary"])
    _write_json(schema, output_paths["schema"])
    _write_json(training_spec, output_paths["training_spec"])
    _write_json(guard_contract, output_paths["guard_contract"])
    output_paths["report"].write_text(
        _report(summary=summary, rows=rows),
        encoding="utf-8",
    )
    return {
        "summary": summary,
        "schema": schema,
        "rows": rows,
        "training_spec": training_spec,
        "guard_contract": guard_contract,
        "paths": {key: str(value) for key, value in output_paths.items()},
    }


def _source_paths(
    *,
    training: Path,
    augmented: Path,
    guarded: Path,
    submission: Path,
) -> dict[str, Path]:
    return {
        "training_integration_summary": (
            training / "csr_blocked_gap_training_integration_summary.json"
        ),
        "training_tensor_summary": training / "csr_blocked_gap_training_tensor_summary.json",
        "training_tensors": training / "csr_blocked_gap_training_tensors.json",
        "request_index": training / "csr_blocked_gap_training_request_index.jsonl",
        "model_requests": training / "csr_blocked_gap_training_model_requests.jsonl",
        "model_targets": training / "csr_blocked_gap_training_model_targets.jsonl",
        "selector_rows": training / "combined_csr_selector_rows.jsonl",
        "positive_membership": training / "csr_blocked_gap_training_positive_membership.jsonl",
        "training_manifest": training / "artifact_manifest.json",
        "augmented_ranker_summary": augmented / "csr_blocked_gap_augmented_ranker_summary.json",
        "augmented_ranker_model": augmented / "csr_blocked_gap_augmented_ranker_model.json",
        "augmented_ranker_predictions": (
            augmented / "csr_blocked_gap_augmented_ranker_predictions.jsonl"
        ),
        "augmented_ranker_positive_predictions": (
            augmented / "csr_blocked_gap_augmented_ranker_positive_predictions.jsonl"
        ),
        "augmented_ranker_manifest": augmented / "artifact_manifest.json",
        "guarded_replay_summary": guarded / "csr_blocked_gap_guarded_replay_summary.json",
        "guarded_replay_rows": guarded / "csr_blocked_gap_guarded_replay_rows.jsonl",
        "guarded_replay_blocked_quality_gate": (
            guarded / "csr_blocked_gap_guarded_replay_blocked_quality_gate.json"
        ),
        "guarded_replay_counterfactual_quality_gate": (
            guarded / "csr_blocked_gap_guarded_replay_counterfactual_quality_gate.json"
        ),
        "guarded_replay_manifest": guarded / "artifact_manifest.json",
        "model_submission_summary": submission / "csr_policy_model_submission_summary.json",
        "model_submission_schema": submission / "csr_policy_model_submission_schema.json",
        "model_submission_manifest": submission / "csr_policy_model_submission_manifest.json",
        "model_contribution_terms": Path("MODEL_CONTRIBUTION_TERMS.md"),
        "contributor_license_agreement": Path("CONTRIBUTOR_LICENSE_AGREEMENT.md"),
    }


def _source_kind(label: str) -> str:
    if label in {
        "training_tensors",
        "request_index",
        "model_requests",
        "model_targets",
        "selector_rows",
    }:
        return "training_input"
    if label.startswith("augmented_ranker"):
        return "shadow_baseline"
    if label.startswith("guarded_replay"):
        return "runtime_safety"
    if label.startswith("model_submission") or label in {
        "model_contribution_terms",
        "contributor_license_agreement",
    }:
        return "contribution_contract"
    return "source_metadata"


def _file_row(*, label: str, path: Path, kind: str) -> dict[str, Any]:
    exists = path.exists()
    data = path.read_bytes() if exists else b""
    return {
        "schema_version": CSR_TRANSFORMER_HANDOFF_SCHEMA_VERSION,
        "row_kind": "source_file",
        "source_kind": kind,
        "label": label,
        "path": str(path),
        "required": True,
        "exists": exists,
        "size_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest() if exists else None,
    }


def _validation_errors(
    *,
    file_rows: tuple[dict[str, Any], ...],
    training_summary: dict[str, Any],
    tensor_summary: dict[str, Any],
    ranker_summary: dict[str, Any],
    guarded_summary: dict[str, Any],
    submission_summary: dict[str, Any],
    request_index: tuple[dict[str, Any], ...],
    model_requests: tuple[dict[str, Any], ...],
    model_targets: tuple[dict[str, Any], ...],
    guarded_rows: tuple[dict[str, Any], ...],
) -> tuple[str, ...]:
    errors: list[str] = []
    if any(row["exists"] is not True for row in file_rows):
        errors.append("missing_handoff_source_file")
    if len({row["label"] for row in file_rows}) != len(file_rows):
        errors.append("duplicate_handoff_source_label")
    if training_summary.get("status") != "passed":
        errors.append("training_integration_not_passed")
    if training_summary.get("runtime_selector_changed") is not False:
        errors.append("training_integration_changed_runtime_selector")
    if training_summary.get("generic_queue_merge") is not False:
        errors.append("training_integration_generic_queue_merge")
    if tensor_summary.get("status") != "passed":
        errors.append("training_tensor_summary_not_passed")
    if tensor_summary.get("runtime_selector_changed") is not False:
        errors.append("training_tensor_changed_runtime_selector")
    if ranker_summary.get("status") != "passed":
        errors.append("augmented_ranker_not_passed")
    if ranker_summary.get("runtime_selector_changed") is not False:
        errors.append("augmented_ranker_changed_runtime_selector")
    if ranker_summary.get("shadow_only") is not True:
        errors.append("augmented_ranker_not_shadow_only")
    if guarded_summary.get("status") != "passed":
        errors.append("guarded_replay_not_passed")
    if guarded_summary.get("runtime_selector_changed") is not False:
        errors.append("guarded_replay_changed_runtime_selector")
    if guarded_summary.get("executes_gpu") is not False:
        errors.append("guarded_replay_executed_gpu")
    if submission_summary.get("status") != "passed":
        errors.append("submission_contract_not_passed")
    if submission_summary.get("runtime_selector_changed") is not False:
        errors.append("submission_contract_changed_runtime_selector")
    if len(request_index) != int(training_summary.get("integrated_tensor_requests", -1)):
        errors.append("request_index_count_mismatch")
    if len(model_requests) != int(training_summary.get("integrated_model_requests", -1)):
        errors.append("model_request_count_mismatch")
    if len(model_targets) != len(model_requests):
        errors.append("model_target_count_mismatch")
    if len(guarded_rows) != int(guarded_summary.get("num_replay_rows", -1)):
        errors.append("guarded_replay_row_count_mismatch")
    if guarded_summary.get("actual_runtime_selector_changes") != 0:
        errors.append("guarded_replay_actual_runtime_change")
    if guarded_summary.get("actual_quality_gate_blocks") != 5:
        errors.append("guarded_replay_quality_gate_block_mismatch")
    if guarded_summary.get("fallback_chain_enforced_rows") != 5:
        errors.append("guarded_replay_fallback_chain_mismatch")
    return tuple(errors)


def _summary(
    *,
    validation_errors: tuple[str, ...],
    training_summary: dict[str, Any],
    tensor_summary: dict[str, Any],
    ranker_summary: dict[str, Any],
    guarded_summary: dict[str, Any],
    submission_summary: dict[str, Any],
    file_rows: tuple[dict[str, Any], ...],
    request_index: tuple[dict[str, Any], ...],
    model_family: str,
) -> dict[str, Any]:
    handoff_ready = not validation_errors
    train_rows = sum(1 for row in request_index if str(row.get("split")) == "train")
    eval_rows = sum(1 for row in request_index if str(row.get("split")) == "eval")
    return {
        "status": "passed" if handoff_ready else "failed",
        "schema_version": CSR_TRANSFORMER_HANDOFF_SCHEMA_VERSION,
        "handoff_ready": handoff_ready,
        "model_family": model_family,
        "model_training_required": True,
        "model_trained": False,
        "runtime_selector_changed": False,
        "executes_gpu": False,
        "shadow_only": True,
        "num_source_files": len(file_rows),
        "all_source_files_present": all(row["exists"] is True for row in file_rows),
        "source_file_total_size_bytes": sum(int(row["size_bytes"]) for row in file_rows),
        "training_selector_rows": int(training_summary["integrated_selector_rows"]),
        "training_model_requests": int(training_summary["integrated_model_requests"]),
        "training_tensor_requests": int(training_summary["integrated_tensor_requests"]),
        "training_train_requests": train_rows,
        "training_eval_requests": eval_rows,
        "training_global_candidates": int(training_summary["integrated_global_candidates"]),
        "training_success_rows": int(training_summary["integrated_success_rows"]),
        "training_oracle_rows": int(training_summary["integrated_oracle_rows"]),
        "positive_selector_rows": int(training_summary["positive_selector_rows"]),
        "positive_success_rows": int(training_summary["positive_success_rows"]),
        "positive_oracle_rows": int(training_summary["positive_oracle_rows"]),
        "new_global_candidate_ids": tuple(training_summary["new_global_candidate_ids"]),
        "matrix_feature_dim": int(tensor_summary["matrix_feature_dim"]),
        "candidate_feature_dim": int(tensor_summary["candidate_feature_dim"]),
        "ranker_model_id": ranker_summary["model_id"],
        "ranker_num_requests": int(ranker_summary["ranker_num_requests"]),
        "ranker_eval_oracle_top1_accuracy": ranker_summary[
            "ranker_eval_oracle_top1_accuracy"
        ],
        "ranker_eval_profiled_success_selection_rate": ranker_summary[
            "ranker_eval_profiled_success_selection_rate"
        ],
        "ranker_eval_non_success_selection_count": int(
            ranker_summary["ranker_eval_non_success_selection_count"]
        ),
        "blocked_gap_success_predictions": int(
            ranker_summary["blocked_gap_success_predictions"]
        ),
        "blocked_gap_non_success_predictions": int(
            ranker_summary["blocked_gap_non_success_predictions"]
        ),
        "blocked_gap_oracle_matches": int(ranker_summary["blocked_gap_oracle_matches"]),
        "blocked_gap_positive_candidate_coverage_complete": bool(
            ranker_summary["blocked_gap_positive_candidate_coverage_complete"]
        ),
        "guard_actual_quality_gate_blocks": int(
            guarded_summary["actual_quality_gate_blocks"]
        ),
        "guard_actual_runtime_selector_changes": int(
            guarded_summary["actual_runtime_selector_changes"]
        ),
        "guard_fallback_chain_enforced_rows": int(
            guarded_summary["fallback_chain_enforced_rows"]
        ),
        "guard_max_learned_regret_vs_artifact_ms": guarded_summary[
            "max_learned_regret_vs_artifact_ms"
        ],
        "counterfactual_only": bool(guarded_summary["counterfactual_only"]),
        "submission_contract_ready": bool(submission_summary["submission_ready"]),
        "submission_shadow_ready": bool(submission_summary["shadow_submission_ready"]),
        "submission_runtime_promotion_ready": bool(
            submission_summary["runtime_promotion_ready"]
        ),
        "terms_documents_required": tuple(submission_summary["required_terms_documents"]),
        "validation_error_count": len(validation_errors),
        "validation_errors": validation_errors,
        "next_step": (
            "run external large-scale Transformer training against this handoff "
            "bundle, then submit the resulting model through artifact, replay, "
            "quality-gate, acceptance, and guarded runtime checks"
        ),
    }


def _rows(
    *,
    file_rows: tuple[dict[str, Any], ...],
    summary: dict[str, Any],
) -> tuple[dict[str, Any], ...]:
    return (
        {
            "schema_version": summary["schema_version"],
            "row_kind": "handoff_summary",
            "handoff_ready": summary["handoff_ready"],
            "model_training_required": summary["model_training_required"],
            "model_trained": summary["model_trained"],
            "runtime_selector_changed": summary["runtime_selector_changed"],
        },
        {
            "schema_version": summary["schema_version"],
            "row_kind": "training_tensor_contract",
            "training_model_requests": summary["training_model_requests"],
            "training_tensor_requests": summary["training_tensor_requests"],
            "training_global_candidates": summary["training_global_candidates"],
            "matrix_feature_dim": summary["matrix_feature_dim"],
            "candidate_feature_dim": summary["candidate_feature_dim"],
        },
        {
            "schema_version": summary["schema_version"],
            "row_kind": "shadow_baseline_contract",
            "ranker_model_id": summary["ranker_model_id"],
            "ranker_eval_non_success_selection_count": summary[
                "ranker_eval_non_success_selection_count"
            ],
            "blocked_gap_success_predictions": summary[
                "blocked_gap_success_predictions"
            ],
        },
        {
            "schema_version": summary["schema_version"],
            "row_kind": "guard_contract",
            "actual_quality_gate_blocks": summary["guard_actual_quality_gate_blocks"],
            "actual_runtime_selector_changes": summary[
                "guard_actual_runtime_selector_changes"
            ],
            "fallback_chain_enforced_rows": summary[
                "guard_fallback_chain_enforced_rows"
            ],
        },
        {
            "schema_version": summary["schema_version"],
            "row_kind": "submission_contract",
            "submission_shadow_ready": summary["submission_shadow_ready"],
            "submission_runtime_promotion_ready": summary[
                "submission_runtime_promotion_ready"
            ],
            "terms_documents_required": summary["terms_documents_required"],
        },
        *file_rows,
    )


def _training_spec(
    *,
    summary: dict[str, Any],
    source_paths: dict[str, Path],
) -> dict[str, Any]:
    return {
        "schema_version": CSR_TRANSFORMER_HANDOFF_SCHEMA_VERSION,
        "job_kind": "external_csr_transformer_policy_training",
        "model_family": summary["model_family"],
        "runtime_selector_changed": False,
        "model_trained_by_this_artifact": False,
        "inputs": {
            "tensor_arrays": str(source_paths["training_tensors"]),
            "request_index": str(source_paths["request_index"]),
            "label_free_model_requests": str(source_paths["model_requests"]),
            "offline_targets": str(source_paths["model_targets"]),
            "selector_rows": str(source_paths["selector_rows"]),
        },
        "baseline_context": {
            "shadow_ranker_summary": str(source_paths["augmented_ranker_summary"]),
            "shadow_ranker_predictions": str(source_paths["augmented_ranker_predictions"]),
            "guarded_replay_summary": str(source_paths["guarded_replay_summary"]),
            "model_submission_contract": str(source_paths["model_submission_schema"]),
        },
        "expected_outputs": {
            "saved_model": "future_csr_transformer_policy_model.json",
            "predictions": "future_csr_transformer_policy_predictions.jsonl",
            "replay_summary": "future_csr_transformer_policy_replay_summary.json",
            "quality_gate_summary": "future_csr_transformer_quality_gate_summary.json",
            "policy_model_artifact": "future_csr_policy_model_artifact.json",
            "submission_manifest": "future_csr_policy_model_submission_manifest.json",
        },
        "required_post_training_checks": (
            "saved_model_schema_validation",
            "exact_saved_model_replay",
            "selector_model_quality_gate",
            "policy_model_artifact_contract",
            "policy_model_acceptance",
            "policy_model_submission_terms",
            "learned_guard_shadow_mode",
            "profiled_success_fallback_chain_enforcement",
        ),
        "non_goals": (
            "do_not_train_inside_handoff_artifact",
            "do_not_change_runtime_selector",
            "do_not_execute_gpu_benchmarks",
            "do_not_publish_internal_paper_or_milestone_notes",
        ),
    }


def _guard_contract(
    *,
    summary: dict[str, Any],
    ranker_summary: dict[str, Any],
    guarded_summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": CSR_TRANSFORMER_HANDOFF_SCHEMA_VERSION,
        "guard_required": True,
        "default_mode": "shadow",
        "runtime_selector_changed": False,
        "min_confidence": guarded_summary["min_confidence"],
        "current_shadow_baseline": {
            "model_id": ranker_summary["model_id"],
            "runtime_eligible": False,
            "eval_non_success_selection_count": ranker_summary[
                "ranker_eval_non_success_selection_count"
            ],
            "blocked_gap_success_predictions": ranker_summary[
                "blocked_gap_success_predictions"
            ],
        },
        "actual_guard_replay": {
            "quality_gate_blocks": summary["guard_actual_quality_gate_blocks"],
            "runtime_selector_changes": summary[
                "guard_actual_runtime_selector_changes"
            ],
            "fallback_chain_enforced_rows": summary[
                "guard_fallback_chain_enforced_rows"
            ],
        },
        "promotion_requires": (
            "quality_gate_runtime_eligible",
            "confidence_at_or_above_threshold",
            "exact_matrix_context_profiled_success_candidate",
            "fallback_chain_contains_only_profiled_success_candidates",
            "runtime_timeout_guard",
            "accepted_model_submission",
        ),
        "counterfactual_rows_are_runtime_evidence": False,
    }


def _schema(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": CSR_TRANSFORMER_HANDOFF_SCHEMA_VERSION,
        "task": "package_augmented_csr_transformer_training_handoff",
        "handoff_ready": summary["handoff_ready"],
        "runtime_boundary": {
            "runtime_selector_changed": False,
            "executes_gpu": False,
            "model_trained": False,
            "shadow_only": True,
        },
        "input_contract": {
            "tensor_schema": "phase1_csr_training_tensors_v1",
            "model_request_schema": "phase1_csr_model_contract_v1",
            "guard_replay_schema": "phase1_csr_blocked_gap_guarded_replay_v1",
            "model_submission_schema": "phase1_csr_policy_model_submission_v1",
        },
        "acceptance_gate": {
            "requires_training_integration_passed": True,
            "requires_augmented_ranker_passed": True,
            "requires_guarded_replay_passed": True,
            "requires_actual_runtime_selector_changes": 0,
            "requires_all_source_file_checksums": True,
        },
    }


def _report(*, summary: dict[str, Any], rows: tuple[dict[str, Any], ...]) -> str:
    lines = [
        "# CSR Transformer Handoff Bundle",
        "",
        f"- status: `{summary['status']}`",
        f"- handoff_ready: `{summary['handoff_ready']}`",
        f"- model_family: `{summary['model_family']}`",
        f"- model_training_required: `{summary['model_training_required']}`",
        f"- model_trained: `{summary['model_trained']}`",
        f"- runtime_selector_changed: `{summary['runtime_selector_changed']}`",
        f"- executes_gpu: `{summary['executes_gpu']}`",
        f"- source_files: `{summary['num_source_files']}`",
        f"- training_model_requests: `{summary['training_model_requests']}`",
        f"- training_global_candidates: `{summary['training_global_candidates']}`",
        f"- positive_success_rows: `{summary['positive_success_rows']}`",
        f"- ranker_model_id: `{summary['ranker_model_id']}`",
        f"- ranker_eval_non_success_selection_count: `{summary['ranker_eval_non_success_selection_count']}`",
        f"- guard_actual_quality_gate_blocks: `{summary['guard_actual_quality_gate_blocks']}`",
        f"- guard_actual_runtime_selector_changes: `{summary['guard_actual_runtime_selector_changes']}`",
        f"- submission_shadow_ready: `{summary['submission_shadow_ready']}`",
        f"- submission_runtime_promotion_ready: `{summary['submission_runtime_promotion_ready']}`",
        f"- validation_error_count: `{summary['validation_error_count']}`",
        f"- next_step: `{summary['next_step']}`",
        "",
        "| label | kind | exists | size_bytes |",
        "|---|---|---|---:|",
    ]
    for row in rows:
        if row["row_kind"] != "source_file":
            continue
        lines.append(
            "| "
            f"{row['label']} | "
            f"{row['source_kind']} | "
            f"{row['exists']} | "
            f"{row['size_bytes']} |"
        )
    lines.append("")
    return "\n".join(lines)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(data: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
