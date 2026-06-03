"""End-to-end intake runner for external CSR policy model checkpoints."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from transsolvestack.datasets.csr import csr_matrix_from_record
from transsolvestack.policies.csr_external_model_adapter import (
    adapt_csr_external_ranker_checkpoint_from_files,
    build_reference_csr_external_ranker_checkpoint_from_files,
    write_csr_external_model_adapter_model,
    write_csr_external_model_adapter_predictions,
    write_csr_external_model_adapter_ranker_summary,
    write_csr_external_model_adapter_report,
    write_csr_external_model_adapter_rows,
    write_csr_external_model_adapter_schema,
    write_csr_external_model_adapter_summary,
    write_csr_external_ranker_checkpoint,
)
from transsolvestack.policies.csr_learned_guard import plan_csr_with_learned_guard
from transsolvestack.policies.csr_policy_model_acceptance import (
    ACCEPTANCE_MATRIX_IDS,
    build_csr_policy_model_acceptance_from_files,
    write_csr_policy_model_acceptance_report,
    write_csr_policy_model_acceptance_rows,
    write_csr_policy_model_acceptance_schema,
    write_csr_policy_model_acceptance_summary,
)
from transsolvestack.policies.csr_policy_model_artifact import (
    build_csr_policy_model_artifact_from_files,
    write_csr_policy_model_artifact,
    write_csr_policy_model_artifact_report,
    write_csr_policy_model_artifact_rows,
    write_csr_policy_model_artifact_schema,
    write_csr_policy_model_artifact_summary,
)
from transsolvestack.policies.csr_policy_model_submission import (
    build_csr_policy_model_submission_from_files,
    write_csr_policy_model_submission_manifest,
    write_csr_policy_model_submission_model_card,
    write_csr_policy_model_submission_report,
    write_csr_policy_model_submission_rows,
    write_csr_policy_model_submission_schema,
    write_csr_policy_model_submission_summary,
)
from transsolvestack.policies.csr_selector_model_eval import (
    evaluate_csr_selector_models_from_files,
    write_csr_selector_model_eval_report,
    write_csr_selector_model_eval_rows,
    write_csr_selector_model_eval_schema,
    write_csr_selector_model_eval_summary,
)
from transsolvestack.policies.csr_transformer_model_replay import (
    build_csr_transformer_model_replay_from_files,
    write_csr_transformer_model_replay_comparison_rows,
    write_csr_transformer_model_replay_predictions,
    write_csr_transformer_model_replay_report,
    write_csr_transformer_model_replay_schema,
    write_csr_transformer_model_replay_summary,
)
from transsolvestack.profiling.artifacts import read_jsonl, write_jsonl


CSR_EXTERNAL_MODEL_INTAKE_SCHEMA_VERSION = "phase1_csr_external_model_intake_v1"


def run_csr_external_model_intake_from_files(
    checkpoint_path: str | Path | None = None,
    output_dir: str | Path = "runs/phase1_csr_external_model_intake",
    *,
    source_model_path: str | Path = (
        "runs/phase1_csr_transformer_ranker/csr_transformer_ranker_model.json"
    ),
    tensor_path: str | Path = (
        "runs/phase1_csr_transformer_ready/csr_transformer_training_tensors.json"
    ),
    request_index_path: str | Path = (
        "runs/phase1_csr_transformer_ready/csr_transformer_request_index.jsonl"
    ),
    baseline_summary_path: str | Path = (
        "runs/phase1_csr_transformer_ready/combined_csr_learning_summary.json"
    ),
    baseline_predictions_path: str | Path = (
        "runs/phase1_csr_transformer_ready/combined_csr_baseline_predictions.jsonl"
    ),
    csr_path: str | Path = "runs/phase1_suitesparse_csr_import/csr_matrices.jsonl",
    selector_path: str | Path = "runs/phase1_csr_selector_readiness/csr_selector_rows.jsonl",
    guarded_auto_solve_summary_path: str | Path = (
        "runs/phase1_csr_guarded_auto_solve/csr_guarded_auto_solve_summary.json"
    ),
    guarded_auto_solve_results_path: str | Path = (
        "runs/phase1_csr_guarded_auto_solve/csr_guarded_auto_solve_results.jsonl"
    ),
    contributor_id: str = "wei_cui_reference",
    contributor_name: str = "Wei CUI",
    contribution_name: str = "csr_external_ranker_reference_intake",
    contribution_version: str = "phase1-reference",
    min_eval_oracle_requests: int = 4,
    min_runtime_oracle_top1_accuracy: float = 0.5,
    min_runtime_profiled_success_rate: float = 0.8,
    min_confidence: float = 0.75,
) -> dict[str, Any]:
    """Run external checkpoint intake through all model-safety boundaries."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = _paths(output)
    active_checkpoint_path = _prepare_checkpoint(
        checkpoint_path=checkpoint_path,
        output_checkpoint_path=paths["checkpoint"],
        source_model_path=Path(source_model_path),
        tensor_path=Path(tensor_path),
        request_index_path=Path(request_index_path),
        contributor_id=contributor_id,
        contributor_name=contributor_name,
    )

    adapter = adapt_csr_external_ranker_checkpoint_from_files(
        active_checkpoint_path,
        tensor_path,
        request_index_path,
    )
    _write_adapter(adapter, paths)

    replay = build_csr_transformer_model_replay_from_files(
        paths["adapted_model"],
        tensor_path,
        request_index_path,
        paths["adapted_predictions"],
    )
    _write_replay(replay, paths)

    quality_gate = evaluate_csr_selector_models_from_files(
        baseline_summary_path,
        baseline_predictions_path,
        paths["adapted_ranker_summary"],
        paths["adapted_predictions"],
        min_eval_oracle_requests=min_eval_oracle_requests,
        min_runtime_oracle_top1_accuracy=min_runtime_oracle_top1_accuracy,
        min_runtime_profiled_success_rate=min_runtime_profiled_success_rate,
    )
    _write_quality_gate(quality_gate, paths)

    model_artifact = build_csr_policy_model_artifact_from_files(
        paths["adapted_model"],
        tensor_path,
        request_index_path,
        paths["quality_gate_summary"],
        paths["replay_summary"],
    )
    _write_policy_model_artifact(model_artifact, paths)

    learned_guard = _build_learned_guard_checks(
        model_artifact_path=paths["policy_model_artifact"],
        quality_gate_summary_path=paths["quality_gate_summary"],
        csr_path=Path(csr_path),
        selector_path=Path(selector_path),
        min_confidence=min_confidence,
    )
    _write_learned_guard_checks(learned_guard, paths)

    acceptance = build_csr_policy_model_acceptance_from_files(
        paths["policy_model_artifact"],
        csr_path,
        selector_path,
        paths["learned_guard_summary"],
        guarded_auto_solve_summary_path,
        guarded_auto_solve_results_path,
        min_confidence=min_confidence,
    )
    _write_acceptance(acceptance, paths)

    submission = build_csr_policy_model_submission_from_files(
        paths["policy_model_artifact"],
        paths["acceptance_summary"],
        paths["acceptance_rows"],
        contributor_id=contributor_id,
        contributor_name=contributor_name,
        contribution_name=contribution_name,
        contribution_version=contribution_version,
        contributor_terms_acknowledged=True,
    )
    _write_submission(submission, paths)

    summary = _summary(
        adapter=adapter["summary"],
        replay=asdict(replay.summary),
        quality_gate=asdict(quality_gate.summary),
        model_artifact=model_artifact["summary"],
        learned_guard=learned_guard["summary"],
        acceptance=acceptance["summary"],
        submission=submission["summary"],
        paths=paths,
    )
    schema = _schema(summary)
    rows = _rows(summary)
    report = _report(summary)
    _write_json(summary, paths["summary"])
    _write_json(schema, paths["schema"])
    write_jsonl(rows, paths["rows"])
    paths["report"].write_text(report, encoding="utf-8")
    return {
        "paths": {key: str(value) for key, value in paths.items()},
        "summary": summary,
        "schema": schema,
        "rows": rows,
        "report": report,
        "adapter": adapter,
        "acceptance": acceptance,
        "submission": submission,
    }


def _prepare_checkpoint(
    *,
    checkpoint_path: str | Path | None,
    output_checkpoint_path: Path,
    source_model_path: Path,
    tensor_path: Path,
    request_index_path: Path,
    contributor_id: str,
    contributor_name: str,
) -> Path:
    if checkpoint_path is None:
        checkpoint = build_reference_csr_external_ranker_checkpoint_from_files(
            source_model_path,
            tensor_path,
            request_index_path,
            contributor_id=contributor_id,
            contributor_name=contributor_name,
        )
    else:
        checkpoint = json.loads(Path(checkpoint_path).read_text(encoding="utf-8"))
    write_csr_external_ranker_checkpoint(checkpoint, output_checkpoint_path)
    return output_checkpoint_path


def _write_adapter(data: dict[str, Any], paths: dict[str, Path]) -> None:
    write_csr_external_model_adapter_model(data, paths["adapted_model"])
    write_csr_external_model_adapter_predictions(
        data["predictions"],
        paths["adapted_predictions"],
    )
    write_csr_external_model_adapter_ranker_summary(
        data,
        paths["adapted_ranker_summary"],
    )
    write_csr_external_model_adapter_rows(data["rows"], paths["adapter_rows"])
    write_csr_external_model_adapter_summary(data, paths["adapter_summary"])
    write_csr_external_model_adapter_schema(data, paths["adapter_schema"])
    write_csr_external_model_adapter_report(data, paths["adapter_report"])


def _write_replay(data: Any, paths: dict[str, Path]) -> None:
    write_csr_transformer_model_replay_predictions(
        data.predictions,
        paths["replay_predictions"],
    )
    write_csr_transformer_model_replay_comparison_rows(
        data.comparison_rows,
        paths["replay_comparison"],
    )
    write_csr_transformer_model_replay_summary(data.summary, paths["replay_summary"])
    write_csr_transformer_model_replay_schema(data.schema, paths["replay_schema"])
    write_csr_transformer_model_replay_report(data, paths["replay_report"])


def _write_quality_gate(data: Any, paths: dict[str, Path]) -> None:
    write_csr_selector_model_eval_rows(data.rows, paths["quality_gate_rows"])
    write_csr_selector_model_eval_summary(data.summary, paths["quality_gate_summary"])
    write_csr_selector_model_eval_schema(data.schema, paths["quality_gate_schema"])
    write_csr_selector_model_eval_report(data, paths["quality_gate_report"])


def _write_policy_model_artifact(data: dict[str, Any], paths: dict[str, Path]) -> None:
    write_csr_policy_model_artifact(data, paths["policy_model_artifact"])
    write_csr_policy_model_artifact_rows(data["rows"], paths["policy_model_rows"])
    write_csr_policy_model_artifact_summary(data, paths["policy_model_summary"])
    write_csr_policy_model_artifact_schema(data, paths["policy_model_schema"])
    write_csr_policy_model_artifact_report(data, paths["policy_model_report"])


def _write_acceptance(data: dict[str, Any], paths: dict[str, Path]) -> None:
    write_csr_policy_model_acceptance_rows(data["rows"], paths["acceptance_rows"])
    write_csr_policy_model_acceptance_summary(data, paths["acceptance_summary"])
    write_csr_policy_model_acceptance_schema(data, paths["acceptance_schema"])
    write_csr_policy_model_acceptance_report(data, paths["acceptance_report"])


def _write_submission(data: dict[str, Any], paths: dict[str, Path]) -> None:
    write_csr_policy_model_submission_manifest(data, paths["submission_manifest"])
    write_csr_policy_model_submission_rows(data["rows"], paths["submission_rows"])
    write_csr_policy_model_submission_summary(data, paths["submission_summary"])
    write_csr_policy_model_submission_schema(data, paths["submission_schema"])
    write_csr_policy_model_submission_model_card(data, paths["submission_model_card"])
    write_csr_policy_model_submission_report(data, paths["submission_report"])


def _build_learned_guard_checks(
    *,
    model_artifact_path: Path,
    quality_gate_summary_path: Path,
    csr_path: Path,
    selector_path: Path,
    min_confidence: float,
) -> dict[str, Any]:
    csr_rows = {str(row["matrix_id"]): row for row in read_jsonl(csr_path)}
    decisions = []
    for mode in ("shadow", "promote_if_safe"):
        matrix_id = ACCEPTANCE_MATRIX_IDS[-1]
        matrix = csr_matrix_from_record(csr_rows[matrix_id])
        decisions.append(
            plan_csr_with_learned_guard(
                matrix,
                selector_path=selector_path,
                learned_model_artifact_path=model_artifact_path,
                quality_gate_summary_path=quality_gate_summary_path,
                mode=mode,
                min_confidence=min_confidence,
            )
        )
    rows = tuple(
        {
            "schema_version": "phase1_csr_external_model_intake_guard_v1",
            "row_kind": "learned_guard_check",
            "mode": decision.mode,
            "guard_status": decision.guard_status,
            "runtime_selection_source": decision.runtime_selection_source,
            "runtime_selector_changed": decision.runtime_selector_changed,
            "fallback_chain_enforced": decision.fallback_chain_enforced,
            "learned_policy_source": asdict(decision.learned_policy_source),
            "guard_reasons": tuple(decision.guard_reasons),
        }
        for decision in decisions
    )
    shadow = rows[0]
    promote = rows[1]
    summary = {
        "status": "passed"
        if (
            shadow["guard_status"] == "shadow_only"
            and promote["guard_status"] == "blocked_quality_gate"
            and shadow["learned_policy_source"]["source_kind"] == "model_artifact"
            and promote["learned_policy_source"]["source_kind"] == "model_artifact"
            and shadow["learned_policy_source"]["model_loaded"] is True
            and promote["learned_policy_source"]["model_loaded"] is True
            and shadow["runtime_selector_changed"] is False
            and promote["runtime_selector_changed"] is False
        )
        else "failed",
        "schema_version": "phase1_csr_external_model_intake_guard_v1",
        "model_artifact_shadow_checked": True,
        "model_artifact_quality_gate_checked": True,
        "runtime_selector_changed": False,
        "num_checks": len(rows),
    }
    schema = {
        "schema_version": summary["schema_version"],
        "checks": ("shadow", "promote_if_safe_quality_gate_block"),
        "runtime_selector_changed": False,
    }
    report = "\n".join(
        [
            "# Intake Learned Guard Checks",
            "",
            f"- status: `{summary['status']}`",
            f"- checks: `{summary['num_checks']}`",
            "- source: `model_artifact`",
            "- runtime_selector_changed: `False`",
            "",
        ]
    )
    return {"summary": summary, "schema": schema, "rows": rows, "report": report}


def _write_learned_guard_checks(data: dict[str, Any], paths: dict[str, Path]) -> None:
    write_jsonl(data["rows"], paths["learned_guard_rows"])
    _write_json(data["summary"], paths["learned_guard_summary"])
    _write_json(data["schema"], paths["learned_guard_schema"])
    paths["learned_guard_report"].write_text(data["report"], encoding="utf-8")


def _summary(
    *,
    adapter: dict[str, Any],
    replay: dict[str, Any],
    quality_gate: dict[str, Any],
    model_artifact: dict[str, Any],
    learned_guard: dict[str, Any],
    acceptance: dict[str, Any],
    submission: dict[str, Any],
    paths: dict[str, Path],
) -> dict[str, Any]:
    stage_statuses = {
        "adapter": adapter["status"],
        "replay": replay["status"],
        "quality_gate": quality_gate["status"],
        "policy_model_artifact": model_artifact["status"],
        "learned_guard": learned_guard["status"],
        "acceptance": acceptance["status"],
        "submission": submission["status"],
    }
    intake_ready = all(status == "passed" for status in stage_statuses.values())
    shadow_ready = intake_ready and bool(submission["shadow_submission_ready"])
    runtime_promotion_ready = shadow_ready and bool(submission["runtime_promotion_ready"])
    return {
        "status": "passed" if intake_ready else "failed",
        "schema_version": CSR_EXTERNAL_MODEL_INTAKE_SCHEMA_VERSION,
        "intake_ready": intake_ready,
        "shadow_submission_ready": shadow_ready,
        "runtime_promotion_ready": runtime_promotion_ready,
        "default_runtime_mode": "shadow" if shadow_ready else "rejected",
        "stage_statuses": stage_statuses,
        "adapter_ready": adapter["adapter_ready"],
        "replay_exact": replay["exact_replay"],
        "quality_gate_runtime_eligible": quality_gate["challenger_runtime_eligible"],
        "policy_model_artifact_ready": model_artifact["artifact_ready"],
        "acceptance_status": acceptance["status"],
        "accepted_for_shadow": acceptance["accepted_for_shadow"],
        "accepted_for_runtime_promotion": acceptance["accepted_for_runtime_promotion"],
        "submission_ready": submission["submission_ready"],
        "guarded_gpu_shadow_smoke_checked": acceptance[
            "guarded_gpu_shadow_smoke_checked"
        ],
        "guarded_gpu_smoke_successes": acceptance["guarded_gpu_smoke_successes"],
        "promotion_blockers": acceptance["promotion_blockers"],
        "num_predictions": adapter["num_predictions"],
        "runtime_selector_changed": False,
        "artifact_paths": {
            key: str(value)
            for key, value in paths.items()
            if key
            in {
                "checkpoint",
                "adapted_model",
                "quality_gate_summary",
                "policy_model_artifact",
                "acceptance_summary",
                "submission_manifest",
            }
        },
    }


def _schema(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": CSR_EXTERNAL_MODEL_INTAKE_SCHEMA_VERSION,
        "task": "external_checkpoint_to_guarded_shadow_submission",
        "stages": tuple(summary["stage_statuses"]),
        "runtime_boundary": {
            "default_mode": summary["default_runtime_mode"],
            "guard_required": True,
            "runtime_selector_changed": False,
        },
    }


def _rows(summary: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    rows = [
        {
            "schema_version": summary["schema_version"],
            "row_kind": "stage_status",
            "stage": stage,
            "status": status,
        }
        for stage, status in summary["stage_statuses"].items()
    ]
    rows.append(
        {
            "schema_version": summary["schema_version"],
            "row_kind": "intake_decision",
            "status": summary["status"],
            "shadow_submission_ready": summary["shadow_submission_ready"],
            "runtime_promotion_ready": summary["runtime_promotion_ready"],
            "runtime_selector_changed": summary["runtime_selector_changed"],
        }
    )
    return tuple(rows)


def _report(summary: dict[str, Any]) -> str:
    stage_lines = [
        f"- {stage}: `{status}`" for stage, status in summary["stage_statuses"].items()
    ]
    blocker_lines = [
        f"- `{blocker}`" for blocker in summary["promotion_blockers"]
    ] or ["- none"]
    return "\n".join(
        [
            "# CSR External Model Intake Review",
            "",
            f"- status: `{summary['status']}`",
            f"- intake_ready: `{summary['intake_ready']}`",
            f"- shadow_submission_ready: `{summary['shadow_submission_ready']}`",
            f"- runtime_promotion_ready: `{summary['runtime_promotion_ready']}`",
            f"- default_runtime_mode: `{summary['default_runtime_mode']}`",
            f"- num_predictions: `{summary['num_predictions']}`",
            f"- guarded_gpu_shadow_smoke_checked: `{summary['guarded_gpu_shadow_smoke_checked']}`",
            f"- guarded_gpu_smoke_successes: `{summary['guarded_gpu_smoke_successes']}`",
            f"- runtime_selector_changed: `{summary['runtime_selector_changed']}`",
            "",
            "## Stages",
            "",
            *stage_lines,
            "",
            "## Promotion Blockers",
            "",
            *blocker_lines,
            "",
        ]
    )


def _paths(output: Path) -> dict[str, Path]:
    return {
        "checkpoint": output / "external_csr_ranker_checkpoint.json",
        "adapted_model": output / "adapted_csr_transformer_ranker_model.json",
        "adapted_predictions": output / "adapted_csr_transformer_ranker_predictions.jsonl",
        "adapted_ranker_summary": output / "adapted_csr_transformer_ranker_summary.json",
        "adapter_rows": output / "csr_external_model_adapter_rows.jsonl",
        "adapter_summary": output / "csr_external_model_adapter_summary.json",
        "adapter_schema": output / "csr_external_model_adapter_schema.json",
        "adapter_report": output / "csr_external_model_adapter_report.md",
        "replay_predictions": output / "csr_external_model_replay_predictions.jsonl",
        "replay_comparison": output / "csr_external_model_replay_comparison.jsonl",
        "replay_summary": output / "csr_external_model_replay_summary.json",
        "replay_schema": output / "csr_external_model_replay_schema.json",
        "replay_report": output / "csr_external_model_replay_report.md",
        "quality_gate_rows": output / "csr_external_model_quality_gate_rows.jsonl",
        "quality_gate_summary": output / "csr_external_model_quality_gate_summary.json",
        "quality_gate_schema": output / "csr_external_model_quality_gate_schema.json",
        "quality_gate_report": output / "csr_external_model_quality_gate_report.md",
        "policy_model_artifact": output / "csr_policy_model_artifact.json",
        "policy_model_rows": output / "csr_policy_model_artifact_rows.jsonl",
        "policy_model_summary": output / "csr_policy_model_artifact_summary.json",
        "policy_model_schema": output / "csr_policy_model_artifact_schema.json",
        "policy_model_report": output / "csr_policy_model_artifact_report.md",
        "learned_guard_rows": output / "csr_external_model_learned_guard_rows.jsonl",
        "learned_guard_summary": output / "csr_external_model_learned_guard_summary.json",
        "learned_guard_schema": output / "csr_external_model_learned_guard_schema.json",
        "learned_guard_report": output / "csr_external_model_learned_guard_report.md",
        "acceptance_rows": output / "csr_policy_model_acceptance_rows.jsonl",
        "acceptance_summary": output / "csr_policy_model_acceptance_summary.json",
        "acceptance_schema": output / "csr_policy_model_acceptance_schema.json",
        "acceptance_report": output / "csr_policy_model_acceptance_report.md",
        "submission_manifest": output / "csr_policy_model_submission_manifest.json",
        "submission_rows": output / "csr_policy_model_submission_rows.jsonl",
        "submission_summary": output / "csr_policy_model_submission_summary.json",
        "submission_schema": output / "csr_policy_model_submission_schema.json",
        "submission_model_card": output / "MODEL_CARD.md",
        "submission_report": output / "csr_policy_model_submission_report.md",
        "rows": output / "csr_external_model_intake_rows.jsonl",
        "summary": output / "csr_external_model_intake_summary.json",
        "schema": output / "csr_external_model_intake_schema.json",
        "report": output / "csr_external_model_intake_report.md",
    }


def _write_json(data: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
