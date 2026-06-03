"""Guarded replay for the blocked-gap augmented shadow ranker."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from transsolvestack.core.types import SolveContext
from transsolvestack.datasets.csr import CsrMatrix
from transsolvestack.policies.csr_learned_guard import CsrLearnedGuardDecision
from transsolvestack.policies.csr_learned_guard import (
    plan_csr_with_learned_guard as _plan_with_guard,
)
from transsolvestack.profiling.artifacts import read_jsonl, write_jsonl


CSR_BLOCKED_GAP_GUARDED_REPLAY_SCHEMA_VERSION = (
    "phase1_csr_blocked_gap_guarded_replay_v1"
)
POSITIVE_CONTEXT_ID = "phase1_csr_blocked_gap_positive_search"
MODEL_ID = "csr_blocked_gap_augmented_shadow_ranker_v1"


def build_csr_blocked_gap_guarded_replay_from_files(
    *,
    selector_rows_path: str | Path = (
        "runs/phase1_csr_blocked_gap_training_integration/"
        "combined_csr_selector_rows.jsonl"
    ),
    predictions_path: str | Path = (
        "runs/phase1_csr_blocked_gap_augmented_ranker/"
        "csr_blocked_gap_augmented_ranker_predictions.jsonl"
    ),
    positive_predictions_path: str | Path = (
        "runs/phase1_csr_blocked_gap_augmented_ranker/"
        "csr_blocked_gap_augmented_ranker_positive_predictions.jsonl"
    ),
    ranker_summary_path: str | Path = (
        "runs/phase1_csr_blocked_gap_augmented_ranker/"
        "csr_blocked_gap_augmented_ranker_summary.json"
    ),
    output_dir: str | Path = "runs/phase1_csr_blocked_gap_guarded_replay",
    min_confidence: float = 0.75,
) -> dict[str, Any]:
    """Replay D20 predictions through the existing learned runtime guard."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    ranker_summary = _read_json(Path(ranker_summary_path))
    positive_predictions = tuple(read_jsonl(positive_predictions_path))
    blocked_gate = _blocked_quality_gate(ranker_summary)
    eligible_gate = _counterfactual_eligible_gate(ranker_summary)
    paths = {
        "blocked_quality_gate": output / "csr_blocked_gap_guarded_replay_blocked_quality_gate.json",
        "counterfactual_quality_gate": (
            output / "csr_blocked_gap_guarded_replay_counterfactual_quality_gate.json"
        ),
        "rows": output / "csr_blocked_gap_guarded_replay_rows.jsonl",
        "summary": output / "csr_blocked_gap_guarded_replay_summary.json",
        "schema": output / "csr_blocked_gap_guarded_replay_schema.json",
        "report": output / "csr_blocked_gap_guarded_replay_report.md",
    }
    _write_json(blocked_gate, paths["blocked_quality_gate"])
    _write_json(eligible_gate, paths["counterfactual_quality_gate"])
    rows = tuple(
        _replay_row(
            positive_prediction,
            selector_rows_path=Path(selector_rows_path),
            predictions_path=Path(predictions_path),
            blocked_quality_gate_path=paths["blocked_quality_gate"],
            counterfactual_quality_gate_path=paths["counterfactual_quality_gate"],
            min_confidence=min_confidence,
        )
        for positive_prediction in positive_predictions
    )
    summary = _summary(
        rows,
        ranker_summary=ranker_summary,
        selector_rows_path=Path(selector_rows_path),
        predictions_path=Path(predictions_path),
        positive_predictions_path=Path(positive_predictions_path),
        ranker_summary_path=Path(ranker_summary_path),
        min_confidence=min_confidence,
    )
    schema = _schema(summary)
    write_jsonl(rows, paths["rows"])
    _write_json(summary, paths["summary"])
    _write_json(schema, paths["schema"])
    paths["report"].write_text(_report(summary, rows), encoding="utf-8")
    return {
        "summary": summary,
        "schema": schema,
        "rows": rows,
        "quality_gates": {
            "blocked": blocked_gate,
            "counterfactual_eligible": eligible_gate,
        },
        "paths": {key: str(value) for key, value in paths.items()},
    }


def _replay_row(
    positive_prediction: dict[str, Any],
    *,
    selector_rows_path: Path,
    predictions_path: Path,
    blocked_quality_gate_path: Path,
    counterfactual_quality_gate_path: Path,
    min_confidence: float,
) -> dict[str, Any]:
    matrix_id = str(positive_prediction["matrix_id"])
    context_id = str(positive_prediction["context_id"])
    csr = _stub_csr(matrix_id)
    shadow = _plan_with_guard(
        csr,
        selector_path=selector_rows_path,
        learned_predictions_path=predictions_path,
        quality_gate_summary_path=blocked_quality_gate_path,
        context_id=context_id,
        mode="shadow",
        min_confidence=min_confidence,
    )
    blocked = _plan_with_guard(
        csr,
        selector_path=selector_rows_path,
        learned_predictions_path=predictions_path,
        quality_gate_summary_path=blocked_quality_gate_path,
        context_id=context_id,
        mode="promote_if_safe",
        min_confidence=min_confidence,
    )
    counterfactual = _plan_with_guard(
        csr,
        selector_path=selector_rows_path,
        learned_predictions_path=predictions_path,
        quality_gate_summary_path=counterfactual_quality_gate_path,
        context_id=context_id,
        mode="promote_if_safe",
        min_confidence=min_confidence,
    )
    learned_prediction = blocked.learned_prediction
    if learned_prediction is None:
        raise ValueError(f"missing D20 prediction for {matrix_id}:{context_id}")
    artifact_profile = blocked.selection.profile
    artifact_median_ms = (
        None if artifact_profile is None else artifact_profile.median_solve_time_ms
    )
    learned_median_ms = _optional_float(positive_prediction.get("regret_vs_oracle_ms"))
    if learned_median_ms is not None and artifact_median_ms is not None:
        learned_median_ms = artifact_median_ms + learned_median_ms
    return {
        "schema_version": CSR_BLOCKED_GAP_GUARDED_REPLAY_SCHEMA_VERSION,
        "request_id": str(positive_prediction["request_id"]),
        "matrix_id": matrix_id,
        "context_id": context_id,
        "split": str(positive_prediction["split"]),
        "learned_model_id": learned_prediction.model_id,
        "learned_candidate_id": learned_prediction.selected_candidate_id,
        "learned_selected_target_status": learned_prediction.selected_target_status,
        "learned_evaluation_status": learned_prediction.evaluation_status,
        "learned_is_oracle": bool(positive_prediction["selected_is_oracle"]),
        "learned_is_profiled_success": bool(
            positive_prediction["selected_is_profiled_success"]
        ),
        "learned_confidence": learned_prediction.confidence,
        "learned_score_margin": _finite_or_none(learned_prediction.score_margin),
        "learned_score_margin_is_infinite": (
            learned_prediction.score_margin is not None
            and math.isinf(float(learned_prediction.score_margin))
        ),
        "learned_regret_vs_artifact_ms": _optional_float(
            positive_prediction.get("regret_vs_oracle_ms")
        ),
        "learned_median_solve_time_ms": learned_median_ms,
        "artifact_candidate_id": blocked.artifact_candidate_id,
        "artifact_is_oracle": None if artifact_profile is None else artifact_profile.is_oracle,
        "artifact_median_solve_time_ms": artifact_median_ms,
        "learned_matches_artifact": (
            learned_prediction.selected_candidate_id == blocked.artifact_candidate_id
        ),
        "shadow_guard_status": shadow.guard_status,
        "shadow_runtime_candidate_id": shadow.runtime_candidate_id,
        "shadow_runtime_selector_changed": shadow.runtime_selector_changed,
        "blocked_guard_status": blocked.guard_status,
        "blocked_guard_reasons": tuple(blocked.guard_reasons),
        "blocked_runtime_candidate_id": blocked.runtime_candidate_id,
        "blocked_runtime_selection_source": blocked.runtime_selection_source,
        "blocked_runtime_selector_changed": blocked.runtime_selector_changed,
        "blocked_fallback_candidate_ids": blocked.fallback_candidate_ids,
        "blocked_fallback_chain_enforced": blocked.fallback_chain_enforced,
        "counterfactual_guard_status": counterfactual.guard_status,
        "counterfactual_runtime_candidate_id": counterfactual.runtime_candidate_id,
        "counterfactual_runtime_selection_source": (
            counterfactual.runtime_selection_source
        ),
        "counterfactual_runtime_selector_changed": (
            counterfactual.runtime_selector_changed
        ),
        "counterfactual_fallback_candidate_ids": (
            counterfactual.fallback_candidate_ids
        ),
        "counterfactual_fallback_chain_enforced": (
            counterfactual.fallback_chain_enforced
        ),
        "counterfactual_only": True,
    }


def _summary(
    rows: tuple[dict[str, Any], ...],
    *,
    ranker_summary: dict[str, Any],
    selector_rows_path: Path,
    predictions_path: Path,
    positive_predictions_path: Path,
    ranker_summary_path: Path,
    min_confidence: float,
) -> dict[str, Any]:
    actual_blocked = tuple(row for row in rows if row["blocked_guard_status"] == "blocked_quality_gate")
    shadow_only = tuple(row for row in rows if row["shadow_guard_status"] == "shadow_only")
    fallback_enforced = tuple(row for row in rows if row["blocked_fallback_chain_enforced"])
    learned_success = tuple(
        row for row in rows if row["learned_selected_target_status"] == "success"
    )
    learned_oracle = tuple(row for row in rows if row["learned_is_oracle"])
    learned_differs = tuple(row for row in rows if not row["learned_matches_artifact"])
    counterfactual_promotions = tuple(
        row for row in rows if row["counterfactual_guard_status"] == "promoted"
    )
    confidence_values = [float(row["learned_confidence"]) for row in rows]
    regrets = [
        float(row["learned_regret_vs_artifact_ms"])
        for row in rows
        if row["learned_regret_vs_artifact_ms"] is not None
    ]
    blocked_reason_counts = _reason_counts(rows)
    status = (
        "passed"
        if len(rows) == 5
        and len(actual_blocked) == len(rows)
        and len(shadow_only) == len(rows)
        and len(fallback_enforced) == len(rows)
        and len(learned_success) == len(rows)
        and len(learned_oracle) == 4
        and len(learned_differs) == 1
        and len(counterfactual_promotions) == len(rows)
        and all(row["blocked_runtime_selector_changed"] is False for row in rows)
        and all(row["blocked_runtime_selection_source"] == "artifact" for row in rows)
        and bool(ranker_summary.get("blocked_gap_positive_candidate_coverage_complete"))
        else "failed"
    )
    return {
        "status": status,
        "schema_version": CSR_BLOCKED_GAP_GUARDED_REPLAY_SCHEMA_VERSION,
        "runtime_selector_changed": False,
        "shadow_only": True,
        "executes_gpu": False,
        "source_selector_rows_path": str(selector_rows_path),
        "source_predictions_path": str(predictions_path),
        "source_positive_predictions_path": str(positive_predictions_path),
        "source_ranker_summary_path": str(ranker_summary_path),
        "source_ranker_status": ranker_summary.get("status"),
        "source_ranker_model_id": ranker_summary.get("model_id"),
        "source_ranker_eval_profiled_success_selection_rate": (
            ranker_summary.get("ranker_eval_profiled_success_selection_rate")
        ),
        "source_ranker_eval_non_success_selection_count": (
            ranker_summary.get("ranker_eval_non_success_selection_count")
        ),
        "min_confidence": min_confidence,
        "num_replay_rows": len(rows),
        "shadow_only_decisions": len(shadow_only),
        "actual_quality_gate_blocks": len(actual_blocked),
        "actual_runtime_selector_changes": sum(
            1 for row in rows if row["blocked_runtime_selector_changed"]
        ),
        "actual_artifact_runtime_selections": sum(
            1 for row in rows if row["blocked_runtime_selection_source"] == "artifact"
        ),
        "fallback_chain_enforced_rows": len(fallback_enforced),
        "learned_success_predictions": len(learned_success),
        "learned_oracle_predictions": len(learned_oracle),
        "learned_profiled_success_non_oracle_predictions": (
            len(rows) - len(learned_oracle)
        ),
        "learned_differs_from_artifact_rows": len(learned_differs),
        "learned_differs_from_artifact_request_ids": tuple(
            row["request_id"] for row in learned_differs
        ),
        "max_learned_regret_vs_artifact_ms": max(regrets) if regrets else None,
        "min_learned_confidence": min(confidence_values) if confidence_values else 0.0,
        "max_learned_confidence": max(confidence_values) if confidence_values else 0.0,
        "counterfactual_only": True,
        "counterfactual_eligible_gate_promotions": len(counterfactual_promotions),
        "counterfactual_runtime_selector_changes": sum(
            1 for row in rows if row["counterfactual_runtime_selector_changed"]
        ),
        "blocked_guard_reason_counts": blocked_reason_counts,
        "acceptance_note": (
            "Actual replay requires the D20 model to remain blocked by quality gate; "
            "counterfactual promotion rows only verify existing profiled-success and "
            "fallback-chain enforcement."
        ),
        "next_step": (
            "add a transformer handoff bundle that packages tensors, D20/D21 "
            "summaries, and guard thresholds for external model training"
        ),
    }


def _schema(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": CSR_BLOCKED_GAP_GUARDED_REPLAY_SCHEMA_VERSION,
        "task": "replay_blocked_gap_augmented_ranker_through_learned_guard",
        "runtime_boundary": {
            "runtime_selector_changed": False,
            "shadow_only": True,
            "executes_gpu": False,
        },
        "input_contract": {
            "selector_rows": summary["source_selector_rows_path"],
            "ranker_predictions": summary["source_predictions_path"],
            "positive_predictions": summary["source_positive_predictions_path"],
        },
        "output_contract": {
            "rows": "csr_blocked_gap_guarded_replay_rows.jsonl",
            "summary": "csr_blocked_gap_guarded_replay_summary.json",
            "blocked_quality_gate": (
                "csr_blocked_gap_guarded_replay_blocked_quality_gate.json"
            ),
            "counterfactual_quality_gate": (
                "csr_blocked_gap_guarded_replay_counterfactual_quality_gate.json"
            ),
        },
        "acceptance_gate": {
            "requires_shadow_mode_decisions": 5,
            "requires_actual_quality_gate_blocks": 5,
            "requires_actual_runtime_selector_changes": 0,
            "requires_fallback_chain_enforced": 5,
            "requires_counterfactual_rows_marked_non_runtime": True,
        },
    }


def _report(summary: dict[str, Any], rows: tuple[dict[str, Any], ...]) -> str:
    lines = [
        "# CSR Blocked Gap Guarded Replay",
        "",
        f"- status: `{summary['status']}`",
        f"- shadow_only: `{summary['shadow_only']}`",
        f"- runtime_selector_changed: `{summary['runtime_selector_changed']}`",
        f"- replay_rows: `{summary['num_replay_rows']}`",
        f"- actual_quality_gate_blocks: `{summary['actual_quality_gate_blocks']}`",
        f"- actual_runtime_selector_changes: `{summary['actual_runtime_selector_changes']}`",
        f"- fallback_chain_enforced_rows: `{summary['fallback_chain_enforced_rows']}`",
        f"- learned_success_predictions: `{summary['learned_success_predictions']}`",
        f"- learned_oracle_predictions: `{summary['learned_oracle_predictions']}`",
        f"- learned_differs_from_artifact_rows: `{summary['learned_differs_from_artifact_rows']}`",
        f"- max_learned_regret_vs_artifact_ms: `{_fmt(summary['max_learned_regret_vs_artifact_ms'])}`",
        f"- counterfactual_eligible_gate_promotions: `{summary['counterfactual_eligible_gate_promotions']}`",
        f"- acceptance_note: `{summary['acceptance_note']}`",
        f"- next_step: `{summary['next_step']}`",
        "",
        "| matrix | learned | artifact | blocked_status | counterfactual_status | regret_ms |",
        "|---|---|---|---|---|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['matrix_id']} | "
            f"{row['learned_candidate_id']} | "
            f"{row['artifact_candidate_id']} | "
            f"{row['blocked_guard_status']} | "
            f"{row['counterfactual_guard_status']} | "
            f"{_fmt(row['learned_regret_vs_artifact_ms'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def _blocked_quality_gate(ranker_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "passed",
        "schema_version": "phase1_csr_selector_model_eval_v1",
        "evaluation_id": "csr_blocked_gap_guarded_replay_quality_gate_v1",
        "baseline_model_id": "artifact_profiled_success_oracle",
        "challenger_model_id": MODEL_ID,
        "best_offline_model_id": "artifact_profiled_success_oracle",
        "runtime_selected_model_id": None,
        "runtime_selector_changed": False,
        "challenger_beats_baseline": False,
        "challenger_runtime_eligible": False,
        "challenger_gate_failures": [
            "no_dedicated_runtime_quality_gate",
            "overall_eval_non_success_selections",
            "below_runtime_profiled_success_rate",
        ],
        "num_eval_predictions": int(ranker_summary["ranker_num_eval_requests"]),
        "num_eval_oracle_requests": int(ranker_summary["ranker_num_eval_oracle_requests"]),
    }


def _counterfactual_eligible_gate(ranker_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "passed",
        "schema_version": "phase1_csr_selector_model_eval_v1",
        "evaluation_id": "csr_blocked_gap_guarded_replay_counterfactual_gate_v1",
        "baseline_model_id": "counterfactual_fixture",
        "challenger_model_id": MODEL_ID,
        "best_offline_model_id": MODEL_ID,
        "runtime_selected_model_id": MODEL_ID,
        "runtime_selector_changed": False,
        "challenger_beats_baseline": True,
        "challenger_runtime_eligible": True,
        "challenger_gate_failures": [],
        "num_eval_predictions": int(ranker_summary["ranker_num_eval_requests"]),
        "num_eval_oracle_requests": int(ranker_summary["ranker_num_eval_oracle_requests"]),
        "counterfactual_only": True,
    }


def _reason_counts(rows: tuple[dict[str, Any], ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        for reason in row["blocked_guard_reasons"]:
            counts[str(reason)] = counts.get(str(reason), 0) + 1
    return dict(sorted(counts.items()))


def _stub_csr(matrix_id: str) -> CsrMatrix:
    return CsrMatrix(
        matrix_id=matrix_id,
        n_rows=1,
        n_cols=1,
        row_ptr=(0, 1),
        col_ind=(0,),
        values=(1.0,),
        field="real",
        symmetry="general",
        source_path="memory://blocked_gap_guarded_replay",
    )


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _finite_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _fmt(value: Any) -> str:
    return "" if value is None else f"{float(value):.6g}"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(data: dict[str, Any], path: Path) -> Path:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
