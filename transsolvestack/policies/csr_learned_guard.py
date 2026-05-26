"""Runtime guard for learned CSR selector recommendations."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from transsolvestack.policies.csr_artifact_selector import (
    CsrArtifactPolicySelector,
    CsrCandidateSelection,
)
from transsolvestack.datasets.csr import CsrMatrix, csr_matrix_from_record
from transsolvestack.profiling.artifacts import read_jsonl, write_jsonl


CSR_LEARNED_GUARD_SCHEMA_VERSION = "phase1_csr_learned_runtime_guard_v1"
LEARNED_GUARD_ID = "csr_learned_policy_runtime_guard_v1"
LEARNED_GUARD_MODES = {"shadow", "promote_if_safe"}


@dataclass(frozen=True)
class CsrLearnedShadowPrediction:
    model_id: str
    request_id: str
    matrix_id: str
    context_id: str
    selected_candidate_id: str
    selected_target_status: str | None
    evaluation_status: str | None
    confidence: float
    score_margin: float | None
    ranked_candidate_ids: tuple[str, ...]


@dataclass(frozen=True)
class CsrLearnedGuardDecision:
    schema_version: str
    guard_id: str
    mode: str
    guard_status: str
    guard_reasons: tuple[str, ...]
    runtime_selector_changed: bool
    runtime_candidate_id: str
    runtime_selection_source: str
    artifact_candidate_id: str
    fallback_candidate_ids: tuple[str, ...]
    fallback_chain_enforced: bool
    min_confidence: float
    learned_prediction: CsrLearnedShadowPrediction | None
    quality_gate_summary: dict[str, Any]
    selection: CsrCandidateSelection


def plan_csr_with_learned_guard(
    csr: CsrMatrix | dict[str, Any],
    *,
    selector_path: str | Path = "runs/phase1_csr_selector_readiness/csr_selector_rows.jsonl",
    learned_predictions_path: str | Path = "runs/phase1_csr_transformer_ranker/csr_transformer_ranker_predictions.jsonl",
    quality_gate_summary_path: str | Path = "runs/phase1_csr_transformer_quality_gate/csr_transformer_quality_gate_summary.json",
    context_id: str = "phase1_csr_selector",
    objective: str = "min_solve_time_success",
    fallback_candidate_id: str | None = None,
    mode: str = "shadow",
    min_confidence: float = 0.75,
) -> CsrLearnedGuardDecision:
    """Plan CSR execution with learned-policy shadowing and safety gates.

    The default mode is shadow-only. `promote_if_safe` still requires a passing
    offline quality gate, sufficient confidence, and a learned candidate that is
    present as a profiled-success artifact row for the exact matrix/context.
    """

    if mode not in LEARNED_GUARD_MODES:
        raise ValueError(f"unsupported CSR learned guard mode: {mode}")
    if min_confidence < 0.0 or min_confidence > 1.0:
        raise ValueError("min_confidence must be in [0, 1]")

    matrix = _coerce_csr(csr)
    artifact_selector = CsrArtifactPolicySelector.from_selector_rows(
        selector_path,
        fallback_candidate_id=fallback_candidate_id,
    )
    artifact_selection = artifact_selector.select(
        matrix,
        context_id=context_id,
        objective=objective,
    )
    quality_gate_summary = json.loads(
        Path(quality_gate_summary_path).read_text(encoding="utf-8")
    )
    learned_prediction = _load_shadow_prediction(
        learned_predictions_path,
        matrix_id=matrix.matrix_id,
        context_id=context_id,
    )

    reasons: list[str] = []
    promoted_selection: CsrCandidateSelection | None = None
    guard_status = "shadow_only"
    runtime_source = "artifact"

    if mode == "shadow":
        reasons.append("shadow_mode")
    elif learned_prediction is None:
        guard_status = "blocked_no_prediction"
        reasons.append("no_learned_prediction_for_matrix_context")
    elif not _quality_gate_runtime_eligible(quality_gate_summary, learned_prediction.model_id):
        guard_status = "blocked_quality_gate"
        gate_failures = quality_gate_summary.get("challenger_gate_failures", ())
        if gate_failures:
            reasons.extend(f"quality_gate:{item}" for item in gate_failures)
        else:
            reasons.append("quality_gate_not_runtime_eligible")
    elif learned_prediction.confidence < min_confidence:
        guard_status = "blocked_confidence"
        reasons.append(
            f"confidence_below_threshold:{learned_prediction.confidence:.6g}<{min_confidence:.6g}"
        )
    else:
        try:
            promoted_selection = artifact_selector.select_candidate(
                matrix,
                learned_prediction.selected_candidate_id,
                context_id=context_id,
                reason="learned_guard_promoted",
            )
        except ValueError as exc:
            guard_status = "blocked_non_success_candidate"
            reasons.append(str(exc))
        else:
            guard_status = "promoted"
            runtime_source = "learned"
            reasons.append("quality_gate_confidence_and_artifact_success_passed")

    selection = promoted_selection or artifact_selection
    fallback_enforced = _fallback_chain_is_enforced(selection)
    if not fallback_enforced:
        guard_status = "blocked_fallback_chain"
        runtime_source = "artifact"
        selection = artifact_selection
        reasons.append("fallback_chain_contains_unprofiled_or_duplicate_candidate")

    return CsrLearnedGuardDecision(
        schema_version=CSR_LEARNED_GUARD_SCHEMA_VERSION,
        guard_id=LEARNED_GUARD_ID,
        mode=mode,
        guard_status=guard_status,
        guard_reasons=tuple(reasons),
        runtime_selector_changed=runtime_source == "learned",
        runtime_candidate_id=selection.candidate_id,
        runtime_selection_source=runtime_source,
        artifact_candidate_id=artifact_selection.candidate_id,
        fallback_candidate_ids=selection.fallback_candidate_ids,
        fallback_chain_enforced=fallback_enforced,
        min_confidence=min_confidence,
        learned_prediction=learned_prediction,
        quality_gate_summary=_quality_gate_projection(quality_gate_summary),
        selection=selection,
    )


def decision_to_record(decision: CsrLearnedGuardDecision) -> dict[str, Any]:
    payload = asdict(decision)
    return payload


def write_csr_learned_guard_decisions(
    decisions: tuple[CsrLearnedGuardDecision, ...],
    path: str | Path,
) -> Path:
    return write_jsonl((decision_to_record(decision) for decision in decisions), path)


def _load_shadow_prediction(
    predictions_path: str | Path,
    *,
    matrix_id: str,
    context_id: str,
) -> CsrLearnedShadowPrediction | None:
    candidates = [
        row
        for row in read_jsonl(predictions_path)
        if str(row.get("matrix_id")) == matrix_id
        and str(row.get("context_id")) == context_id
    ]
    if not candidates:
        return None
    row = sorted(candidates, key=lambda item: str(item.get("request_id", "")))[0]
    confidence, margin = _prediction_confidence(row)
    return CsrLearnedShadowPrediction(
        model_id=str(row["model_id"]),
        request_id=str(row["request_id"]),
        matrix_id=str(row["matrix_id"]),
        context_id=str(row["context_id"]),
        selected_candidate_id=str(row["selected_candidate_id"]),
        selected_target_status=(
            None
            if row.get("selected_target_status") is None
            else str(row.get("selected_target_status"))
        ),
        evaluation_status=(
            None if row.get("evaluation_status") is None else str(row.get("evaluation_status"))
        ),
        confidence=confidence,
        score_margin=margin,
        ranked_candidate_ids=tuple(str(item) for item in row.get("ranked_candidate_ids", ())),
    )


def _prediction_confidence(row: dict[str, Any]) -> tuple[float, float | None]:
    scores = {str(key): float(value) for key, value in row.get("scores", {}).items()}
    ranked = tuple(str(item) for item in row.get("ranked_candidate_ids", ()))
    if not ranked:
        return 0.0, None
    if len(ranked) == 1:
        return 1.0, math.inf
    top = ranked[0]
    second = ranked[1]
    if top not in scores or second not in scores:
        return 0.0, None
    margin = scores[top] - scores[second]
    if not math.isfinite(margin):
        return 0.0, margin
    clamped = max(min(margin, 60.0), -60.0)
    confidence = 1.0 / (1.0 + math.exp(-clamped))
    return confidence, margin


def _quality_gate_runtime_eligible(
    summary: dict[str, Any],
    model_id: str,
) -> bool:
    return (
        summary.get("status") == "passed"
        and summary.get("challenger_runtime_eligible") is True
        and str(summary.get("challenger_model_id")) == model_id
        and str(summary.get("runtime_selected_model_id")) == model_id
        and summary.get("runtime_selector_changed") is False
    )


def _quality_gate_projection(summary: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "status",
        "schema_version",
        "evaluation_id",
        "baseline_model_id",
        "challenger_model_id",
        "best_offline_model_id",
        "runtime_selected_model_id",
        "runtime_selector_changed",
        "challenger_beats_baseline",
        "challenger_runtime_eligible",
        "challenger_gate_failures",
        "num_eval_predictions",
        "num_eval_oracle_requests",
    )
    return {key: summary.get(key) for key in keys if key in summary}


def _fallback_chain_is_enforced(selection: CsrCandidateSelection) -> bool:
    fallback_ids = tuple(item.get("candidate_id") for item in selection.plan.fallback_chain)
    if len(fallback_ids) != len(set(fallback_ids)):
        return False
    if selection.candidate_id in set(fallback_ids):
        return False
    if tuple(fallback_ids) != selection.fallback_candidate_ids:
        return False
    return True


def _coerce_csr(csr: CsrMatrix | dict[str, Any]) -> CsrMatrix:
    if isinstance(csr, CsrMatrix):
        return csr
    return csr_matrix_from_record(csr)
