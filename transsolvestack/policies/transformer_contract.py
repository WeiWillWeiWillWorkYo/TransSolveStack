"""Transformer policy integration contracts.

This module intentionally contains no model implementation. It defines the
stable request/response shape that a future Transformer ranker must satisfy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TransformerPolicyRequest:
    schema_version: str
    system_id: str
    context_id: str
    candidate_ids: tuple[str, ...]
    features: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TransformerPolicyPrediction:
    schema_version: str
    system_id: str
    context_id: str
    ranked_candidate_ids: tuple[str, ...]
    scores: dict[str, float]
    model_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


def validate_transformer_prediction(
    request: TransformerPolicyRequest,
    prediction: TransformerPolicyPrediction,
) -> None:
    if prediction.schema_version != request.schema_version:
        raise ValueError("Transformer prediction schema version mismatch")
    if prediction.system_id != request.system_id:
        raise ValueError("Transformer prediction system_id mismatch")
    if prediction.context_id != request.context_id:
        raise ValueError("Transformer prediction context_id mismatch")
    requested = set(request.candidate_ids)
    ranked = tuple(prediction.ranked_candidate_ids)
    if set(ranked) != requested:
        raise ValueError("Transformer prediction must rank every requested candidate")
    if len(ranked) != len(request.candidate_ids):
        raise ValueError("Transformer prediction contains duplicate candidates")
    missing_scores = requested - set(prediction.scores)
    if missing_scores:
        raise ValueError(f"Transformer prediction missing scores: {sorted(missing_scores)}")
