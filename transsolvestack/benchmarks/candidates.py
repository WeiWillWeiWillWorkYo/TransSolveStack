"""Candidate set contracts and YAML loader."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from transsolvestack.config import load_yaml


@dataclass(frozen=True)
class CandidateConfig:
    candidate_id: str
    solver: dict[str, Any]
    preconditioner: dict[str, Any] = field(default_factory=dict)
    reuse: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def max_iter(self) -> int | None:
        value = self.solver.get("max_iter")
        return int(value) if value is not None else None


@dataclass(frozen=True)
class CandidateSet:
    candidate_set_id: str
    backend: str
    candidates: tuple[CandidateConfig, ...]
    path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def load_candidate_set(path: str | Path) -> CandidateSet:
    data = load_yaml(path)
    candidates = tuple(
        CandidateConfig(
            candidate_id=item["candidate_id"],
            solver=dict(item.get("solver", {})),
            preconditioner=dict(item.get("preconditioner", {})),
            reuse=dict(item.get("reuse", {})),
            metadata=dict(item.get("metadata", {})),
        )
        for item in data.get("candidates", [])
    )
    if not candidates:
        raise ValueError(f"candidate set has no candidates: {path}")
    return CandidateSet(
        candidate_set_id=data["candidate_set_id"],
        backend=data["backend"],
        candidates=candidates,
        path=str(path),
        metadata=dict(data.get("metadata", {})),
    )

