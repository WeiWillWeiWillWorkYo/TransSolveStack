"""Repeated-solve sequence workload configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from transsolvestack.config import load_yaml


@dataclass(frozen=True)
class SequenceStepConfig:
    step_id: str
    ax: float
    ay: float
    az: float = 0.0


@dataclass(frozen=True)
class SequenceSolveContextConfig:
    context_id: str
    tolerance_abs: float
    tolerance_rel: float
    max_iter: int
    precision: str


@dataclass(frozen=True)
class SequenceWorkloadConfig:
    workload_id: str
    backend: str
    family_id: str
    operator_kind: str
    size: tuple[int, ...]
    context: SequenceSolveContextConfig
    sequence_id: str
    steps: tuple[SequenceStepConfig, ...]
    candidate_ids: tuple[str, ...]
    candidate_set_path: str
    resource_limits_path: str
    path: str | None = None


def load_sequence_workload_config(path: str | Path) -> SequenceWorkloadConfig:
    data = load_yaml(path)
    context = data["context"]
    sequence = data["sequence"]
    steps = tuple(
        SequenceStepConfig(
            step_id=item["step_id"],
            ax=float(item["ax"]),
            ay=float(item["ay"]),
            az=float(item.get("az", 0.0)),
        )
        for item in sequence["steps"]
    )
    if not steps:
        raise ValueError(f"sequence workload has no steps: {path}")
    return SequenceWorkloadConfig(
        workload_id=data["workload_id"],
        backend=data["backend"],
        family_id=data["family_id"],
        operator_kind=data["operator_kind"],
        size=tuple(int(dim) for dim in data["size"]),
        context=SequenceSolveContextConfig(
            context_id=context["context_id"],
            tolerance_abs=float(context["tolerance_abs"]),
            tolerance_rel=float(context["tolerance_rel"]),
            max_iter=int(context["max_iter"]),
            precision=context["precision"],
        ),
        sequence_id=sequence["sequence_id"],
        steps=steps,
        candidate_ids=tuple(data.get("candidate_ids", ())),
        candidate_set_path=data["candidate_set"],
        resource_limits_path=data["resource_limits"],
        path=str(path),
    )

