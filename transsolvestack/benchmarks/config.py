"""Benchmark workload configuration loader."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from transsolvestack.config import load_yaml


@dataclass(frozen=True)
class WorkloadFamily:
    family_id: str
    operator_kind: str
    sizes: tuple[tuple[int, ...], ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def problem_sizes(self) -> tuple[int, ...]:
        values: list[int] = []
        for size in self.sizes:
            n = 1
            for dim in size:
                n *= int(dim)
            values.append(n)
        return tuple(values)


@dataclass(frozen=True)
class WorkloadContext:
    context_id: str
    tolerance_abs: float
    tolerance_rel: float
    max_iter: int
    precision: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WorkloadConfig:
    workload_id: str
    backend: str
    families: tuple[WorkloadFamily, ...]
    contexts: tuple[WorkloadContext, ...]
    candidate_set_path: str
    resource_limits_path: str | None = None
    benchmark_protocol: dict[str, Any] = field(default_factory=dict)
    path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def load_workload_config(path: str | Path) -> WorkloadConfig:
    data = load_yaml(path)
    families = tuple(
        WorkloadFamily(
            family_id=item["family_id"],
            operator_kind=item["operator_kind"],
            sizes=tuple(tuple(int(dim) for dim in size) for size in item.get("sizes", [])),
            metadata=dict(item.get("metadata", {})),
        )
        for item in data.get("families", [])
    )
    contexts = tuple(
        WorkloadContext(
            context_id=item["context_id"],
            tolerance_abs=float(item["tolerance_abs"]),
            tolerance_rel=float(item["tolerance_rel"]),
            max_iter=int(item["max_iter"]),
            precision=item["precision"],
            metadata=dict(item.get("metadata", {})),
        )
        for item in data.get("contexts", [])
    )
    if not families:
        raise ValueError(f"workload has no families: {path}")
    if not contexts:
        raise ValueError(f"workload has no contexts: {path}")
    return WorkloadConfig(
        workload_id=data["workload_id"],
        backend=data["backend"],
        families=families,
        contexts=contexts,
        candidate_set_path=data["candidate_set"],
        resource_limits_path=data.get("resource_limits"),
        benchmark_protocol=dict(data.get("benchmark_protocol", {})),
        path=str(path),
        metadata=dict(data.get("metadata", {})),
    )
