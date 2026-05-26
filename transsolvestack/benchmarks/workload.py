"""Benchmark workload contracts."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class BenchmarkWorkload:
    workload_id: str
    family_id: str
    systems: tuple[str, ...]
    contexts: tuple[str, ...]
    candidate_set: str
    metadata: dict = field(default_factory=dict)


@dataclass
class BenchmarkReport:
    workload_id: str
    artifact_dir: str
    summary: dict = field(default_factory=dict)

