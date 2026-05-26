"""Expand benchmark workload configs into metadata-only system specs."""

from __future__ import annotations

from transsolvestack.benchmarks.config import WorkloadConfig
from transsolvestack.operators.synthetic import (
    SyntheticSystemSpec,
    build_synthetic_system,
)


def expand_workload_systems(workload: WorkloadConfig) -> tuple[SyntheticSystemSpec, ...]:
    """Expand a workload into LinearSystem metadata without materializing matrices."""

    systems: list[SyntheticSystemSpec] = []
    for family in workload.families:
        for size in family.sizes:
            systems.append(
                build_synthetic_system(
                    family_id=family.family_id,
                    size=size,
                    dtype=_workload_default_dtype(workload),
                    expected_operator_kind=family.operator_kind,
                )
            )
    return tuple(systems)


def _workload_default_dtype(workload: WorkloadConfig) -> str:
    if not workload.contexts:
        return "float32"
    return workload.contexts[0].precision

