"""Dry-run validation for workloads and candidate sets."""

from __future__ import annotations

from dataclasses import dataclass, field

from transsolvestack.benchmarks.candidates import CandidateSet
from transsolvestack.benchmarks.config import WorkloadConfig
from transsolvestack.benchmarks.expand import expand_workload_systems
from transsolvestack.runtime.resource_budget import ResourceBudget


@dataclass(frozen=True)
class BenchmarkDryRunReport:
    workload_id: str
    candidate_set_id: str
    backend: str
    num_families: int
    num_contexts: int
    num_candidates: int
    max_problem_n: int
    max_candidate_iter: int
    budget_mode: str
    num_systems: int = 0
    max_effective_nnz: int = 0
    warnings: tuple[str, ...] = field(default_factory=tuple)


def validate_workload_for_budget(
    workload: WorkloadConfig,
    candidate_set: CandidateSet,
    budget: ResourceBudget,
) -> BenchmarkDryRunReport:
    """Validate a workload/candidate set without running numerical kernels."""

    warnings: list[str] = []
    if workload.backend != candidate_set.backend:
        raise ValueError(
            f"workload backend {workload.backend} does not match candidate backend "
            f"{candidate_set.backend}"
        )
    if workload.backend != "taichi_gpu":
        warnings.append("workload backend is not taichi_gpu")

    systems = expand_workload_systems(workload)
    max_problem_n = 0
    max_effective_nnz = 0
    for system in systems:
        max_problem_n = max(max_problem_n, system.estimated_unknowns)
        max_effective_nnz = max(max_effective_nnz, system.estimated_effective_nnz)
        budget.validate_problem_size(
            n=system.estimated_unknowns,
            effective_nnz=system.estimated_effective_nnz,
        )

    max_iter = 0
    for context in workload.contexts:
        max_iter = max(max_iter, context.max_iter)
        budget.validate_iterations(context.max_iter)
    for candidate in candidate_set.candidates:
        if candidate.max_iter is not None:
            max_iter = max(max_iter, candidate.max_iter)
            budget.validate_iterations(candidate.max_iter)

    return BenchmarkDryRunReport(
        workload_id=workload.workload_id,
        candidate_set_id=candidate_set.candidate_set_id,
        backend=workload.backend,
        num_families=len(workload.families),
        num_contexts=len(workload.contexts),
        num_candidates=len(candidate_set.candidates),
        num_systems=len(systems),
        max_problem_n=max_problem_n,
        max_effective_nnz=max_effective_nnz,
        max_candidate_iter=max_iter,
        budget_mode=budget.mode,
        warnings=tuple(warnings),
    )
