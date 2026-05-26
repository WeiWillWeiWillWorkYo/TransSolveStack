"""Profile runner skeleton with non-executing dry-run support."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from transsolvestack.benchmarks.candidates import CandidateConfig, CandidateSet
from transsolvestack.benchmarks.config import WorkloadConfig, WorkloadContext
from transsolvestack.benchmarks.expand import expand_workload_systems
from transsolvestack.benchmarks.report import write_dry_run_report
from transsolvestack.benchmarks.validation import (
    BenchmarkDryRunReport,
    validate_workload_for_budget,
)
from transsolvestack.core.result import PolicyPlan, RunTrace
from transsolvestack.core.types import LinearSystem, SolveContext
from transsolvestack.profiling.artifacts import (
    CandidatePerformanceRecord,
    performance_from_trace,
    write_candidate_performance,
    write_jsonl,
    write_run_traces,
)
from transsolvestack.runtime.plan_resolver import PlanResolver
from transsolvestack.runtime.resource_budget import ResourceBudget


@dataclass(frozen=True)
class ProfileRunPlan:
    run_id: str
    workload_id: str
    system_id: str
    context_id: str
    candidate_id: str
    policy_plan: PolicyPlan


@dataclass(frozen=True)
class ProfileDryRunResult:
    validation: BenchmarkDryRunReport
    run_plans: tuple[ProfileRunPlan, ...]
    traces: tuple[RunTrace, ...]
    performance: tuple[CandidatePerformanceRecord, ...]


class ProfileRunner:
    """Build profile execution plans without launching numerical kernels."""

    def __init__(self, resolver: PlanResolver | None = None) -> None:
        self.resolver = resolver or PlanResolver()

    def build_run_plans(
        self,
        workload: WorkloadConfig,
        candidate_set: CandidateSet,
    ) -> tuple[ProfileRunPlan, ...]:
        systems = expand_workload_systems(workload)
        run_plans: list[ProfileRunPlan] = []
        for system_spec in systems:
            for workload_context in workload.contexts:
                solve_context = solve_context_from_workload(
                    workload_context,
                    backend=workload.backend,
                )
                for candidate in candidate_set.candidates:
                    policy_plan = policy_plan_from_candidate(
                        system=system_spec.system,
                        context=solve_context,
                        candidate=candidate,
                        backend=candidate_set.backend,
                    )
                    run_id = _run_id(
                        workload.workload_id,
                        system_spec.system.system_id,
                        workload_context.context_id,
                        candidate.candidate_id,
                    )
                    run_plans.append(
                        ProfileRunPlan(
                            run_id=run_id,
                            workload_id=workload.workload_id,
                            system_id=system_spec.system.system_id,
                            context_id=workload_context.context_id,
                            candidate_id=candidate.candidate_id,
                            policy_plan=policy_plan,
                        )
                    )
        return tuple(run_plans)

    def dry_run(
        self,
        workload: WorkloadConfig,
        candidate_set: CandidateSet,
        budget: ResourceBudget,
    ) -> ProfileDryRunResult:
        validation = validate_workload_for_budget(workload, candidate_set, budget)
        run_plans = self.build_run_plans(workload, candidate_set)
        traces: list[RunTrace] = []
        performance: list[CandidatePerformanceRecord] = []
        for run_plan in run_plans:
            resolved = self.resolver.resolve(run_plan.policy_plan)
            trace = RunTrace(
                run_id=run_plan.run_id,
                plan_id=run_plan.policy_plan.plan_id,
                backend=resolved.backend.name,
                status="dry_run",
                metadata={
                    "workload_id": run_plan.workload_id,
                    "system_id": run_plan.system_id,
                    "context_id": run_plan.context_id,
                    "candidate_id": run_plan.candidate_id,
                    "solver": resolved.solver.name,
                    "preconditioner": resolved.preconditioner.name,
                    "executed": False,
                },
            )
            traces.append(trace)
            performance.append(
                performance_from_trace(
                    trace,
                    candidate_id=run_plan.candidate_id,
                    context_id=run_plan.context_id,
                    system_id=run_plan.system_id,
                    run_trace_uri="run_trace.jsonl",
                )
            )
        return ProfileDryRunResult(
            validation=validation,
            run_plans=run_plans,
            traces=tuple(traces),
            performance=tuple(performance),
        )

    def write_dry_run_artifacts(
        self,
        result: ProfileDryRunResult,
        output_dir: str | Path,
    ) -> dict[str, Path]:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        paths = {
            "run_plans": output / "run_plans.jsonl",
            "run_traces": output / "run_trace.jsonl",
            "candidate_performance": output / "candidate_performance.jsonl",
            "report": output / "dry_run_report.md",
        }
        write_jsonl(
            (asdict(run_plan) for run_plan in result.run_plans),
            paths["run_plans"],
        )
        write_run_traces(result.traces, paths["run_traces"])
        write_candidate_performance(
            result.performance,
            paths["candidate_performance"],
        )
        write_dry_run_report(result.validation, paths["report"])
        return paths


def solve_context_from_workload(
    context: WorkloadContext,
    backend: str = "taichi_gpu",
) -> SolveContext:
    return SolveContext(
        context_id=context.context_id,
        tolerance_abs=context.tolerance_abs,
        tolerance_rel=context.tolerance_rel,
        max_iter=context.max_iter,
        precision=context.precision,
        required_backend=backend,
        metadata={"source": "workload_config", **context.metadata},
    )


def policy_plan_from_candidate(
    system: LinearSystem,
    context: SolveContext,
    candidate: CandidateConfig,
    backend: str,
) -> PolicyPlan:
    solver = dict(candidate.solver)
    preconditioner = dict(candidate.preconditioner) or {"name": "none"}
    return PolicyPlan(
        plan_id=_plan_id(system.system_id, context.context_id, candidate.candidate_id),
        backend=backend,
        solver=solver,
        preconditioner=preconditioner,
        reuse=dict(candidate.reuse),
        audit={
            "source": "candidate_config",
            "system_id": system.system_id,
            "operator_id": system.operator.operator_id,
            "context_id": context.context_id,
            "candidate_id": candidate.candidate_id,
        },
    )


def _plan_id(system_id: str, context_id: str, candidate_id: str) -> str:
    return f"plan:{_safe(system_id)}:{_safe(context_id)}:{_safe(candidate_id)}"


def _run_id(
    workload_id: str,
    system_id: str,
    context_id: str,
    candidate_id: str,
) -> str:
    return (
        f"run:{_safe(workload_id)}:{_safe(system_id)}:"
        f"{_safe(context_id)}:{_safe(candidate_id)}"
    )


def _safe(value: str) -> str:
    return (
        value.replace("/", "_")
        .replace(" ", "_")
        .replace(":", "_")
        .replace("[", "")
        .replace("]", "")
    )
