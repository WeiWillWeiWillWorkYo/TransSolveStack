"""Functional smoke tests for newly registered Taichi solvers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from transsolvestack.benchmarks.candidates import CandidateSet
from transsolvestack.benchmarks.config import WorkloadConfig
from transsolvestack.benchmarks.expand import expand_workload_systems
from transsolvestack.operators.taichi_diffusion2d import (
    diffusion_operator_from_system,
    ensure_taichi_cuda,
)
from transsolvestack.profiling.artifacts import write_jsonl
from transsolvestack.profiling.provenance import (
    CORE_SOLVER_FUNCTIONAL_PROVENANCE_FILES,
    build_artifact_manifest,
    git_commit_or_unknown,
    write_manifest,
)
from transsolvestack.profiling.runner import (
    policy_plan_from_candidate,
    solve_context_from_workload,
)
from transsolvestack.runtime.engine import TaichiExecutionEngine
from transsolvestack.runtime.resource_budget import ResourceBudget


@dataclass(frozen=True)
class SolverFunctionalRecord:
    workload_id: str
    system_id: str
    context_id: str
    candidate_id: str
    solver: str
    preconditioner: str
    status: str
    total_time_ms: float | None
    num_iterations: int | None
    final_residual_norm: float | None
    relative_error_to_true: float | None
    residual_drop: float | None
    metadata: dict[str, Any]


class TaichiSolverFunctionalRunner:
    def __init__(self, device_memory_gb: float = 0.5) -> None:
        ensure_taichi_cuda(device_memory_gb=device_memory_gb)
        self.engine = TaichiExecutionEngine()

    def run(
        self,
        workload: WorkloadConfig,
        candidate_set: CandidateSet,
        budget: ResourceBudget,
        output_dir: str | Path,
    ) -> dict[str, Path]:
        records: list[SolverFunctionalRecord] = []
        for system_spec in expand_workload_systems(workload):
            budget.validate_problem_size(
                n=system_spec.estimated_unknowns,
                effective_nnz=system_spec.estimated_effective_nnz,
            )
            operator = diffusion_operator_from_system(system_spec.system)
            for workload_context in workload.contexts:
                budget.validate_iterations(workload_context.max_iter)
                context = solve_context_from_workload(
                    workload_context,
                    backend=workload.backend,
                )
                for candidate in candidate_set.candidates:
                    plan = policy_plan_from_candidate(
                        system=system_spec.system,
                        context=context,
                        candidate=candidate,
                        backend=candidate_set.backend,
                    )
                    result = self.engine.solve(
                        operator=operator,
                        rhs=None,
                        context=context,
                        plan=plan,
                    )
                    trace = result.trace
                    initial_residual = (
                        trace.residual_history[0]
                        if trace.residual_history
                        else None
                    )
                    final_residual = trace.final_residual_norm
                    residual_drop = (
                        None
                        if initial_residual is None or final_residual is None
                        else final_residual / max(initial_residual, 1.0e-30)
                    )
                    records.append(
                        SolverFunctionalRecord(
                            workload_id=workload.workload_id,
                            system_id=system_spec.system.system_id,
                            context_id=context.context_id,
                            candidate_id=candidate.candidate_id,
                            solver=str(candidate.solver["name"]),
                            preconditioner=str(
                                candidate.preconditioner.get("name", "none")
                            ),
                            status=trace.status,
                            total_time_ms=trace.total_time_ms,
                            num_iterations=trace.num_iterations,
                            final_residual_norm=final_residual,
                            relative_error_to_true=trace.metadata.get(
                                "relative_error_to_true"
                            ),
                            residual_drop=residual_drop,
                            metadata=dict(trace.metadata),
                        )
                    )
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        paths = {
            "results": output / "solver_functional_results.jsonl",
            "report": output / "solver_functional_report.md",
            "manifest": output / "artifact_manifest.json",
        }
        write_jsonl((asdict(record) for record in records), paths["results"])
        _write_solver_functional_report(records, paths["report"])
        write_manifest(
            build_artifact_manifest(
                artifact_kind="solver_functional_smoke",
                command="scripts/tss_solver_functional_smoke.py",
                tracked_files=CORE_SOLVER_FUNCTIONAL_PROVENANCE_FILES,
                metadata={
                    "git_commit": git_commit_or_unknown(),
                    "num_records": len(records),
                    "num_success": sum(1 for row in records if row.status == "success"),
                    "candidate_set": candidate_set.candidate_set_id,
                },
            ),
            paths["manifest"],
        )
        return paths


def _write_solver_functional_report(
    records: list[SolverFunctionalRecord],
    path: str | Path,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Solver Functional Smoke",
        "",
        f"- rows: `{len(records)}`",
        f"- success: `{sum(1 for row in records if row.status == 'success')}`",
        "",
        "| candidate | solver | preconditioner | status | iters | residual | rel_error | residual_drop |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for row in records:
        lines.append(
            "| "
            f"{row.candidate_id} | "
            f"{row.solver} | "
            f"{row.preconditioner} | "
            f"{row.status} | "
            f"{row.num_iterations} | "
            f"{_fmt(row.final_residual_norm)} | "
            f"{_fmt(row.relative_error_to_true)} | "
            f"{_fmt(row.residual_drop)} |"
        )
    lines.append("")
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def _fmt(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6g}"
