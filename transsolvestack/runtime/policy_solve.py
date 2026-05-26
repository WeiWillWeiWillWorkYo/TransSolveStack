"""End-to-end policy-driven Taichi solve runner."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from transsolvestack.benchmarks.candidates import CandidateSet
from transsolvestack.benchmarks.config import WorkloadConfig
from transsolvestack.benchmarks.expand import expand_workload_systems
from transsolvestack.core.result import RunTrace
from transsolvestack.operators.taichi_diffusion2d import (
    diffusion_operator_from_system,
    ensure_taichi_cuda,
)
from transsolvestack.policies.artifact_selector import BenchmarkArtifactPolicySelector
from transsolvestack.profiling.artifacts import write_jsonl
from transsolvestack.profiling.provenance import (
    CORE_POLICY_SOLVE_PROVENANCE_FILES,
    build_artifact_manifest,
    git_commit_or_unknown,
    write_manifest,
)
from transsolvestack.profiling.runner import solve_context_from_workload
from transsolvestack.profiling.trace import trace_to_record
from transsolvestack.runtime.engine import TaichiExecutionEngine
from transsolvestack.runtime.resource_budget import ResourceBudget


@dataclass(frozen=True)
class PolicySolveRecord:
    workload_id: str
    system_id: str
    context_id: str
    selected_candidate_id: str
    selection_reason: str
    status: str
    total_time_ms: float | None
    num_iterations: int | None
    final_residual_norm: float | None
    relative_error_to_true: float | None
    fallback_candidate_ids: tuple[str, ...]
    policy_plan_id: str
    trace_run_id: str
    audit: dict[str, Any] = field(default_factory=dict)


class PolicyDrivenTaichiSolveRunner:
    """Run exact workload systems through artifact-backed policy selection."""

    def __init__(self, device_memory_gb: float = 0.5) -> None:
        ensure_taichi_cuda(device_memory_gb=device_memory_gb)
        self.engine = TaichiExecutionEngine()

    def run(
        self,
        workload: WorkloadConfig,
        candidate_set: CandidateSet,
        budget: ResourceBudget,
        evaluation_path: str | Path,
        output_dir: str | Path,
        fallback_candidate_id: str | None = None,
    ) -> dict[str, Path]:
        selector = BenchmarkArtifactPolicySelector.from_evaluation_artifact(
            candidate_set=candidate_set,
            evaluation_path=evaluation_path,
            fallback_candidate_id=fallback_candidate_id,
        )
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        records: list[PolicySolveRecord] = []
        traces: list[RunTrace] = []
        run_plans: list[dict[str, Any]] = []

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
                selection = selector.select(system_spec.system.system_id, context.context_id)
                result = self.engine.solve(
                    operator=operator,
                    rhs=None,
                    context=context,
                    plan=selection.plan,
                )
                trace = result.trace
                trace.run_id = (
                    f"policy_solve:{workload.workload_id}:"
                    f"{system_spec.system.system_id}:"
                    f"{context.context_id}:{selection.candidate_id}"
                ).replace(":", "_")
                trace.metadata.update(
                    {
                        "workload_id": workload.workload_id,
                        "system_id": system_spec.system.system_id,
                        "context_id": context.context_id,
                        "selected_candidate_id": selection.candidate_id,
                        "selection_reason": selection.reason,
                    }
                )
                traces.append(trace)
                run_plans.append(
                    {
                        "run_id": trace.run_id,
                        "workload_id": workload.workload_id,
                        "system_id": system_spec.system.system_id,
                        "context_id": context.context_id,
                        "selected_candidate_id": selection.candidate_id,
                        "policy_plan": asdict(selection.plan),
                    }
                )
                records.append(
                    PolicySolveRecord(
                        workload_id=workload.workload_id,
                        system_id=system_spec.system.system_id,
                        context_id=context.context_id,
                        selected_candidate_id=selection.candidate_id,
                        selection_reason=selection.reason,
                        status=trace.status,
                        total_time_ms=trace.total_time_ms,
                        num_iterations=trace.num_iterations,
                        final_residual_norm=trace.final_residual_norm,
                        relative_error_to_true=trace.metadata.get(
                            "relative_error_to_true"
                        ),
                        fallback_candidate_ids=selection.fallback_candidate_ids,
                        policy_plan_id=selection.plan.plan_id,
                        trace_run_id=trace.run_id,
                        audit=dict(selection.plan.audit),
                    )
                )

        paths = {
            "run_plans": output / "policy_run_plans.jsonl",
            "run_traces": output / "policy_run_traces.jsonl",
            "results": output / "policy_solve_results.jsonl",
            "report": output / "policy_solve_report.md",
            "manifest": output / "artifact_manifest.json",
        }
        write_jsonl(run_plans, paths["run_plans"])
        write_jsonl((trace_to_record(trace) for trace in traces), paths["run_traces"])
        write_jsonl((asdict(record) for record in records), paths["results"])
        _write_policy_solve_report(records, paths["report"])
        write_manifest(
            build_artifact_manifest(
                artifact_kind="policy_driven_taichi_solve",
                command="scripts/tss_policy_solve_smoke.py",
                tracked_files=CORE_POLICY_SOLVE_PROVENANCE_FILES,
                metadata={
                    "git_commit": git_commit_or_unknown(),
                    "workload_id": workload.workload_id,
                    "evaluation": str(evaluation_path),
                    "num_results": len(records),
                    "num_success": sum(1 for record in records if record.status == "success"),
                },
            ),
            paths["manifest"],
        )
        return paths


def _write_policy_solve_report(
    records: list[PolicySolveRecord],
    path: str | Path,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Policy-Driven Solve Report",
        "",
        f"- rows: `{len(records)}`",
        f"- success: `{sum(1 for record in records if record.status == 'success')}`",
        "",
        "| system | context | candidate | status | reason | ms | iters | residual | rel_error | fallbacks |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        lines.append(
            "| "
            f"{record.system_id} | "
            f"{record.context_id} | "
            f"{record.selected_candidate_id} | "
            f"{record.status} | "
            f"{record.selection_reason} | "
            f"{_fmt(record.total_time_ms)} | "
            f"{record.num_iterations} | "
            f"{_fmt(record.final_residual_norm)} | "
            f"{_fmt(record.relative_error_to_true)} | "
            f"{len(record.fallback_candidate_ids)} |"
        )
    lines.append("")
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def _fmt(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6g}"
