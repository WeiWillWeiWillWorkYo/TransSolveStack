"""Repeated-solve Taichi sequence benchmark."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from transsolvestack.benchmarks.candidates import CandidateSet
from transsolvestack.benchmarks.sequence_config import SequenceWorkloadConfig
from transsolvestack.core.result import PolicyPlan
from transsolvestack.core.types import SolveContext
from transsolvestack.operators.synthetic import build_synthetic_system
from transsolvestack.operators.taichi_diffusion2d import (
    diffusion_operator_from_system,
    ensure_taichi_cuda,
)
from transsolvestack.profiling.artifacts import write_jsonl
from transsolvestack.profiling.provenance import (
    CORE_SEQUENCE_PROVENANCE_FILES,
    build_artifact_manifest,
    git_commit_or_unknown,
    write_manifest,
)
from transsolvestack.runtime.engine import TaichiExecutionEngine
from transsolvestack.runtime.resource_budget import ResourceBudget


@dataclass(frozen=True)
class SequenceStepRecord:
    sequence_id: str
    step_id: str
    system_id: str
    candidate_id: str
    status: str
    ax: float
    ay: float
    az: float
    total_time_ms: float | None
    num_iterations: int | None
    final_residual_norm: float | None
    relative_error_to_true: float | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SequenceCandidateSummary:
    sequence_id: str
    candidate_id: str
    status: str
    num_steps: int
    total_sequence_time_ms: float
    total_iterations: int
    max_final_residual_norm: float
    max_relative_error_to_true: float


class TaichiSequenceBenchmarkRunner:
    def __init__(self, device_memory_gb: float = 0.5) -> None:
        ensure_taichi_cuda(device_memory_gb=device_memory_gb)
        self.engine = TaichiExecutionEngine()

    def run(
        self,
        workload: SequenceWorkloadConfig,
        candidate_set: CandidateSet,
        budget: ResourceBudget,
        output_dir: str | Path,
    ) -> dict[str, Path]:
        system_spec = build_synthetic_system(
            family_id=workload.family_id,
            size=workload.size,
            dtype=workload.context.precision,
            expected_operator_kind=workload.operator_kind,
        )
        budget.validate_problem_size(
            n=system_spec.estimated_unknowns,
            effective_nnz=system_spec.estimated_effective_nnz,
        )
        budget.validate_iterations(workload.context.max_iter)
        selected_candidates = [
            item
            for item in candidate_set.candidates
            if not workload.candidate_ids or item.candidate_id in workload.candidate_ids
        ]
        if not selected_candidates:
            raise ValueError("sequence workload selected no candidates")
        operator = diffusion_operator_from_system(system_spec.system)
        context = SolveContext(
            context_id=workload.context.context_id,
            tolerance_abs=workload.context.tolerance_abs,
            tolerance_rel=workload.context.tolerance_rel,
            max_iter=workload.context.max_iter,
            precision=workload.context.precision,
            required_backend=workload.backend,
        )
        step_records: list[SequenceStepRecord] = []
        summaries: list[SequenceCandidateSummary] = []
        for candidate in selected_candidates:
            candidate_steps: list[SequenceStepRecord] = []
            for step in workload.steps:
                operator.update_coefficients(step.ax, step.ay, step.az)
                plan = PolicyPlan(
                    plan_id=(
                        f"sequence:{workload.sequence_id}:"
                        f"{step.step_id}:{candidate.candidate_id}"
                    ),
                    backend=workload.backend,
                    solver=dict(candidate.solver),
                    preconditioner=dict(candidate.preconditioner) or {"name": "none"},
                    reuse={
                        **dict(candidate.reuse),
                        "sequence_id": workload.sequence_id,
                        "step_id": step.step_id,
                        "same_operator_buffers": True,
                    },
                    audit={
                        "system_id": system_spec.system.system_id,
                        "sequence_id": workload.sequence_id,
                        "step_id": step.step_id,
                        "candidate_id": candidate.candidate_id,
                    },
                )
                result = self.engine.solve(operator, rhs=None, context=context, plan=plan)
                trace = result.trace
                record = SequenceStepRecord(
                    sequence_id=workload.sequence_id,
                    step_id=step.step_id,
                    system_id=system_spec.system.system_id,
                    candidate_id=candidate.candidate_id,
                    status=trace.status,
                    ax=step.ax,
                    ay=step.ay,
                    az=step.az,
                    total_time_ms=trace.total_time_ms,
                    num_iterations=trace.num_iterations,
                    final_residual_norm=trace.final_residual_norm,
                    relative_error_to_true=trace.metadata.get("relative_error_to_true"),
                    metadata=dict(trace.metadata),
                )
                candidate_steps.append(record)
                step_records.append(record)
            summaries.append(_summarize_candidate_sequence(candidate_steps))

        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        paths = {
            "steps": output / "sequence_steps.jsonl",
            "summary": output / "sequence_summary.jsonl",
            "report": output / "sequence_report.md",
            "manifest": output / "artifact_manifest.json",
        }
        write_jsonl((asdict(record) for record in step_records), paths["steps"])
        write_jsonl((asdict(record) for record in summaries), paths["summary"])
        _write_sequence_report(summaries, step_records, paths["report"])
        write_manifest(
            build_artifact_manifest(
                artifact_kind="taichi_sequence_benchmark",
                command="scripts/tss_sequence_benchmark.py",
                tracked_files=CORE_SEQUENCE_PROVENANCE_FILES,
                metadata={
                    "git_commit": git_commit_or_unknown(),
                    "sequence_id": workload.sequence_id,
                    "num_steps": len(workload.steps),
                    "num_candidates": len(selected_candidates),
                    "num_step_records": len(step_records),
                },
            ),
            paths["manifest"],
        )
        return paths


def _summarize_candidate_sequence(
    records: list[SequenceStepRecord],
) -> SequenceCandidateSummary:
    status = "success" if all(record.status == "success" for record in records) else "failed"
    return SequenceCandidateSummary(
        sequence_id=records[0].sequence_id,
        candidate_id=records[0].candidate_id,
        status=status,
        num_steps=len(records),
        total_sequence_time_ms=sum(float(record.total_time_ms or 0.0) for record in records),
        total_iterations=sum(int(record.num_iterations or 0) for record in records),
        max_final_residual_norm=max(float(record.final_residual_norm or 0.0) for record in records),
        max_relative_error_to_true=max(
            float(record.relative_error_to_true or 0.0) for record in records
        ),
    )


def _write_sequence_report(
    summaries: list[SequenceCandidateSummary],
    steps: list[SequenceStepRecord],
    path: Path,
) -> Path:
    lines = [
        "# Sequence Benchmark Report",
        "",
        f"- candidates: `{len(summaries)}`",
        f"- step_records: `{len(steps)}`",
        "",
        "## Candidate Summary",
        "",
        "| candidate | status | steps | total_ms | iterations | max_residual | max_rel_error |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        lines.append(
            f"| {item.candidate_id} | {item.status} | {item.num_steps} | "
            f"{item.total_sequence_time_ms:.6g} | {item.total_iterations} | "
            f"{item.max_final_residual_norm:.6g} | {item.max_relative_error_to_true:.6g} |"
        )
    lines.extend(["", "## Steps", ""])
    for step in steps:
        lines.append(
            f"- {step.candidate_id} / {step.step_id}: status={step.status}, "
            f"ax={step.ax}, total_ms={step.total_time_ms}, "
            f"iters={step.num_iterations}, residual={step.final_residual_norm}"
        )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path

