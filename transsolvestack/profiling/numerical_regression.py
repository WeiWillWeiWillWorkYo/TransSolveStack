"""Numerical regression checks for real Taichi solve paths."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from transsolvestack.benchmarks.candidates import CandidateSet
from transsolvestack.benchmarks.config import WorkloadConfig
from transsolvestack.benchmarks.expand import expand_workload_systems
from transsolvestack.benchmarks.validation import validate_workload_for_budget
from transsolvestack.operators.taichi_diffusion2d import (
    diffusion_operator_from_system,
    ensure_taichi_cuda,
)
from transsolvestack.profiling.artifacts import write_jsonl
from transsolvestack.profiling.provenance import (
    CORE_REGRESSION_PROVENANCE_FILES,
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
class NumericalRegressionThresholds:
    max_final_residual: float = 1.0e-6
    max_relative_error: float = 5.0e-3
    max_residual_growth: float = 1.0e-3


@dataclass(frozen=True)
class NumericalRegressionRecord:
    run_id: str
    system_id: str
    context_id: str
    candidate_id: str
    solver: str
    preconditioner: str
    status: str
    passed: bool
    failure_reasons: tuple[str, ...]
    final_residual_norm: float | None
    initial_residual_norm: float | None
    residual_drop: float | None
    num_iterations: int | None
    max_iter: int
    relative_error_to_true: float | None
    metadata: dict[str, Any] = field(default_factory=dict)


class TaichiNumericalRegressionRunner:
    def __init__(self, device_memory_gb: float = 0.5) -> None:
        ensure_taichi_cuda(device_memory_gb=device_memory_gb)
        self.engine = TaichiExecutionEngine()

    def run(
        self,
        workload: WorkloadConfig,
        candidate_set: CandidateSet,
        budget: ResourceBudget,
        output_dir: str | Path,
        thresholds: NumericalRegressionThresholds | None = None,
    ) -> dict[str, Path]:
        thresholds = thresholds or NumericalRegressionThresholds()
        validate_workload_for_budget(workload, candidate_set, budget)
        records: list[NumericalRegressionRecord] = []
        for system_spec in expand_workload_systems(workload):
            operator = diffusion_operator_from_system(system_spec.system)
            for workload_context in workload.contexts:
                context = solve_context_from_workload(
                    workload_context,
                    backend=workload.backend,
                )
                for candidate in candidate_set.candidates:
                    plan = policy_plan_from_candidate(
                        system_spec.system,
                        context,
                        candidate,
                        candidate_set.backend,
                    )
                    result = self.engine.solve(
                        operator,
                        rhs=None,
                        context=context,
                        plan=plan,
                    )
                    trace = result.trace
                    run_id = (
                        f"regression:{system_spec.system.system_id}:"
                        f"{context.context_id}:{candidate.candidate_id}"
                    ).replace(":", "_")
                    records.append(
                        _record_from_trace(
                            run_id=run_id,
                            trace=trace,
                            system_id=system_spec.system.system_id,
                            context_id=context.context_id,
                            candidate_id=candidate.candidate_id,
                            max_iter=context.max_iter,
                            thresholds=thresholds,
                        )
                    )
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        records_path = write_jsonl(
            (asdict(record) for record in records),
            output / "numerical_regression.jsonl",
        )
        report_path = _write_regression_report(records, output / "numerical_regression.md")
        manifest_path = write_manifest(
            build_artifact_manifest(
                artifact_kind="taichi_numerical_regression",
                command="scripts/tss_numerical_regression.py",
                tracked_files=CORE_REGRESSION_PROVENANCE_FILES,
                metadata={
                    "git_commit": git_commit_or_unknown(),
                    "num_records": len(records),
                    "num_passed": sum(1 for record in records if record.passed),
                },
            ),
            output / "artifact_manifest.json",
        )
        return {"records": records_path, "report": report_path, "manifest": manifest_path}


def _record_from_trace(
    run_id: str,
    trace,
    system_id: str,
    context_id: str,
    candidate_id: str,
    max_iter: int,
    thresholds: NumericalRegressionThresholds,
) -> NumericalRegressionRecord:
    reasons: list[str] = []
    residual_history = tuple(float(value) for value in trace.residual_history)
    initial_residual = residual_history[0] if residual_history else None
    final_residual = trace.final_residual_norm
    relative_error = trace.metadata.get("relative_error_to_true")
    residual_drop = None
    if initial_residual is not None and final_residual is not None:
        residual_drop = final_residual / max(initial_residual, 1.0e-30)

    if trace.status != "success":
        reasons.append(f"status={trace.status}")
    if final_residual is None or not math.isfinite(final_residual):
        reasons.append("final_residual_not_finite")
    elif final_residual > thresholds.max_final_residual:
        reasons.append(f"final_residual>{thresholds.max_final_residual}")
    if relative_error is None or not math.isfinite(float(relative_error)):
        reasons.append("relative_error_not_finite")
    elif float(relative_error) > thresholds.max_relative_error:
        reasons.append(f"relative_error>{thresholds.max_relative_error}")
    if trace.num_iterations is None:
        reasons.append("missing_iterations")
    elif trace.num_iterations > max_iter:
        reasons.append("iterations_exceed_max")
    if residual_drop is None or not math.isfinite(residual_drop):
        reasons.append("residual_drop_not_finite")
    elif residual_drop > thresholds.max_residual_growth:
        reasons.append(f"residual_drop>{thresholds.max_residual_growth}")
    if any(not math.isfinite(value) for value in residual_history):
        reasons.append("residual_history_contains_nonfinite")

    return NumericalRegressionRecord(
        run_id=run_id,
        system_id=system_id,
        context_id=context_id,
        candidate_id=candidate_id,
        solver=str(trace.metadata.get("solver")),
        preconditioner=str(trace.metadata.get("preconditioner")),
        status=trace.status,
        passed=not reasons,
        failure_reasons=tuple(reasons),
        final_residual_norm=final_residual,
        initial_residual_norm=initial_residual,
        residual_drop=residual_drop,
        num_iterations=trace.num_iterations,
        max_iter=max_iter,
        relative_error_to_true=relative_error,
        metadata=dict(trace.metadata),
    )


def _write_regression_report(
    records: list[NumericalRegressionRecord],
    path: Path,
) -> Path:
    passed = sum(1 for record in records if record.passed)
    failed = len(records) - passed
    lines = [
        "# Numerical Regression",
        "",
        f"- records: `{len(records)}`",
        f"- passed: `{passed}`",
        f"- failed: `{failed}`",
        "",
        "## Cases",
        "",
    ]
    for record in records:
        status = "pass" if record.passed else "fail"
        lines.append(
            f"- {status}: {record.system_id} / {record.candidate_id} "
            f"res={record.final_residual_norm} err={record.relative_error_to_true} "
            f"iters={record.num_iterations}"
        )
        if record.failure_reasons:
            lines.append(f"  reasons: {', '.join(record.failure_reasons)}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
