"""Actual Taichi smoke benchmark runner."""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from statistics import median, pstdev

from transsolvestack.benchmarks.candidates import CandidateSet
from transsolvestack.benchmarks.config import WorkloadConfig
from transsolvestack.benchmarks.expand import expand_workload_systems
from transsolvestack.benchmarks.oracle import (
    build_oracles_for_contexts,
    write_oracle_plans,
)
from transsolvestack.benchmarks.summary import (
    summarize_candidate_performance,
    write_performance_summary_report,
)
from transsolvestack.benchmarks.validation import validate_workload_for_budget
from transsolvestack.operators.taichi_diffusion2d import (
    diffusion_operator_from_system,
    ensure_taichi_cuda,
)
from transsolvestack.profiling.artifacts import (
    CandidatePerformanceRecord,
    write_candidate_performance,
    write_jsonl,
    write_run_traces,
)
from transsolvestack.profiling.protocol import BenchmarkProtocol
from transsolvestack.profiling.provenance import (
    CORE_BENCHMARK_PROVENANCE_FILES,
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


class TaichiSmokeBenchmarkRunner:
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
        validation = validate_workload_for_budget(workload, candidate_set, budget)
        protocol = BenchmarkProtocol(**workload.benchmark_protocol)
        protocol.validate()
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        systems = expand_workload_systems(workload)
        traces = []
        performance: list[CandidatePerformanceRecord] = []
        run_plans = []

        for system_spec in systems:
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
                    warmup_traces = []
                    for _ in range(protocol.warmup_runs):
                        warmup_traces.append(
                            self.engine.solve(
                                operator,
                                rhs=None,
                                context=context,
                                plan=plan,
                            ).trace
                        )
                    measurement_traces = [
                        self.engine.solve(
                            operator,
                            rhs=None,
                            context=context,
                            plan=plan,
                        ).trace
                        for _ in range(protocol.measurement_repeats)
                    ]
                    trace = _aggregate_measurement_traces(
                        measurement_traces,
                        warmup_traces,
                        protocol,
                    )
                    trace.run_id = (
                        f"run:{workload.workload_id}:"
                        f"{system_spec.system.system_id}:"
                        f"{context.context_id}:{candidate.candidate_id}"
                    ).replace(":", "_")
                    trace.metadata.update(
                        {
                            "workload_id": workload.workload_id,
                            "system_id": system_spec.system.system_id,
                            "context_id": context.context_id,
                            "candidate_id": candidate.candidate_id,
                        }
                    )
                    traces.append(trace)
                    performance.append(
                        _performance_from_aggregated_trace(
                            candidate_id=candidate.candidate_id,
                            context_id=context.context_id,
                            system_id=system_spec.system.system_id,
                            trace=trace,
                            run_trace_uri="run_trace.jsonl",
                            protocol=protocol,
                            measurement_traces=measurement_traces,
                        )
                    )
                    run_plans.append(
                        {
                            "run_id": trace.run_id,
                            "workload_id": workload.workload_id,
                            "system_id": system_spec.system.system_id,
                            "context_id": context.context_id,
                            "candidate_id": candidate.candidate_id,
                            "policy_plan": asdict(plan),
                        }
                    )

        oracles = build_oracles_for_contexts(performance)
        summary = summarize_candidate_performance(performance)
        paths = {
            "run_plans": output / "run_plans.jsonl",
            "run_traces": output / "run_trace.jsonl",
            "candidate_performance": output / "candidate_performance.jsonl",
            "oracle": output / "oracle_plan.jsonl",
            "summary": output / "performance_summary.md",
            "manifest": output / "artifact_manifest.json",
        }
        write_jsonl(run_plans, paths["run_plans"])
        write_run_traces(traces, paths["run_traces"])
        write_candidate_performance(performance, paths["candidate_performance"])
        write_oracle_plans(oracles, paths["oracle"])
        write_performance_summary_report(summary, oracles, paths["summary"])
        write_jsonl([{"validation": validation.__dict__}], output / "validation.jsonl")
        write_manifest(
            build_artifact_manifest(
                artifact_kind="taichi_smoke_benchmark",
                command="scripts/tss_taichi_smoke_benchmark.py",
                tracked_files=_benchmark_provenance_files(workload, candidate_set),
                metadata={
                    "git_commit": git_commit_or_unknown(),
                    "workload": workload.path,
                    "candidate_set": candidate_set.path,
                    "warmup_runs": protocol.warmup_runs,
                    "measurement_repeats": protocol.measurement_repeats,
                    "num_performance_records": len(performance),
                    "num_oracle_records": len(oracles),
                },
            ),
            paths["manifest"],
        )
        return paths


def _aggregate_measurement_traces(
    measurement_traces: list,
    warmup_traces: list,
    protocol: BenchmarkProtocol,
):
    if not measurement_traces:
        raise ValueError("measurement_traces must not be empty")
    best = min(
        measurement_traces,
        key=lambda trace: float(trace.total_time_ms or float("inf")),
    )
    total_values = [float(trace.total_time_ms or 0.0) for trace in measurement_traces]
    kernel_values = [
        float(trace.gpu_kernel_time_ms or 0.0) for trace in measurement_traces
    ]
    metadata = dict(best.metadata)
    metadata.update(
        {
            "warmup_runs": protocol.warmup_runs,
            "measurement_repeats": protocol.measurement_repeats,
            "measurement_total_time_ms": total_values,
            "measurement_gpu_kernel_time_ms": kernel_values,
            "measurement_total_time_ms_min": min(total_values),
            "measurement_total_time_ms_median": float(median(total_values)),
            "measurement_total_time_ms_std": float(pstdev(total_values)),
            "measurement_gpu_kernel_time_ms_min": min(kernel_values),
            "measurement_gpu_kernel_time_ms_median": float(median(kernel_values)),
            "measurement_gpu_kernel_time_ms_std": float(pstdev(kernel_values)),
            "warmup_statuses": [trace.status for trace in warmup_traces],
        }
    )
    return replace(
        best,
        total_time_ms=min(total_values),
        solve_time_ms=min(kernel_values),
        gpu_kernel_time_ms=min(kernel_values),
        metadata=metadata,
    )


def _performance_from_aggregated_trace(
    trace,
    candidate_id: str,
    context_id: str,
    system_id: str,
    run_trace_uri: str,
    protocol: BenchmarkProtocol,
    measurement_traces: list,
) -> CandidatePerformanceRecord:
    total_values = [float(item.total_time_ms or 0.0) for item in measurement_traces]
    kernel_values = [float(item.gpu_kernel_time_ms or 0.0) for item in measurement_traces]
    return CandidatePerformanceRecord(
        run_id=trace.run_id,
        system_id=system_id,
        candidate_id=candidate_id,
        context_id=context_id,
        backend=trace.backend,
        status=trace.status,
        failure_class=trace.failure_class,
        setup_time_ms=trace.setup_time_ms,
        solve_time_ms=trace.solve_time_ms,
        total_time_ms=trace.total_time_ms,
        total_time_ms_min=min(total_values),
        total_time_ms_median=float(median(total_values)),
        total_time_ms_std=float(pstdev(total_values)),
        gpu_kernel_time_ms=trace.gpu_kernel_time_ms,
        gpu_kernel_time_ms_min=min(kernel_values),
        gpu_kernel_time_ms_median=float(median(kernel_values)),
        gpu_kernel_time_ms_std=float(pstdev(kernel_values)),
        transfer_time_ms=trace.transfer_time_ms,
        warmup_runs=protocol.warmup_runs,
        measurement_repeats=protocol.measurement_repeats,
        measurement_total_time_ms=tuple(total_values),
        measurement_gpu_kernel_time_ms=tuple(kernel_values),
        peak_device_memory_mb=trace.peak_device_memory_mb,
        num_iterations=trace.num_iterations,
        final_residual_norm=trace.final_residual_norm,
        run_trace_uri=run_trace_uri,
        metadata=dict(trace.metadata),
    )


def _benchmark_provenance_files(
    workload: WorkloadConfig,
    candidate_set: CandidateSet,
) -> tuple[str, ...]:
    config_paths = (
        workload.path or "configs/workloads/phase1_smoke.yaml",
        candidate_set.path or "configs/candidates/phase1_taichi_gpu.yaml",
    )
    static_paths = tuple(
        path
        for path in CORE_BENCHMARK_PROVENANCE_FILES
        if path
        not in {
            "configs/workloads/phase1_smoke.yaml",
            "configs/candidates/phase1_taichi_gpu.yaml",
        }
    )
    return tuple(dict.fromkeys((*config_paths, *static_paths)))
