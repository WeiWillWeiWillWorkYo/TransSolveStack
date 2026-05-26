"""Profiling artifact records and writers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from transsolvestack.core.result import RunTrace
from transsolvestack.profiling.trace import trace_to_record


@dataclass(frozen=True)
class CandidatePerformanceRecord:
    run_id: str
    candidate_id: str
    context_id: str
    backend: str
    status: str
    system_id: str = "unknown"
    failure_class: str | None = None
    setup_time_ms: float | None = None
    solve_time_ms: float | None = None
    total_time_ms: float | None = None
    total_time_ms_min: float | None = None
    total_time_ms_median: float | None = None
    total_time_ms_std: float | None = None
    gpu_kernel_time_ms: float | None = None
    gpu_kernel_time_ms_min: float | None = None
    gpu_kernel_time_ms_median: float | None = None
    gpu_kernel_time_ms_std: float | None = None
    transfer_time_ms: float | None = None
    warmup_runs: int = 0
    measurement_repeats: int = 1
    measurement_total_time_ms: tuple[float, ...] = field(default_factory=tuple)
    measurement_gpu_kernel_time_ms: tuple[float, ...] = field(default_factory=tuple)
    peak_device_memory_mb: float | None = None
    num_iterations: int | None = None
    final_residual_norm: float | None = None
    run_trace_uri: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def candidate_performance_from_record(
    record: dict[str, Any],
) -> CandidatePerformanceRecord:
    """Build a CandidatePerformanceRecord from a JSON-compatible mapping."""

    return CandidatePerformanceRecord(
        run_id=record["run_id"],
        system_id=record.get("system_id")
        or record.get("metadata", {}).get("system_id", "unknown"),
        candidate_id=record["candidate_id"],
        context_id=record["context_id"],
        backend=record["backend"],
        status=record["status"],
        failure_class=record.get("failure_class"),
        setup_time_ms=record.get("setup_time_ms"),
        solve_time_ms=record.get("solve_time_ms"),
        total_time_ms=record.get("total_time_ms"),
        total_time_ms_min=record.get("total_time_ms_min"),
        total_time_ms_median=record.get("total_time_ms_median"),
        total_time_ms_std=record.get("total_time_ms_std"),
        gpu_kernel_time_ms=record.get("gpu_kernel_time_ms"),
        gpu_kernel_time_ms_min=record.get("gpu_kernel_time_ms_min"),
        gpu_kernel_time_ms_median=record.get("gpu_kernel_time_ms_median"),
        gpu_kernel_time_ms_std=record.get("gpu_kernel_time_ms_std"),
        transfer_time_ms=record.get("transfer_time_ms"),
        warmup_runs=int(record.get("warmup_runs", 0)),
        measurement_repeats=int(record.get("measurement_repeats", 1)),
        measurement_total_time_ms=tuple(record.get("measurement_total_time_ms", ())),
        measurement_gpu_kernel_time_ms=tuple(
            record.get("measurement_gpu_kernel_time_ms", ())
        ),
        peak_device_memory_mb=record.get("peak_device_memory_mb"),
        num_iterations=record.get("num_iterations"),
        final_residual_norm=record.get("final_residual_norm"),
        run_trace_uri=record.get("run_trace_uri"),
        metadata=dict(record.get("metadata", {})),
    )


def performance_from_trace(
    trace: RunTrace,
    candidate_id: str,
    context_id: str,
    system_id: str = "unknown",
    run_trace_uri: str | None = None,
) -> CandidatePerformanceRecord:
    """Build a candidate performance row from a RunTrace."""

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
        gpu_kernel_time_ms=trace.gpu_kernel_time_ms,
        transfer_time_ms=trace.transfer_time_ms,
        peak_device_memory_mb=trace.peak_device_memory_mb,
        num_iterations=trace.num_iterations,
        final_residual_norm=trace.final_residual_norm,
        run_trace_uri=run_trace_uri,
        metadata=dict(trace.metadata),
    )


def write_jsonl(records: Iterable[dict[str, Any]], path: str | Path) -> Path:
    """Write JSONL records with parent directory creation."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    return output


def read_jsonl(path: str | Path) -> tuple[dict[str, Any], ...]:
    input_path = Path(path)
    rows: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return tuple(rows)


def read_candidate_performance(
    path: str | Path,
) -> tuple[CandidatePerformanceRecord, ...]:
    return tuple(
        candidate_performance_from_record(record) for record in read_jsonl(path)
    )


def write_run_traces(traces: Iterable[RunTrace], path: str | Path) -> Path:
    return write_jsonl((trace_to_record(trace) for trace in traces), path)


def write_candidate_performance(
    records: Iterable[CandidatePerformanceRecord],
    path: str | Path,
) -> Path:
    return write_jsonl((asdict(record) for record in records), path)
