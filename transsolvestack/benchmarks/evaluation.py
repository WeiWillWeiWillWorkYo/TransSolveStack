"""Benchmark evaluation tables, including regret versus oracle."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from transsolvestack.benchmarks.oracle import OraclePlanRecord
from transsolvestack.profiling.artifacts import (
    CandidatePerformanceRecord,
    write_jsonl,
)


@dataclass(frozen=True)
class CandidateEvaluationRecord:
    system_id: str
    context_id: str
    candidate_id: str
    status: str
    is_oracle: bool
    total_time_ms: float | None
    oracle_total_time_ms: float | None
    regret_vs_oracle: float | None
    num_iterations: int | None
    final_residual_norm: float | None
    relative_error_to_true: float | None
    total_time_ms_median: float | None
    total_time_ms_std: float | None
    gpu_kernel_time_ms: float | None
    measurement_repeats: int


def build_candidate_evaluations(
    records: Iterable[CandidatePerformanceRecord],
    oracles: Iterable[OraclePlanRecord],
) -> tuple[CandidateEvaluationRecord, ...]:
    oracle_by_key = {
        (oracle.system_id, oracle.context_id): oracle for oracle in oracles
    }
    evaluations: list[CandidateEvaluationRecord] = []
    for record in sorted(
        records,
        key=lambda item: (item.system_id, item.context_id, item.candidate_id),
    ):
        oracle = oracle_by_key.get((record.system_id, record.context_id))
        oracle_time = oracle.best_total_time_ms if oracle else None
        regret = _regret(record.total_time_ms, oracle_time, record.status)
        evaluations.append(
            CandidateEvaluationRecord(
                system_id=record.system_id,
                context_id=record.context_id,
                candidate_id=record.candidate_id,
                status=record.status,
                is_oracle=bool(
                    oracle and record.candidate_id == oracle.best_candidate_id
                ),
                total_time_ms=record.total_time_ms,
                oracle_total_time_ms=oracle_time,
                regret_vs_oracle=regret,
                num_iterations=record.num_iterations,
                final_residual_norm=record.final_residual_norm,
                relative_error_to_true=record.metadata.get("relative_error_to_true"),
                total_time_ms_median=record.total_time_ms_median,
                total_time_ms_std=record.total_time_ms_std,
                gpu_kernel_time_ms=record.gpu_kernel_time_ms,
                measurement_repeats=record.measurement_repeats,
            )
        )
    return tuple(evaluations)


def write_candidate_evaluations(
    evaluations: Iterable[CandidateEvaluationRecord],
    path: str | Path,
) -> Path:
    return write_jsonl((asdict(record) for record in evaluations), path)


def write_benchmark_report(
    evaluations: Iterable[CandidateEvaluationRecord],
    path: str | Path,
) -> Path:
    rows = tuple(evaluations)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Benchmark Evaluation Report",
        "",
        f"- rows: `{len(rows)}`",
        f"- systems: `{len({row.system_id for row in rows})}`",
        f"- candidates: `{len({row.candidate_id for row in rows})}`",
        "",
        "## Per-System Candidate Table",
        "",
        "| system | candidate | status | oracle | time_ms | regret | iters | residual | rel_error | repeats |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row.system_id} | "
            f"{row.candidate_id} | "
            f"{row.status} | "
            f"{'yes' if row.is_oracle else 'no'} | "
            f"{_fmt(row.total_time_ms)} | "
            f"{_fmt(row.regret_vs_oracle)} | "
            f"{row.num_iterations} | "
            f"{_fmt(row.final_residual_norm)} | "
            f"{_fmt(row.relative_error_to_true)} | "
            f"{row.measurement_repeats} |"
        )
    lines.append("")
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def _regret(
    total_time_ms: float | None,
    oracle_total_time_ms: float | None,
    status: str,
) -> float | None:
    if status not in {"success", "fallback_success"}:
        return None
    if total_time_ms is None or oracle_total_time_ms is None:
        return None
    if oracle_total_time_ms <= 0:
        return None
    return (total_time_ms - oracle_total_time_ms) / oracle_total_time_ms


def _fmt(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6g}"

