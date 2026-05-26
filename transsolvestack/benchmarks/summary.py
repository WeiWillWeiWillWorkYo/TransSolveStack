"""Profile artifact aggregation and summary reports."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from transsolvestack.benchmarks.oracle import OraclePlanRecord
from transsolvestack.profiling.artifacts import CandidatePerformanceRecord


@dataclass(frozen=True)
class CandidatePerformanceSummary:
    num_records: int
    status_counts: dict[str, int]
    backend_counts: dict[str, int]
    system_counts: dict[str, int]
    context_counts: dict[str, int]
    candidate_counts: dict[str, int]
    num_success: int
    num_failed: int
    num_dry_run: int
    best_total_time_ms: float | None
    best_candidate_id: str | None
    metadata: dict[str, Any] = field(default_factory=dict)


def summarize_candidate_performance(
    records: Iterable[CandidatePerformanceRecord],
) -> CandidatePerformanceSummary:
    rows = tuple(records)
    status_counts = Counter(record.status for record in rows)
    backend_counts = Counter(record.backend for record in rows)
    system_counts = Counter(record.system_id for record in rows)
    context_counts = Counter(record.context_id for record in rows)
    candidate_counts = Counter(record.candidate_id for record in rows)
    feasible = [
        record
        for record in rows
        if record.status in {"success", "fallback_success"}
        and record.total_time_ms is not None
    ]
    best = min(feasible, key=lambda record: float(record.total_time_ms), default=None)
    return CandidatePerformanceSummary(
        num_records=len(rows),
        status_counts=dict(status_counts),
        backend_counts=dict(backend_counts),
        system_counts=dict(system_counts),
        context_counts=dict(context_counts),
        candidate_counts=dict(candidate_counts),
        num_success=status_counts.get("success", 0)
        + status_counts.get("fallback_success", 0),
        num_failed=status_counts.get("failed", 0)
        + status_counts.get("fallback_failed", 0),
        num_dry_run=status_counts.get("dry_run", 0),
        best_total_time_ms=best.total_time_ms if best else None,
        best_candidate_id=best.candidate_id if best else None,
    )


def write_performance_summary_report(
    summary: CandidatePerformanceSummary,
    oracles: Iterable[OraclePlanRecord],
    path: str | Path,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    oracle_rows = tuple(oracles)
    lines = [
        "# Candidate Performance Summary",
        "",
        f"- num_records: `{summary.num_records}`",
        f"- num_success: `{summary.num_success}`",
        f"- num_failed: `{summary.num_failed}`",
        f"- num_dry_run: `{summary.num_dry_run}`",
        f"- best_candidate_id: `{summary.best_candidate_id}`",
        f"- best_total_time_ms: `{summary.best_total_time_ms}`",
        f"- num_systems: `{len(summary.system_counts)}`",
        "",
        "## Status Counts",
        "",
    ]
    lines.extend(f"- {key}: `{value}`" for key, value in summary.status_counts.items())
    lines.extend(["", "## Oracle Plans", ""])
    if oracle_rows:
        for oracle in oracle_rows:
            lines.append(
                f"- {oracle.system_id} / {oracle.context_id}: "
                f"`{oracle.best_candidate_id}` "
                f"({oracle.best_total_time_ms})"
            )
    else:
        lines.append("- none")
    lines.extend(["", "## Raw Summary", "", "```text"])
    lines.extend(f"{key}: {value}" for key, value in asdict(summary).items())
    lines.extend(["```", ""])
    output.write_text("\n".join(lines), encoding="utf-8")
    return output
