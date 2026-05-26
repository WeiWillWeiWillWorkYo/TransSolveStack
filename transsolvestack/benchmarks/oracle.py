"""Oracle plan schema and builder for profiled candidates."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from transsolvestack.profiling.artifacts import (
    CandidatePerformanceRecord,
    write_jsonl,
)


@dataclass(frozen=True)
class OraclePlanRecord:
    system_id: str
    context_id: str
    oracle_plan_id: str
    objective: str
    best_candidate_id: str | None
    best_total_time_ms: float | None
    safe_candidate_id: str | None = None
    fallback_candidate_id: str | None = None
    regret_baselines: dict[str, Any] = field(default_factory=dict)


def build_fastest_success_oracle(
    system_id: str,
    context_id: str,
    records: Iterable[CandidatePerformanceRecord],
    objective: str = "min_total_time_success",
) -> OraclePlanRecord:
    """Select the fastest successful candidate with known total time."""

    feasible = [
        record
        for record in records
        if record.system_id == system_id
        and record.context_id == context_id
        and record.status in {"success", "fallback_success"}
        and record.total_time_ms is not None
    ]
    if not feasible:
        return OraclePlanRecord(
            system_id=system_id,
            context_id=context_id,
            oracle_plan_id=f"oracle:{system_id}:{context_id}:empty",
            objective=objective,
            best_candidate_id=None,
            best_total_time_ms=None,
        )
    best = min(feasible, key=lambda record: float(record.total_time_ms))
    return OraclePlanRecord(
        system_id=system_id,
        context_id=context_id,
        oracle_plan_id=f"oracle:{system_id}:{context_id}:{best.candidate_id}",
        objective=objective,
        best_candidate_id=best.candidate_id,
        best_total_time_ms=best.total_time_ms,
        safe_candidate_id=best.candidate_id,
    )


def build_oracles_for_contexts(
    records: Iterable[CandidatePerformanceRecord],
    objective: str = "min_total_time_success",
) -> tuple[OraclePlanRecord, ...]:
    rows = tuple(records)
    keys = sorted({(record.system_id, record.context_id) for record in rows})
    return tuple(
        build_fastest_success_oracle(system_id, context_id, rows, objective=objective)
        for system_id, context_id in keys
    )


def write_oracle_plans(
    records: Iterable[OraclePlanRecord],
    path: str | Path,
) -> Path:
    return write_jsonl((asdict(record) for record in records), path)
