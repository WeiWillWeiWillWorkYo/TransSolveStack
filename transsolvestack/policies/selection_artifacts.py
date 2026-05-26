"""Artifacts for benchmark-backed policy selections."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from transsolvestack.policies.artifact_selector import (
    BenchmarkArtifactPolicySelector,
)
from transsolvestack.profiling.artifacts import read_jsonl, write_jsonl


@dataclass(frozen=True)
class SelectedPolicyRecord:
    system_id: str
    context_id: str
    candidate_id: str
    reason: str
    fallback_candidate_ids: tuple[str, ...]
    plan_id: str
    backend: str
    solver: dict[str, Any]
    preconditioner: dict[str, Any]
    reuse: dict[str, Any]
    fallback_chain: list[dict[str, Any]]
    budget: dict[str, Any]
    audit: dict[str, Any] = field(default_factory=dict)


def build_selected_policy_records(
    selector: BenchmarkArtifactPolicySelector,
    evaluation_path: str | Path,
) -> tuple[SelectedPolicyRecord, ...]:
    keys = sorted(
        {
            (str(row["system_id"]), str(row["context_id"]))
            for row in read_jsonl(evaluation_path)
        }
    )
    records: list[SelectedPolicyRecord] = []
    for system_id, context_id in keys:
        selection = selector.select(system_id, context_id)
        plan = selection.plan
        records.append(
            SelectedPolicyRecord(
                system_id=system_id,
                context_id=context_id,
                candidate_id=selection.candidate_id,
                reason=selection.reason,
                fallback_candidate_ids=selection.fallback_candidate_ids,
                plan_id=plan.plan_id,
                backend=plan.backend,
                solver=dict(plan.solver),
                preconditioner=dict(plan.preconditioner),
                reuse=dict(plan.reuse),
                fallback_chain=list(plan.fallback_chain),
                budget=dict(plan.budget),
                audit=dict(plan.audit),
            )
        )
    return tuple(records)


def write_selected_policy_records(
    records: Iterable[SelectedPolicyRecord],
    path: str | Path,
) -> Path:
    return write_jsonl((asdict(record) for record in records), path)


def write_policy_selection_report(
    records: Iterable[SelectedPolicyRecord],
    path: str | Path,
) -> Path:
    rows = tuple(records)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Policy Selection Report",
        "",
        f"- rows: `{len(rows)}`",
        f"- systems: `{len({row.system_id for row in rows})}`",
        "",
        "| system | context | candidate | reason | oracle | time_ms | regret | fallbacks |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row.system_id} | "
            f"{row.context_id} | "
            f"{row.candidate_id} | "
            f"{row.reason} | "
            f"{'yes' if row.audit.get('is_oracle') else 'no'} | "
            f"{_fmt(row.audit.get('total_time_ms'))} | "
            f"{_fmt(row.audit.get('regret_vs_oracle'))} | "
            f"{len(row.fallback_candidate_ids)} |"
        )
    lines.append("")
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):.6g}"
