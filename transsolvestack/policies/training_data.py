"""Policy training row export for learned selector integration."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from transsolvestack.benchmarks.config import WorkloadConfig
from transsolvestack.benchmarks.expand import expand_workload_systems
from transsolvestack.profiling.artifacts import read_jsonl, write_jsonl


FEATURE_SCHEMA_VERSION = "phase1_policy_features_v1"


@dataclass(frozen=True)
class PolicyTrainingRow:
    schema_version: str
    system_id: str
    context_id: str
    candidate_id: str
    label_is_oracle: bool
    target_regret_vs_oracle: float
    target_total_time_ms: float
    target_status: str
    features: dict[str, Any] = field(default_factory=dict)


def build_policy_training_rows(
    workload: WorkloadConfig,
    evaluation_path: str | Path,
) -> tuple[PolicyTrainingRow, ...]:
    system_features = {
        spec.system.system_id: _system_features(spec) for spec in expand_workload_systems(workload)
    }
    rows: list[PolicyTrainingRow] = []
    for row in read_jsonl(evaluation_path):
        total_time = row.get("total_time_ms")
        regret = row.get("regret_vs_oracle")
        if total_time is None or regret is None:
            continue
        system_id = str(row["system_id"])
        features = dict(system_features.get(system_id, {}))
        features.update(
            {
                "candidate_id": row["candidate_id"],
                "context_id": row["context_id"],
                "num_iterations": row.get("num_iterations"),
                "final_residual_norm": row.get("final_residual_norm"),
                "relative_error_to_true": row.get("relative_error_to_true"),
                "measurement_repeats": row.get("measurement_repeats"),
            }
        )
        rows.append(
            PolicyTrainingRow(
                schema_version=FEATURE_SCHEMA_VERSION,
                system_id=system_id,
                context_id=str(row["context_id"]),
                candidate_id=str(row["candidate_id"]),
                label_is_oracle=bool(row["is_oracle"]),
                target_regret_vs_oracle=float(regret),
                target_total_time_ms=float(total_time),
                target_status=str(row["status"]),
                features=features,
            )
        )
    return tuple(rows)


def write_policy_training_rows(
    rows: tuple[PolicyTrainingRow, ...],
    path: str | Path,
) -> Path:
    return write_jsonl((asdict(row) for row in rows), path)


def write_transformer_feature_schema(path: str | Path) -> Path:
    schema = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "task": "rank_solver_preconditioner_candidates",
        "input_features": {
            "system": [
                "family_id",
                "operator_kind",
                "n_rows",
                "n_cols",
                "estimated_unknowns",
                "estimated_effective_nnz",
                "symmetry",
                "dtype",
                "grid_size",
                "dofs_per_node",
            ],
            "context": ["context_id"],
            "candidate": ["candidate_id"],
            "observed_numeric": [
                "num_iterations",
                "final_residual_norm",
                "relative_error_to_true",
                "measurement_repeats",
            ],
        },
        "targets": [
            "label_is_oracle",
            "target_regret_vs_oracle",
            "target_total_time_ms",
            "target_status",
        ],
        "integration_boundary": {
            "status": "ready_for_dataset_scaleout",
            "model_required": False,
            "next_step": "train_or_connect_transformer_policy_ranker",
        },
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(schema, indent=2, sort_keys=True), encoding="utf-8")
    return output


def write_policy_training_report(
    rows: tuple[PolicyTrainingRow, ...],
    path: str | Path,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    systems = {row.system_id for row in rows}
    candidates = {row.candidate_id for row in rows}
    oracle_rows = sum(1 for row in rows if row.label_is_oracle)
    lines = [
        "# Transformer Readiness Export",
        "",
        f"- schema_version: `{FEATURE_SCHEMA_VERSION}`",
        f"- rows: `{len(rows)}`",
        f"- systems: `{len(systems)}`",
        f"- candidates: `{len(candidates)}`",
        f"- oracle_rows: `{oracle_rows}`",
        "",
        "| system | candidate | oracle | regret | total_ms |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row.system_id} | "
            f"{row.candidate_id} | "
            f"{'yes' if row.label_is_oracle else 'no'} | "
            f"{row.target_regret_vs_oracle:.6g} | "
            f"{row.target_total_time_ms:.6g} |"
        )
    lines.append("")
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def _system_features(spec) -> dict[str, Any]:
    operator = spec.system.operator
    metadata = operator.metadata
    return {
        "family_id": metadata.get("family_id"),
        "operator_kind": operator.kind,
        "n_rows": operator.shape[0],
        "n_cols": operator.shape[1],
        "estimated_unknowns": spec.estimated_unknowns,
        "estimated_effective_nnz": spec.estimated_effective_nnz,
        "symmetry": operator.symmetry,
        "dtype": operator.dtype,
        "grid_size": tuple(metadata.get("grid_size", ())),
        "dofs_per_node": metadata.get("dofs_per_node"),
    }
