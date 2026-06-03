"""Accumulate completed CSR queue-batch selector rows into a training pool."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from transsolvestack.profiling.artifacts import read_jsonl, write_jsonl


CSR_QUEUE_TRAINING_POOL_SCHEMA_VERSION = "phase1_csr_queue_training_pool_v1"


def build_csr_queue_training_pool_from_files(
    *,
    full_queue_batch_path: str | Path = (
        "runs/phase1_csr_full_dataset_queue/csr_full_dataset_queue_batches.jsonl"
    ),
    base_selector_paths: Iterable[str | Path] = (
        "runs/phase1_csr_selector_readiness/csr_selector_rows.jsonl",
        "runs/phase1_csr_micro_campaign/csr_micro_selector_rows.jsonl",
    ),
    runs_root: str | Path = "runs",
    queue_batch_dir_prefix: str = "phase1_csr_queue_batch_",
) -> dict[str, Any]:
    """Build an append-only selector-row pool from completed queue batches."""

    batch_rows = tuple(read_jsonl(full_queue_batch_path))
    base_paths = tuple(Path(path) for path in base_selector_paths)
    queue_sources = _queue_batch_sources(Path(runs_root), queue_batch_dir_prefix)
    source_rows = tuple(
        [
            *(
                _source_row(
                    source_id=f"base_{index + 1:02d}",
                    source_kind="base_selector",
                    selector_rows_path=path,
                    manifest_path=None,
                    batch_summary=None,
                )
                for index, path in enumerate(base_paths)
            ),
            *queue_sources,
        ]
    )
    selector_rows, membership_rows = _combine_selector_rows(source_rows)
    completed_batch_ids = tuple(
        str(row["batch_id"])
        for row in source_rows
        if row["source_kind"] == "queue_batch_execution"
    )
    summary = _summary(
        selector_rows,
        source_rows=source_rows,
        batch_rows=batch_rows,
        completed_batch_ids=completed_batch_ids,
        full_queue_batch_path=full_queue_batch_path,
    )
    return {
        "selector_rows": selector_rows,
        "source_rows": source_rows,
        "membership_rows": membership_rows,
        "summary": summary,
        "schema": _schema(summary),
        "state": _state(summary),
        "report": _report(summary),
    }


def write_csr_queue_training_pool_artifacts(
    pool: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "selector_rows": output / "csr_queue_training_pool_selector_rows.jsonl",
        "sources": output / "csr_queue_training_pool_sources.jsonl",
        "membership": output / "csr_queue_training_pool_membership.jsonl",
        "summary": output / "csr_queue_training_pool_summary.json",
        "schema": output / "csr_queue_training_pool_schema.json",
        "state": output / "csr_queue_training_pool_state.json",
        "report": output / "csr_queue_training_pool_report.md",
    }
    write_jsonl(pool["selector_rows"], paths["selector_rows"])
    write_jsonl(pool["source_rows"], paths["sources"])
    write_jsonl(pool["membership_rows"], paths["membership"])
    _write_json(pool["summary"], paths["summary"])
    _write_json(pool["schema"], paths["schema"])
    _write_json(pool["state"], paths["state"])
    paths["report"].write_text(pool["report"], encoding="utf-8")
    return paths


def _queue_batch_sources(runs_root: Path, prefix: str) -> tuple[dict[str, Any], ...]:
    sources: list[dict[str, Any]] = []
    if not runs_root.exists():
        return ()
    for path in sorted(runs_root.iterdir(), key=lambda item: item.name):
        if not path.is_dir() or not path.name.startswith(prefix):
            continue
        selector_path = path / "csr_micro_selector_rows.jsonl"
        summary_path = path / "csr_queue_batch_execution_summary.json"
        manifest_path = path / "artifact_manifest.json"
        if not selector_path.exists() or not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") != "passed":
            continue
        sources.append(
            _source_row(
                source_id=f"queue_{summary['batch_id']}",
                source_kind="queue_batch_execution",
                selector_rows_path=selector_path,
                manifest_path=manifest_path if manifest_path.exists() else None,
                batch_summary=summary,
            )
        )
    return tuple(sources)


def _source_row(
    *,
    source_id: str,
    source_kind: str,
    selector_rows_path: Path,
    manifest_path: Path | None,
    batch_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    rows = tuple(read_jsonl(selector_rows_path))
    matrix_ids = {str(row["matrix_id"]) for row in rows}
    return {
        "schema_version": CSR_QUEUE_TRAINING_POOL_SCHEMA_VERSION,
        "source_id": source_id,
        "source_kind": source_kind,
        "selector_rows_path": str(selector_rows_path),
        "manifest_path": str(manifest_path) if manifest_path else None,
        "queue_id": batch_summary.get("queue_id") if batch_summary else None,
        "batch_id": batch_summary.get("batch_id") if batch_summary else None,
        "source_status": batch_summary.get("status", "passed") if batch_summary else "passed",
        "batch_outcome": batch_summary.get("batch_outcome") if batch_summary else None,
        "completed_without_oracle": (
            bool(batch_summary.get("completed_without_oracle", False))
            if batch_summary
            else False
        ),
        "training_pool_role": (
            batch_summary.get("training_pool_role") if batch_summary else "base_selector_rows"
        ),
        "runtime_selector_changed": (
            bool(batch_summary.get("runtime_selector_changed", False))
            if batch_summary
            else False
        ),
        "executes_gpu": bool(batch_summary.get("executes_gpu", False))
        if batch_summary
        else False,
        "imports_matrices": bool(batch_summary.get("imports_matrices", False))
        if batch_summary
        else False,
        "selector_rows": len(rows),
        "matrices": len(matrix_ids),
        "success_rows": _count_status(rows, "success"),
        "screened_out_rows": _count_status(rows, "screened_out"),
        "oracle_rows": sum(1 for row in rows if bool(row["label_is_oracle"])),
    }


def _combine_selector_rows(
    source_rows: tuple[dict[str, Any], ...],
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    rows: list[dict[str, Any]] = []
    membership: list[dict[str, Any]] = []
    seen: dict[tuple[str, str, str], str] = {}
    for source in source_rows:
        source_id = str(source["source_id"])
        for row in read_jsonl(source["selector_rows_path"]):
            key = (
                str(row["matrix_id"]),
                str(row["context_id"]),
                str(row["candidate_id"]),
            )
            if key in seen:
                raise ValueError(
                    f"duplicate CSR training pool selector row: {key} from "
                    f"{source_id} and {seen[key]}"
                )
            seen[key] = source_id
            rows.append(dict(row))
            membership.append(
                {
                    "schema_version": CSR_QUEUE_TRAINING_POOL_SCHEMA_VERSION,
                    "source_id": source_id,
                    "matrix_id": key[0],
                    "context_id": key[1],
                    "candidate_id": key[2],
                }
            )
    sort_key = lambda row: (
        str(row["matrix_id"]),
        str(row["context_id"]),
        str(row["candidate_id"]),
    )
    return tuple(sorted(rows, key=sort_key)), tuple(sorted(membership, key=sort_key))


def _summary(
    selector_rows: tuple[dict[str, Any], ...],
    *,
    source_rows: tuple[dict[str, Any], ...],
    batch_rows: tuple[dict[str, Any], ...],
    completed_batch_ids: tuple[str, ...],
    full_queue_batch_path: str | Path,
) -> dict[str, Any]:
    matrix_ids = {str(row["matrix_id"]) for row in selector_rows}
    completed_batch_set = set(completed_batch_ids)
    pending_batch_ids = tuple(
        str(row["batch_id"])
        for row in batch_rows
        if str(row["batch_id"]) not in completed_batch_set
    )
    return {
        "status": "passed" if selector_rows and source_rows else "failed",
        "schema_version": CSR_QUEUE_TRAINING_POOL_SCHEMA_VERSION,
        "source_queue_batch_path": str(full_queue_batch_path),
        "runtime_selector_changed": False,
        "executes_gpu": False,
        "imports_matrices": False,
        "append_only": True,
        "training_pool_ready": True,
        "source_count": len(source_rows),
        "base_source_count": sum(
            1 for row in source_rows if row["source_kind"] == "base_selector"
        ),
        "queue_batch_source_count": sum(
            1 for row in source_rows if row["source_kind"] == "queue_batch_execution"
        ),
        "completed_queue_batches": len(completed_batch_ids),
        "completed_batch_ids": completed_batch_ids,
        "queue_batch_oracle_batches": sum(
            1
            for row in source_rows
            if row["source_kind"] == "queue_batch_execution"
            and row.get("batch_outcome") == "profiled_with_gpu_success"
        ),
        "queue_batch_screen_only_batches": sum(
            1
            for row in source_rows
            if row["source_kind"] == "queue_batch_execution"
            and row.get("batch_outcome") == "screen_only_no_oracle"
        ),
        "full_queue_batches": len(batch_rows),
        "remaining_queue_batches": len(pending_batch_ids),
        "next_pending_batch_id": pending_batch_ids[0] if pending_batch_ids else None,
        "selector_rows": len(selector_rows),
        "membership_rows": len(selector_rows),
        "matrices": len(matrix_ids),
        "success_rows": _count_status(selector_rows, "success"),
        "screened_out_rows": _count_status(selector_rows, "screened_out"),
        "not_profiled_rows": _count_status(selector_rows, "not_profiled"),
        "not_applicable_rows": _count_status(selector_rows, "not_applicable"),
        "oracle_rows": sum(1 for row in selector_rows if bool(row["label_is_oracle"])),
        "queue_batch_selector_rows": sum(
            int(row["selector_rows"])
            for row in source_rows
            if row["source_kind"] == "queue_batch_execution"
        ),
        "non_queue_selector_rows": sum(
            int(row["selector_rows"])
            for row in source_rows
            if row["source_kind"] != "queue_batch_execution"
        ),
        "next_step": "use_pool_selector_rows_for_transformer_ready_bundle",
    }


def _schema(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": CSR_QUEUE_TRAINING_POOL_SCHEMA_VERSION,
        "task": "accumulate_completed_csr_queue_batch_selector_rows",
        "runtime_selector_changed": False,
        "executes_gpu": False,
        "imports_matrices": False,
        "append_policy": {
            "dedupe_key": ["matrix_id", "context_id", "candidate_id"],
            "duplicate_policy": "fail_closed",
            "completed_batch_filter": (
                "csr_queue_batch_execution_summary.status == passed; "
                "batch_outcome may be profiled_with_gpu_success or screen_only_no_oracle"
            ),
        },
        "outputs": {
            "selector_rows": "csr_queue_training_pool_selector_rows.jsonl",
            "sources": "csr_queue_training_pool_sources.jsonl",
            "membership": "csr_queue_training_pool_membership.jsonl",
            "state": "csr_queue_training_pool_state.json",
        },
        "resume": {
            "completed_batch_ids": summary["completed_batch_ids"],
            "next_pending_batch_id": summary["next_pending_batch_id"],
        },
    }


def _state(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": summary["schema_version"],
        "state_kind": "csr_queue_training_pool_state",
        "training_pool_ready": summary["training_pool_ready"],
        "runtime_selector_changed": False,
        "completed_batch_ids": summary["completed_batch_ids"],
        "queue_batch_screen_only_batches": summary["queue_batch_screen_only_batches"],
        "next_pending_batch_id": summary["next_pending_batch_id"],
        "selector_rows": summary["selector_rows"],
        "next_step": summary["next_step"],
    }


def _report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# CSR Queue Training Pool",
            "",
            f"- status: `{summary['status']}`",
            f"- schema_version: `{summary['schema_version']}`",
            f"- training_pool_ready: `{summary['training_pool_ready']}`",
            f"- runtime_selector_changed: `{summary['runtime_selector_changed']}`",
            f"- executes_gpu: `{summary['executes_gpu']}`",
            f"- imports_matrices: `{summary['imports_matrices']}`",
            f"- sources: `{summary['source_count']}`",
            f"- completed_queue_batches: `{summary['completed_queue_batches']}`",
            f"- completed_batch_ids: `{summary['completed_batch_ids']}`",
            f"- queue_batch_oracle_batches: `{summary['queue_batch_oracle_batches']}`",
            f"- queue_batch_screen_only_batches: `{summary['queue_batch_screen_only_batches']}`",
            f"- next_pending_batch_id: `{summary['next_pending_batch_id']}`",
            f"- selector_rows: `{summary['selector_rows']}`",
            f"- queue_batch_selector_rows: `{summary['queue_batch_selector_rows']}`",
            f"- matrices: `{summary['matrices']}`",
            f"- success_rows: `{summary['success_rows']}`",
            f"- screened_out_rows: `{summary['screened_out_rows']}`",
            f"- oracle_rows: `{summary['oracle_rows']}`",
            f"- next_step: `{summary['next_step']}`",
            "",
        ]
    )


def _count_status(rows: Iterable[dict[str, Any]], status: str) -> int:
    return sum(1 for row in rows if str(row.get("target_status")) == status)


def _write_json(data: dict[str, Any], path: Path) -> Path:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
