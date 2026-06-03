"""Recoverable full-dataset CSR benchmark/training queue planning."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from transsolvestack.datasets.suitesparse_index import (
    SuiteSparseIndexEntry,
    read_suitesparse_index,
)
from transsolvestack.profiling.artifacts import read_jsonl, write_jsonl


CSR_FULL_DATASET_QUEUE_SCHEMA_VERSION = "phase1_csr_full_dataset_queue_v1"


@dataclass(frozen=True)
class CsrFullDatasetQueueMatrix:
    schema_version: str
    queue_id: str
    matrix_id: str
    queue_rank: int
    batch_id: str | None
    group: str
    name: str
    n_rows: int
    n_cols: int
    nnz: int
    kind: str
    size_bucket: str
    candidate_profile: str
    local_path: str
    archive_size_bytes: int
    already_profiled: bool
    queue_status: str
    skipped_reason: str | None


@dataclass(frozen=True)
class CsrFullDatasetQueueJob:
    schema_version: str
    queue_id: str
    job_id: str
    batch_id: str
    matrix_id: str
    candidate_id: str
    solver: str
    preconditioner: str
    precision: str
    solver_parameters: dict[str, Any]
    measurement_repeats: int
    max_iter: int
    tolerance_rel: float
    requires_import: bool
    requires_cpu_screen: bool
    planned_gpu_solve_attempts: int
    estimated_matvecs: int
    estimated_nnz_visits: int
    queue_status: str


@dataclass(frozen=True)
class CsrFullDatasetQueueBatch:
    schema_version: str
    queue_id: str
    batch_id: str
    batch_rank: int
    matrix_count: int
    job_count: int
    planned_gpu_solve_attempts: int
    total_nnz: int
    max_matrix_nnz: int
    total_archive_size_bytes: int
    queue_status: str
    resume_token: str


@dataclass(frozen=True)
class CsrFullDatasetQueueSummary:
    status: str
    schema_version: str
    queue_id: str
    source_index_path: str
    source_selector_rows_path: str
    resource_limits_path: str
    storage_paths_path: str
    runtime_selector_changed: bool
    executes_gpu: bool
    imports_matrices: bool
    resumable: bool
    external_drive_required: bool
    external_drive_path: str | None
    index_matrices: int
    index_present_archives: int
    already_profiled_matrices: int
    eligible_matrices: int
    queued_matrices: int
    queued_jobs: int
    queued_batches: int
    planned_gpu_solve_attempts: int
    measurement_repeats: int
    batch_matrix_count: int
    max_rows: int
    max_cols: int
    max_nnz: int
    max_archive_size_bytes: int
    max_iter: int
    tolerance_rel: float
    total_queued_nnz: int
    estimated_total_nnz_visits: int
    by_candidate_profile: dict[str, int]
    by_solver: dict[str, int]
    skipped_non_real_matrices: int
    skipped_non_square_matrices: int
    skipped_missing_archives: int
    skipped_resource_limit_matrices: int
    first_pending_batch_id: str | None
    next_step: str


@dataclass(frozen=True)
class CsrFullDatasetQueue:
    matrix_rows: tuple[CsrFullDatasetQueueMatrix, ...]
    job_rows: tuple[CsrFullDatasetQueueJob, ...]
    batch_rows: tuple[CsrFullDatasetQueueBatch, ...]
    state: dict[str, Any]
    summary: CsrFullDatasetQueueSummary
    schema: dict[str, Any]


def build_csr_full_dataset_queue_from_files(
    index_path: str | Path,
    selector_rows_path: str | Path,
    resource_limits_path: str | Path,
    storage_paths_path: str | Path,
    *,
    queue_id: str = "phase1_csr_full_dataset_queue_m83",
    batch_matrix_count: int = 8,
    measurement_repeats: int = 1,
    max_rows: int | None = None,
    max_cols: int | None = None,
    max_nnz: int | None = None,
    max_archive_size_bytes: int = 256_000_000,
    max_iter: int | None = None,
    tolerance_rel: float = 1.0e-5,
    max_queue_matrices: int | None = None,
) -> CsrFullDatasetQueue:
    entries = read_suitesparse_index(index_path)
    selector_rows = tuple(read_jsonl(selector_rows_path))
    resource_limits = _read_yaml_like(resource_limits_path)
    storage_paths = _read_yaml_like(storage_paths_path)
    limits = dict(resource_limits.get("limits", resource_limits))
    active_max_rows = int(max_rows or limits.get("max_problem_n", 65_536))
    active_max_cols = int(max_cols or limits.get("max_problem_n", 65_536))
    active_max_nnz = int(max_nnz or limits.get("max_effective_nnz", 1_000_000))
    active_max_iter = int(max_iter or limits.get("max_default_iterations", 300))
    _validate_limits(
        limits,
        batch_matrix_count=batch_matrix_count,
        measurement_repeats=measurement_repeats,
        max_rows=active_max_rows,
        max_cols=active_max_cols,
        max_nnz=active_max_nnz,
        max_iter=active_max_iter,
    )
    return build_csr_full_dataset_queue(
        entries,
        selector_rows,
        storage_paths,
        source_index_path=str(index_path),
        source_selector_rows_path=str(selector_rows_path),
        resource_limits_path=str(resource_limits_path),
        storage_paths_path=str(storage_paths_path),
        queue_id=queue_id,
        batch_matrix_count=batch_matrix_count,
        measurement_repeats=measurement_repeats,
        max_rows=active_max_rows,
        max_cols=active_max_cols,
        max_nnz=active_max_nnz,
        max_archive_size_bytes=max_archive_size_bytes,
        max_iter=active_max_iter,
        tolerance_rel=tolerance_rel,
        max_queue_matrices=max_queue_matrices,
    )


def build_csr_full_dataset_queue(
    entries: Iterable[SuiteSparseIndexEntry],
    selector_rows: Iterable[dict[str, Any]],
    storage_paths: dict[str, Any],
    *,
    source_index_path: str,
    source_selector_rows_path: str,
    resource_limits_path: str,
    storage_paths_path: str,
    queue_id: str = "phase1_csr_full_dataset_queue_m83",
    batch_matrix_count: int = 8,
    measurement_repeats: int = 1,
    max_rows: int = 65_536,
    max_cols: int = 65_536,
    max_nnz: int = 1_000_000,
    max_archive_size_bytes: int = 256_000_000,
    max_iter: int = 300,
    tolerance_rel: float = 1.0e-5,
    max_queue_matrices: int | None = None,
) -> CsrFullDatasetQueue:
    all_entries = tuple(entries)
    profiled_ids = {str(row["matrix_id"]) for row in selector_rows}
    eligible_entries = tuple(
        entry
        for entry in all_entries
        if _is_eligible(
            entry,
            max_rows=max_rows,
            max_cols=max_cols,
            max_nnz=max_nnz,
            max_archive_size_bytes=max_archive_size_bytes,
        )
    )
    unprofiled_entries = tuple(
        entry for entry in eligible_entries if entry.matrix_id not in profiled_ids
    )
    queued_source = (
        unprofiled_entries[:max_queue_matrices]
        if max_queue_matrices is not None
        else unprofiled_entries
    )
    queued_ids = {entry.matrix_id for entry in queued_source}

    batch_by_matrix = _batch_assignments(queued_source, batch_matrix_count=batch_matrix_count)
    matrix_rows = tuple(
        _matrix_row(
            entry,
            queue_id=queue_id,
            queue_rank=index + 1,
            batch_id=batch_by_matrix.get(entry.matrix_id),
            already_profiled=entry.matrix_id in profiled_ids,
            queued=entry.matrix_id in queued_ids,
            skipped_reason=_skipped_reason(
                entry,
                profiled_ids=profiled_ids,
                queued_ids=queued_ids,
                max_rows=max_rows,
                max_cols=max_cols,
                max_nnz=max_nnz,
                max_archive_size_bytes=max_archive_size_bytes,
            ),
        )
        for index, entry in enumerate(all_entries)
        if entry.matrix_id in queued_ids or entry.matrix_id in profiled_ids
    )
    job_rows = _job_rows(
        queued_source,
        queue_id=queue_id,
        batch_by_matrix=batch_by_matrix,
        measurement_repeats=measurement_repeats,
        max_iter=max_iter,
        tolerance_rel=tolerance_rel,
    )
    batch_rows = _batch_rows(
        queued_source,
        job_rows,
        queue_id=queue_id,
        batch_matrix_count=batch_matrix_count,
        batch_by_matrix=batch_by_matrix,
    )
    summary = _summary(
        all_entries=all_entries,
        eligible_entries=eligible_entries,
        profiled_ids=profiled_ids,
        queued_source=queued_source,
        job_rows=job_rows,
        batch_rows=batch_rows,
        storage_paths=storage_paths,
        source_index_path=source_index_path,
        source_selector_rows_path=source_selector_rows_path,
        resource_limits_path=resource_limits_path,
        storage_paths_path=storage_paths_path,
        queue_id=queue_id,
        batch_matrix_count=batch_matrix_count,
        measurement_repeats=measurement_repeats,
        max_rows=max_rows,
        max_cols=max_cols,
        max_nnz=max_nnz,
        max_archive_size_bytes=max_archive_size_bytes,
        max_iter=max_iter,
        tolerance_rel=tolerance_rel,
    )
    return CsrFullDatasetQueue(
        matrix_rows=matrix_rows,
        job_rows=job_rows,
        batch_rows=batch_rows,
        state=_state(summary, batch_rows),
        summary=summary,
        schema=_schema(summary),
    )


def write_csr_full_dataset_queue_matrices(
    rows: Iterable[CsrFullDatasetQueueMatrix],
    path: str | Path,
) -> Path:
    return write_jsonl((asdict(row) for row in rows), path)


def write_csr_full_dataset_queue_jobs(
    rows: Iterable[CsrFullDatasetQueueJob],
    path: str | Path,
) -> Path:
    return write_jsonl((asdict(row) for row in rows), path)


def write_csr_full_dataset_queue_batches(
    rows: Iterable[CsrFullDatasetQueueBatch],
    path: str | Path,
) -> Path:
    return write_jsonl((asdict(row) for row in rows), path)


def write_csr_full_dataset_queue_state(state: dict[str, Any], path: str | Path) -> Path:
    return _write_json(state, Path(path))


def write_csr_full_dataset_queue_summary(
    summary: CsrFullDatasetQueueSummary,
    path: str | Path,
) -> Path:
    return _write_json(asdict(summary), Path(path))


def write_csr_full_dataset_queue_schema(schema: dict[str, Any], path: str | Path) -> Path:
    return _write_json(schema, Path(path))


def write_csr_full_dataset_queue_report(
    queue: CsrFullDatasetQueue,
    path: str | Path,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = queue.summary
    lines = [
        "# CSR Full Dataset Queue",
        "",
        f"- status: `{summary.status}`",
        f"- schema_version: `{summary.schema_version}`",
        f"- queue_id: `{summary.queue_id}`",
        f"- runtime_selector_changed: `{summary.runtime_selector_changed}`",
        f"- executes_gpu: `{summary.executes_gpu}`",
        f"- imports_matrices: `{summary.imports_matrices}`",
        f"- resumable: `{summary.resumable}`",
        f"- external_drive_required: `{summary.external_drive_required}`",
        f"- external_drive_path: `{summary.external_drive_path}`",
        f"- index_matrices: `{summary.index_matrices}`",
        f"- index_present_archives: `{summary.index_present_archives}`",
        f"- already_profiled_matrices: `{summary.already_profiled_matrices}`",
        f"- eligible_matrices: `{summary.eligible_matrices}`",
        f"- queued_matrices: `{summary.queued_matrices}`",
        f"- queued_jobs: `{summary.queued_jobs}`",
        f"- queued_batches: `{summary.queued_batches}`",
        f"- planned_gpu_solve_attempts: `{summary.planned_gpu_solve_attempts}`",
        f"- total_queued_nnz: `{summary.total_queued_nnz}`",
        f"- estimated_total_nnz_visits: `{summary.estimated_total_nnz_visits}`",
        f"- by_candidate_profile: `{summary.by_candidate_profile}`",
        f"- by_solver: `{summary.by_solver}`",
        f"- first_pending_batch_id: `{summary.first_pending_batch_id}`",
        f"- next_step: `{summary.next_step}`",
        "",
        "## Batches",
        "",
        "| batch | matrices | jobs | gpu_attempts | total_nnz | archive_mb | status |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for batch in queue.batch_rows[:24]:
        lines.append(
            "| "
            f"{batch.batch_id} | "
            f"{batch.matrix_count} | "
            f"{batch.job_count} | "
            f"{batch.planned_gpu_solve_attempts} | "
            f"{batch.total_nnz} | "
            f"{batch.total_archive_size_bytes / 1_000_000:.3f} | "
            f"{batch.queue_status} |"
        )
    if len(queue.batch_rows) > 24:
        lines.append(f"| ... | {len(queue.batch_rows) - 24} more | | | | | |")
    lines.extend(
        [
            "",
            "## Execution Boundary",
            "",
            "- This artifact is a queue only.",
            "- Matrix import, CPU screening, and Taichi GPU solves remain separate explicit steps.",
            "- The default runtime selector is not changed by this queue.",
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def _matrix_row(
    entry: SuiteSparseIndexEntry,
    *,
    queue_id: str,
    queue_rank: int,
    batch_id: str | None,
    already_profiled: bool,
    queued: bool,
    skipped_reason: str | None,
) -> CsrFullDatasetQueueMatrix:
    return CsrFullDatasetQueueMatrix(
        schema_version=CSR_FULL_DATASET_QUEUE_SCHEMA_VERSION,
        queue_id=queue_id,
        matrix_id=entry.matrix_id,
        queue_rank=queue_rank,
        batch_id=batch_id,
        group=entry.group,
        name=entry.name,
        n_rows=entry.n_rows,
        n_cols=entry.n_cols,
        nnz=entry.nnz,
        kind=entry.kind,
        size_bucket=_size_bucket(entry.nnz),
        candidate_profile=_candidate_profile(entry),
        local_path=entry.local_path,
        archive_size_bytes=int(entry.archive_size_bytes or 0),
        already_profiled=already_profiled,
        queue_status=(
            "already_profiled" if already_profiled else "queued" if queued else "skipped"
        ),
        skipped_reason=skipped_reason,
    )


def _job_rows(
    entries: tuple[SuiteSparseIndexEntry, ...],
    *,
    queue_id: str,
    batch_by_matrix: dict[str, str],
    measurement_repeats: int,
    max_iter: int,
    tolerance_rel: float,
) -> tuple[CsrFullDatasetQueueJob, ...]:
    rows: list[CsrFullDatasetQueueJob] = []
    for entry in entries:
        for candidate in _candidate_templates(entry):
            job_id = _job_id(batch_by_matrix[entry.matrix_id], entry, candidate)
            solver = str(candidate["solver"])
            estimated_matvecs = _estimated_matvecs(solver, max_iter)
            rows.append(
                CsrFullDatasetQueueJob(
                    schema_version=CSR_FULL_DATASET_QUEUE_SCHEMA_VERSION,
                    queue_id=queue_id,
                    job_id=job_id,
                    batch_id=batch_by_matrix[entry.matrix_id],
                    matrix_id=entry.matrix_id,
                    candidate_id=str(candidate["candidate_id"]),
                    solver=solver,
                    preconditioner=str(candidate["preconditioner"]),
                    precision="float64",
                    solver_parameters=dict(candidate.get("solver_parameters", {})),
                    measurement_repeats=measurement_repeats,
                    max_iter=max_iter,
                    tolerance_rel=tolerance_rel,
                    requires_import=True,
                    requires_cpu_screen=True,
                    planned_gpu_solve_attempts=measurement_repeats,
                    estimated_matvecs=estimated_matvecs,
                    estimated_nnz_visits=(
                        entry.nnz * estimated_matvecs * measurement_repeats
                    ),
                    queue_status="pending_import",
                )
            )
    return tuple(rows)


def _batch_rows(
    entries: tuple[SuiteSparseIndexEntry, ...],
    job_rows: tuple[CsrFullDatasetQueueJob, ...],
    *,
    queue_id: str,
    batch_matrix_count: int,
    batch_by_matrix: dict[str, str],
) -> tuple[CsrFullDatasetQueueBatch, ...]:
    entries_by_batch: dict[str, list[SuiteSparseIndexEntry]] = {}
    for entry in entries:
        entries_by_batch.setdefault(batch_by_matrix[entry.matrix_id], []).append(entry)
    jobs_by_batch = Counter(row.batch_id for row in job_rows)
    attempts_by_batch = Counter()
    for row in job_rows:
        attempts_by_batch[row.batch_id] += row.planned_gpu_solve_attempts
    rows = []
    for rank, batch_id in enumerate(sorted(entries_by_batch), start=1):
        batch_entries = entries_by_batch[batch_id]
        rows.append(
            CsrFullDatasetQueueBatch(
                schema_version=CSR_FULL_DATASET_QUEUE_SCHEMA_VERSION,
                queue_id=queue_id,
                batch_id=batch_id,
                batch_rank=rank,
                matrix_count=len(batch_entries),
                job_count=jobs_by_batch[batch_id],
                planned_gpu_solve_attempts=attempts_by_batch[batch_id],
                total_nnz=sum(entry.nnz for entry in batch_entries),
                max_matrix_nnz=max(entry.nnz for entry in batch_entries),
                total_archive_size_bytes=sum(
                    int(entry.archive_size_bytes or 0) for entry in batch_entries
                ),
                queue_status="pending",
                resume_token=f"{queue_id}:{batch_id}:pending",
            )
        )
    return tuple(rows)


def _summary(
    *,
    all_entries: tuple[SuiteSparseIndexEntry, ...],
    eligible_entries: tuple[SuiteSparseIndexEntry, ...],
    profiled_ids: set[str],
    queued_source: tuple[SuiteSparseIndexEntry, ...],
    job_rows: tuple[CsrFullDatasetQueueJob, ...],
    batch_rows: tuple[CsrFullDatasetQueueBatch, ...],
    storage_paths: dict[str, Any],
    source_index_path: str,
    source_selector_rows_path: str,
    resource_limits_path: str,
    storage_paths_path: str,
    queue_id: str,
    batch_matrix_count: int,
    measurement_repeats: int,
    max_rows: int,
    max_cols: int,
    max_nnz: int,
    max_archive_size_bytes: int,
    max_iter: int,
    tolerance_rel: float,
) -> CsrFullDatasetQueueSummary:
    eligible_ids = {entry.matrix_id for entry in eligible_entries}
    queued_ids = {entry.matrix_id for entry in queued_source}
    total_nnz = sum(entry.nnz for entry in queued_source)
    return CsrFullDatasetQueueSummary(
        status="passed" if queued_source and job_rows and batch_rows else "failed",
        schema_version=CSR_FULL_DATASET_QUEUE_SCHEMA_VERSION,
        queue_id=queue_id,
        source_index_path=source_index_path,
        source_selector_rows_path=source_selector_rows_path,
        resource_limits_path=resource_limits_path,
        storage_paths_path=storage_paths_path,
        runtime_selector_changed=False,
        executes_gpu=False,
        imports_matrices=False,
        resumable=True,
        external_drive_required=bool(
            storage_paths.get("policy", {}).get("use_external_drive_for_large_tss_data")
        ),
        external_drive_path=str(
            storage_paths.get("external_drive", {}).get("expected_mountpoint")
            or storage_paths.get("data_root")
            or ""
        ),
        index_matrices=len(all_entries),
        index_present_archives=sum(1 for entry in all_entries if entry.download_present),
        already_profiled_matrices=len(profiled_ids & {entry.matrix_id for entry in all_entries}),
        eligible_matrices=len(eligible_entries),
        queued_matrices=len(queued_source),
        queued_jobs=len(job_rows),
        queued_batches=len(batch_rows),
        planned_gpu_solve_attempts=sum(row.planned_gpu_solve_attempts for row in job_rows),
        measurement_repeats=measurement_repeats,
        batch_matrix_count=batch_matrix_count,
        max_rows=max_rows,
        max_cols=max_cols,
        max_nnz=max_nnz,
        max_archive_size_bytes=max_archive_size_bytes,
        max_iter=max_iter,
        tolerance_rel=tolerance_rel,
        total_queued_nnz=total_nnz,
        estimated_total_nnz_visits=sum(row.estimated_nnz_visits for row in job_rows),
        by_candidate_profile=dict(Counter(_candidate_profile(entry) for entry in queued_source)),
        by_solver=dict(Counter(row.solver for row in job_rows)),
        skipped_non_real_matrices=sum(1 for entry in all_entries if not entry.is_real),
        skipped_non_square_matrices=sum(1 for entry in all_entries if entry.n_rows != entry.n_cols),
        skipped_missing_archives=sum(1 for entry in all_entries if not entry.download_present),
        skipped_resource_limit_matrices=sum(
            1
            for entry in all_entries
            if entry.matrix_id not in eligible_ids
            and entry.matrix_id not in queued_ids
            and entry.is_real
            and entry.n_rows == entry.n_cols
            and entry.download_present
        ),
        first_pending_batch_id=batch_rows[0].batch_id if batch_rows else None,
        next_step="execute_queue_batches_with_explicit_import_cpu_screen_and_taichi_gpu_benchmark",
    )


def _state(
    summary: CsrFullDatasetQueueSummary,
    batch_rows: tuple[CsrFullDatasetQueueBatch, ...],
) -> dict[str, Any]:
    return {
        "schema_version": summary.schema_version,
        "queue_id": summary.queue_id,
        "state_kind": "csr_full_dataset_queue_resume_state",
        "current_status": summary.status,
        "runtime_selector_changed": False,
        "executes_gpu": False,
        "imports_matrices": False,
        "resume_policy": {
            "unit": "batch",
            "first_pending_batch_id": summary.first_pending_batch_id,
            "completed_batch_ids": (),
            "failed_batch_ids": (),
            "pending_batch_ids": tuple(row.batch_id for row in batch_rows),
            "retry_failed_batches_first": True,
        },
        "execution_requires": (
            "external_drive_mounted",
            "explicit_user_start_for_gpu_benchmark",
            "cpu_screen_before_gpu_solve",
            "append_only_selector_rows",
        ),
        "failure_policy": (
            "record_failed_job",
            "continue_next_job_in_batch",
            "do_not_change_runtime_selector",
            "do_not_delete_previous_selector_rows",
        ),
    }


def _schema(summary: CsrFullDatasetQueueSummary) -> dict[str, Any]:
    return {
        "schema_version": summary.schema_version,
        "task": "plan_recoverable_full_suitesparse_csr_training_queue",
        "runtime_selector_changed": False,
        "executes_gpu": False,
        "imports_matrices": False,
        "queue_files": {
            "matrices": "csr_full_dataset_queue_matrices.jsonl",
            "jobs": "csr_full_dataset_queue_jobs.jsonl",
            "batches": "csr_full_dataset_queue_batches.jsonl",
            "state": "csr_full_dataset_queue_state.json",
        },
        "execution_boundary": {
            "automatic_execution": False,
            "requires_explicit_gpu_benchmark_start": True,
            "requires_cpu_screen": True,
            "append_only_training_rows": True,
        },
        "resource_limits": {
            "batch_matrix_count": summary.batch_matrix_count,
            "measurement_repeats": summary.measurement_repeats,
            "max_rows": summary.max_rows,
            "max_cols": summary.max_cols,
            "max_nnz": summary.max_nnz,
            "max_archive_size_bytes": summary.max_archive_size_bytes,
            "max_iter": summary.max_iter,
        },
    }


def _batch_assignments(
    entries: tuple[SuiteSparseIndexEntry, ...],
    *,
    batch_matrix_count: int,
) -> dict[str, str]:
    return {
        entry.matrix_id: f"batch_{index // batch_matrix_count + 1:05d}"
        for index, entry in enumerate(entries)
    }


def _is_eligible(
    entry: SuiteSparseIndexEntry,
    *,
    max_rows: int,
    max_cols: int,
    max_nnz: int,
    max_archive_size_bytes: int,
) -> bool:
    return (
        entry.download_present
        and entry.is_real
        and entry.n_rows == entry.n_cols
        and entry.n_rows <= max_rows
        and entry.n_cols <= max_cols
        and entry.nnz <= max_nnz
        and int(entry.archive_size_bytes or 0) <= max_archive_size_bytes
        and Path(entry.local_path).exists()
    )


def _skipped_reason(
    entry: SuiteSparseIndexEntry,
    *,
    profiled_ids: set[str],
    queued_ids: set[str],
    max_rows: int,
    max_cols: int,
    max_nnz: int,
    max_archive_size_bytes: int,
) -> str | None:
    if entry.matrix_id in profiled_ids:
        return "already_profiled"
    if entry.matrix_id in queued_ids:
        return None
    if not entry.download_present or not Path(entry.local_path).exists():
        return "missing_archive"
    if not entry.is_real:
        return "non_real"
    if entry.n_rows != entry.n_cols:
        return "non_square"
    if (
        entry.n_rows > max_rows
        or entry.n_cols > max_cols
        or entry.nnz > max_nnz
        or int(entry.archive_size_bytes or 0) > max_archive_size_bytes
    ):
        return "resource_limit"
    return "queue_limit"


def _candidate_templates(entry: SuiteSparseIndexEntry) -> tuple[dict[str, Any], ...]:
    if _candidate_profile(entry) == "symmetric":
        return (
            {
                "candidate_id": "taichi_csr_cg_none_float64",
                "solver": "cg",
                "preconditioner": "none",
            },
            {
                "candidate_id": "taichi_csr_pcg_jacobi_float64",
                "solver": "pcg",
                "preconditioner": "jacobi",
            },
            {
                "candidate_id": "taichi_csr_richardson_jacobi_float64",
                "solver": "richardson",
                "preconditioner": "jacobi",
                "solver_parameters": {"omega_source": "cpu_screen_required"},
            },
        )
    return (
        {
            "candidate_id": "taichi_csr_bicgstab_none_float64",
            "solver": "bicgstab",
            "preconditioner": "none",
        },
        {
            "candidate_id": "taichi_csr_bicgstab_jacobi_float64",
            "solver": "bicgstab",
            "preconditioner": "jacobi",
        },
        {
            "candidate_id": "taichi_csr_gmres_jacobi_restart16_float64",
            "solver": "gmres",
            "preconditioner": "jacobi",
            "solver_parameters": {"restart": 16},
        },
    )


def _candidate_profile(entry: SuiteSparseIndexEntry) -> str:
    if entry.is_pos_def:
        return "symmetric"
    if entry.pattern_symmetry >= 0.999 and entry.numerical_symmetry >= 0.999:
        return "symmetric"
    return "general"


def _estimated_matvecs(solver: str, max_iter: int) -> int:
    if solver == "bicgstab":
        return max_iter * 2
    if solver == "gmres":
        return max_iter + max_iter // 16
    return max_iter


def _job_id(
    batch_id: str,
    entry: SuiteSparseIndexEntry,
    candidate: dict[str, Any],
) -> str:
    safe_matrix = entry.matrix_id.replace(":", "_").replace("/", "_")
    return f"{batch_id}:{safe_matrix}:{candidate['candidate_id']}"


def _size_bucket(nnz: int) -> str:
    if nnz < 10_000:
        return "tiny"
    if nnz < 100_000:
        return "small"
    if nnz < 1_000_000:
        return "medium"
    return "large"


def _validate_limits(
    limits: dict[str, Any],
    *,
    batch_matrix_count: int,
    measurement_repeats: int,
    max_rows: int,
    max_cols: int,
    max_nnz: int,
    max_iter: int,
) -> None:
    if batch_matrix_count < 1:
        raise ValueError("batch_matrix_count must be >= 1")
    if measurement_repeats < 1:
        raise ValueError("measurement_repeats must be >= 1")
    if bool(limits.get("allow_background_benchmark", False)):
        raise ValueError("full dataset queue expects background benchmark disabled")
    if max_rows > int(limits.get("max_problem_n", max_rows)):
        raise ValueError("max_rows exceeds resource limit")
    if max_cols > int(limits.get("max_problem_n", max_cols)):
        raise ValueError("max_cols exceeds resource limit")
    if max_nnz > int(limits.get("max_effective_nnz", max_nnz)):
        raise ValueError("max_nnz exceeds resource limit")
    if max_iter > int(limits.get("max_default_iterations", max_iter)):
        raise ValueError("max_iter exceeds resource limit")


def _read_yaml_like(path: str | Path) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
    except Exception:
        return _read_simple_yaml(text)
    return dict(yaml.safe_load(text))


def _read_simple_yaml(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for line in text.splitlines():
        if not line.strip() or line.strip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, raw_value = line.strip().split(":", 1)
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        value = raw_value.strip()
        if not value:
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(value)
    return root


def _parse_scalar(value: str) -> Any:
    value = value.strip().strip('"')
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _write_json(data: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
