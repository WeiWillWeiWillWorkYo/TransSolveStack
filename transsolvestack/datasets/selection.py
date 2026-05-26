"""Matrix subset selection for real-matrix benchmark scaleout."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from transsolvestack.core.types import LinearOperatorSpec, LinearSystem
from transsolvestack.datasets.suitesparse_index import (
    SuiteSparseIndexEntry,
    read_suitesparse_index,
)
from transsolvestack.profiling.artifacts import write_jsonl


@dataclass(frozen=True)
class MatrixSelectionCriteria:
    max_matrices: int = 64
    max_per_kind: int = 4
    min_rows: int = 1
    max_rows: int | None = 250_000
    max_cols: int | None = 250_000
    max_nnz: int | None = 5_000_000
    max_archive_size_bytes: int | None = 512_000_000
    require_present: bool = True
    require_real: bool = True
    require_square: bool = True
    require_positive_definite: bool | None = None
    allowed_kinds: tuple[str, ...] = ()
    denied_kinds: tuple[str, ...] = ()


@dataclass(frozen=True)
class SelectedMatrixRecord:
    selection_rank: int
    matrix_id: str
    source: str
    group: str
    name: str
    url: str | None
    local_path: str
    file_format: str
    n_rows: int
    n_cols: int
    nnz: int
    kind: str
    size_bucket: str
    symmetry_class: str
    archive_size_bytes: int | None
    is_real: bool
    is_binary: bool
    is_pos_def: bool
    pattern_symmetry: float
    numerical_symmetry: float
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MatrixSelectionSummary:
    source_index_path: str | None
    status: str
    candidates_considered: int
    candidates_after_filter: int
    selected_matrices: int
    criteria: dict[str, Any]
    by_kind: dict[str, int]
    by_size_bucket: dict[str, int]
    total_selected_archive_size_bytes: int

    @property
    def total_selected_archive_size_gb(self) -> float:
        return self.total_selected_archive_size_bytes / 1_000_000_000


@dataclass(frozen=True)
class MatrixSelection:
    records: tuple[SelectedMatrixRecord, ...]
    summary: MatrixSelectionSummary


def select_matrix_subset(
    entries: Iterable[SuiteSparseIndexEntry],
    criteria: MatrixSelectionCriteria | None = None,
    *,
    source_index_path: str | Path | None = None,
) -> MatrixSelection:
    active_criteria = criteria or MatrixSelectionCriteria()
    all_entries = tuple(entries)
    filtered = tuple(
        entry for entry in all_entries if _entry_matches(entry, active_criteria)
    )
    selected_entries = _stratified_select(filtered, active_criteria)
    records = tuple(
        _record_from_entry(rank=index + 1, entry=entry)
        for index, entry in enumerate(selected_entries)
    )
    status = "ready" if records else "empty"
    by_kind = dict(Counter(record.kind for record in records))
    by_size_bucket = dict(Counter(record.size_bucket for record in records))
    total_bytes = sum(record.archive_size_bytes or 0 for record in records)
    return MatrixSelection(
        records=records,
        summary=MatrixSelectionSummary(
            source_index_path=str(source_index_path) if source_index_path else None,
            status=status,
            candidates_considered=len(all_entries),
            candidates_after_filter=len(filtered),
            selected_matrices=len(records),
            criteria=asdict(active_criteria),
            by_kind=by_kind,
            by_size_bucket=by_size_bucket,
            total_selected_archive_size_bytes=total_bytes,
        ),
    )


def select_matrix_subset_from_index(
    index_path: str | Path,
    criteria: MatrixSelectionCriteria | None = None,
) -> MatrixSelection:
    return select_matrix_subset(
        read_suitesparse_index(index_path),
        criteria=criteria,
        source_index_path=index_path,
    )


def build_external_matrix_systems(
    records: Iterable[SelectedMatrixRecord],
    *,
    dtype: str = "float32",
) -> tuple[LinearSystem, ...]:
    systems: list[LinearSystem] = []
    for record in records:
        operator = LinearOperatorSpec(
            operator_id=f"external:{record.matrix_id}",
            kind="assembled_sparse",
            shape=(record.n_rows, record.n_cols),
            dtype=dtype,
            symmetry=record.symmetry_class,
            device_resident=False,
            metadata={
                "source": record.source,
                "group": record.group,
                "name": record.name,
                "url": record.url,
                "local_path": record.local_path,
                "file_format": record.file_format,
                "nnz": record.nnz,
                "matrix_kind": record.kind,
                "size_bucket": record.size_bucket,
                "archive_size_bytes": record.archive_size_bytes,
                "import_status": "archive_indexed_not_loaded",
            },
        )
        systems.append(
            LinearSystem(
                system_id=record.matrix_id,
                operator=operator,
                family_id=f"external:suitesparse:{record.kind}",
                metadata={
                    "selection_rank": record.selection_rank,
                    "tags": record.tags,
                    "is_real": record.is_real,
                    "is_binary": record.is_binary,
                    "is_pos_def": record.is_pos_def,
                    "pattern_symmetry": record.pattern_symmetry,
                    "numerical_symmetry": record.numerical_symmetry,
                },
            )
        )
    return tuple(systems)


def write_matrix_selection(selection: MatrixSelection, path: str | Path) -> Path:
    return write_jsonl((asdict(record) for record in selection.records), path)


def write_external_matrix_systems(
    systems: tuple[LinearSystem, ...],
    path: str | Path,
) -> Path:
    return write_jsonl((_linear_system_record(system) for system in systems), path)


def write_matrix_selection_summary(
    summary: MatrixSelectionSummary,
    path: str | Path,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


def write_matrix_selection_report(
    selection: MatrixSelection,
    path: str | Path,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = selection.summary
    lines = [
        "# SuiteSparse Matrix Subset Selection",
        "",
        f"- status: `{summary.status}`",
        f"- candidates_considered: `{summary.candidates_considered}`",
        f"- candidates_after_filter: `{summary.candidates_after_filter}`",
        f"- selected_matrices: `{summary.selected_matrices}`",
        f"- total_selected_archive_size_gb: `{summary.total_selected_archive_size_gb:.3f}`",
        f"- source_index_path: `{summary.source_index_path}`",
        "",
        "## Size Buckets",
        "",
        "| bucket | count |",
        "|---|---:|",
    ]
    for bucket, count in sorted(summary.by_size_bucket.items()):
        lines.append(f"| {bucket} | {count} |")
    lines.extend(
        [
            "",
            "## Kinds",
            "",
            "| kind | count |",
            "|---|---:|",
        ]
    )
    for kind, count in sorted(summary.by_kind.items()):
        lines.append(f"| {kind} | {count} |")
    lines.extend(
        [
            "",
            "## Selected Matrices",
            "",
            "| rank | matrix | shape | nnz | bucket | kind | archive_mb |",
            "|---:|---|---:|---:|---|---|---:|",
        ]
    )
    for record in selection.records:
        archive_mb = (
            record.archive_size_bytes / 1_000_000
            if record.archive_size_bytes is not None
            else 0.0
        )
        lines.append(
            "| "
            f"{record.selection_rank} | "
            f"{record.matrix_id} | "
            f"{record.n_rows}x{record.n_cols} | "
            f"{record.nnz} | "
            f"{record.size_bucket} | "
            f"{record.kind} | "
            f"{archive_mb:.3f} |"
        )
    lines.append("")
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def _entry_matches(
    entry: SuiteSparseIndexEntry,
    criteria: MatrixSelectionCriteria,
) -> bool:
    if criteria.require_present and not entry.download_present:
        return False
    if criteria.require_real and not entry.is_real:
        return False
    if criteria.require_square and entry.n_rows != entry.n_cols:
        return False
    if criteria.require_positive_definite is not None:
        if entry.is_pos_def != criteria.require_positive_definite:
            return False
    if entry.n_rows < criteria.min_rows:
        return False
    if criteria.max_rows is not None and entry.n_rows > criteria.max_rows:
        return False
    if criteria.max_cols is not None and entry.n_cols > criteria.max_cols:
        return False
    if criteria.max_nnz is not None and entry.nnz > criteria.max_nnz:
        return False
    if (
        criteria.max_archive_size_bytes is not None
        and entry.archive_size_bytes is not None
        and entry.archive_size_bytes > criteria.max_archive_size_bytes
    ):
        return False
    if criteria.allowed_kinds and entry.kind not in criteria.allowed_kinds:
        return False
    if criteria.denied_kinds and entry.kind in criteria.denied_kinds:
        return False
    return True


def _stratified_select(
    entries: tuple[SuiteSparseIndexEntry, ...],
    criteria: MatrixSelectionCriteria,
) -> tuple[SuiteSparseIndexEntry, ...]:
    grouped: dict[str, list[SuiteSparseIndexEntry]] = defaultdict(list)
    for entry in entries:
        grouped[entry.kind or "unknown"].append(entry)
    for group_entries in grouped.values():
        group_entries.sort(key=_selection_sort_key)

    selected: list[SuiteSparseIndexEntry] = []
    kind_counts: Counter[str] = Counter()
    kinds = sorted(grouped)
    while len(selected) < criteria.max_matrices:
        progressed = False
        for kind in kinds:
            if len(selected) >= criteria.max_matrices:
                break
            if kind_counts[kind] >= criteria.max_per_kind:
                continue
            bucket = grouped[kind]
            if not bucket:
                continue
            selected.append(bucket.pop(0))
            kind_counts[kind] += 1
            progressed = True
        if not progressed:
            break
    return tuple(selected)


def _selection_sort_key(entry: SuiteSparseIndexEntry) -> tuple[Any, ...]:
    return (
        _size_bucket_rank(entry.n_rows, entry.nnz),
        entry.archive_size_bytes if entry.archive_size_bytes is not None else 0,
        entry.nnz,
        entry.n_rows,
        entry.group,
        entry.name,
    )


def _record_from_entry(
    rank: int,
    entry: SuiteSparseIndexEntry,
) -> SelectedMatrixRecord:
    return SelectedMatrixRecord(
        selection_rank=rank,
        matrix_id=entry.matrix_id,
        source=entry.source,
        group=entry.group,
        name=entry.name,
        url=entry.url,
        local_path=entry.local_path,
        file_format=entry.file_format,
        n_rows=entry.n_rows,
        n_cols=entry.n_cols,
        nnz=entry.nnz,
        kind=entry.kind or "unknown",
        size_bucket=_size_bucket(entry.n_rows, entry.nnz),
        symmetry_class=_symmetry_class(entry),
        archive_size_bytes=entry.archive_size_bytes,
        is_real=entry.is_real,
        is_binary=entry.is_binary,
        is_pos_def=entry.is_pos_def,
        pattern_symmetry=entry.pattern_symmetry,
        numerical_symmetry=entry.numerical_symmetry,
        tags=entry.tags,
        metadata={"nnz_per_row": entry.nnz / max(entry.n_rows, 1)},
    )


def _size_bucket(n_rows: int, nnz: int) -> str:
    rank = _size_bucket_rank(n_rows, nnz)
    if rank == 0:
        return "tiny"
    if rank == 1:
        return "small"
    if rank == 2:
        return "medium"
    return "large"


def _size_bucket_rank(n_rows: int, nnz: int) -> int:
    if n_rows <= 10_000 and nnz <= 100_000:
        return 0
    if n_rows <= 100_000 and nnz <= 1_000_000:
        return 1
    if n_rows <= 1_000_000 and nnz <= 10_000_000:
        return 2
    return 3


def _symmetry_class(entry: SuiteSparseIndexEntry) -> str:
    if entry.is_pos_def:
        return "spd"
    if entry.pattern_symmetry >= 1.0 and entry.numerical_symmetry >= 1.0:
        return "symmetric"
    if entry.pattern_symmetry > 0.0 or entry.numerical_symmetry > 0.0:
        return "partially_symmetric"
    return "unsymmetric"


def _linear_system_record(system: LinearSystem) -> dict[str, Any]:
    operator = system.operator
    return {
        "system_id": system.system_id,
        "family_id": system.family_id,
        "rhs_id": system.rhs_id,
        "operator": {
            "operator_id": operator.operator_id,
            "kind": operator.kind,
            "shape": operator.shape,
            "dtype": operator.dtype,
            "symmetry": operator.symmetry,
            "device_resident": operator.device_resident,
            "metadata": operator.metadata,
        },
        "metadata": system.metadata,
    }
