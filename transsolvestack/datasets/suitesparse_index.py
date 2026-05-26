"""SuiteSparse Matrix Collection archive indexing.

The indexer is intentionally metadata-only. It aligns SuiteSparse's
``ssstats.csv`` rows, the Matrix Market URL queue, and the local ``.tar.gz``
archives without extracting matrix payloads or hashing the full dataset.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import unquote, urlparse

from transsolvestack.profiling.artifacts import write_jsonl


SSTATS_COLUMN_COUNT = 13


@dataclass(frozen=True)
class SuiteSparseStatsRow:
    group: str
    name: str
    n_rows: int
    n_cols: int
    nnz: int
    is_real: bool
    is_binary: bool
    is_nd: bool
    is_pos_def: bool
    pattern_symmetry: float
    numerical_symmetry: float
    kind: str
    nnzdiag: int

    @property
    def matrix_id(self) -> str:
        return f"suitesparse:{self.group}/{self.name}"


@dataclass(frozen=True)
class SuiteSparseStats:
    expected_matrices: int
    source_timestamp: str
    rows: tuple[SuiteSparseStatsRow, ...]


@dataclass(frozen=True)
class SuiteSparseIndexEntry:
    matrix_id: str
    source: str
    group: str
    name: str
    url: str | None
    file_format: str
    n_rows: int
    n_cols: int
    nnz: int
    is_real: bool
    is_binary: bool
    is_nd: bool
    is_pos_def: bool
    pattern_symmetry: float
    numerical_symmetry: float
    kind: str
    nnzdiag: int
    local_path: str
    archive_size_bytes: int | None
    download_present: bool
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class SuiteSparseIndexSummary:
    status: str
    stats_path: str
    urls_path: str
    mm_root: str
    expected_matrices: int
    stats_rows: int
    indexed_matrices: int
    expected_archive_files: int
    present_archive_files: int
    actual_archive_files: int
    missing_archive_files: int
    missing_url_entries: int
    extra_archive_files: int
    total_archive_size_bytes: int
    stats_source_timestamp: str
    missing_matrix_ids: tuple[str, ...]
    extra_archive_paths: tuple[str, ...]

    @property
    def total_archive_size_gb(self) -> float:
        return self.total_archive_size_bytes / 1_000_000_000


@dataclass(frozen=True)
class SuiteSparseIndex:
    entries: tuple[SuiteSparseIndexEntry, ...]
    summary: SuiteSparseIndexSummary


def parse_suitesparse_stats(path: str | Path) -> SuiteSparseStats:
    stats_path = Path(path)
    with stats_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            expected_matrices = int(next(reader)[0])
            source_timestamp = next(reader)[0]
        except StopIteration as exc:
            raise ValueError(f"SuiteSparse stats file is incomplete: {stats_path}") from exc

        rows: list[SuiteSparseStatsRow] = []
        for line_number, row in enumerate(reader, start=3):
            if not row:
                continue
            if len(row) != SSTATS_COLUMN_COUNT:
                raise ValueError(
                    f"SuiteSparse stats row {line_number} has {len(row)} columns; "
                    f"expected {SSTATS_COLUMN_COUNT}"
                )
            rows.append(_stats_row_from_csv(row))

    if len(rows) != expected_matrices:
        raise ValueError(
            f"SuiteSparse stats expected {expected_matrices} rows but found {len(rows)}"
        )

    return SuiteSparseStats(
        expected_matrices=expected_matrices,
        source_timestamp=source_timestamp,
        rows=tuple(rows),
    )


def load_suitesparse_mm_urls(path: str | Path) -> dict[tuple[str, str], str]:
    urls: dict[tuple[str, str], str] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            url = line.strip()
            if not url:
                continue
            group, name = _group_name_from_mm_url(url, line_number)
            urls[(group, name)] = url
    return urls


def build_suitesparse_index(
    stats_path: str | Path,
    urls_path: str | Path,
    mm_root: str | Path,
) -> SuiteSparseIndex:
    stats = parse_suitesparse_stats(stats_path)
    urls = load_suitesparse_mm_urls(urls_path)
    matrix_keys = {(row.group, row.name) for row in stats.rows}
    actual_archives = _archive_paths(Path(mm_root))

    entries: list[SuiteSparseIndexEntry] = []
    missing_matrix_ids: list[str] = []
    missing_url_entries = 0
    total_archive_size_bytes = 0
    expected_archive_paths: set[Path] = set()

    for row in stats.rows:
        archive_path = Path(mm_root) / row.group / f"{row.name}.tar.gz"
        expected_archive_paths.add(archive_path)
        present = archive_path.is_file()
        archive_size_bytes = archive_path.stat().st_size if present else None
        if archive_size_bytes is not None:
            total_archive_size_bytes += archive_size_bytes
        if not present:
            missing_matrix_ids.append(row.matrix_id)
        url = urls.get((row.group, row.name))
        if url is None:
            missing_url_entries += 1
        entries.append(
            SuiteSparseIndexEntry(
                matrix_id=row.matrix_id,
                source="suitesparse",
                group=row.group,
                name=row.name,
                url=url,
                file_format="matrix_market_tar_gz",
                n_rows=row.n_rows,
                n_cols=row.n_cols,
                nnz=row.nnz,
                is_real=row.is_real,
                is_binary=row.is_binary,
                is_nd=row.is_nd,
                is_pos_def=row.is_pos_def,
                pattern_symmetry=row.pattern_symmetry,
                numerical_symmetry=row.numerical_symmetry,
                kind=row.kind,
                nnzdiag=row.nnzdiag,
                local_path=str(archive_path),
                archive_size_bytes=archive_size_bytes,
                download_present=present,
                tags=_tags_for_stats_row(row),
            )
        )

    extra_archive_paths = tuple(
        str(path)
        for path in sorted(actual_archives - expected_archive_paths, key=lambda p: str(p))
        if _group_name_from_archive_path(path) not in matrix_keys
    )
    status = "complete"
    if missing_matrix_ids or missing_url_entries:
        status = "incomplete"
    elif extra_archive_paths:
        status = "complete_with_extra_files"

    summary = SuiteSparseIndexSummary(
        status=status,
        stats_path=str(stats_path),
        urls_path=str(urls_path),
        mm_root=str(mm_root),
        expected_matrices=stats.expected_matrices,
        stats_rows=len(stats.rows),
        indexed_matrices=len(entries),
        expected_archive_files=len(expected_archive_paths),
        present_archive_files=sum(1 for entry in entries if entry.download_present),
        actual_archive_files=len(actual_archives),
        missing_archive_files=len(missing_matrix_ids),
        missing_url_entries=missing_url_entries,
        extra_archive_files=len(extra_archive_paths),
        total_archive_size_bytes=total_archive_size_bytes,
        stats_source_timestamp=stats.source_timestamp,
        missing_matrix_ids=tuple(missing_matrix_ids),
        extra_archive_paths=extra_archive_paths,
    )
    return SuiteSparseIndex(entries=tuple(entries), summary=summary)


def write_suitesparse_index(index: SuiteSparseIndex, path: str | Path) -> Path:
    return write_jsonl((asdict(entry) for entry in index.entries), path)


def write_suitesparse_summary(
    summary: SuiteSparseIndexSummary,
    path: str | Path,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


def write_suitesparse_report(
    summary: SuiteSparseIndexSummary,
    path: str | Path,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# SuiteSparse Full Index",
        "",
        f"- status: `{summary.status}`",
        f"- stats_rows: `{summary.stats_rows}`",
        f"- expected_matrices: `{summary.expected_matrices}`",
        f"- indexed_matrices: `{summary.indexed_matrices}`",
        f"- present_archive_files: `{summary.present_archive_files}`",
        f"- actual_archive_files: `{summary.actual_archive_files}`",
        f"- missing_archive_files: `{summary.missing_archive_files}`",
        f"- missing_url_entries: `{summary.missing_url_entries}`",
        f"- extra_archive_files: `{summary.extra_archive_files}`",
        f"- total_archive_size_gb: `{summary.total_archive_size_gb:.3f}`",
        f"- stats_source_timestamp: `{summary.stats_source_timestamp}`",
        f"- mm_root: `{summary.mm_root}`",
        "",
    ]
    if summary.missing_matrix_ids:
        lines.extend(["## Missing Archives", ""])
        for matrix_id in summary.missing_matrix_ids[:50]:
            lines.append(f"- `{matrix_id}`")
        if len(summary.missing_matrix_ids) > 50:
            lines.append(f"- ... {len(summary.missing_matrix_ids) - 50} more")
        lines.append("")
    if summary.extra_archive_paths:
        lines.extend(["## Extra Archives", ""])
        for archive_path in summary.extra_archive_paths[:50]:
            lines.append(f"- `{archive_path}`")
        if len(summary.extra_archive_paths) > 50:
            lines.append(f"- ... {len(summary.extra_archive_paths) - 50} more")
        lines.append("")
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def read_suitesparse_index(path: str | Path) -> tuple[SuiteSparseIndexEntry, ...]:
    entries: list[SuiteSparseIndexEntry] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            data = json.loads(stripped)
            data["tags"] = tuple(data.get("tags", ()))
            entries.append(SuiteSparseIndexEntry(**data))
    return tuple(entries)


def _stats_row_from_csv(row: list[str]) -> SuiteSparseStatsRow:
    return SuiteSparseStatsRow(
        group=row[0],
        name=row[1],
        n_rows=int(row[2]),
        n_cols=int(row[3]),
        nnz=int(row[4]),
        is_real=_csv_bool(row[5]),
        is_binary=_csv_bool(row[6]),
        is_nd=_csv_bool(row[7]),
        is_pos_def=_csv_bool(row[8]),
        pattern_symmetry=float(row[9]),
        numerical_symmetry=float(row[10]),
        kind=row[11],
        nnzdiag=int(row[12]),
    )


def _csv_bool(value: str) -> bool:
    if value == "1":
        return True
    if value == "0":
        return False
    raise ValueError(f"expected SuiteSparse boolean column to be 0 or 1, got {value!r}")


def _group_name_from_mm_url(url: str, line_number: int) -> tuple[str, str]:
    parts = [unquote(part) for part in Path(urlparse(url).path).parts]
    if len(parts) < 3 or parts[-3] != "MM" or not parts[-1].endswith(".tar.gz"):
        raise ValueError(f"invalid SuiteSparse Matrix Market URL at line {line_number}: {url}")
    return parts[-2], parts[-1][: -len(".tar.gz")]


def _archive_paths(mm_root: Path) -> set[Path]:
    if not mm_root.exists():
        return set()
    return {path for path in mm_root.glob("*/*.tar.gz") if path.is_file()}


def _group_name_from_archive_path(path: Path) -> tuple[str, str]:
    return path.parent.name, path.name[: -len(".tar.gz")]


def _tags_for_stats_row(row: SuiteSparseStatsRow) -> tuple[str, ...]:
    tags: list[str] = []
    if row.n_rows == row.n_cols:
        tags.append("square")
    else:
        tags.append("rectangular")
    if row.is_real:
        tags.append("real")
    if row.is_binary:
        tags.append("binary")
    if row.is_pos_def:
        tags.append("pos_def")
    if row.pattern_symmetry >= 1.0:
        tags.append("pattern_symmetric")
    if row.numerical_symmetry >= 1.0:
        tags.append("numerically_symmetric")
    return tuple(tags)
