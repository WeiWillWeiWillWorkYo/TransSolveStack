"""CPU-side CSR import boundary for Matrix Market archives.

This module is a data-loading boundary, not a solver backend. It provides a
small, deterministic CSR representation that later Taichi sparse operators can
consume after an explicit device-transfer path is implemented.
"""

from __future__ import annotations

import io
import json
import math
import tarfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, TextIO

from transsolvestack.datasets.matrix_market import MatrixMarketHeader
from transsolvestack.profiling.artifacts import read_jsonl, write_jsonl


SUPPORTED_FIELDS = {"real", "integer", "pattern"}
SUPPORTED_SYMMETRIES = {"general", "symmetric", "skew-symmetric", "hermitian"}


@dataclass(frozen=True)
class CsrMatrix:
    matrix_id: str
    n_rows: int
    n_cols: int
    row_ptr: tuple[int, ...]
    col_ind: tuple[int, ...]
    values: tuple[float, ...]
    field: str
    symmetry: str
    source_path: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def nnz(self) -> int:
        return len(self.values)

    def validate(self) -> None:
        if len(self.row_ptr) != self.n_rows + 1:
            raise ValueError("CSR row_ptr length must be n_rows + 1")
        if self.row_ptr[0] != 0:
            raise ValueError("CSR row_ptr must start at zero")
        if self.row_ptr[-1] != self.nnz:
            raise ValueError("CSR row_ptr final value must equal nnz")
        if len(self.col_ind) != self.nnz:
            raise ValueError("CSR col_ind length must equal nnz")
        if any(self.row_ptr[i] > self.row_ptr[i + 1] for i in range(self.n_rows)):
            raise ValueError("CSR row_ptr must be nondecreasing")
        if any(col < 0 or col >= self.n_cols for col in self.col_ind):
            raise ValueError("CSR column index out of bounds")
        if any(not math.isfinite(value) for value in self.values):
            raise ValueError("CSR values must be finite")

    def matvec(self, x: Iterable[float]) -> tuple[float, ...]:
        vector = tuple(float(value) for value in x)
        if len(vector) != self.n_cols:
            raise ValueError(f"matvec expected x length {self.n_cols}, got {len(vector)}")
        y: list[float] = []
        for row in range(self.n_rows):
            total = 0.0
            for offset in range(self.row_ptr[row], self.row_ptr[row + 1]):
                total += self.values[offset] * vector[self.col_ind[offset]]
            y.append(total)
        return tuple(y)


@dataclass(frozen=True)
class CsrImportRecord:
    selection_rank: int
    matrix_id: str
    source_archive_path: str
    member_path: str | None
    status: str
    error: str | None
    n_rows: int | None
    n_cols: int | None
    stored_entries: int | None
    csr_nnz: int | None
    field: str | None
    symmetry: str | None
    duplicate_entries_merged: int
    explicit_zeros_dropped: int
    expanded_entries_added: int
    row_ptr: tuple[int, ...]
    col_ind: tuple[int, ...]
    values: tuple[float, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CsrImportSummary:
    source_selection_path: str | None
    source_header_probe_path: str | None
    status: str
    eligible_rows: int
    attempted_imports: int
    imported_matrices: int
    failed_imports: int
    total_csr_nnz: int
    max_rows: int
    max_cols: int
    max_stored_entries: int
    max_archive_size_bytes: int


@dataclass(frozen=True)
class CsrImportBatch:
    records: tuple[CsrImportRecord, ...]
    summary: CsrImportSummary


def read_matrix_market_coordinate_csr(
    handle: TextIO,
    *,
    matrix_id: str,
    source_path: str,
    drop_explicit_zeros: bool = True,
) -> CsrMatrix:
    csr, _metadata = _read_matrix_market_coordinate_csr_with_metadata(
        handle,
        matrix_id=matrix_id,
        source_path=source_path,
        drop_explicit_zeros=drop_explicit_zeros,
    )
    return csr


def import_archive_matrix_market_csr(
    archive_path: str | Path,
    *,
    matrix_id: str,
    expected_name: str | None = None,
) -> tuple[str, CsrMatrix]:
    archive = Path(archive_path)
    with tarfile.open(archive, mode="r:gz") as handle:
        members = tuple(member for member in handle.getmembers() if member.isfile())
        member = _choose_matrix_member(members, expected_name=expected_name)
        if member is None:
            raise ValueError(f"archive has no Matrix Market .mtx member: {archive}")
        extracted = handle.extractfile(member)
        if extracted is None:
            raise ValueError(f"could not read archive member {member.name}: {archive}")
        with extracted:
            text = io.TextIOWrapper(extracted, encoding="utf-8")
            csr = read_matrix_market_coordinate_csr(
                text,
                matrix_id=matrix_id,
                source_path=f"{archive}!{member.name}",
            )
    return member.name, csr


def import_selected_archives_to_csr(
    selection_rows: Iterable[dict[str, Any]],
    *,
    header_probe_rows: Iterable[dict[str, Any]] = (),
    source_selection_path: str | Path | None = None,
    source_header_probe_path: str | Path | None = None,
    max_matrices: int = 12,
    max_rows: int = 10_000,
    max_cols: int = 10_000,
    max_stored_entries: int = 100_000,
    max_archive_size_bytes: int = 8_000_000,
) -> CsrImportBatch:
    header_by_id = {
        str(row["matrix_id"]): row for row in header_probe_rows if row.get("status") == "success"
    }
    eligible = tuple(
        row
        for row in selection_rows
        if _eligible_for_csr_import(
            row,
            header_by_id.get(str(row["matrix_id"])),
            max_rows=max_rows,
            max_cols=max_cols,
            max_stored_entries=max_stored_entries,
            max_archive_size_bytes=max_archive_size_bytes,
        )
    )
    selected = eligible[:max_matrices]
    records = tuple(_import_one(row) for row in selected)
    imported = sum(1 for record in records if record.status == "success")
    failed = len(records) - imported
    status = "passed" if records and failed == 0 else "failed"
    return CsrImportBatch(
        records=records,
        summary=CsrImportSummary(
            source_selection_path=str(source_selection_path)
            if source_selection_path
            else None,
            source_header_probe_path=str(source_header_probe_path)
            if source_header_probe_path
            else None,
            status=status,
            eligible_rows=len(eligible),
            attempted_imports=len(records),
            imported_matrices=imported,
            failed_imports=failed,
            total_csr_nnz=sum(record.csr_nnz or 0 for record in records),
            max_rows=max_rows,
            max_cols=max_cols,
            max_stored_entries=max_stored_entries,
            max_archive_size_bytes=max_archive_size_bytes,
        ),
    )


def import_selected_archives_to_csr_from_files(
    selection_path: str | Path,
    *,
    header_probe_path: str | Path | None = None,
    max_matrices: int = 12,
    max_rows: int = 10_000,
    max_cols: int = 10_000,
    max_stored_entries: int = 100_000,
    max_archive_size_bytes: int = 8_000_000,
) -> CsrImportBatch:
    header_rows = read_jsonl(header_probe_path) if header_probe_path else ()
    return import_selected_archives_to_csr(
        read_jsonl(selection_path),
        header_probe_rows=header_rows,
        source_selection_path=selection_path,
        source_header_probe_path=header_probe_path,
        max_matrices=max_matrices,
        max_rows=max_rows,
        max_cols=max_cols,
        max_stored_entries=max_stored_entries,
        max_archive_size_bytes=max_archive_size_bytes,
    )


def write_csr_import_records(batch: CsrImportBatch, path: str | Path) -> Path:
    return write_jsonl((asdict(record) for record in batch.records), path)


def write_csr_import_summary(summary: CsrImportSummary, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


def write_csr_import_report(batch: CsrImportBatch, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = batch.summary
    lines = [
        "# SuiteSparse CSR Import Boundary",
        "",
        f"- status: `{summary.status}`",
        f"- eligible_rows: `{summary.eligible_rows}`",
        f"- attempted_imports: `{summary.attempted_imports}`",
        f"- imported_matrices: `{summary.imported_matrices}`",
        f"- failed_imports: `{summary.failed_imports}`",
        f"- total_csr_nnz: `{summary.total_csr_nnz}`",
        f"- source_selection_path: `{summary.source_selection_path}`",
        f"- source_header_probe_path: `{summary.source_header_probe_path}`",
        "",
        "| rank | matrix | status | shape | stored_entries | csr_nnz | field | symmetry | duplicates | zeros_dropped |",
        "|---:|---|---|---:|---:|---:|---|---|---:|---:|",
    ]
    for record in batch.records:
        lines.append(
            "| "
            f"{record.selection_rank} | "
            f"{record.matrix_id} | "
            f"{record.status} | "
            f"{_shape_text(record)} | "
            f"{record.stored_entries if record.stored_entries is not None else ''} | "
            f"{record.csr_nnz if record.csr_nnz is not None else ''} | "
            f"{record.field or ''} | "
            f"{record.symmetry or ''} | "
            f"{record.duplicate_entries_merged} | "
            f"{record.explicit_zeros_dropped} |"
        )
    lines.append("")
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def csr_matrix_from_record(row: dict[str, Any]) -> CsrMatrix:
    """Build a CsrMatrix from a JSONL CSR import record."""

    csr = CsrMatrix(
        matrix_id=str(row["matrix_id"]),
        n_rows=int(row["n_rows"]),
        n_cols=int(row["n_cols"]),
        row_ptr=tuple(int(value) for value in row["row_ptr"]),
        col_ind=tuple(int(value) for value in row["col_ind"]),
        values=tuple(float(value) for value in row["values"]),
        field=str(row["field"]),
        symmetry=str(row["symmetry"]),
        source_path=str(row.get("source_archive_path") or row.get("source_path")),
        metadata=dict(row.get("metadata", {})),
    )
    csr.validate()
    return csr


def _read_matrix_market_coordinate_csr_with_metadata(
    handle: TextIO,
    *,
    matrix_id: str,
    source_path: str,
    drop_explicit_zeros: bool,
) -> tuple[CsrMatrix, dict[str, int]]:
    banner = _next_data_line(handle, allow_banner=True)
    header = _parse_banner_and_shape(handle, banner=banner, path=source_path)
    if header.object_type != "matrix":
        raise ValueError(f"Matrix Market object must be matrix: {source_path}")
    if header.storage_format != "coordinate":
        raise ValueError(f"CSR import requires coordinate storage: {source_path}")
    if header.field not in SUPPORTED_FIELDS:
        raise ValueError(f"unsupported Matrix Market field for CSR import: {header.field}")
    if header.symmetry not in SUPPORTED_SYMMETRIES:
        raise ValueError(
            f"unsupported Matrix Market symmetry for CSR import: {header.symmetry}"
        )

    entries: dict[tuple[int, int], float] = {}
    stored_entries_read = 0
    duplicate_entries_merged = 0
    expanded_entries_added = 0
    for line in handle:
        stripped = line.strip()
        if not stripped or stripped.startswith("%"):
            continue
        stored_entries_read += 1
        row, col, value = _parse_coordinate_entry(
            stripped,
            field=header.field,
            source_path=source_path,
            line_index=stored_entries_read,
        )
        _validate_coordinate(row, col, header)
        duplicate_entries_merged += _accumulate(entries, row, col, value)
        if header.symmetry != "general" and row != col:
            expanded_row, expanded_col, expanded_value = _expanded_coordinate(
                row,
                col,
                value,
                symmetry=header.symmetry,
            )
            duplicate_entries_merged += _accumulate(
                entries,
                expanded_row,
                expanded_col,
                expanded_value,
            )
            expanded_entries_added += 1
        elif header.symmetry == "skew-symmetric" and row == col and value != 0.0:
            raise ValueError(f"skew-symmetric diagonal entry must be zero: {source_path}")
    if stored_entries_read != header.nnz:
        raise ValueError(
            f"Matrix Market stored entry count mismatch in {source_path}: "
            f"header={header.nnz}, read={stored_entries_read}"
        )

    explicit_zeros_dropped = 0
    sorted_entries: list[tuple[int, int, float]] = []
    for (row, col), value in sorted(entries.items()):
        if drop_explicit_zeros and value == 0.0:
            explicit_zeros_dropped += 1
            continue
        sorted_entries.append((row, col, value))

    row_ptr = [0 for _ in range(header.n_rows + 1)]
    col_ind: list[int] = []
    values: list[float] = []
    current_row = 0
    for row, col, value in sorted_entries:
        while current_row < row:
            row_ptr[current_row + 1] = len(values)
            current_row += 1
        col_ind.append(col)
        values.append(value)
    while current_row < header.n_rows:
        row_ptr[current_row + 1] = len(values)
        current_row += 1

    csr = CsrMatrix(
        matrix_id=matrix_id,
        n_rows=header.n_rows,
        n_cols=header.n_cols,
        row_ptr=tuple(row_ptr),
        col_ind=tuple(col_ind),
        values=tuple(values),
        field=header.field,
        symmetry=header.symmetry,
        source_path=source_path,
        metadata={
            "stored_entries": header.nnz,
            "duplicate_entries_merged": duplicate_entries_merged,
            "explicit_zeros_dropped": explicit_zeros_dropped,
            "expanded_entries_added": expanded_entries_added,
        },
    )
    csr.validate()
    return csr, {
        "stored_entries": header.nnz,
        "duplicate_entries_merged": duplicate_entries_merged,
        "explicit_zeros_dropped": explicit_zeros_dropped,
        "expanded_entries_added": expanded_entries_added,
    }


def _parse_banner_and_shape(
    handle: TextIO,
    *,
    banner: str,
    path: str,
) -> MatrixMarketHeader:
    tokens = banner.split()
    if len(tokens) != 5 or tokens[0] != "%%MatrixMarket":
        raise ValueError(f"invalid Matrix Market banner in {path}")
    object_type, storage_format, field, symmetry = (
        tokens[1].lower(),
        tokens[2].lower(),
        tokens[3].lower(),
        tokens[4].lower(),
    )
    shape_line = _next_data_line(handle, allow_banner=False)
    shape = shape_line.split()
    if storage_format != "coordinate":
        raise ValueError(f"CSR import requires coordinate storage: {path}")
    if len(shape) != 3:
        raise ValueError(f"coordinate Matrix Market shape needs 3 ints: {shape_line}")
    return MatrixMarketHeader(
        path=path,
        object_type=object_type,
        storage_format=storage_format,
        field=field,
        symmetry=symmetry,
        n_rows=int(shape[0]),
        n_cols=int(shape[1]),
        nnz=int(shape[2]),
        compressed=False,
    )


def _parse_coordinate_entry(
    line: str,
    *,
    field: str,
    source_path: str,
    line_index: int,
) -> tuple[int, int, float]:
    tokens = line.split()
    if field == "pattern":
        if len(tokens) < 2:
            raise ValueError(f"pattern row {line_index} needs 2 columns: {source_path}")
        return int(tokens[0]) - 1, int(tokens[1]) - 1, 1.0
    if field in {"real", "integer"}:
        if len(tokens) < 3:
            raise ValueError(f"{field} row {line_index} needs 3 columns: {source_path}")
        return int(tokens[0]) - 1, int(tokens[1]) - 1, float(tokens[2])
    raise ValueError(f"unsupported Matrix Market field: {field}")


def _validate_coordinate(row: int, col: int, header: MatrixMarketHeader) -> None:
    if row < 0 or row >= header.n_rows:
        raise ValueError(f"row index out of bounds: {row + 1}")
    if col < 0 or col >= header.n_cols:
        raise ValueError(f"column index out of bounds: {col + 1}")


def _accumulate(
    entries: dict[tuple[int, int], float],
    row: int,
    col: int,
    value: float,
) -> int:
    key = (row, col)
    if key in entries:
        entries[key] += value
        return 1
    entries[key] = value
    return 0


def _expanded_coordinate(
    row: int,
    col: int,
    value: float,
    *,
    symmetry: str,
) -> tuple[int, int, float]:
    if symmetry in {"symmetric", "hermitian"}:
        return col, row, value
    if symmetry == "skew-symmetric":
        return col, row, -value
    raise ValueError(f"unsupported expansion symmetry: {symmetry}")


def _eligible_for_csr_import(
    row: dict[str, Any],
    header_row: dict[str, Any] | None,
    *,
    max_rows: int,
    max_cols: int,
    max_stored_entries: int,
    max_archive_size_bytes: int,
) -> bool:
    if int(row.get("archive_size_bytes") or 0) > max_archive_size_bytes:
        return False
    if int(row["n_rows"]) > max_rows or int(row["n_cols"]) > max_cols:
        return False
    stored_entries = (
        int(header_row["stored_nnz"])
        if header_row and header_row.get("stored_nnz") is not None
        else int(row["nnz"])
    )
    if stored_entries > max_stored_entries:
        return False
    if header_row and header_row.get("shape_matches") is not True:
        return False
    return True


def _import_one(row: dict[str, Any]) -> CsrImportRecord:
    try:
        member_path, csr = import_archive_matrix_market_csr(
            row["local_path"],
            matrix_id=str(row["matrix_id"]),
            expected_name=str(row.get("name") or ""),
        )
    except Exception as exc:
        return CsrImportRecord(
            selection_rank=int(row["selection_rank"]),
            matrix_id=str(row["matrix_id"]),
            source_archive_path=str(row["local_path"]),
            member_path=None,
            status="failed",
            error=str(exc),
            n_rows=None,
            n_cols=None,
            stored_entries=None,
            csr_nnz=None,
            field=None,
            symmetry=None,
            duplicate_entries_merged=0,
            explicit_zeros_dropped=0,
            expanded_entries_added=0,
            row_ptr=(),
            col_ind=(),
            values=(),
            metadata={},
        )
    return CsrImportRecord(
        selection_rank=int(row["selection_rank"]),
        matrix_id=str(row["matrix_id"]),
        source_archive_path=str(row["local_path"]),
        member_path=member_path,
        status="success",
        error=None,
        n_rows=csr.n_rows,
        n_cols=csr.n_cols,
        stored_entries=int(csr.metadata["stored_entries"]),
        csr_nnz=csr.nnz,
        field=csr.field,
        symmetry=csr.symmetry,
        duplicate_entries_merged=int(csr.metadata["duplicate_entries_merged"]),
        explicit_zeros_dropped=int(csr.metadata["explicit_zeros_dropped"]),
        expanded_entries_added=int(csr.metadata["expanded_entries_added"]),
        row_ptr=csr.row_ptr,
        col_ind=csr.col_ind,
        values=csr.values,
        metadata={
            "source_kind": row.get("kind"),
            "size_bucket": row.get("size_bucket"),
            "source_archive_size_bytes": row.get("archive_size_bytes"),
        },
    )


def _choose_matrix_member(
    members: tuple[tarfile.TarInfo, ...],
    *,
    expected_name: str | None,
) -> tarfile.TarInfo | None:
    matrix_members = tuple(member for member in members if member.name.endswith(".mtx"))
    if not matrix_members:
        return None
    if expected_name:
        expected_suffix = f"/{expected_name}.mtx"
        for member in matrix_members:
            if member.name == f"{expected_name}.mtx" or member.name.endswith(
                expected_suffix
            ):
                return member
    return sorted(matrix_members, key=lambda member: member.name)[0]


def _next_data_line(handle: TextIO, allow_banner: bool) -> str:
    for line in handle:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("%") and not (
            allow_banner and stripped.startswith("%%MatrixMarket")
        ):
            continue
        return stripped
    raise ValueError("Matrix Market file ended before required metadata")


def _shape_text(record: CsrImportRecord) -> str:
    if record.n_rows is None or record.n_cols is None:
        return ""
    return f"{record.n_rows}x{record.n_cols}"
