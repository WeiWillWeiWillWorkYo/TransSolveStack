"""Matrix Market header probes for downloaded SuiteSparse archives."""

from __future__ import annotations

import io
import json
import tarfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from transsolvestack.datasets.matrix_market import (
    MatrixMarketHeader,
    inspect_matrix_market_header_from_text,
)
from transsolvestack.profiling.artifacts import read_jsonl, write_jsonl


@dataclass(frozen=True)
class ArchiveHeaderProbeRecord:
    selection_rank: int
    matrix_id: str
    archive_path: str
    member_path: str | None
    status: str
    error: str | None
    object_type: str | None
    storage_format: str | None
    field: str | None
    symmetry: str | None
    n_rows: int | None
    n_cols: int | None
    stored_nnz: int | None
    expected_n_rows: int
    expected_n_cols: int
    expected_nnz: int
    expected_field: str
    shape_matches: bool | None
    stored_nnz_matches: bool | None
    field_matches_expected: bool | None
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ArchiveHeaderProbeSummary:
    source_selection_path: str | None
    status: str
    selected_matrices: int
    probed_archives: int
    success_count: int
    failure_count: int
    shape_mismatch_count: int
    stored_nnz_mismatch_count: int
    field_mismatch_count: int
    total_archive_size_bytes: int


@dataclass(frozen=True)
class ArchiveHeaderProbe:
    records: tuple[ArchiveHeaderProbeRecord, ...]
    summary: ArchiveHeaderProbeSummary


def inspect_archive_matrix_market_header(
    archive_path: str | Path,
    *,
    expected_name: str | None = None,
) -> tuple[str, MatrixMarketHeader]:
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
            header = inspect_matrix_market_header_from_text(
                text,
                path=f"{archive}!{member.name}",
                compressed=True,
            )
        return member.name, header


def probe_selected_matrix_archives(
    selection_rows: Iterable[dict[str, Any]],
    *,
    source_selection_path: str | Path | None = None,
) -> ArchiveHeaderProbe:
    rows = tuple(selection_rows)
    records = tuple(_probe_one(row) for row in rows)
    success_count = sum(1 for record in records if record.status == "success")
    failure_count = len(records) - success_count
    shape_mismatch_count = sum(1 for record in records if record.shape_matches is False)
    stored_nnz_mismatch_count = sum(
        1 for record in records if record.stored_nnz_matches is False
    )
    field_mismatch_count = sum(
        1 for record in records if record.field_matches_expected is False
    )
    status = "passed"
    if failure_count or shape_mismatch_count:
        status = "failed"
    return ArchiveHeaderProbe(
        records=records,
        summary=ArchiveHeaderProbeSummary(
            source_selection_path=str(source_selection_path)
            if source_selection_path
            else None,
            status=status,
            selected_matrices=len(rows),
            probed_archives=len(records),
            success_count=success_count,
            failure_count=failure_count,
            shape_mismatch_count=shape_mismatch_count,
            stored_nnz_mismatch_count=stored_nnz_mismatch_count,
            field_mismatch_count=field_mismatch_count,
            total_archive_size_bytes=sum(
                int(row.get("archive_size_bytes") or 0) for row in rows
            ),
        ),
    )


def probe_selected_matrix_archives_from_file(
    selection_path: str | Path,
) -> ArchiveHeaderProbe:
    return probe_selected_matrix_archives(
        read_jsonl(selection_path),
        source_selection_path=selection_path,
    )


def write_archive_header_probe(
    probe: ArchiveHeaderProbe,
    path: str | Path,
) -> Path:
    return write_jsonl((asdict(record) for record in probe.records), path)


def write_archive_header_summary(
    summary: ArchiveHeaderProbeSummary,
    path: str | Path,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


def write_archive_header_report(
    probe: ArchiveHeaderProbe,
    path: str | Path,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = probe.summary
    lines = [
        "# SuiteSparse Archive Header Probe",
        "",
        f"- status: `{summary.status}`",
        f"- selected_matrices: `{summary.selected_matrices}`",
        f"- probed_archives: `{summary.probed_archives}`",
        f"- success_count: `{summary.success_count}`",
        f"- failure_count: `{summary.failure_count}`",
        f"- shape_mismatch_count: `{summary.shape_mismatch_count}`",
        f"- stored_nnz_mismatch_count: `{summary.stored_nnz_mismatch_count}`",
        f"- field_mismatch_count: `{summary.field_mismatch_count}`",
        f"- source_selection_path: `{summary.source_selection_path}`",
        "",
        "| rank | matrix | status | member | field | symmetry | shape | stored_nnz | shape_ok | nnz_ok |",
        "|---:|---|---|---|---|---|---:|---:|---:|---:|",
    ]
    for record in probe.records:
        lines.append(
            "| "
            f"{record.selection_rank} | "
            f"{record.matrix_id} | "
            f"{record.status} | "
            f"{record.member_path or ''} | "
            f"{record.field or ''} | "
            f"{record.symmetry or ''} | "
            f"{_shape_text(record)} | "
            f"{record.stored_nnz if record.stored_nnz is not None else ''} | "
            f"{_bool_text(record.shape_matches)} | "
            f"{_bool_text(record.stored_nnz_matches)} |"
        )
    lines.append("")
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def _probe_one(row: dict[str, Any]) -> ArchiveHeaderProbeRecord:
    expected_field = _expected_field(row)
    try:
        member_path, header = inspect_archive_matrix_market_header(
            row["local_path"],
            expected_name=str(row.get("name") or ""),
        )
    except Exception as exc:
        return ArchiveHeaderProbeRecord(
            selection_rank=int(row["selection_rank"]),
            matrix_id=str(row["matrix_id"]),
            archive_path=str(row["local_path"]),
            member_path=None,
            status="failed",
            error=str(exc),
            object_type=None,
            storage_format=None,
            field=None,
            symmetry=None,
            n_rows=None,
            n_cols=None,
            stored_nnz=None,
            expected_n_rows=int(row["n_rows"]),
            expected_n_cols=int(row["n_cols"]),
            expected_nnz=int(row["nnz"]),
            expected_field=expected_field,
            shape_matches=None,
            stored_nnz_matches=None,
            field_matches_expected=None,
            metadata={},
        )
    shape_matches = header.n_rows == int(row["n_rows"]) and header.n_cols == int(
        row["n_cols"]
    )
    return ArchiveHeaderProbeRecord(
        selection_rank=int(row["selection_rank"]),
        matrix_id=str(row["matrix_id"]),
        archive_path=str(row["local_path"]),
        member_path=member_path,
        status="success",
        error=None,
        object_type=header.object_type,
        storage_format=header.storage_format,
        field=header.field,
        symmetry=header.symmetry,
        n_rows=header.n_rows,
        n_cols=header.n_cols,
        stored_nnz=header.nnz,
        expected_n_rows=int(row["n_rows"]),
        expected_n_cols=int(row["n_cols"]),
        expected_nnz=int(row["nnz"]),
        expected_field=expected_field,
        shape_matches=shape_matches,
        stored_nnz_matches=header.nnz == int(row["nnz"]),
        field_matches_expected=_field_matches(header.field, expected_field),
        metadata={
            "kind": row.get("kind"),
            "size_bucket": row.get("size_bucket"),
            "archive_size_bytes": row.get("archive_size_bytes"),
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


def _expected_field(row: dict[str, Any]) -> str:
    if bool(row.get("is_binary")):
        return "pattern"
    if bool(row.get("is_real")):
        return "real"
    return "unknown"


def _field_matches(actual: str, expected: str) -> bool | None:
    if expected == "unknown":
        return None
    return actual == expected


def _shape_text(record: ArchiveHeaderProbeRecord) -> str:
    if record.n_rows is None or record.n_cols is None:
        return ""
    return f"{record.n_rows}x{record.n_cols}"


def _bool_text(value: bool | None) -> str:
    if value is None:
        return ""
    return "yes" if value else "no"
