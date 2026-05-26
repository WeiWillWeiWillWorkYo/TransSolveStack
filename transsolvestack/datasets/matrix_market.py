"""Matrix Market metadata inspection without materializing matrices."""

from __future__ import annotations

import gzip
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TextIO

from transsolvestack.profiling.artifacts import write_jsonl


@dataclass(frozen=True)
class MatrixMarketHeader:
    path: str
    object_type: str
    storage_format: str
    field: str
    symmetry: str
    n_rows: int
    n_cols: int
    nnz: int
    compressed: bool

    @property
    def is_sparse_coordinate(self) -> bool:
        return self.object_type == "matrix" and self.storage_format == "coordinate"


def inspect_matrix_market_header(path: str | Path) -> MatrixMarketHeader:
    matrix_path = Path(path)
    with _open_text(matrix_path) as handle:
        return inspect_matrix_market_header_from_text(
            handle,
            path=str(matrix_path),
            compressed=matrix_path.suffix == ".gz",
        )


def inspect_matrix_market_header_from_text(
    handle: TextIO,
    *,
    path: str,
    compressed: bool = False,
) -> MatrixMarketHeader:
    banner = _next_data_line(handle, allow_banner=True)
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
    if storage_format == "coordinate":
        if len(shape) != 3:
            raise ValueError(f"coordinate Matrix Market shape needs 3 ints: {shape_line}")
        n_rows, n_cols, nnz = (int(shape[0]), int(shape[1]), int(shape[2]))
    elif storage_format == "array":
        if len(shape) != 2:
            raise ValueError(f"array Matrix Market shape needs 2 ints: {shape_line}")
        n_rows, n_cols = (int(shape[0]), int(shape[1]))
        nnz = n_rows * n_cols
    else:
        raise ValueError(f"unsupported Matrix Market storage format: {storage_format}")
    return MatrixMarketHeader(
        path=path,
        object_type=object_type,
        storage_format=storage_format,
        field=field,
        symmetry=symmetry,
        n_rows=n_rows,
        n_cols=n_cols,
        nnz=nnz,
        compressed=compressed,
    )


def write_matrix_market_metadata(
    headers: tuple[MatrixMarketHeader, ...],
    path: str | Path,
) -> Path:
    return write_jsonl((asdict(header) for header in headers), path)


def write_matrix_market_report(
    headers: tuple[MatrixMarketHeader, ...],
    path: str | Path,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Matrix Market Metadata Probe",
        "",
        f"- matrices: `{len(headers)}`",
        "",
        "| path | format | field | symmetry | shape | nnz | compressed |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for header in headers:
        lines.append(
            "| "
            f"{header.path} | "
            f"{header.storage_format} | "
            f"{header.field} | "
            f"{header.symmetry} | "
            f"{header.n_rows}x{header.n_cols} | "
            f"{header.nnz} | "
            f"{'yes' if header.compressed else 'no'} |"
        )
    lines.append("")
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def _open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


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
