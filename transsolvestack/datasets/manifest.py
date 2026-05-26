"""Dataset manifest records."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MatrixManifestEntry:
    matrix_id: str
    source: str
    group: str
    name: str
    url: str
    file_format: str
    n_rows: int | None = None
    n_cols: int | None = None
    nnz: int | None = None
    symmetry: str = "unknown"
    license_note: str | None = None
    checksum: str | None = None
    local_path: str | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)

