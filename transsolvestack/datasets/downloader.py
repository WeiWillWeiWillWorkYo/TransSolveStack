"""Download planning for matrix cache files.

The actual download command is intentionally not run by import. Large datasets
must be pulled explicitly into TSS_DATA_ROOT or data/raw.
"""

from __future__ import annotations

from pathlib import Path

from transsolvestack.datasets.manifest import MatrixManifestEntry


def planned_local_path(
    entry: MatrixManifestEntry,
    data_root: str | Path = "data/raw",
) -> Path:
    root = Path(data_root)
    suffixes = Path(entry.url).suffixes
    suffix = "".join(suffixes) if suffixes else ".dat"
    return root / entry.source / entry.group / f"{entry.name}{suffix}"
