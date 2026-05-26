"""External matrix catalog loading and download planning."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from transsolvestack.config import load_yaml
from transsolvestack.profiling.artifacts import write_jsonl
from transsolvestack.runtime.resource_budget import ResourceBudget


@dataclass(frozen=True)
class MatrixCatalogEntry:
    matrix_id: str
    source: str
    group: str
    name: str
    url: str
    format: str
    n_rows: int
    n_cols: int
    nnz: int
    symmetry: str
    license: str
    checksum: str | None
    estimated_download_gb: float
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def local_relative_path(self) -> Path:
        suffix = _suffix_for_format(self.format)
        return Path(self.source_key) / self.group / f"{self.name}{suffix}"

    @property
    def source_key(self) -> str:
        return (
            self.source.lower()
            .replace(" ", "_")
            .replace("/", "_")
            .replace("-", "_")
        )


@dataclass(frozen=True)
class MatrixCatalog:
    catalog_id: str
    matrices: tuple[MatrixCatalogEntry, ...]
    resource_limits_path: str | None = None
    default_data_root: str = "data/raw"
    path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def estimated_download_gb(self) -> float:
        return sum(item.estimated_download_gb for item in self.matrices)


@dataclass(frozen=True)
class DownloadPlanEntry:
    matrix_id: str
    source: str
    url: str
    local_path: str
    format: str
    n_rows: int
    n_cols: int
    nnz: int
    symmetry: str
    estimated_download_gb: float
    download_required: bool
    checksum: str | None
    tags: tuple[str, ...]


@dataclass(frozen=True)
class DownloadPlan:
    catalog_id: str
    data_root: str
    total_estimated_download_gb: float
    entries: tuple[DownloadPlanEntry, ...]
    dry_run: bool = True


def load_matrix_catalog(path: str | Path) -> MatrixCatalog:
    data = load_yaml(path)
    matrices = tuple(
        MatrixCatalogEntry(
            matrix_id=str(item["matrix_id"]),
            source=str(item["source"]),
            group=str(item["group"]),
            name=str(item["name"]),
            url=str(item["url"]),
            format=str(item["format"]),
            n_rows=int(item["n_rows"]),
            n_cols=int(item["n_cols"]),
            nnz=int(item["nnz"]),
            symmetry=str(item.get("symmetry", "unknown")),
            license=str(item.get("license", "unknown")),
            checksum=item.get("checksum"),
            estimated_download_gb=float(item["estimated_download_gb"]),
            tags=tuple(str(tag) for tag in item.get("tags", ())),
            metadata=dict(item.get("metadata", {})),
        )
        for item in data.get("matrices", ())
    )
    if not matrices:
        raise ValueError(f"matrix catalog has no matrices: {path}")
    return MatrixCatalog(
        catalog_id=str(data["catalog_id"]),
        matrices=matrices,
        resource_limits_path=data.get("resource_limits"),
        default_data_root=str(data.get("default_data_root", "data/raw")),
        path=str(path),
        metadata=dict(data.get("metadata", {})),
    )


def build_download_plan(
    catalog: MatrixCatalog,
    budget: ResourceBudget,
    data_root: str | Path | None = None,
) -> DownloadPlan:
    root = Path(data_root or os.environ.get("TSS_DATA_ROOT") or catalog.default_data_root)
    budget.validate_download_size(catalog.estimated_download_gb)
    entries: list[DownloadPlanEntry] = []
    for matrix in catalog.matrices:
        budget.validate_download_size(matrix.estimated_download_gb)
        local_path = root / matrix.local_relative_path
        entries.append(
            DownloadPlanEntry(
                matrix_id=matrix.matrix_id,
                source=matrix.source,
                url=matrix.url,
                local_path=str(local_path),
                format=matrix.format,
                n_rows=matrix.n_rows,
                n_cols=matrix.n_cols,
                nnz=matrix.nnz,
                symmetry=matrix.symmetry,
                estimated_download_gb=matrix.estimated_download_gb,
                download_required=not local_path.exists(),
                checksum=matrix.checksum,
                tags=matrix.tags,
            )
        )
    return DownloadPlan(
        catalog_id=catalog.catalog_id,
        data_root=str(root),
        total_estimated_download_gb=catalog.estimated_download_gb,
        entries=tuple(entries),
    )


def write_download_plan(plan: DownloadPlan, path: str | Path) -> Path:
    return write_jsonl((asdict(entry) for entry in plan.entries), path)


def write_download_plan_report(plan: DownloadPlan, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Dataset Download Plan",
        "",
        f"- catalog: `{plan.catalog_id}`",
        f"- data_root: `{plan.data_root}`",
        f"- dry_run: `{plan.dry_run}`",
        f"- matrices: `{len(plan.entries)}`",
        f"- estimated_download_gb: `{plan.total_estimated_download_gb:.6g}`",
        "",
        "| matrix | source | format | shape | nnz | gb | required | local_path |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for entry in plan.entries:
        lines.append(
            "| "
            f"{entry.matrix_id} | "
            f"{entry.source} | "
            f"{entry.format} | "
            f"{entry.n_rows}x{entry.n_cols} | "
            f"{entry.nnz} | "
            f"{entry.estimated_download_gb:.6g} | "
            f"{'yes' if entry.download_required else 'no'} | "
            f"{entry.local_path} |"
        )
    lines.append("")
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def _suffix_for_format(format_name: str) -> str:
    if format_name == "matrix_market_tar_gz":
        return ".tar.gz"
    if format_name == "matrix_market_gz":
        return ".mtx.gz"
    if format_name == "matrix_market":
        return ".mtx"
    raise ValueError(f"unsupported matrix catalog format: {format_name}")
