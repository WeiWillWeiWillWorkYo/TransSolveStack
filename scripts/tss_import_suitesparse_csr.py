"""Import a small selected SuiteSparse subset into CPU-side CSR fixtures."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transsolvestack.datasets.csr import (
    import_selected_archives_to_csr_from_files,
    write_csr_import_records,
    write_csr_import_report,
    write_csr_import_summary,
)
from transsolvestack.profiling.provenance import (
    CORE_SUITESPARSE_CSR_IMPORT_PROVENANCE_FILES,
    build_artifact_manifest,
    git_commit_or_unknown,
    write_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selection",
        default="runs/phase1_suitesparse_selection/selected_matrices.jsonl",
    )
    parser.add_argument(
        "--header-probe",
        default="runs/phase1_suitesparse_header_probe/archive_header_probe.jsonl",
    )
    parser.add_argument("--out", default="runs/phase1_suitesparse_csr_import")
    parser.add_argument("--max-matrices", type=int, default=12)
    parser.add_argument("--max-rows", type=int, default=10_000)
    parser.add_argument("--max-cols", type=int, default=10_000)
    parser.add_argument("--max-stored-entries", type=int, default=100_000)
    parser.add_argument("--max-archive-mb", type=float, default=8.0)
    args = parser.parse_args()

    batch = import_selected_archives_to_csr_from_files(
        args.selection,
        header_probe_path=args.header_probe,
        max_matrices=args.max_matrices,
        max_rows=args.max_rows,
        max_cols=args.max_cols,
        max_stored_entries=args.max_stored_entries,
        max_archive_size_bytes=int(args.max_archive_mb * 1_000_000),
    )
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "csr": output / "csr_matrices.jsonl",
        "summary": output / "csr_import_summary.json",
        "report": output / "csr_import_report.md",
        "manifest": output / "artifact_manifest.json",
    }
    write_csr_import_records(batch, paths["csr"])
    write_csr_import_summary(batch.summary, paths["summary"])
    write_csr_import_report(batch, paths["report"])
    write_manifest(
        build_artifact_manifest(
            artifact_kind="suitesparse_csr_import_boundary",
            command="scripts/tss_import_suitesparse_csr.py",
            tracked_files=CORE_SUITESPARSE_CSR_IMPORT_PROVENANCE_FILES,
            metadata={
                "git_commit": git_commit_or_unknown(),
                "status": batch.summary.status,
                "eligible_rows": batch.summary.eligible_rows,
                "attempted_imports": batch.summary.attempted_imports,
                "imported_matrices": batch.summary.imported_matrices,
                "failed_imports": batch.summary.failed_imports,
                "total_csr_nnz": batch.summary.total_csr_nnz,
                "max_matrices": args.max_matrices,
                "max_rows": args.max_rows,
                "max_cols": args.max_cols,
                "max_stored_entries": args.max_stored_entries,
            },
        ),
        paths["manifest"],
    )
    for key, value in paths.items():
        print(f"{key}: {value}")
    print(f"status: {batch.summary.status}")
    if batch.summary.status != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
