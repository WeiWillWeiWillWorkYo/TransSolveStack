"""Build a metadata-only index for the downloaded SuiteSparse archive set."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transsolvestack.datasets.suitesparse_index import (
    build_suitesparse_index,
    write_suitesparse_index,
    write_suitesparse_report,
    write_suitesparse_summary,
)
from transsolvestack.profiling.provenance import (
    CORE_SUITESPARSE_INDEX_PROVENANCE_FILES,
    build_artifact_manifest,
    git_commit_or_unknown,
    write_manifest,
)


DEFAULT_DATASET_ROOT = Path(
    "/mnt/tss_external/TransSolveStack/datasets/suitesparse_full"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--stats", default=None)
    parser.add_argument("--urls", default=None)
    parser.add_argument("--mm-root", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Write artifacts even when expected archives or URL entries are missing.",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    stats_path = Path(args.stats) if args.stats else dataset_root / "metadata" / "ssstats.csv"
    urls_path = Path(args.urls) if args.urls else dataset_root / "metadata" / "mm_urls.txt"
    mm_root = Path(args.mm_root) if args.mm_root else dataset_root / "MM"
    output = Path(args.out) if args.out else dataset_root / "index"

    index = build_suitesparse_index(
        stats_path=stats_path,
        urls_path=urls_path,
        mm_root=mm_root,
    )
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "index": output / "matrix_manifest.jsonl",
        "summary": output / "index_summary.json",
        "report": output / "index_report.md",
        "manifest": output / "artifact_manifest.json",
    }
    write_suitesparse_index(index, paths["index"])
    write_suitesparse_summary(index.summary, paths["summary"])
    write_suitesparse_report(index.summary, paths["report"])
    write_manifest(
        build_artifact_manifest(
            artifact_kind="suitesparse_matrix_manifest_index",
            command="scripts/tss_build_suitesparse_index.py",
            tracked_files=CORE_SUITESPARSE_INDEX_PROVENANCE_FILES,
            metadata={
                "git_commit": git_commit_or_unknown(),
                "status": index.summary.status,
                "expected_matrices": index.summary.expected_matrices,
                "indexed_matrices": index.summary.indexed_matrices,
                "present_archive_files": index.summary.present_archive_files,
                "actual_archive_files": index.summary.actual_archive_files,
                "missing_archive_files": index.summary.missing_archive_files,
                "missing_url_entries": index.summary.missing_url_entries,
                "extra_archive_files": index.summary.extra_archive_files,
                "total_archive_size_bytes": index.summary.total_archive_size_bytes,
                "stats_source_timestamp": index.summary.stats_source_timestamp,
            },
        ),
        paths["manifest"],
    )

    for key, value in paths.items():
        print(f"{key}: {value}")
    print(f"status: {index.summary.status}")

    if index.summary.status == "incomplete" and not args.allow_incomplete:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
