"""Select a conservative real-matrix subset from the SuiteSparse full index."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transsolvestack.datasets.selection import (
    MatrixSelectionCriteria,
    build_external_matrix_systems,
    select_matrix_subset_from_index,
    write_external_matrix_systems,
    write_matrix_selection,
    write_matrix_selection_report,
    write_matrix_selection_summary,
)
from transsolvestack.profiling.provenance import (
    CORE_SUITESPARSE_SELECTION_PROVENANCE_FILES,
    build_artifact_manifest,
    git_commit_or_unknown,
    write_manifest,
)


DEFAULT_INDEX_PATH = Path(
    "/mnt/tss_external/TransSolveStack/datasets/"
    "suitesparse_full/index/matrix_manifest.jsonl"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default=str(DEFAULT_INDEX_PATH))
    parser.add_argument("--out", default="runs/phase1_suitesparse_selection")
    parser.add_argument("--max-matrices", type=int, default=64)
    parser.add_argument("--max-per-kind", type=int, default=4)
    parser.add_argument("--max-rows", type=int, default=250_000)
    parser.add_argument("--max-cols", type=int, default=250_000)
    parser.add_argument("--max-nnz", type=int, default=5_000_000)
    parser.add_argument("--max-archive-mb", type=float, default=512.0)
    parser.add_argument("--include-rectangular", action="store_true")
    parser.add_argument("--include-non-real", action="store_true")
    parser.add_argument(
        "--require-spd",
        action="store_true",
        help="Only select SuiteSparse rows marked positive definite.",
    )
    args = parser.parse_args()

    criteria = MatrixSelectionCriteria(
        max_matrices=args.max_matrices,
        max_per_kind=args.max_per_kind,
        max_rows=args.max_rows,
        max_cols=args.max_cols,
        max_nnz=args.max_nnz,
        max_archive_size_bytes=int(args.max_archive_mb * 1_000_000),
        require_square=not args.include_rectangular,
        require_real=not args.include_non_real,
        require_positive_definite=True if args.require_spd else None,
    )
    selection = select_matrix_subset_from_index(args.index, criteria)
    systems = build_external_matrix_systems(selection.records)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "selection": output / "selected_matrices.jsonl",
        "systems": output / "selected_systems.jsonl",
        "summary": output / "selection_summary.json",
        "report": output / "selection_report.md",
        "manifest": output / "artifact_manifest.json",
    }
    write_matrix_selection(selection, paths["selection"])
    write_external_matrix_systems(systems, paths["systems"])
    write_matrix_selection_summary(selection.summary, paths["summary"])
    write_matrix_selection_report(selection, paths["report"])
    write_manifest(
        build_artifact_manifest(
            artifact_kind="suitesparse_matrix_subset_selection",
            command="scripts/tss_select_suitesparse_subset.py",
            tracked_files=CORE_SUITESPARSE_SELECTION_PROVENANCE_FILES,
            metadata={
                "git_commit": git_commit_or_unknown(),
                "status": selection.summary.status,
                "source_index_path": str(args.index),
                "candidates_considered": selection.summary.candidates_considered,
                "candidates_after_filter": selection.summary.candidates_after_filter,
                "selected_matrices": selection.summary.selected_matrices,
                "total_selected_archive_size_bytes": (
                    selection.summary.total_selected_archive_size_bytes
                ),
                "max_matrices": criteria.max_matrices,
                "max_per_kind": criteria.max_per_kind,
            },
        ),
        paths["manifest"],
    )
    for key, value in paths.items():
        print(f"{key}: {value}")
    print(f"status: {selection.summary.status}")

    if selection.summary.status != "ready":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
