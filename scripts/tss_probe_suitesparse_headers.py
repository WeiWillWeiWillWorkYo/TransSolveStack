"""Probe Matrix Market headers inside selected SuiteSparse archives."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transsolvestack.datasets.archive_probe import (
    probe_selected_matrix_archives_from_file,
    write_archive_header_probe,
    write_archive_header_report,
    write_archive_header_summary,
)
from transsolvestack.profiling.provenance import (
    CORE_SUITESPARSE_HEADER_PROBE_PROVENANCE_FILES,
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
    parser.add_argument("--out", default="runs/phase1_suitesparse_header_probe")
    args = parser.parse_args()

    probe = probe_selected_matrix_archives_from_file(args.selection)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "probe": output / "archive_header_probe.jsonl",
        "summary": output / "archive_header_summary.json",
        "report": output / "archive_header_report.md",
        "manifest": output / "artifact_manifest.json",
    }
    write_archive_header_probe(probe, paths["probe"])
    write_archive_header_summary(probe.summary, paths["summary"])
    write_archive_header_report(probe, paths["report"])
    write_manifest(
        build_artifact_manifest(
            artifact_kind="suitesparse_archive_header_probe",
            command="scripts/tss_probe_suitesparse_headers.py",
            tracked_files=CORE_SUITESPARSE_HEADER_PROBE_PROVENANCE_FILES,
            metadata={
                "git_commit": git_commit_or_unknown(),
                "status": probe.summary.status,
                "selected_matrices": probe.summary.selected_matrices,
                "probed_archives": probe.summary.probed_archives,
                "success_count": probe.summary.success_count,
                "failure_count": probe.summary.failure_count,
                "shape_mismatch_count": probe.summary.shape_mismatch_count,
                "stored_nnz_mismatch_count": probe.summary.stored_nnz_mismatch_count,
                "field_mismatch_count": probe.summary.field_mismatch_count,
            },
        ),
        paths["manifest"],
    )
    for key, value in paths.items():
        print(f"{key}: {value}")
    print(f"status: {probe.summary.status}")
    if probe.summary.status != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
