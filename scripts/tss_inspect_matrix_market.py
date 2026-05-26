"""Inspect Matrix Market headers without loading matrix values."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transsolvestack.datasets.matrix_market import (
    inspect_matrix_market_header,
    write_matrix_market_metadata,
    write_matrix_market_report,
)
from transsolvestack.profiling.provenance import (
    CORE_MATRIX_MARKET_PROBE_PROVENANCE_FILES,
    build_artifact_manifest,
    git_commit_or_unknown,
    write_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matrix",
        action="append",
        default=None,
        help="Matrix Market file path. Can be passed multiple times.",
    )
    parser.add_argument("--out", default="runs/phase1_matrix_market_probe")
    args = parser.parse_args()

    matrix_paths = args.matrix or ["tests/fixtures/tiny_spd.mtx"]
    headers = tuple(inspect_matrix_market_header(path) for path in matrix_paths)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "metadata": output / "matrix_metadata.jsonl",
        "report": output / "matrix_metadata.md",
        "manifest": output / "artifact_manifest.json",
    }
    write_matrix_market_metadata(headers, paths["metadata"])
    write_matrix_market_report(headers, paths["report"])
    write_manifest(
        build_artifact_manifest(
            artifact_kind="matrix_market_metadata_probe",
            command="scripts/tss_inspect_matrix_market.py",
            tracked_files=CORE_MATRIX_MARKET_PROBE_PROVENANCE_FILES,
            metadata={
                "git_commit": git_commit_or_unknown(),
                "num_matrices": len(headers),
            },
        ),
        paths["manifest"],
    )
    for key, value in paths.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
