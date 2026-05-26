"""Execute the planned real CSR benchmark micro-campaign."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transsolvestack.profiling.csr_micro_campaign import (
    run_csr_micro_campaign_from_files,
    write_csr_micro_campaign_artifacts,
)
from transsolvestack.profiling.provenance import (
    CORE_CSR_MICRO_CAMPAIGN_PROVENANCE_FILES,
    build_artifact_manifest,
    git_commit_or_unknown,
    write_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matrix-queue",
        default="runs/phase1_csr_benchmark_expansion_plan/csr_benchmark_matrix_queue.jsonl",
    )
    parser.add_argument(
        "--candidate-queue",
        default="runs/phase1_csr_benchmark_expansion_plan/csr_benchmark_candidate_queue.jsonl",
    )
    parser.add_argument("--out", default="runs/phase1_csr_micro_campaign")
    parser.add_argument("--device-memory-gb", type=float, default=0.25)
    parser.add_argument("--max-matrices", type=int, default=8)
    parser.add_argument("--max-rows", type=int, default=10_000)
    parser.add_argument("--max-cols", type=int, default=10_000)
    parser.add_argument("--max-stored-entries", type=int, default=100_000)
    parser.add_argument("--max-archive-size-bytes", type=int, default=8_000_000)
    args = parser.parse_args()

    export = run_csr_micro_campaign_from_files(
        args.matrix_queue,
        args.candidate_queue,
        device_memory_gb=args.device_memory_gb,
        max_matrices=args.max_matrices,
        max_rows=args.max_rows,
        max_cols=args.max_cols,
        max_stored_entries=args.max_stored_entries,
        max_archive_size_bytes=args.max_archive_size_bytes,
    )
    paths = write_csr_micro_campaign_artifacts(export, args.out)
    manifest_path = Path(args.out) / "artifact_manifest.json"
    write_manifest(
        build_artifact_manifest(
            artifact_kind="csr_micro_campaign",
            command="scripts/tss_csr_micro_campaign.py",
            tracked_files=CORE_CSR_MICRO_CAMPAIGN_PROVENANCE_FILES,
            metadata={
                "git_commit": git_commit_or_unknown(),
                "status": export["summary"]["status"],
                "schema_version": export["summary"]["schema_version"],
                "imported_matrices": export["summary"]["imported_matrices"],
                "candidate_jobs": export["summary"]["candidate_jobs"],
                "gpu_success_rows": export["summary"]["gpu_success_rows"],
                "cpu_screened_out_rows": export["summary"]["cpu_screened_out_rows"],
                "gpu_failed_rows": export["summary"]["gpu_failed_rows"],
                "selector_rows": export["summary"]["selector_rows"],
                "selector_oracle_rows": export["summary"]["selector_oracle_rows"],
                "runtime_selector_changed": export["summary"]["runtime_selector_changed"],
                "executes_gpu": export["summary"]["executes_gpu"],
            },
        ),
        manifest_path,
    )
    for key, value in {**paths, "manifest": manifest_path}.items():
        print(f"{key}: {value}")
    print(f"status: {export['summary']['status']}")
    if export["summary"]["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
