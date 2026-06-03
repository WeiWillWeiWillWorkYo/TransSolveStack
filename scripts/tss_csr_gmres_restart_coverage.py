"""Run bounded GMRES restart coverage probe for queue-ready candidate gaps."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transsolvestack.profiling.csr_gmres_restart_coverage import (  # noqa: E402
    run_csr_gmres_restart_coverage_from_files,
)
from transsolvestack.profiling.provenance import (  # noqa: E402
    CORE_CSR_GMRES_RESTART_COVERAGE_PROVENANCE_FILES,
    build_artifact_manifest,
    git_commit_or_unknown,
    write_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matrix-queue",
        default="runs/phase1_csr_queue_batch_00010/csr_queue_batch_matrix_queue.jsonl",
    )
    parser.add_argument(
        "--coverage-summary",
        default=(
            "runs/phase1_csr_queue_candidate_coverage/"
            "csr_queue_candidate_coverage_summary.json"
        ),
    )
    parser.add_argument("--out", default="runs/phase1_csr_gmres_restart_coverage")
    parser.add_argument("--device-memory-gb", type=float, default=0.20)
    parser.add_argument("--max-matrices", type=int, default=8)
    parser.add_argument("--max-rows", type=int, default=65_536)
    parser.add_argument("--max-cols", type=int, default=65_536)
    parser.add_argument("--max-stored-entries", type=int, default=1_000_000)
    parser.add_argument("--max-archive-size-bytes", type=int, default=256_000_000)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--tolerance-rel", type=float, default=1.0e-5)
    args = parser.parse_args()

    export = run_csr_gmres_restart_coverage_from_files(
        matrix_queue_path=args.matrix_queue,
        coverage_summary_path=args.coverage_summary,
        output_dir=args.out,
        device_memory_gb=args.device_memory_gb,
        max_matrices=args.max_matrices,
        max_rows=args.max_rows,
        max_cols=args.max_cols,
        max_stored_entries=args.max_stored_entries,
        max_archive_size_bytes=args.max_archive_size_bytes,
        max_iter=args.max_iter,
        tolerance_rel=args.tolerance_rel,
    )
    summary = export["summary"]
    paths = export["paths"]
    manifest_path = Path(args.out) / "artifact_manifest.json"
    write_manifest(
        build_artifact_manifest(
            artifact_kind="csr_gmres_restart_coverage",
            command="scripts/tss_csr_gmres_restart_coverage.py",
            tracked_files=CORE_CSR_GMRES_RESTART_COVERAGE_PROVENANCE_FILES,
            metadata={
                "git_commit": git_commit_or_unknown(),
                "status": summary["status"],
                "schema_version": summary["schema_version"],
                "coverage_source_latest_batch_id": (
                    summary["coverage_source_latest_batch_id"]
                ),
                "coverage_outcome": summary["coverage_outcome"],
                "selected_matrices": summary["selected_matrices"],
                "candidate_jobs": summary["candidate_jobs"],
                "gpu_success_rows": summary["gpu_success_rows"],
                "cpu_screened_out_rows": summary["cpu_screened_out_rows"],
                "gpu_failed_rows": summary["gpu_failed_rows"],
                "selector_oracle_rows": summary["selector_oracle_rows"],
                "queue_merge_ready": summary["queue_merge_ready"],
                "runtime_selector_changed": summary["runtime_selector_changed"],
                "executes_gpu": summary["executes_gpu"],
                "imports_matrices": summary["imports_matrices"],
            },
        ),
        manifest_path,
    )
    for key, value in {**paths, "manifest": str(manifest_path)}.items():
        print(f"{key}: {value}")
    print(f"status: {summary['status']}")
    print(f"coverage_outcome: {summary['coverage_outcome']}")
    print(f"selected_matrices: {summary['selected_matrices']}")
    print(f"candidate_jobs: {summary['candidate_jobs']}")
    print(f"gpu_success_rows: {summary['gpu_success_rows']}")
    print(f"cpu_screened_out_rows: {summary['cpu_screened_out_rows']}")
    print(f"queue_merge_ready: {summary['queue_merge_ready']}")
    if summary["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
