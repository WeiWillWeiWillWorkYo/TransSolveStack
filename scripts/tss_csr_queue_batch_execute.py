"""Execute one full-dataset CSR queue batch with CPU screen and Taichi GPU solves."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transsolvestack.profiling.csr_queue_batch_execution import (  # noqa: E402
    execute_csr_queue_batch_from_files,
)
from transsolvestack.profiling.provenance import (  # noqa: E402
    CORE_CSR_QUEUE_BATCH_EXECUTION_PROVENANCE_FILES,
    build_artifact_manifest,
    git_commit_or_unknown,
    write_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matrix-queue",
        default="runs/phase1_csr_full_dataset_queue/csr_full_dataset_queue_matrices.jsonl",
    )
    parser.add_argument(
        "--job-queue",
        default="runs/phase1_csr_full_dataset_queue/csr_full_dataset_queue_jobs.jsonl",
    )
    parser.add_argument(
        "--batch-queue",
        default="runs/phase1_csr_full_dataset_queue/csr_full_dataset_queue_batches.jsonl",
    )
    parser.add_argument("--batch-id", default="batch_00001")
    parser.add_argument("--out", default="runs/phase1_csr_queue_batch_00001")
    parser.add_argument("--device-memory-gb", type=float, default=0.20)
    parser.add_argument("--max-rows", type=int, default=65_536)
    parser.add_argument("--max-cols", type=int, default=65_536)
    parser.add_argument("--max-stored-entries", type=int, default=1_000_000)
    parser.add_argument("--max-archive-size-bytes", type=int, default=256_000_000)
    args = parser.parse_args()

    export = execute_csr_queue_batch_from_files(
        matrix_queue_path=args.matrix_queue,
        job_queue_path=args.job_queue,
        batch_queue_path=args.batch_queue,
        output_dir=args.out,
        batch_id=args.batch_id,
        device_memory_gb=args.device_memory_gb,
        max_rows=args.max_rows,
        max_cols=args.max_cols,
        max_stored_entries=args.max_stored_entries,
        max_archive_size_bytes=args.max_archive_size_bytes,
    )
    summary = export["summary"]
    manifest_path = Path(args.out) / "artifact_manifest.json"
    write_manifest(
        build_artifact_manifest(
            artifact_kind="csr_queue_batch_execution",
            command="scripts/tss_csr_queue_batch_execute.py",
            tracked_files=CORE_CSR_QUEUE_BATCH_EXECUTION_PROVENANCE_FILES,
            metadata={
                "git_commit": git_commit_or_unknown(),
                "status": summary["status"],
                "schema_version": summary["schema_version"],
                "queue_id": summary["queue_id"],
                "batch_id": summary["batch_id"],
                "batch_outcome": summary["batch_outcome"],
                "micro_campaign_status": summary["micro_campaign_status"],
                "completed_without_oracle": summary["completed_without_oracle"],
                "training_pool_role": summary["training_pool_role"],
                "planned_matrices": summary["planned_matrices"],
                "planned_jobs": summary["planned_jobs"],
                "executed_matrices": summary["executed_matrices"],
                "executed_jobs": summary["executed_jobs"],
                "imported_matrices": summary["imported_matrices"],
                "candidate_jobs": summary["candidate_jobs"],
                "gpu_success_rows": summary["gpu_success_rows"],
                "cpu_screened_out_rows": summary["cpu_screened_out_rows"],
                "gpu_failed_rows": summary["gpu_failed_rows"],
                "selector_rows": summary["selector_rows"],
                "selector_oracle_rows": summary["selector_oracle_rows"],
                "executes_gpu": summary["executes_gpu"],
                "imports_matrices": summary["imports_matrices"],
                "cpu_screen_required": summary["cpu_screen_required"],
                "runtime_selector_changed": summary["runtime_selector_changed"],
            },
        ),
        manifest_path,
    )
    for key, value in {**export["paths"], "manifest": str(manifest_path)}.items():
        print(f"{key}: {value}")
    print(f"status: {summary['status']}")
    print(f"batch_id: {summary['batch_id']}")
    print(f"batch_outcome: {summary['batch_outcome']}")
    print(f"gpu_success_rows: {summary['gpu_success_rows']}")
    print(f"cpu_screened_out_rows: {summary['cpu_screened_out_rows']}")
    if summary["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
