"""Build a resumable full SuiteSparse CSR benchmark/training queue."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transsolvestack.profiling.csr_full_dataset_queue import (
    build_csr_full_dataset_queue_from_files,
    write_csr_full_dataset_queue_batches,
    write_csr_full_dataset_queue_jobs,
    write_csr_full_dataset_queue_matrices,
    write_csr_full_dataset_queue_report,
    write_csr_full_dataset_queue_schema,
    write_csr_full_dataset_queue_state,
    write_csr_full_dataset_queue_summary,
)
from transsolvestack.profiling.provenance import (
    CORE_CSR_FULL_DATASET_QUEUE_PROVENANCE_FILES,
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
    parser.add_argument(
        "--selector-rows",
        default="runs/phase1_csr_selector_readiness/csr_selector_rows.jsonl",
    )
    parser.add_argument("--resource-limits", default="configs/runtime/resource_limits.yaml")
    parser.add_argument("--storage-paths", default="configs/runtime/storage_paths.yaml")
    parser.add_argument("--out", default="runs/phase1_csr_full_dataset_queue")
    parser.add_argument("--queue-id", default="phase1_csr_full_dataset_queue_m83")
    parser.add_argument("--batch-matrix-count", type=int, default=8)
    parser.add_argument("--measurement-repeats", type=int, default=1)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--max-cols", type=int, default=None)
    parser.add_argument("--max-nnz", type=int, default=None)
    parser.add_argument("--max-archive-mb", type=float, default=256.0)
    parser.add_argument("--max-iter", type=int, default=None)
    parser.add_argument("--tolerance-rel", type=float, default=1.0e-5)
    parser.add_argument("--max-queue-matrices", type=int, default=None)
    args = parser.parse_args()

    queue = build_csr_full_dataset_queue_from_files(
        args.index,
        args.selector_rows,
        args.resource_limits,
        args.storage_paths,
        queue_id=args.queue_id,
        batch_matrix_count=args.batch_matrix_count,
        measurement_repeats=args.measurement_repeats,
        max_rows=args.max_rows,
        max_cols=args.max_cols,
        max_nnz=args.max_nnz,
        max_archive_size_bytes=int(args.max_archive_mb * 1_000_000),
        max_iter=args.max_iter,
        tolerance_rel=args.tolerance_rel,
        max_queue_matrices=args.max_queue_matrices,
    )
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "matrices": output / "csr_full_dataset_queue_matrices.jsonl",
        "jobs": output / "csr_full_dataset_queue_jobs.jsonl",
        "batches": output / "csr_full_dataset_queue_batches.jsonl",
        "state": output / "csr_full_dataset_queue_state.json",
        "summary": output / "csr_full_dataset_queue_summary.json",
        "schema": output / "csr_full_dataset_queue_schema.json",
        "report": output / "csr_full_dataset_queue_report.md",
        "manifest": output / "artifact_manifest.json",
    }
    write_csr_full_dataset_queue_matrices(queue.matrix_rows, paths["matrices"])
    write_csr_full_dataset_queue_jobs(queue.job_rows, paths["jobs"])
    write_csr_full_dataset_queue_batches(queue.batch_rows, paths["batches"])
    write_csr_full_dataset_queue_state(queue.state, paths["state"])
    write_csr_full_dataset_queue_summary(queue.summary, paths["summary"])
    write_csr_full_dataset_queue_schema(queue.schema, paths["schema"])
    write_csr_full_dataset_queue_report(queue, paths["report"])
    write_manifest(
        build_artifact_manifest(
            artifact_kind="csr_full_dataset_queue",
            command="scripts/tss_csr_full_dataset_queue.py",
            tracked_files=CORE_CSR_FULL_DATASET_QUEUE_PROVENANCE_FILES,
            metadata={
                "git_commit": git_commit_or_unknown(),
                "status": queue.summary.status,
                "schema_version": queue.summary.schema_version,
                "queue_id": queue.summary.queue_id,
                "index_matrices": queue.summary.index_matrices,
                "eligible_matrices": queue.summary.eligible_matrices,
                "queued_matrices": queue.summary.queued_matrices,
                "queued_jobs": queue.summary.queued_jobs,
                "queued_batches": queue.summary.queued_batches,
                "planned_gpu_solve_attempts": (
                    queue.summary.planned_gpu_solve_attempts
                ),
                "runtime_selector_changed": queue.summary.runtime_selector_changed,
                "executes_gpu": queue.summary.executes_gpu,
                "imports_matrices": queue.summary.imports_matrices,
                "resumable": queue.summary.resumable,
            },
        ),
        paths["manifest"],
    )
    for key, value in paths.items():
        print(f"{key}: {value}")
    print(f"status: {queue.summary.status}")
    print(f"queued_batches: {queue.summary.queued_batches}")
    print(f"queued_jobs: {queue.summary.queued_jobs}")
    if queue.summary.status != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
