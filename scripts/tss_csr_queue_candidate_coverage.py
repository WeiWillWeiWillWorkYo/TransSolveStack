"""Diagnose solver/candidate coverage gaps after rolling CSR queue batches."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transsolvestack.profiling.csr_queue_candidate_coverage import (  # noqa: E402
    build_csr_queue_candidate_coverage_from_files,
    write_csr_queue_candidate_coverage_artifacts,
)
from transsolvestack.profiling.provenance import (  # noqa: E402
    CORE_CSR_QUEUE_CANDIDATE_COVERAGE_PROVENANCE_FILES,
    build_artifact_manifest,
    git_commit_or_unknown,
    write_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue-batches-root", default="runs")
    parser.add_argument(
        "--full-queue-jobs",
        default="runs/phase1_csr_full_dataset_queue/csr_full_dataset_queue_jobs.jsonl",
    )
    parser.add_argument(
        "--queue-training-pool-summary",
        default="runs/phase1_csr_queue_training_pool/csr_queue_training_pool_summary.json",
    )
    parser.add_argument("--recent-window", type=int, default=2)
    parser.add_argument("--low-success-threshold", type=float, default=0.5)
    parser.add_argument("--out", default="runs/phase1_csr_queue_candidate_coverage")
    args = parser.parse_args()

    coverage = build_csr_queue_candidate_coverage_from_files(
        queue_batches_root=args.queue_batches_root,
        full_queue_jobs_path=args.full_queue_jobs,
        queue_training_pool_summary_path=args.queue_training_pool_summary,
        recent_window=args.recent_window,
        low_success_threshold=args.low_success_threshold,
    )
    paths = write_csr_queue_candidate_coverage_artifacts(coverage, args.out)
    manifest_path = Path(args.out) / "artifact_manifest.json"
    write_manifest(
        build_artifact_manifest(
            artifact_kind="csr_queue_candidate_coverage",
            command="scripts/tss_csr_queue_candidate_coverage.py",
            tracked_files=CORE_CSR_QUEUE_CANDIDATE_COVERAGE_PROVENANCE_FILES,
            metadata={
                "git_commit": git_commit_or_unknown(),
                "status": coverage.summary.status,
                "schema_version": coverage.summary.schema_version,
                "completed_queue_batches": coverage.summary.completed_queue_batches,
                "latest_batch_id": coverage.summary.latest_batch_id,
                "latest_batch_outcome": coverage.summary.latest_batch_outcome,
                "trigger_candidate_coverage": (
                    coverage.summary.trigger_candidate_coverage
                ),
                "planned_gap_count": coverage.summary.planned_gap_count,
                "queue_ready_gap_count": coverage.summary.queue_ready_gap_count,
                "cpu_screen_blocked_gap_count": (
                    coverage.summary.cpu_screen_blocked_gap_count
                ),
                "runtime_selector_changed": coverage.summary.runtime_selector_changed,
                "executes_gpu": coverage.summary.executes_gpu,
                "imports_matrices": coverage.summary.imports_matrices,
            },
        ),
        manifest_path,
    )
    for key, value in {**paths, "manifest": manifest_path}.items():
        print(f"{key}: {value}")
    print(f"status: {coverage.summary.status}")
    print(f"completed_queue_batches: {coverage.summary.completed_queue_batches}")
    print(f"latest_batch_id: {coverage.summary.latest_batch_id}")
    print(f"trigger_candidate_coverage: {coverage.summary.trigger_candidate_coverage}")
    print(f"planned_gap_count: {coverage.summary.planned_gap_count}")
    if coverage.summary.status != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
