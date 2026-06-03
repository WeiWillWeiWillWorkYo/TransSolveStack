"""Build the append-only CSR queue training pool."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transsolvestack.profiling.csr_queue_training_pool import (  # noqa: E402
    build_csr_queue_training_pool_from_files,
    write_csr_queue_training_pool_artifacts,
)
from transsolvestack.profiling.provenance import (  # noqa: E402
    CORE_CSR_QUEUE_TRAINING_POOL_PROVENANCE_FILES,
    build_artifact_manifest,
    git_commit_or_unknown,
    write_manifest,
)


DEFAULT_BASE_SELECTOR_ROWS = (
    "runs/phase1_csr_selector_readiness/csr_selector_rows.jsonl",
    "runs/phase1_csr_micro_campaign/csr_micro_selector_rows.jsonl",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full-queue-batches",
        default="runs/phase1_csr_full_dataset_queue/csr_full_dataset_queue_batches.jsonl",
    )
    parser.add_argument(
        "--base-selector-rows",
        action="append",
        default=list(DEFAULT_BASE_SELECTOR_ROWS),
    )
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--out", default="runs/phase1_csr_queue_training_pool")
    args = parser.parse_args()

    pool = build_csr_queue_training_pool_from_files(
        full_queue_batch_path=args.full_queue_batches,
        base_selector_paths=args.base_selector_rows,
        runs_root=args.runs_root,
    )
    paths = write_csr_queue_training_pool_artifacts(pool, args.out)
    summary = pool["summary"]
    manifest_path = Path(args.out) / "artifact_manifest.json"
    dynamic_source_files = []
    for source in pool["source_rows"]:
        if source.get("source_kind") != "queue_batch_execution":
            continue
        dynamic_source_files.append(str(source["selector_rows_path"]))
        manifest = source.get("manifest_path")
        if manifest:
            dynamic_source_files.append(str(manifest))
    tracked_files = tuple(
        dict.fromkeys(
            (
                *CORE_CSR_QUEUE_TRAINING_POOL_PROVENANCE_FILES,
                *dynamic_source_files,
            )
        )
    )
    write_manifest(
        build_artifact_manifest(
            artifact_kind="csr_queue_training_pool",
            command="scripts/tss_csr_queue_training_pool.py",
            tracked_files=tracked_files,
            metadata={
                "git_commit": git_commit_or_unknown(),
                "status": summary["status"],
                "schema_version": summary["schema_version"],
                "training_pool_ready": summary["training_pool_ready"],
                "runtime_selector_changed": summary["runtime_selector_changed"],
                "source_count": summary["source_count"],
                "completed_queue_batches": summary["completed_queue_batches"],
                "completed_batch_ids": summary["completed_batch_ids"],
                "queue_batch_oracle_batches": summary["queue_batch_oracle_batches"],
                "queue_batch_screen_only_batches": (
                    summary["queue_batch_screen_only_batches"]
                ),
                "next_pending_batch_id": summary["next_pending_batch_id"],
                "selector_rows": summary["selector_rows"],
                "queue_batch_selector_rows": summary["queue_batch_selector_rows"],
                "matrices": summary["matrices"],
                "success_rows": summary["success_rows"],
                "screened_out_rows": summary["screened_out_rows"],
                "oracle_rows": summary["oracle_rows"],
            },
        ),
        manifest_path,
    )
    for key, value in {**paths, "manifest": manifest_path}.items():
        print(f"{key}: {value}")
    print(f"status: {summary['status']}")
    print(f"selector_rows: {summary['selector_rows']}")
    print(f"completed_queue_batches: {summary['completed_queue_batches']}")
    print(f"next_pending_batch_id: {summary['next_pending_batch_id']}")
    if summary["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
