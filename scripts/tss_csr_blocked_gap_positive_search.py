"""Run blocked-gap positive GPU evidence search."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transsolvestack.profiling.csr_blocked_gap_positive_search import (  # noqa: E402
    DEFAULT_CSR_RECORD_PATHS,
    run_csr_blocked_gap_positive_search_from_files,
)
from transsolvestack.profiling.provenance import (  # noqa: E402
    CORE_CSR_BLOCKED_GAP_POSITIVE_SEARCH_PROVENANCE_FILES,
    build_artifact_manifest,
    git_commit_or_unknown,
    write_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csr-records",
        action="append",
        default=list(DEFAULT_CSR_RECORD_PATHS),
        help="CSR JSONL record file. May be provided more than once.",
    )
    parser.add_argument("--out", default="runs/phase1_csr_blocked_gap_positive_search")
    parser.add_argument("--max-rows", type=int, default=2_048)
    parser.add_argument("--max-cols", type=int, default=2_048)
    parser.add_argument("--max-nnz", type=int, default=20_000)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--tolerance-rel", type=float, default=1.0e-5)
    parser.add_argument("--max-positive-per-gap", type=int, default=2)
    parser.add_argument("--device-memory-gb", type=float, default=0.20)
    args = parser.parse_args()

    export = run_csr_blocked_gap_positive_search_from_files(
        csr_record_paths=tuple(args.csr_records),
        output_dir=args.out,
        max_rows=args.max_rows,
        max_cols=args.max_cols,
        max_nnz=args.max_nnz,
        max_iter=args.max_iter,
        tolerance_rel=args.tolerance_rel,
        max_positive_per_gap=args.max_positive_per_gap,
        device_memory_gb=args.device_memory_gb,
    )
    summary = export["summary"]
    paths = export["paths"]
    manifest_path = Path(args.out) / "artifact_manifest.json"
    write_manifest(
        build_artifact_manifest(
            artifact_kind="csr_blocked_gap_positive_search",
            command="scripts/tss_csr_blocked_gap_positive_search.py",
            tracked_files=CORE_CSR_BLOCKED_GAP_POSITIVE_SEARCH_PROVENANCE_FILES,
            metadata={
                "git_commit": git_commit_or_unknown(),
                "status": summary["status"],
                "schema_version": summary["schema_version"],
                "positive_evidence_found": summary["positive_evidence_found"],
                "candidate_jobs": summary["candidate_jobs"],
                "gpu_success_rows": summary["gpu_success_rows"],
                "gpu_failed_rows": summary["gpu_failed_rows"],
                "positive_gpu_gap_ids": summary["positive_gpu_gap_ids"],
                "queue_merge_ready": summary["queue_merge_ready"],
                "runtime_selector_changed": summary["runtime_selector_changed"],
                "executes_gpu": summary["executes_gpu"],
            },
        ),
        manifest_path,
    )
    for key, value in {**paths, "manifest": str(manifest_path)}.items():
        print(f"{key}: {value}")
    print(f"status: {summary['status']}")
    print(f"positive_evidence_found: {summary['positive_evidence_found']}")
    print(f"candidate_jobs: {summary['candidate_jobs']}")
    print(f"gpu_success_rows: {summary['gpu_success_rows']}")
    print(f"gpu_failed_rows: {summary['gpu_failed_rows']}")
    print(f"positive_gpu_gap_ids: {summary['positive_gpu_gap_ids']}")
    print(f"queue_merge_ready: {summary['queue_merge_ready']}")
    if summary["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
