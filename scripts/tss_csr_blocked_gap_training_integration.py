"""Build a training-only bundle with blocked-gap positive CSR rows."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transsolvestack.profiling.csr_blocked_gap_training_integration import (  # noqa: E402
    build_csr_blocked_gap_training_integration_from_files,
)
from transsolvestack.profiling.provenance import (  # noqa: E402
    CORE_CSR_BLOCKED_GAP_TRAINING_INTEGRATION_PROVENANCE_FILES,
    build_artifact_manifest,
    git_commit_or_unknown,
    write_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-selector-rows",
        default=(
            "runs/phase1_csr_queue_training_pool/"
            "csr_queue_training_pool_selector_rows.jsonl"
        ),
    )
    parser.add_argument(
        "--base-pool-summary",
        default="runs/phase1_csr_queue_training_pool/csr_queue_training_pool_summary.json",
    )
    parser.add_argument(
        "--positive-selector-rows",
        default=(
            "runs/phase1_csr_blocked_gap_positive_search/"
            "csr_blocked_gap_positive_selector_rows.jsonl"
        ),
    )
    parser.add_argument(
        "--positive-summary",
        default=(
            "runs/phase1_csr_blocked_gap_positive_search/"
            "csr_blocked_gap_positive_search_summary.json"
        ),
    )
    parser.add_argument(
        "--out",
        default="runs/phase1_csr_blocked_gap_training_integration",
    )
    parser.add_argument("--eval-fraction", type=float, default=0.25)
    parser.add_argument("--min-eval-matrices", type=int, default=4)
    args = parser.parse_args()

    export = build_csr_blocked_gap_training_integration_from_files(
        base_selector_rows_path=args.base_selector_rows,
        base_pool_summary_path=args.base_pool_summary,
        positive_selector_rows_path=args.positive_selector_rows,
        positive_summary_path=args.positive_summary,
        output_dir=args.out,
        eval_fraction=args.eval_fraction,
        min_eval_matrices=args.min_eval_matrices,
    )
    summary = export["summary"]
    manifest_path = Path(args.out) / "artifact_manifest.json"
    write_manifest(
        build_artifact_manifest(
            artifact_kind="csr_blocked_gap_training_integration",
            command="scripts/tss_csr_blocked_gap_training_integration.py",
            tracked_files=CORE_CSR_BLOCKED_GAP_TRAINING_INTEGRATION_PROVENANCE_FILES,
            metadata={
                "git_commit": git_commit_or_unknown(),
                "status": summary["status"],
                "schema_version": summary["schema_version"],
                "runtime_selector_changed": summary["runtime_selector_changed"],
                "generic_queue_merge": summary["generic_queue_merge"],
                "base_selector_rows": summary["base_selector_rows"],
                "positive_selector_rows": summary["positive_selector_rows"],
                "integrated_selector_rows": summary["integrated_selector_rows"],
                "integrated_model_requests": summary["integrated_model_requests"],
                "integrated_global_candidates": summary[
                    "integrated_global_candidates"
                ],
                "new_global_candidate_ids": summary["new_global_candidate_ids"],
                "positive_success_rows": summary["positive_success_rows"],
                "positive_oracle_rows": summary["positive_oracle_rows"],
                "validation_error_count": summary["validation_error_count"],
            },
        ),
        manifest_path,
    )
    for key, value in {**export["paths"], "manifest": str(manifest_path)}.items():
        print(f"{key}: {value}")
    print(f"status: {summary['status']}")
    print(f"integrated_selector_rows: {summary['integrated_selector_rows']}")
    print(f"integrated_model_requests: {summary['integrated_model_requests']}")
    print(f"integrated_global_candidates: {summary['integrated_global_candidates']}")
    if summary["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
