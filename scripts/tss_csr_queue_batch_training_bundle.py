"""Build a Transformer-ready CSR training bundle from the queue training pool."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transsolvestack.policies.csr_transformer_ready import (  # noqa: E402
    build_csr_transformer_ready_bundle_from_files,
)
from transsolvestack.profiling.artifacts import read_jsonl  # noqa: E402
from transsolvestack.profiling.provenance import (  # noqa: E402
    CORE_CSR_QUEUE_BATCH_TRAINING_BUNDLE_PROVENANCE_FILES,
    build_artifact_manifest,
    git_commit_or_unknown,
    write_manifest,
)


DEFAULT_SELECTOR_ROWS = (
    "runs/phase1_csr_queue_training_pool/csr_queue_training_pool_selector_rows.jsonl",
)
DEFAULT_QUEUE_POOL_SUMMARY = (
    "runs/phase1_csr_queue_training_pool/csr_queue_training_pool_summary.json"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selector-rows", action="append", default=list(DEFAULT_SELECTOR_ROWS))
    parser.add_argument("--queue-pool-summary", default=DEFAULT_QUEUE_POOL_SUMMARY)
    parser.add_argument("--out", default="runs/phase1_csr_queue_batch_training_bundle")
    parser.add_argument("--eval-fraction", type=float, default=0.25)
    parser.add_argument("--min-eval-matrices", type=int, default=4)
    args = parser.parse_args()

    bundle = build_csr_transformer_ready_bundle_from_files(
        args.selector_rows,
        args.out,
        eval_fraction=args.eval_fraction,
        min_eval_matrices=args.min_eval_matrices,
    )
    summary = bundle["summary"]
    pool_summary_path = Path(args.queue_pool_summary)
    pool_summary = (
        json.loads(pool_summary_path.read_text(encoding="utf-8"))
        if pool_summary_path.exists()
        else {}
    )
    queue_selector_rows = int(
        pool_summary.get("queue_batch_selector_rows", len(read_jsonl(args.selector_rows[-1])))
    )
    manifest_path = Path(args.out) / "artifact_manifest.json"
    write_manifest(
        build_artifact_manifest(
            artifact_kind="csr_queue_batch_training_bundle",
            command="scripts/tss_csr_queue_batch_training_bundle.py",
            tracked_files=CORE_CSR_QUEUE_BATCH_TRAINING_BUNDLE_PROVENANCE_FILES,
            metadata={
                "git_commit": git_commit_or_unknown(),
                "status": summary["status"],
                "schema_version": summary["schema_version"],
                "transformer_connectable": summary["transformer_connectable"],
                "runtime_selector_changed": summary["runtime_selector_changed"],
                "source_selector_paths": summary["source_selector_paths"],
                "queue_pool_summary": str(pool_summary_path),
                "queue_pool_ready": bool(pool_summary.get("training_pool_ready", False)),
                "queue_batch_selector_rows": queue_selector_rows,
                "num_selector_rows": summary["num_selector_rows"],
                "num_model_requests": summary["num_model_requests"],
                "num_global_candidates": summary["num_global_candidates"],
                "matrix_feature_dim": summary["matrix_feature_dim"],
                "candidate_feature_dim": summary["candidate_feature_dim"],
                "validation_error_count": summary["validation_error_count"],
            },
        ),
        manifest_path,
    )
    for key, value in {**bundle["paths"], "manifest": manifest_path}.items():
        print(f"{key}: {value}")
    print(f"status: {summary['status']}")
    print(f"selector_rows: {summary['num_selector_rows']}")
    print(f"model_requests: {summary['num_model_requests']}")
    if summary["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
