"""Build the CSR Transformer-ready training bundle."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transsolvestack.policies.csr_transformer_ready import (
    build_csr_transformer_ready_bundle_from_files,
)
from transsolvestack.profiling.provenance import (
    CORE_CSR_TRANSFORMER_READY_PROVENANCE_FILES,
    build_artifact_manifest,
    git_commit_or_unknown,
    write_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selector-rows",
        action="append",
        default=[
            "runs/phase1_csr_selector_readiness/csr_selector_rows.jsonl",
            "runs/phase1_csr_micro_campaign/csr_micro_selector_rows.jsonl",
        ],
    )
    parser.add_argument("--out", default="runs/phase1_csr_transformer_ready")
    parser.add_argument("--eval-fraction", type=float, default=0.25)
    parser.add_argument("--min-eval-matrices", type=int, default=4)
    args = parser.parse_args()

    bundle = build_csr_transformer_ready_bundle_from_files(
        args.selector_rows,
        args.out,
        eval_fraction=args.eval_fraction,
        min_eval_matrices=args.min_eval_matrices,
    )
    manifest_path = Path(args.out) / "artifact_manifest.json"
    summary = bundle["summary"]
    write_manifest(
        build_artifact_manifest(
            artifact_kind="csr_transformer_ready_bundle",
            command="scripts/tss_csr_transformer_ready.py",
            tracked_files=CORE_CSR_TRANSFORMER_READY_PROVENANCE_FILES,
            metadata={
                "git_commit": git_commit_or_unknown(),
                "status": summary["status"],
                "schema_version": summary["schema_version"],
                "transformer_connectable": summary["transformer_connectable"],
                "runtime_selector_changed": summary["runtime_selector_changed"],
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
    if summary["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
