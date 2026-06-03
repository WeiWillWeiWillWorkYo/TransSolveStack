"""Train a shadow ranker on the blocked-gap augmented CSR tensor bundle."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transsolvestack.profiling.csr_blocked_gap_augmented_ranker import (  # noqa: E402
    build_csr_blocked_gap_augmented_ranker_from_files,
)
from transsolvestack.profiling.provenance import (  # noqa: E402
    CORE_CSR_BLOCKED_GAP_AUGMENTED_RANKER_PROVENANCE_FILES,
    build_artifact_manifest,
    git_commit_or_unknown,
    write_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tensors",
        default=(
            "runs/phase1_csr_blocked_gap_training_integration/"
            "csr_blocked_gap_training_tensors.json"
        ),
    )
    parser.add_argument(
        "--request-index",
        default=(
            "runs/phase1_csr_blocked_gap_training_integration/"
            "csr_blocked_gap_training_request_index.jsonl"
        ),
    )
    parser.add_argument(
        "--integration-summary",
        default=(
            "runs/phase1_csr_blocked_gap_training_integration/"
            "csr_blocked_gap_training_integration_summary.json"
        ),
    )
    parser.add_argument(
        "--positive-membership",
        default=(
            "runs/phase1_csr_blocked_gap_training_integration/"
            "csr_blocked_gap_training_positive_membership.jsonl"
        ),
    )
    parser.add_argument(
        "--out",
        default="runs/phase1_csr_blocked_gap_augmented_ranker",
    )
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--l2-regularization", type=float, default=1.0e-4)
    parser.add_argument("--d-model", type=int, default=24)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--feedforward-dim", type=int, default=48)
    parser.add_argument("--seed", type=int, default=20)
    args = parser.parse_args()

    artifact = build_csr_blocked_gap_augmented_ranker_from_files(
        tensors_path=args.tensors,
        request_index_path=args.request_index,
        integration_summary_path=args.integration_summary,
        positive_membership_path=args.positive_membership,
        output_dir=args.out,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2_regularization=args.l2_regularization,
        d_model=args.d_model,
        num_attention_heads=args.attention_heads,
        feedforward_dim=args.feedforward_dim,
        seed=args.seed,
    )
    summary = artifact["summary"]
    manifest_path = Path(args.out) / "artifact_manifest.json"
    write_manifest(
        build_artifact_manifest(
            artifact_kind="csr_blocked_gap_augmented_ranker",
            command="scripts/tss_csr_blocked_gap_augmented_ranker.py",
            tracked_files=CORE_CSR_BLOCKED_GAP_AUGMENTED_RANKER_PROVENANCE_FILES,
            metadata={
                "git_commit": git_commit_or_unknown(),
                "status": summary["status"],
                "schema_version": summary["schema_version"],
                "shadow_only": summary["shadow_only"],
                "runtime_selector_changed": summary["runtime_selector_changed"],
                "model_id": summary["model_id"],
                "model_trained": summary["model_trained"],
                "ranker_num_requests": summary["ranker_num_requests"],
                "ranker_num_global_candidates": summary[
                    "ranker_num_global_candidates"
                ],
                "blocked_gap_prediction_rows": summary[
                    "blocked_gap_prediction_rows"
                ],
                "blocked_gap_success_predictions": summary[
                    "blocked_gap_success_predictions"
                ],
                "blocked_gap_non_success_predictions": summary[
                    "blocked_gap_non_success_predictions"
                ],
                "blocked_gap_oracle_matches": summary["blocked_gap_oracle_matches"],
                "blocked_gap_positive_candidate_coverage_complete": summary[
                    "blocked_gap_positive_candidate_coverage_complete"
                ],
            },
        ),
        manifest_path,
    )
    for key, value in {**artifact["paths"], "manifest": str(manifest_path)}.items():
        print(f"{key}: {value}")
    print(f"status: {summary['status']}")
    print(f"ranker_num_requests: {summary['ranker_num_requests']}")
    print(f"ranker_num_global_candidates: {summary['ranker_num_global_candidates']}")
    print(f"blocked_gap_prediction_rows: {summary['blocked_gap_prediction_rows']}")
    print(f"blocked_gap_success_predictions: {summary['blocked_gap_success_predictions']}")
    print(
        "blocked_gap_positive_candidate_coverage_complete: "
        f"{summary['blocked_gap_positive_candidate_coverage_complete']}"
    )
    if summary["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
