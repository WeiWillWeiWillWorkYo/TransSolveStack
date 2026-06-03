"""Replay D20 blocked-gap predictions through the learned runtime guard."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transsolvestack.profiling.csr_blocked_gap_guarded_replay import (  # noqa: E402
    build_csr_blocked_gap_guarded_replay_from_files,
)
from transsolvestack.profiling.provenance import (  # noqa: E402
    CORE_CSR_BLOCKED_GAP_GUARDED_REPLAY_PROVENANCE_FILES,
    build_artifact_manifest,
    git_commit_or_unknown,
    write_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selector-rows",
        default=(
            "runs/phase1_csr_blocked_gap_training_integration/"
            "combined_csr_selector_rows.jsonl"
        ),
    )
    parser.add_argument(
        "--predictions",
        default=(
            "runs/phase1_csr_blocked_gap_augmented_ranker/"
            "csr_blocked_gap_augmented_ranker_predictions.jsonl"
        ),
    )
    parser.add_argument(
        "--positive-predictions",
        default=(
            "runs/phase1_csr_blocked_gap_augmented_ranker/"
            "csr_blocked_gap_augmented_ranker_positive_predictions.jsonl"
        ),
    )
    parser.add_argument(
        "--ranker-summary",
        default=(
            "runs/phase1_csr_blocked_gap_augmented_ranker/"
            "csr_blocked_gap_augmented_ranker_summary.json"
        ),
    )
    parser.add_argument("--out", default="runs/phase1_csr_blocked_gap_guarded_replay")
    parser.add_argument("--min-confidence", type=float, default=0.75)
    args = parser.parse_args()

    artifact = build_csr_blocked_gap_guarded_replay_from_files(
        selector_rows_path=args.selector_rows,
        predictions_path=args.predictions,
        positive_predictions_path=args.positive_predictions,
        ranker_summary_path=args.ranker_summary,
        output_dir=args.out,
        min_confidence=args.min_confidence,
    )
    summary = artifact["summary"]
    manifest_path = Path(args.out) / "artifact_manifest.json"
    write_manifest(
        build_artifact_manifest(
            artifact_kind="csr_blocked_gap_guarded_replay",
            command="scripts/tss_csr_blocked_gap_guarded_replay.py",
            tracked_files=CORE_CSR_BLOCKED_GAP_GUARDED_REPLAY_PROVENANCE_FILES,
            metadata={
                "git_commit": git_commit_or_unknown(),
                "status": summary["status"],
                "schema_version": summary["schema_version"],
                "shadow_only": summary["shadow_only"],
                "runtime_selector_changed": summary["runtime_selector_changed"],
                "executes_gpu": summary["executes_gpu"],
                "num_replay_rows": summary["num_replay_rows"],
                "actual_quality_gate_blocks": summary[
                    "actual_quality_gate_blocks"
                ],
                "actual_runtime_selector_changes": summary[
                    "actual_runtime_selector_changes"
                ],
                "fallback_chain_enforced_rows": summary[
                    "fallback_chain_enforced_rows"
                ],
                "counterfactual_only": summary["counterfactual_only"],
                "counterfactual_eligible_gate_promotions": summary[
                    "counterfactual_eligible_gate_promotions"
                ],
            },
        ),
        manifest_path,
    )
    for key, value in {**artifact["paths"], "manifest": str(manifest_path)}.items():
        print(f"{key}: {value}")
    print(f"status: {summary['status']}")
    print(f"num_replay_rows: {summary['num_replay_rows']}")
    print(f"actual_quality_gate_blocks: {summary['actual_quality_gate_blocks']}")
    print(f"actual_runtime_selector_changes: {summary['actual_runtime_selector_changes']}")
    print(f"fallback_chain_enforced_rows: {summary['fallback_chain_enforced_rows']}")
    if summary["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
