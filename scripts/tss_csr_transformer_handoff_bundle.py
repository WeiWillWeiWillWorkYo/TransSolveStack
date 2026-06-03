"""Build the CSR Transformer training handoff bundle."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transsolvestack.profiling.csr_transformer_handoff_bundle import (  # noqa: E402
    build_csr_transformer_handoff_bundle_from_files,
)
from transsolvestack.profiling.provenance import (  # noqa: E402
    CORE_CSR_TRANSFORMER_HANDOFF_BUNDLE_PROVENANCE_FILES,
    build_artifact_manifest,
    git_commit_or_unknown,
    write_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training-dir",
        default="runs/phase1_csr_blocked_gap_training_integration",
    )
    parser.add_argument(
        "--augmented-ranker-dir",
        default="runs/phase1_csr_blocked_gap_augmented_ranker",
    )
    parser.add_argument(
        "--guarded-replay-dir",
        default="runs/phase1_csr_blocked_gap_guarded_replay",
    )
    parser.add_argument(
        "--submission-contract-dir",
        default="runs/phase1_csr_policy_model_submission",
    )
    parser.add_argument("--out", default="runs/phase1_csr_transformer_handoff_bundle")
    parser.add_argument("--model-family", default="csr_transformer_policy_v2")
    args = parser.parse_args()

    artifact = build_csr_transformer_handoff_bundle_from_files(
        training_dir=args.training_dir,
        augmented_ranker_dir=args.augmented_ranker_dir,
        guarded_replay_dir=args.guarded_replay_dir,
        submission_contract_dir=args.submission_contract_dir,
        output_dir=args.out,
        model_family=args.model_family,
    )
    summary = artifact["summary"]
    manifest_path = Path(args.out) / "artifact_manifest.json"
    write_manifest(
        build_artifact_manifest(
            artifact_kind="csr_transformer_handoff_bundle",
            command="scripts/tss_csr_transformer_handoff_bundle.py",
            tracked_files=CORE_CSR_TRANSFORMER_HANDOFF_BUNDLE_PROVENANCE_FILES,
            metadata={
                "git_commit": git_commit_or_unknown(),
                "status": summary["status"],
                "schema_version": summary["schema_version"],
                "handoff_ready": summary["handoff_ready"],
                "model_training_required": summary["model_training_required"],
                "model_trained": summary["model_trained"],
                "runtime_selector_changed": summary["runtime_selector_changed"],
                "executes_gpu": summary["executes_gpu"],
                "training_model_requests": summary["training_model_requests"],
                "training_global_candidates": summary[
                    "training_global_candidates"
                ],
                "guard_actual_quality_gate_blocks": summary[
                    "guard_actual_quality_gate_blocks"
                ],
                "guard_actual_runtime_selector_changes": summary[
                    "guard_actual_runtime_selector_changes"
                ],
                "num_source_files": summary["num_source_files"],
            },
        ),
        manifest_path,
    )
    for key, value in {**artifact["paths"], "manifest": str(manifest_path)}.items():
        print(f"{key}: {value}")
    print(f"status: {summary['status']}")
    print(f"handoff_ready: {summary['handoff_ready']}")
    print(f"training_model_requests: {summary['training_model_requests']}")
    print(f"training_global_candidates: {summary['training_global_candidates']}")
    print(f"guard_actual_runtime_selector_changes: {summary['guard_actual_runtime_selector_changes']}")
    if summary["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
