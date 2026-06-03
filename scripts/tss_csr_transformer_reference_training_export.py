"""Build the reference CSR Transformer training export checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transsolvestack.policies.csr_transformer_reference_training_export import (
    build_csr_transformer_reference_training_export_from_files,
)
from transsolvestack.profiling.provenance import (
    CORE_CSR_TRANSFORMER_REFERENCE_TRAINING_EXPORT_PROVENANCE_FILES,
    build_artifact_manifest,
    git_commit_or_unknown,
    write_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tensors",
        default="runs/phase1_csr_transformer_ready/csr_transformer_training_tensors.json",
    )
    parser.add_argument(
        "--request-index",
        default="runs/phase1_csr_transformer_ready/csr_transformer_request_index.jsonl",
    )
    parser.add_argument(
        "--training-entrypoint-summary",
        default=(
            "runs/phase1_csr_transformer_training_entrypoint/"
            "csr_transformer_training_entrypoint_summary.json"
        ),
    )
    parser.add_argument(
        "--out",
        default="runs/phase1_csr_transformer_reference_training_export",
    )
    parser.add_argument("--contributor-id", default="wei_cui_reference")
    parser.add_argument("--contributor-name", default="Wei CUI")
    parser.add_argument("--model-id", default="csr_masked_self_attention_ranker_v1")
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--l2-regularization", type=float, default=1.0e-4)
    parser.add_argument("--d-model", type=int, default=24)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--feedforward-dim", type=int, default=48)
    parser.add_argument("--seed", type=int, default=18)
    args = parser.parse_args()

    export = build_csr_transformer_reference_training_export_from_files(
        args.tensors,
        args.request_index,
        args.training_entrypoint_summary,
        args.out,
        contributor_id=args.contributor_id,
        contributor_name=args.contributor_name,
        model_id=args.model_id,
        d_model=args.d_model,
        num_attention_heads=args.attention_heads,
        feedforward_dim=args.feedforward_dim,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2_regularization=args.l2_regularization,
        seed=args.seed,
    )
    summary = export["summary"]
    paths = {key: Path(value) for key, value in export["paths"].items()}
    paths["manifest"] = Path(args.out) / "artifact_manifest.json"
    write_manifest(
        build_artifact_manifest(
            artifact_kind="csr_transformer_reference_training_export",
            command="scripts/tss_csr_transformer_reference_training_export.py",
            tracked_files=CORE_CSR_TRANSFORMER_REFERENCE_TRAINING_EXPORT_PROVENANCE_FILES,
            metadata={
                "git_commit": git_commit_or_unknown(),
                "status": summary["status"],
                "schema_version": summary["schema_version"],
                "reference_training_export_ready": summary[
                    "reference_training_export_ready"
                ],
                "model_trained": summary["model_trained"],
                "external_checkpoint_ready": summary["external_checkpoint_ready"],
                "adapter_ready": summary["adapter_ready"],
                "adapter_roundtrip_exact": summary["adapter_roundtrip_exact"],
                "intake_input_ready": summary["intake_input_ready"],
                "runtime_selector_changed": summary["runtime_selector_changed"],
                "num_predictions": summary["num_predictions"],
            },
        ),
        paths["manifest"],
    )
    for key, value in paths.items():
        print(f"{key}: {value}")
    print(f"status: {summary['status']}")
    print(f"intake_input_ready: {summary['intake_input_ready']}")
    if summary["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
