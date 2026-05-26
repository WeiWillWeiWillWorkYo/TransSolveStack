"""Train and evaluate the CSR pairwise linear ranker baseline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transsolvestack.policies.csr_linear_ranker import (
    train_csr_linear_ranker_from_tensor_file,
    write_csr_linear_ranker_model,
    write_csr_linear_ranker_predictions,
    write_csr_linear_ranker_report,
    write_csr_linear_ranker_schema,
    write_csr_linear_ranker_summary,
)
from transsolvestack.profiling.provenance import (
    CORE_CSR_LINEAR_RANKER_PROVENANCE_FILES,
    build_artifact_manifest,
    git_commit_or_unknown,
    write_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tensors",
        default="runs/phase1_csr_training_tensors/csr_training_tensors.json",
    )
    parser.add_argument(
        "--request-index",
        default="runs/phase1_csr_training_tensors/csr_training_request_index.jsonl",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2-regularization", type=float, default=1.0e-4)
    parser.add_argument("--out", default="runs/phase1_csr_linear_ranker")
    args = parser.parse_args()

    export = train_csr_linear_ranker_from_tensor_file(
        args.tensors,
        args.request_index,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2_regularization=args.l2_regularization,
    )
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "model": output / "csr_linear_ranker_model.json",
        "predictions": output / "csr_linear_ranker_predictions.jsonl",
        "summary": output / "csr_linear_ranker_summary.json",
        "schema": output / "csr_linear_ranker_schema.json",
        "report": output / "csr_linear_ranker_report.md",
        "manifest": output / "artifact_manifest.json",
    }
    write_csr_linear_ranker_model(export.model, paths["model"])
    write_csr_linear_ranker_predictions(export.predictions, paths["predictions"])
    write_csr_linear_ranker_summary(export.summary, paths["summary"])
    write_csr_linear_ranker_schema(export.schema, paths["schema"])
    write_csr_linear_ranker_report(export, paths["report"])
    write_manifest(
        build_artifact_manifest(
            artifact_kind="csr_linear_ranker_baseline",
            command="scripts/tss_csr_linear_ranker.py",
            tracked_files=CORE_CSR_LINEAR_RANKER_PROVENANCE_FILES,
            metadata={
                "git_commit": git_commit_or_unknown(),
                "status": export.summary.status,
                "schema_version": export.summary.schema_version,
                "model_family": export.summary.model_family,
                "model_trained": export.summary.model_trained,
                "runtime_selector_changed": export.summary.runtime_selector_changed,
                "num_predictions": export.summary.num_predictions,
                "num_features": export.summary.num_features,
                "num_pairwise_constraints": export.summary.num_pairwise_constraints,
                "num_pairwise_updates": export.summary.num_pairwise_updates,
                "train_oracle_top1_accuracy": export.summary.train_oracle_top1_accuracy,
                "eval_oracle_top1_accuracy": export.summary.eval_oracle_top1_accuracy,
                "eval_profiled_success_selection_rate": (
                    export.summary.eval_profiled_success_selection_rate
                ),
            },
        ),
        paths["manifest"],
    )
    for key, value in paths.items():
        print(f"{key}: {value}")
    print(f"status: {export.summary.status}")
    if export.summary.status != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
