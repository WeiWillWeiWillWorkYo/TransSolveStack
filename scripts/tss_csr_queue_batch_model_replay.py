"""Replay the queue-batch shadow ranker without retraining."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transsolvestack.policies.csr_transformer_model_replay import (  # noqa: E402
    build_csr_transformer_model_replay_from_files,
    write_csr_transformer_model_replay_comparison_rows,
    write_csr_transformer_model_replay_predictions,
    write_csr_transformer_model_replay_report,
    write_csr_transformer_model_replay_schema,
    write_csr_transformer_model_replay_summary,
)
from transsolvestack.profiling.provenance import (  # noqa: E402
    CORE_CSR_QUEUE_BATCH_MODEL_REPLAY_PROVENANCE_FILES,
    build_artifact_manifest,
    git_commit_or_unknown,
    write_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=(
            "runs/phase1_csr_queue_batch_reference_ranker/"
            "csr_queue_batch_reference_ranker_model.json"
        ),
    )
    parser.add_argument(
        "--tensors",
        default=(
            "runs/phase1_csr_queue_batch_training_bundle/"
            "csr_transformer_training_tensors.json"
        ),
    )
    parser.add_argument(
        "--request-index",
        default=(
            "runs/phase1_csr_queue_batch_training_bundle/"
            "csr_transformer_request_index.jsonl"
        ),
    )
    parser.add_argument(
        "--reference-predictions",
        default=(
            "runs/phase1_csr_queue_batch_reference_ranker/"
            "csr_queue_batch_reference_ranker_predictions.jsonl"
        ),
    )
    parser.add_argument("--out", default="runs/phase1_csr_queue_batch_model_replay")
    args = parser.parse_args()

    export = build_csr_transformer_model_replay_from_files(
        args.model,
        args.tensors,
        args.request_index,
        args.reference_predictions,
    )
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "predictions": output / "csr_queue_batch_model_replay_predictions.jsonl",
        "comparison": output / "csr_queue_batch_model_replay_comparison.jsonl",
        "summary": output / "csr_queue_batch_model_replay_summary.json",
        "schema": output / "csr_queue_batch_model_replay_schema.json",
        "report": output / "csr_queue_batch_model_replay_report.md",
        "manifest": output / "artifact_manifest.json",
    }
    write_csr_transformer_model_replay_predictions(export.predictions, paths["predictions"])
    write_csr_transformer_model_replay_comparison_rows(
        export.comparison_rows,
        paths["comparison"],
    )
    write_csr_transformer_model_replay_summary(export.summary, paths["summary"])
    write_csr_transformer_model_replay_schema(export.schema, paths["schema"])
    write_csr_transformer_model_replay_report(export, paths["report"])
    write_manifest(
        build_artifact_manifest(
            artifact_kind="csr_queue_batch_model_replay",
            command="scripts/tss_csr_queue_batch_model_replay.py",
            tracked_files=CORE_CSR_QUEUE_BATCH_MODEL_REPLAY_PROVENANCE_FILES,
            metadata={
                "git_commit": git_commit_or_unknown(),
                "status": export.summary.status,
                "schema_version": export.summary.schema_version,
                "model_family": export.summary.model_family,
                "model_id": export.summary.model_id,
                "model_loaded": export.summary.model_loaded,
                "model_trained": export.summary.model_trained,
                "runtime_selector_changed": export.summary.runtime_selector_changed,
                "shadow_only": True,
                "num_predictions": export.summary.num_predictions,
                "exact_replay": export.summary.exact_replay,
                "max_abs_score_delta": export.summary.max_abs_score_delta,
            },
        ),
        paths["manifest"],
    )
    for key, value in paths.items():
        print(f"{key}: {value}")
    print(f"status: {export.summary.status}")
    print(f"exact_replay: {export.summary.exact_replay}")
    if export.summary.status != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
