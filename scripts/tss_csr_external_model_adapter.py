"""Adapt an external CSR Transformer checkpoint into a TSS ranker model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transsolvestack.policies.csr_external_model_adapter import (
    adapt_csr_external_ranker_checkpoint_from_files,
    build_reference_csr_external_ranker_checkpoint_from_files,
    write_csr_external_model_adapter_model,
    write_csr_external_model_adapter_predictions,
    write_csr_external_model_adapter_ranker_summary,
    write_csr_external_model_adapter_report,
    write_csr_external_model_adapter_rows,
    write_csr_external_model_adapter_schema,
    write_csr_external_model_adapter_summary,
    write_csr_external_ranker_checkpoint,
)
from transsolvestack.profiling.provenance import (
    CORE_CSR_EXTERNAL_MODEL_ADAPTER_PROVENANCE_FILES,
    build_artifact_manifest,
    git_commit_or_unknown,
    write_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--source-model",
        default="runs/phase1_csr_transformer_ranker/csr_transformer_ranker_model.json",
    )
    parser.add_argument(
        "--tensors",
        default="runs/phase1_csr_transformer_ready/csr_transformer_training_tensors.json",
    )
    parser.add_argument(
        "--request-index",
        default="runs/phase1_csr_transformer_ready/csr_transformer_request_index.jsonl",
    )
    parser.add_argument("--out", default="runs/phase1_csr_external_model_adapter")
    parser.add_argument("--contributor-id", default="wei_cui_reference")
    parser.add_argument("--contributor-name", default="Wei CUI")
    parser.add_argument(
        "--training-statement",
        default="Reference checkpoint exported from the Phase 1 CSR Transformer ranker.",
    )
    args = parser.parse_args()

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else output / "external_csr_ranker_checkpoint.json"
    )
    if args.checkpoint is None:
        checkpoint = build_reference_csr_external_ranker_checkpoint_from_files(
            args.source_model,
            args.tensors,
            args.request_index,
            contributor_id=args.contributor_id,
            contributor_name=args.contributor_name,
            training_statement=args.training_statement,
        )
        write_csr_external_ranker_checkpoint(checkpoint, checkpoint_path)

    data = adapt_csr_external_ranker_checkpoint_from_files(
        checkpoint_path,
        args.tensors,
        args.request_index,
    )
    paths = {
        "checkpoint": checkpoint_path,
        "adapted_model": output / "adapted_csr_transformer_ranker_model.json",
        "adapted_predictions": output / "adapted_csr_transformer_ranker_predictions.jsonl",
        "adapted_ranker_summary": output / "adapted_csr_transformer_ranker_summary.json",
        "rows": output / "csr_external_model_adapter_rows.jsonl",
        "summary": output / "csr_external_model_adapter_summary.json",
        "schema": output / "csr_external_model_adapter_schema.json",
        "report": output / "csr_external_model_adapter_report.md",
        "manifest": output / "artifact_manifest.json",
    }
    write_csr_external_model_adapter_model(data, paths["adapted_model"])
    write_csr_external_model_adapter_predictions(
        data["predictions"],
        paths["adapted_predictions"],
    )
    write_csr_external_model_adapter_ranker_summary(
        data,
        paths["adapted_ranker_summary"],
    )
    write_csr_external_model_adapter_rows(data["rows"], paths["rows"])
    write_csr_external_model_adapter_summary(data, paths["summary"])
    write_csr_external_model_adapter_schema(data, paths["schema"])
    write_csr_external_model_adapter_report(data, paths["report"])
    summary = data["summary"]
    write_manifest(
        build_artifact_manifest(
            artifact_kind="csr_external_model_adapter",
            command="scripts/tss_csr_external_model_adapter.py",
            tracked_files=CORE_CSR_EXTERNAL_MODEL_ADAPTER_PROVENANCE_FILES,
            metadata={
                "git_commit": git_commit_or_unknown(),
                "status": summary["status"],
                "schema_version": summary["schema_version"],
                "adapter": summary["adapter"],
                "adapted_model_id": summary["adapted_model_id"],
                "adapter_ready": summary["adapter_ready"],
                "quality_gate_input_ready": summary["quality_gate_input_ready"],
                "policy_model_artifact_input_ready": summary[
                    "policy_model_artifact_input_ready"
                ],
                "prediction_contract_valid": summary["prediction_contract_valid"],
                "num_predictions": summary["num_predictions"],
                "runtime_selector_changed": summary["runtime_selector_changed"],
            },
        ),
        paths["manifest"],
    )
    for key, value in paths.items():
        print(f"{key}: {value}")
    print(f"status: {summary['status']}")
    print(f"adapter_ready: {summary['adapter_ready']}")
    print(f"quality_gate_input_ready: {summary['quality_gate_input_ready']}")
    if summary["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
