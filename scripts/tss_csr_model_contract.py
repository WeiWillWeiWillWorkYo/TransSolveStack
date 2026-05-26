"""Export CSR learned-selector request/prediction contract artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transsolvestack.policies.csr_model_contract import (
    build_csr_model_contract_export,
    write_csr_model_contract_report,
    write_csr_model_contract_schema,
    write_csr_model_contract_summary,
    write_csr_model_predictions,
    write_csr_model_requests,
    write_csr_model_targets,
)
from transsolvestack.profiling.provenance import (
    CORE_CSR_MODEL_CONTRACT_PROVENANCE_FILES,
    build_artifact_manifest,
    git_commit_or_unknown,
    write_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--learning-rows",
        default="runs/phase1_csr_learning_readiness/csr_learning_rows.jsonl",
    )
    parser.add_argument(
        "--baseline-predictions",
        default="runs/phase1_csr_learning_readiness/csr_baseline_predictions.jsonl",
    )
    parser.add_argument("--out", default="runs/phase1_csr_model_contract")
    args = parser.parse_args()

    export = build_csr_model_contract_export(
        args.learning_rows,
        args.baseline_predictions,
    )
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "requests": output / "csr_model_requests.jsonl",
        "targets": output / "csr_model_targets.jsonl",
        "predictions": output / "csr_model_predictions.jsonl",
        "summary": output / "csr_model_contract_summary.json",
        "schema": output / "csr_model_contract_schema.json",
        "report": output / "csr_model_contract_report.md",
        "manifest": output / "artifact_manifest.json",
    }
    write_csr_model_requests(export.requests, paths["requests"])
    write_csr_model_targets(export.targets, paths["targets"])
    write_csr_model_predictions(export.predictions, paths["predictions"])
    write_csr_model_contract_summary(export.summary, paths["summary"])
    write_csr_model_contract_schema(paths["schema"])
    write_csr_model_contract_report(export, paths["report"])
    write_manifest(
        build_artifact_manifest(
            artifact_kind="csr_model_contract_export",
            command="scripts/tss_csr_model_contract.py",
            tracked_files=CORE_CSR_MODEL_CONTRACT_PROVENANCE_FILES,
            metadata={
                "git_commit": git_commit_or_unknown(),
                "status": export.summary.status,
                "schema_version": export.summary.schema_version,
                "prediction_source": export.summary.prediction_source,
                "model_required": export.summary.model_required,
                "runtime_selector_changed": export.summary.runtime_selector_changed,
                "num_requests": export.summary.num_requests,
                "num_predictions": export.summary.num_predictions,
                "eval_oracle_top1_accuracy": export.summary.eval_oracle_top1_accuracy,
                "eval_profiled_selection_rate": export.summary.eval_profiled_selection_rate,
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
