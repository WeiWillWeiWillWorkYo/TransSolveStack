"""Run the CSR Transformer ranker quality gate."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transsolvestack.policies.csr_selector_model_eval import (
    evaluate_csr_selector_models_from_files,
    write_csr_selector_model_eval_report,
    write_csr_selector_model_eval_rows,
    write_csr_selector_model_eval_schema,
    write_csr_selector_model_eval_summary,
)
from transsolvestack.profiling.provenance import (
    CORE_CSR_TRANSFORMER_QUALITY_GATE_PROVENANCE_FILES,
    build_artifact_manifest,
    git_commit_or_unknown,
    write_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-summary",
        default="runs/phase1_csr_transformer_ready/combined_csr_learning_summary.json",
    )
    parser.add_argument(
        "--baseline-predictions",
        default="runs/phase1_csr_transformer_ready/combined_csr_baseline_predictions.jsonl",
    )
    parser.add_argument(
        "--ranker-summary",
        default="runs/phase1_csr_transformer_ranker/csr_transformer_ranker_summary.json",
    )
    parser.add_argument(
        "--ranker-predictions",
        default="runs/phase1_csr_transformer_ranker/csr_transformer_ranker_predictions.jsonl",
    )
    parser.add_argument("--min-eval-oracle-requests", type=int, default=4)
    parser.add_argument("--min-runtime-oracle-top1-accuracy", type=float, default=0.5)
    parser.add_argument("--min-runtime-profiled-success-rate", type=float, default=0.8)
    parser.add_argument("--out", default="runs/phase1_csr_transformer_quality_gate")
    args = parser.parse_args()

    export = evaluate_csr_selector_models_from_files(
        args.baseline_summary,
        args.baseline_predictions,
        args.ranker_summary,
        args.ranker_predictions,
        min_eval_oracle_requests=args.min_eval_oracle_requests,
        min_runtime_oracle_top1_accuracy=args.min_runtime_oracle_top1_accuracy,
        min_runtime_profiled_success_rate=args.min_runtime_profiled_success_rate,
    )
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "rows": output / "csr_transformer_quality_gate_rows.jsonl",
        "summary": output / "csr_transformer_quality_gate_summary.json",
        "schema": output / "csr_transformer_quality_gate_schema.json",
        "report": output / "csr_transformer_quality_gate_report.md",
        "manifest": output / "artifact_manifest.json",
    }
    write_csr_selector_model_eval_rows(export.rows, paths["rows"])
    write_csr_selector_model_eval_summary(export.summary, paths["summary"])
    write_csr_selector_model_eval_schema(export.schema, paths["schema"])
    write_csr_selector_model_eval_report(export, paths["report"])
    write_manifest(
        build_artifact_manifest(
            artifact_kind="csr_transformer_quality_gate",
            command="scripts/tss_csr_transformer_quality_gate.py",
            tracked_files=CORE_CSR_TRANSFORMER_QUALITY_GATE_PROVENANCE_FILES,
            metadata={
                "git_commit": git_commit_or_unknown(),
                "status": export.summary.status,
                "schema_version": export.summary.schema_version,
                "baseline_model_id": export.summary.baseline_model_id,
                "challenger_model_id": export.summary.challenger_model_id,
                "best_offline_model_id": export.summary.best_offline_model_id,
                "challenger_runtime_eligible": export.summary.challenger_runtime_eligible,
                "runtime_selector_changed": export.summary.runtime_selector_changed,
                "num_eval_predictions": export.summary.num_eval_predictions,
                "num_eval_oracle_requests": export.summary.num_eval_oracle_requests,
                "min_required_eval_oracle_requests": (
                    export.summary.min_required_eval_oracle_requests
                ),
            },
        ),
        paths["manifest"],
    )
    for key, value in paths.items():
        print(f"{key}: {value}")
    print(f"status: {export.summary.status}")
    print(f"challenger_runtime_eligible: {export.summary.challenger_runtime_eligible}")
    if export.summary.status != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
