"""Export CSR selector learning rows and a baseline evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transsolvestack.policies.csr_learning import (
    build_csr_learning_export_from_selector_rows,
    write_csr_baseline_predictions,
    write_csr_learning_report,
    write_csr_learning_rows,
    write_csr_learning_schema,
    write_csr_learning_summary,
)
from transsolvestack.profiling.provenance import (
    CORE_CSR_LEARNING_READINESS_PROVENANCE_FILES,
    build_artifact_manifest,
    git_commit_or_unknown,
    write_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selector-rows",
        default="runs/phase1_csr_selector_readiness/csr_selector_rows.jsonl",
    )
    parser.add_argument("--eval-fraction", type=float, default=0.25)
    parser.add_argument("--min-eval-matrices", type=int, default=2)
    parser.add_argument("--out", default="runs/phase1_csr_learning_readiness")
    args = parser.parse_args()

    export = build_csr_learning_export_from_selector_rows(
        args.selector_rows,
        eval_fraction=args.eval_fraction,
        min_eval_matrices=args.min_eval_matrices,
    )
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "rows": output / "csr_learning_rows.jsonl",
        "predictions": output / "csr_baseline_predictions.jsonl",
        "summary": output / "csr_learning_summary.json",
        "schema": output / "csr_learning_schema.json",
        "report": output / "csr_learning_report.md",
        "manifest": output / "artifact_manifest.json",
    }
    write_csr_learning_rows(export.rows, paths["rows"])
    write_csr_baseline_predictions(export.predictions, paths["predictions"])
    write_csr_learning_summary(export.summary, paths["summary"])
    write_csr_learning_schema(paths["schema"])
    write_csr_learning_report(export, paths["report"])
    write_manifest(
        build_artifact_manifest(
            artifact_kind="csr_learning_readiness_export",
            command="scripts/tss_csr_learning_readiness.py",
            tracked_files=CORE_CSR_LEARNING_READINESS_PROVENANCE_FILES,
            metadata={
                "git_commit": git_commit_or_unknown(),
                "status": export.summary.status,
                "schema_version": export.summary.schema_version,
                "baseline_id": export.summary.baseline_id,
                "num_rows": export.summary.num_rows,
                "num_train_rows": export.summary.num_train_rows,
                "num_eval_rows": export.summary.num_eval_rows,
                "num_eval_predictions": export.summary.num_eval_predictions,
                "eval_oracle_top1_accuracy": export.summary.eval_oracle_top1_accuracy,
                "eval_profiled_success_rate": export.summary.eval_profiled_success_rate,
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
